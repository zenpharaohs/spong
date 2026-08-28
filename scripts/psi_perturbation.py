"""Can a tiny perturbation raise the psi minimum?

A(b) = E[g(bx)^2] involves ONLY g and mu -- f does not appear.  So the psi
margin is controlled by a set of parameters disjoint from the f-linear
handles that move B and N.  Two independent knobs.

At a local minimum b0 of A we have A'(b0) = 0, so by the envelope theorem the
movement of b0 contributes nothing to first order and the gradient of A(b0) is
the partial derivative at FIXED b:

    dA(b0)/dg_k = 2 * b0^k * E[x^k g(b0 x)]
    dA(b0)/dmu_n = sum_{i+j=n} g_i g_j b0^n

Note the identity <g, v> = 2 E[g(b0 x)^2] = 2 A(b0), giving ||v|| >= 2A(b0)/||g||
-- a bound that goes weak exactly when A(b0) is small, which is when the
answer matters, so measure ||v|| directly.

WHY THIS IS THE INTERESTING DEGENERACY.  A(b0) = 0 forces B(b0) = 0 as well
(g(b0 x) = 0 a.s. implies E[f g(b0 x)] = 0), so u = C - B^2/A stays FINITE.
The degeneracy is invisible in the backbone LOSS and shows up entirely in the
backbone's POSITION, a* = B/A.  That is why it never presented as a Morse
problem while making the geometry brutal.

Prefer dg over dmu: mu must remain a valid moment sequence (Hankel PSD), so
dmu is a constrained perturbation, while dg is free.

    python scripts/psi_perturbation.py 953953598 --mode directed
    python scripts/psi_perturbation.py 953953598 --mode directed --target 0.15
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

from qualify import directed_model, random_model        # noqa: E402

EPS = 2.220446049250313e-16


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.146,
                    help="psi margin to aim for (default: the median of the "
                         "CERTIFIED group in the directed ensemble)")
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    built = generate(random.Random(args.seed), args.degree)
    if built is None or built[0] is None:
        raise SystemExit("generator declined this seed")
    m, spec = built
    print(f"seed {args.seed}   {spec}")

    g = np.asarray([float(c) for c in m.g])
    mu = np.asarray([float(c) for c in m.mu])
    A = np.asarray([float(c) for c in m.alpha])[::-1]      # descending
    Ap = np.polyder(A)

    # Where is A smallest?  Over its real critical points, and at 0.
    rr = np.roots(Ap) if len(Ap) > 1 else np.array([])
    rr = rr[np.abs(rr.imag) < 1e-9*np.maximum(1.0, np.abs(rr))].real
    cands = list(rr) + [0.0]
    vals = [(float(b), float(np.polyval(A, b))) for b in cands]
    b0, A0 = min(vals, key=lambda t: t[1])
    Ascale = max(max(abs(v) for _, v in vals), abs(float(A[0])), 1.0)
    print(f"  A minimised at b0 = {b0:.10g}   A(b0) = {A0:.6g}")
    print(f"  A scale = {Ascale:.6g}   ->  psi margin = {A0/Ascale:.6g}")
    print(f"  ||g||_inf = {max(abs(c) for c in g):.6g}")

    # Gradient with respect to g, by the envelope theorem.
    ng = len(g)
    v = np.zeros(ng)
    for k in range(ng):
        # E[x^k g(b0 x)] = sum_j g_j b0^j mu_{k+j}
        s = 0.0
        for j in range(ng):
            if k+j < len(mu):
                s += g[j] * (b0**j) * mu[k+j]
        v[k] = 2.0 * (b0**k) * s
    vn2 = float(v @ v)
    print(f"\n  gradient dA(b0)/dg:")
    for k in range(ng):
        print(f"    d/dg_{k}  {v[k]:>16.8g}")
    print(f"    ||v||_2 = {np.sqrt(vn2):.6g}")
    print(f"    identity check <g,v> = {float(g @ v):.8g}"
          f"   vs 2A(b0) = {2*A0:.8g}")
    print(f"    Cauchy-Schwarz floor 2A(b0)/||g||_2 = "
          f"{2*A0/max(float(np.linalg.norm(g)), 1e-300):.6g}")

    if vn2 <= 0:
        print("\n  gradient vanishes -- no first-order handle here")
        return 0

    # Least-norm dg raising A(b0) to the target margin.
    want = args.target * Ascale
    delta = want - A0
    dg = delta * v / vn2
    gscale = max(abs(c) for c in g) or 1.0
    rel = max(abs(c) for c in dg) / gscale
    print(f"\n  to reach psi margin {args.target:g} (A(b0) = {want:.6g}):")
    print(f"    least-norm dg, ||dg||_inf = {max(abs(c) for c in dg):.6g}")
    print(f"    relative to ||g||_inf: {rel:.6g}   ({rel/EPS:.4g} ulps)")
    print(f"    {'SUB-EPSILON -- free' if rel < EPS else 'NOT sub-epsilon'}")

    # What does that dg actually do?  Recompute A and its true minimum.
    g2 = g + dg
    A2 = np.zeros(2*ng-1)
    for i in range(ng):
        for j in range(ng):
            if i+j < len(mu):
                A2[i+j] += g2[i]*g2[j]*mu[i+j]
    A2d = A2[::-1]
    Ap2 = np.polyder(A2d)
    rr2 = np.roots(Ap2) if len(Ap2) > 1 else np.array([])
    rr2 = rr2[np.abs(rr2.imag) < 1e-9*np.maximum(1.0, np.abs(rr2))].real
    vals2 = [(float(b), float(np.polyval(A2d, b)))
             for b in list(rr2) + [0.0]]
    b0n, A0n = min(vals2, key=lambda t: t[1])
    Ascale2 = max(max(abs(x) for _, x in vals2), abs(float(A2d[0])), 1.0)
    print(f"\n  after the perturbation (recomputed, not linearised):")
    print(f"    A minimised at b0 = {b0n:.10g}   A(b0) = {A0n:.6g}")
    print(f"    psi margin {A0n/Ascale2:.6g}"
          f"   (was {A0/Ascale:.6g}, aimed at {args.target:g})")
    if A0n <= 0:
        print("    WARNING: A now has a zero -- the step overshot into "
              "non-Morse; halve it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
