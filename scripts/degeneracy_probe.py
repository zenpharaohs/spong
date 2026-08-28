"""How close is this model to a degeneracy, and which one?

Three separate boundaries get conflated when a portrait is badly conditioned:

  * NON-MORSE: a double root of B*N, i.e. two critical points about to merge.
    Measured by the gaps between consecutive roots and by |u''| at each
    critical point -- u'' -> 0 is the fold.
  * NOT PSI-NICE: A(b) -> 0 somewhere.  A > 0 is what makes the backbone
    a* = B/A finite and the whole level-curve symmetry work.  A near-zero
    sends a* to infinity, which is what a minimum at a ~ -4.4e7 suggests.
  * ILL-CONDITIONED BUT FINE: neither of the above, just a wide dynamic range.

They want different responses, so name which one before proposing anything.

    python scripts/degeneracy_probe.py 953953598 --mode directed
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

from spong import atlas, sturm                          # noqa: E402
from qualify import directed_model, random_model        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    built = generate(random.Random(args.seed), args.degree)
    if built is None or built[0] is None:
        raise SystemExit("generator declined this seed")
    m, spec = built
    print(f"seed {args.seed}   {spec}")

    A = np.asarray([float(c) for c in m.alpha])[::-1]      # descending
    B = np.asarray([float(c) for c in m.beta])[::-1]
    Ap, App = np.polyder(A), np.polyder(np.polyder(A))
    Bp = np.polyder(B)

    e = sturm.enumerate_critical_points(m)
    print(f"  morse={e.morse}  psi_positive={e.psi_positive}  "
          f"alternates={e.alternates}")

    # --- PSI-NICENESS: how close does A come to zero, and where? -----------
    print("\n  A(b) = E[g(bx)^2], the psi-positivity witness:")
    crit = np.roots(Ap)
    crit = np.sort(crit[np.abs(crit.imag) < 1e-9*np.maximum(1, np.abs(crit))].real)
    Amax = max(abs(float(np.polyval(A, r))) for r in crit) if crit.size else 1.0
    Amax = max(Amax, abs(float(A[0])), 1.0)
    worst = None
    for r in crit:
        v = float(np.polyval(A, r))
        if worst is None or v < worst[1]:
            worst = (float(r), v)
        print(f"    local extremum b = {float(r):>16.8g}   "
              f"A = {v:>14.6g}   A/scale = {v/Amax:>11.4g}")
    print(f"    A(0) = {float(np.polyval(A, 0.0)):.6g}"
          f"   leading A = {float(A[0]):.6g}")
    if worst is not None:
        print(f"    MINIMUM of A over its critical points: "
              f"A = {worst[1]:.6g} at b = {worst[0]:.8g}")
        print(f"    -> psi margin (A_min / A_scale) = {worst[1]/Amax:.6g}")

    # --- MORSE: u'' at each critical point, and root gaps ------------------
    # u = C - B^2/A;  a fold is u'' -> 0.  det H = -2 A u'' (the corrected
    # theorem in MORSE_TECHNOLOGY_INVENTORY), so u'' -> 0 is exactly the
    # Hessian degenerating.
    print("\n  per critical point:")
    print(f"    {'kind':<7}{'b':>16}{'a':>14}{'A(b)':>13}"
          f"{'u\"':>14}{'det H = -2A u\"':>16}")
    pts = sorted(e.points, key=lambda q: float(q.b))
    for q in pts:
        b = float(q.b)
        Av = float(np.polyval(A, b))
        # u'' by exact-ish finite difference on the rational function
        h = max(abs(b), 1.0) * 1e-5
        uf = lambda x: float(m.C) - np.polyval(B, x)**2/np.polyval(A, x)
        u2 = (uf(b+h) - 2*uf(b) + uf(b-h)) / (h*h)
        print(f"    {q.kind:<7}{b:>16.8g}{float(q.a):>14.6g}{Av:>13.5g}"
              f"{u2:>14.5g}{-2*Av*u2:>16.5g}")

    # --- g(0): the check that is binary when it should be a margin --------
    # A(0) = E[g(0*x)^2] = g(0)^2 and B(0) = g(0)*E[f], so
    #     a*(0) = B(0)/A(0) = E[f]/g(0).
    # g(0) = 0 makes A(0) = 0, and then L(a, 0) = E[f^2] = C for EVERY a --
    # the whole line b = 0 is a flat line of critical points, i.e. non-Morse.
    # A strictly positive but tiny g(0) passes the certification and leaves
    # the backbone shooting off to E[f]/g(0), which is what a minimum at
    # a ~ -4.4e7 near b ~ 0 actually is.
    #
    # The scale-invariant margin is |g(0)| / ||g||: under g -> sigma*g (one
    # of the three normalisations available) it is unchanged, so it is a
    # dimensionless distance to the dead-neuron degeneracy rather than a
    # units artifact.
    try:
        gc = [float(c) for c in m.g]
        gscale = max(abs(c) for c in gc) or 1.0
        g0 = gc[0]
        print(f"\n  g(0) = {g0:.6g}   ||g||_inf = {gscale:.6g}")
        print(f"    MARGIN |g(0)|/||g|| = {abs(g0)/gscale:.6g}"
              f"   ({abs(g0)/(gscale*2.220446049250313e-16):.4g} ulps)")
        print(f"    A(0) = g(0)^2 = {g0*g0:.6g}")
        Ef = float(sum(Fraction(c)*Fraction(mu)
                       for c, mu in zip(m.f, m.mu)))
        print(f"    E[f] = {Ef:.6g}   ->  a*(0) = E[f]/g(0) = "
              f"{(Ef/g0 if g0 else float('inf')):.6g}")
    except Exception as exc:                            # noqa: BLE001
        print(f"\n  (g(0) margin unavailable: {type(exc).__name__}: {exc})")

    # --- root gaps of B*N -------------------------------------------------
    Nn = np.polysub(np.polymul(Ap, B), 2*np.polymul(Bp, A))
    for name, poly in (("B", B), ("N", Nn)):
        rr = np.roots(poly) if len(poly) > 1 else np.array([])
        rr = np.sort(rr[np.abs(rr.imag) < 1e-9*np.maximum(1, np.abs(rr))].real)
        if rr.size >= 2:
            gaps = np.diff(rr)
            k = int(np.argmin(np.abs(gaps)))
            print(f"\n  {name}: {rr.size} real roots, closest pair "
                  f"{rr[k]:.8g} and {rr[k+1]:.8g}, gap {gaps[k]:.6g}")
        elif rr.size:
            print(f"\n  {name}: {rr.size} real root at {rr[0]:.8g}")
        else:
            print(f"\n  {name}: no real roots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
