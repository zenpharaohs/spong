"""Is a far critical point real, or removable by a perturbation below eps?

The certification question this serves is not "this portrait is the portrait
of the given system" but "this portrait is the portrait of a system
numerically indistinguishable from the given one" -- with the required
perturbation reported and its size attested.  A critical point that a
sub-epsilon perturbation of (f, g, mu) removes was never a feature of
anything the user could have specified.

THE ALGEBRA.  u = C - B^2/A, so

  * C - u(b) = B(b)^2 / A(b)          -- exact, a ratio of rationals
  * u' = -B*N/A^2 with N = A'B - 2B'A -- critical points are roots of B*N
  * at a root of B, N = -2B'A, and A > 0, so N vanishes there IFF B' = 0

Hence a B-saddle (a root of B, sitting at u = C, always a local maximum of
the backbone loss) merges with an adjacent critical point EXACTLY when B
acquires a double root.  That is the fold, and it is a condition on B alone.

WHY THAT IS THE CONVENIENT CASE.  B(b) = sum_j g_j b^j E[f x^j] is LINEAR in
f, so a perturbation df moves B's coefficients linearly while leaving A -- and
therefore psi-positivity, the Gram structure and the Hankel moment conditions
-- completely untouched.  The minimal removing perturbation is a distance to
the discriminant variety pulled back through a linear map, with dg = dmu = 0.

This reports the exact quantities.  It does not yet SOLVE the minimisation;
the first-order estimate below is a lower bound on how far B must move, and
is meant to say whether the question is worth pursuing per case.

    python scripts/perturbation_probe.py 634753038 --mode directed
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

EPS = 2.220446049250313e-16


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

    A = [Fraction(c) for c in m.alpha]          # ascending powers
    B = [Fraction(c) for c in m.beta]
    C = Fraction(m.C)
    dA, dB = len(A)-1, len(B)-1
    print(f"  deg A {dA}   deg B {dB}   d_eff {atlas.effective_degree(m)}")
    print(f"  C = {float(C):.12g}")

    # u at infinity: C - beta^2/alpha when deg A == 2 deg B, else C.
    if dA == 2*dB:
        u_inf = C - B[-1]*B[-1]/A[-1]
        print(f"  u_inf = C - beta^2/alpha = {float(u_inf):.12g}"
              f"   (C - u_inf = {float(C-u_inf):.6g})")
    elif dA > 2*dB:
        u_inf = C
        print(f"  u_inf = C exactly (deg A > 2 deg B)")
    else:
        u_inf = None
        print("  u_inf undefined for this degree pattern")

    ev = lambda p, x: sum(c * x**k for k, c in enumerate(p))

    e = sturm.enumerate_critical_points(m)
    print(f"\n  {'kind':<8}{'b':>16}{'a':>13}{'C - u = B^2/A':>18}"
          f"{'vs eps*|C|':>14}")
    print("  " + "-"*69)
    pts = sorted(e.points, key=lambda q: q.b)
    floor = EPS * abs(float(C))
    for q in pts:
        b = Fraction(q.b) if not isinstance(q.b, float) else Fraction(q.b)
        Bv, Av = ev(B, b), ev(A, b)
        depth = (Bv*Bv/Av) if Av != 0 else None
        d = float(depth) if depth is not None else float("nan")
        flag = "  BELOW EPS" if d < floor else ""
        print(f"  {q.kind:<8}{float(q.b):>16.8g}{float(q.a):>13.4g}"
              f"{d:>18.6g}{floor:>14.3g}{flag}")

    # How close is B to having a double root?  disc(B) = 0 is the fold.
    Bf = np.asarray([float(c) for c in B])[::-1]        # descending for numpy
    roots = np.roots(Bf) if len(Bf) > 1 else np.array([])
    if roots.size >= 2:
        print("\n  roots of B (the B-saddles are the real ones):")
        for r in np.sort_complex(roots):
            tag = " REAL" if abs(r.imag) < 1e-12*max(1.0, abs(r)) else ""
            print(f"    {r.real:>18.10g} {r.imag:>+18.10g}i{tag}")
        gaps = []
        for i in range(len(roots)):
            for j in range(i+1, len(roots)):
                gaps.append((abs(roots[i]-roots[j]), i, j))
        gap, i, j = min(gaps)
        scale = max(abs(roots[i]), abs(roots[j]), 1.0)
        print(f"\n  closest pair of roots: |gap| = {gap:.6g}"
              f"   relative {gap/scale:.6g}")
        print("  (a double root of B is exactly the fold that removes a"
              " B-saddle;\n   B is LINEAR in f, so dg = dmu = 0 suffices)")

    # First-order feel for how far B's coefficients must move: shifting the
    # outermost real root of B by the gap costs roughly |dB| ~ |B'(r)| * gap.
    real = [r.real for r in roots if abs(r.imag) < 1e-9*max(1.0, abs(r))]
    if real:
        far = max(real, key=abs)
        Bp = np.polyder(Bf)
        slope = abs(np.polyval(Bp, far))
        cscale = max(abs(c) for c in Bf)
        print(f"\n  outermost real root of B: b = {far:.10g}")
        print(f"    |B'(b)| = {slope:.6g}   max|B coeff| = {cscale:.6g}")
        print(f"    moving that root by 1 ulp of b costs "
              f"|dB| ~ {slope*abs(far)*EPS:.6g}"
              f"   relative to coeffs: {slope*abs(far)*EPS/cscale:.6g}")
    # How far out is the backbone loss still DISTINGUISHABLE from its limit?
    # u - u_inf = R/(alpha*A) with R := beta^2 A - alpha B^2.  The leading
    # terms cancel identically, so deg R = 2d-1 and u - u_inf ~ c/b: the decay
    # is only FIRST ORDER, and the interval where it exceeds eps*|C| reaches
    # to b ~ c/(eps*|C|).  Whether that is a useful scan bound or an
    # astronomical one depends entirely on c, so measure it.
    if dA == 2*dB and u_inf is not None:
        alpha, beta = A[-1], B[-1]
        R = [beta*beta*A[k] if k < len(A) else Fraction(0)
             for k in range(len(A))]
        Bsq = [Fraction(0)]*(2*dB+1)
        for i, bi in enumerate(B):
            for j, bj in enumerate(B):
                Bsq[i+j] += bi*bj
        for k, v in enumerate(Bsq):
            R[k] -= alpha*v
        while len(R) > 1 and R[-1] == 0:
            R.pop()
        print(f"\n  R = beta^2 A - alpha B^2 has degree {len(R)-1} "
              f"(deg A = {dA}, so the leading terms cancel as expected)")
        print(f"    leading coeff of R = {float(R[-1]):.6g}")
        if len(R)-1 >= 1:
            c = float(R[-1]) / (float(alpha)**2)
            reach = abs(c) / (EPS*abs(float(C))) if C != 0 else float('inf')
            print(f"    u - u_inf ~ {c:.6g} / b")
            print(f"    distinguishable from u_inf out to |b| ~ {reach:.6g}")

        print(f"\n  EXACT u(b) - u_inf at each critical point:")
        print(f"  {'kind':<8}{'b':>16}{'u - u_inf':>24}{'/ eps|C|':>14}")
        print("  " + "-"*62)
        for q in pts:
            b = Fraction(q.b)
            Av, Bv = ev(A, b), ev(B, b)
            if Av == 0:
                continue
            diff = (C - Bv*Bv/Av) - u_inf          # exact rational
            ratio = float(diff) / floor if floor else float('inf')
            mark = "  INDISTINGUISHABLE" if abs(ratio) < 1.0 else ""
            print(f"  {q.kind:<8}{float(q.b):>16.8g}{float(diff):>24.12g}"
                  f"{ratio:>14.4g}{mark}")

    # A real root of u' can leave in TWO ways, and they want different
    # perturbations.  It can COLLIDE with another (a double root of B*N, the
    # saddle-minimum fold, disc = 0), or it can ESCAPE TO INFINITY -- the
    # leading coefficient vanishing so the degree drops.  The fragile far
    # saddles are the second kind: they have no fragile partner to annihilate
    # with, and what they are close to is the critical point at infinity.
    #
    # N = A'B - 2B'A has leading terms 2d*alpha*beta*b^(3d-1) from A'B and
    # 2d*beta*alpha*b^(3d-1) from 2B'A, which cancel IDENTICALLY -- the same
    # structural cancellation as in R.  So a root escaping is the NEXT
    # coefficient going to zero, and since N is linear in f that is a single
    # linear equation in df: a least-norm solve, not a discriminant
    # minimisation.
    Ap = [k*A[k] for k in range(1, len(A))]
    Bp = [k*B[k] for k in range(1, len(B))]

    def mul(p, q):
        out = [Fraction(0)]*(len(p)+len(q)-1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                out[i+j] += pi*qj
        return out

    t1, t2 = mul(Ap, B), mul(Bp, A)
    N = [(t1[k] if k < len(t1) else Fraction(0))
         - 2*(t2[k] if k < len(t2) else Fraction(0))
         for k in range(max(len(t1), len(t2)))]
    while len(N) > 1 and N[-1] == 0:
        N.pop()
    nominal = dA - 1 + dB
    print(f"\n  N = A'B - 2B'A: degree {len(N)-1}"
          f"  (nominal {nominal}, so {nominal-(len(N)-1)} leading"
          f" coefficient(s) cancel)")
    cscaleN = max(abs(float(c)) for c in N)
    for k in range(len(N)-1, max(len(N)-4, -1), -1):
        print(f"    N[{k}] = {float(N[k]):>16.8g}"
              f"   relative {abs(float(N[k]))/cscaleN:>12.4g}")

    # N's top coefficient as a LINEAR FUNCTIONAL of f, so the least-norm df
    # that sends it to zero is immediate.  B_j = g_j * sum_i f_i mu_{i+j},
    # and A does not involve f at all.
    try:
        g = [Fraction(c) for c in m.g]
        mu = [Fraction(c) for c in m.mu]
        nf = len(m.f)
        grad = []
        for i in range(nf):
            Bi = [g[j]*mu[i+j] if i+j < len(mu) else Fraction(0)
                  for j in range(len(g))]
            Bpi = [k*Bi[k] for k in range(1, len(Bi))]
            u1, u2 = mul(Ap, Bi), mul(Bpi, A)
            top = len(N)-1
            grad.append(float((u1[top] if top < len(u1) else 0)
                              - 2*(u2[top] if top < len(u2) else 0)))
        gn2 = sum(x*x for x in grad)
        if gn2 > 0:
            step = float(N[-1])/gn2
            df = [-step*x for x in grad]
            fscale = max(abs(float(c)) for c in m.f)
            norm = max(abs(x) for x in df)
            print(f"\n  least-norm df sending N's top coefficient to zero"
                  f" (dg = dmu = 0):")
            print(f"    ||df||_inf = {norm:.6g}"
                  f"   relative to ||f||_inf = {fscale:.6g}:"
                  f" {norm/fscale:.6g}")
            print(f"    in ulps of f: {norm/(fscale*EPS):.6g}")
    except Exception as exc:                            # noqa: BLE001
        print(f"\n  (df solve unavailable: {type(exc).__name__}: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
