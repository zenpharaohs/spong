"""Do same-ray stable tails stay provably separated?

Branches escaping along the SAME diagonal ray differ only in the label

    K0 = b^2 - d a^2 + a1 a / sqrt(d)   (plus a c*ln|a| term, see below)

and their separation in b is dK0 / (2 sqrt(d) a) -- a power law, so it
persists and is exactly what a user zooming in would check.  Converting the
enclosure to the label gives dK0 ~ c, CONSTANT in a, where c is the
coefficient in dPhi/da ~ c/a.  So the tails are provably distinct and
correctly ordered iff

    c  <<  min over same-ray pairs of |dK0|

Non-crossing itself is a theorem (distinct orbits never meet), so disjoint
K0 intervals discharge the contact check on the tail rather than performing
it -- which is why this can be stronger than scanning polylines, not merely
cheaper.

Launches from the materialized saddle stubs rather than a full portrait, so
it costs an enumeration and some ascent, not an audit.

    python k0_separation.py tricky-d11
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "src")
from spong import atlas, model, sturm, zoo          # noqa: E402


def ascend(k, a, b, ds, steps, R):
    out = [(a, b)]
    for _ in range(steps):
        try:
            a, b = k.normalized_step(a, b, +ds, 8)
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            break
        a, b = float(a), float(b)
        if not (a == a and b == b):
            break
        out.append((a, b))
        if abs(a) > R or abs(b) > R:
            break
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="tricky-d11")
    ap.add_argument("--reach", type=float, default=2e4)
    ap.add_argument("--ds", type=float, default=0.05)
    args = ap.parse_args(argv)

    case = zoo.get(args.case)
    D = max(len(case.f) - 1, len(case.g) - 1)
    mu = (model.moments_uniform01 if case.moment_dist == "uniform01"
          else model.moments_normal01)(2 * D + 1)
    m = model.build(case.f, case.g, mu)
    Ac = np.asarray([float(x) for x in m.alpha])[::-1]
    Bc = np.asarray([float(x) for x in m.beta])[::-1]
    Ap, Bp = np.polyder(Ac), np.polyder(Bc)
    dA, dB = len(Ac) - 1, len(Bc) - 1
    d = atlas.effective_degree(m)
    a1 = Ac[1] / Ac[0]
    k = getattr(m, "_native_kernel", None)
    if k is None:
        raise SystemExit("no C core")

    print(f"{case.name}: d_eff={d}  a1={a1:.6g}  "
          f"offset c0={-a1/(2*d):.6g}  reach={args.reach:g}  ds={args.ds:g}"
          f"  step cap={int(8.0*args.reach/args.ds)}")

    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))

    def dPhi_da(a, b, sray):
        """sray = sign(a*b): b ~ sray*sqrt(d)*a, and the offset term flips
        with the ray -- b^2 = d a^2 - a1 a/(sray sqrt(d)) + K0."""
        A = np.polyval(Ac, b); B = np.polyval(Bc, b)
        P = b * np.polyval(Ap, b) - dA * A
        Q = b * np.polyval(Bp, b) - dB * B
        ga = 2.0 * (a * A - B)
        if ga == 0.0:
            return float("inf")
        return 2.0 * a * (a * P - 2.0 * Q) / ga + sray * a1 / math.sqrt(d)

    rows = []
    for p in e.points:
        if p.kind != "saddle":
            continue
        for s in p.stubs:
            if s.manifold != "stable":
                continue
            cur = np.asarray(s.curve, dtype=float)
            a0, b0 = cur[-1]
            # Step cap must SCALE with the work requested, or halving ds
            # halves the distance reached instead of refining the same path
            # -- which confounds a ds-refinement test with a reach change.
            cap = int(8.0 * args.reach / args.ds)
            path = ascend(k, a0, b0, args.ds, cap, args.reach)
            A_, B_ = path[:, 0], path[:, 1]
            if np.max(np.abs(B_)) < 0.3 * args.reach:
                rows.append((float(p.b), s.b_direction, None, None, None,
                             float(np.max(np.abs(A_))),
                             float(np.max(np.abs(B_)))))
                continue
            a, b = A_[-1], B_[-1]
            sray = math.copysign(1.0, a * b)
            # c measured directly from the drift, c = a * dPhi/da far out.
            cs = [A_[j] * dPhi_da(A_[j], B_[j], sray)
                  for j in range(int(0.6 * len(A_)), len(A_), max(1, len(A_)//40))]
            cmeas = float(np.median([x for x in cs if math.isfinite(x)]))
            # K0 with the log term folded in, so it settles instead of drifting.
            K0 = (b * b - d * a * a + sray * a1 * a / math.sqrt(d)
                  - cmeas * math.log(abs(a)))
            rows.append((float(p.b), s.b_direction,
                         (math.copysign(1, a), math.copysign(1, b)),
                         K0, cmeas, a, b))

    print()
    print(f"{'saddle b':>11}{'dir':>5}{'ray':>9}{'a_far':>11}{'b_far':>11}"
          f"{'K0':>16}{'c':>12}")
    print("-" * 76)
    live = []
    for sb, dr, ray, K0, cm, a, b in rows:
        if ray is None:
            print(f"{sb:>11.5f}{dr:>5}{'regime 2':>9}{a:>11.4g}{b:>11.4g}"
                  f"{'—':>16}{'—':>12}")
            continue
        print(f"{sb:>11.5f}{dr:>5}  ({int(ray[0]):+d},{int(ray[1]):+d})"
              f"{a:>11.4g}{b:>11.4g}{K0:>16.6f}{cm:>12.4g}")
        live.append((sb, dr, ray, K0, cm))

    print()
    any_pair = False
    for (s1, d1, r1, K1, c1), (s2, d2, r2, K2, c2) in combinations(live, 2):
        if r1 != r2:
            continue
        any_pair = True
        dK = abs(K1 - K2)
        c = max(abs(c1), abs(c2))
        margin = dK / c if c else float("inf")
        verdict = ("SEPARATED" if margin > 10 else
                   "marginal" if margin > 1 else "NOT RESOLVED")
        print(f"saddles b={s1:.5f}({d1:+d}) and b={s2:.5f}({d2:+d})"
              f" share ray ({int(r1[0]):+d},{int(r1[1]):+d})")
        print(f"    |dK0| = {dK:.6g}   c = {c:.4g}"
              f"   margin = {margin:.3g}x   {verdict}")
        for A in (1e2, 1e4, 1e6):
            print(f"      at a={A:>8.0g}: separation db = "
                  f"{dK/(2*math.sqrt(d)*A):.3e}")
    if not any_pair:
        print("no two live branches share a ray in this case")
    print()
    print("margin = |dK0| / c.  Comfortably >1 means the enclosure intervals")
    print("are disjoint, so the tails are provably ordered and non-crossing")
    print("on [handoff, inf) -- a reach the polyline scan cannot have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
