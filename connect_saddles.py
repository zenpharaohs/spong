"""Construct saddle connections deliberately, by Newton on coefficients.

A connection is codimension one, so it is measure zero and no amount of
sampling produces one.  But it is a ROOT, and root finding is better behaved
than minimizing nearest-approach distance.  This drives several separations
to zero at once along a central path, so the connections arrive together and
nothing crosses zero until the end -- which is what prevents attachment
flips en route.

SEPARATION FUNCTION.  For a candidate connection s' -> s (needs u(s') > u(s)),
pick a rational level c strictly between them.  L is strictly monotone along
orbits, so the unstable branch of s' crosses {L = c} exactly once descending,
and the stable branch of s crosses it exactly once ascending.  Both crossings
lie on the same level component; the signed gap between them is smooth in the
coefficients and vanishes exactly at connection.  Raw Euclidean
nearest-approach is NOT smooth -- the nearest point jumps between trace
segments and Newton sees kinks.

CENTRAL PATH.  F(theta, t) = delta(theta) - t * delta(theta_0), t: 1 -> 0.
Every separation shrinks proportionally.  Newton in theta at each t, Jacobian
by finite differences, minimum-norm step via the pseudoinverse so theta stays
near theta_0 -- which also protects the combinatorics, since hyperbolic
critical points persist under small perturbation.

    python connect_saddles.py --case nonnearest-attachment --pairs 1
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from spong import model, sturm, zoo                     # noqa: E402


def build(f, g):
    D = max(len(f) - 1, len(g) - 1)
    return model.build(list(f), list(g), model.moments_uniform01(2 * D + 1))


def skeleton(m):
    e = sturm.enumerate_critical_points(m)
    sad = sorted([p for p in e.points if p.kind == "saddle"],
                 key=lambda p: float(m.L(p.a, p.b)))
    return e, sad


def level_crossing(m, a0, b0, ascend, c, ds=2e-3, cap=400000):
    """March until L crosses c; return the crossing point, or None.

    Ascent for a stable branch, descent for an unstable one.  L is strictly
    monotone along the orbit, so the crossing is unique and bracketing it is
    a sign change, not a search.
    """
    k = getattr(m, "_native_kernel", None)
    if k is None:
        return None
    sgn = +1.0 if ascend else -1.0
    a, b = float(a0), float(b0)
    prev = (a, b, float(m.L(a, b)))
    for _ in range(cap):
        try:
            a, b = k.normalized_step(a, b, sgn * ds, 8)
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            return None
        a, b = float(a), float(b)
        if not (a == a and b == b):
            return None
        L = float(m.L(a, b))
        if (L - c) * (prev[2] - c) <= 0.0:
            # linear interpolation in L across the bracketing step
            w = (c - prev[2]) / (L - prev[2]) if L != prev[2] else 0.0
            return (prev[0] + w * (a - prev[0]), prev[1] + w * (b - prev[1]))
        prev = (a, b, L)
        if abs(a) > 1e4 or abs(b) > 1e4:
            return None
    return None


def stub_launch(m, p, manifold, direction):
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    best = min((q for q in e.points if q.kind == "saddle"),
               key=lambda q: abs(float(q.b) - float(p.b)))
    for s in best.stubs:
        if s.manifold == manifold and s.b_direction == direction:
            return np.asarray(s.curve, dtype=float)[-1]
    return None


def separation(m, src_b, src_dir, tgt_b, tgt_dir):
    """Signed gap on a level between u(target) and u(source)."""
    e, sad = skeleton(m)
    if len(sad) < 2:
        return None
    src = min(sad, key=lambda p: abs(float(p.b) - src_b))
    tgt = min(sad, key=lambda p: abs(float(p.b) - tgt_b))
    us, ut = float(m.L(src.a, src.b)), float(m.L(tgt.a, tgt.b))
    if not (us > ut):
        return None
    c = 0.5 * (us + ut)
    zu = stub_launch(m, src, "unstable", src_dir)
    zs = stub_launch(m, tgt, "stable", tgt_dir)
    if zu is None or zs is None:
        return None
    pu = level_crossing(m, zu[0], zu[1], ascend=False, c=c)
    ps = level_crossing(m, zs[0], zs[1], ascend=True, c=c)
    if pu is None or ps is None:
        return None
    # Signed gap along the level component.  Both points satisfy L = c, so
    # they differ only along the curve; b is a faithful parameter on a sheet.
    same_sheet = math.copysign(1.0, pu[0] - float(np.polyval(
        np.asarray([float(x) for x in m.beta])[::-1], pu[1])) /
        np.polyval(np.asarray([float(x) for x in m.alpha])[::-1], pu[1]))
    return same_sheet * (pu[1] - ps[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="nonnearest-attachment")
    ap.add_argument("--pairs", type=int, default=1,
                    help="how many connections to attempt simultaneously")
    ap.add_argument("--steps", type=int, default=12,
                    help="central-path stages, t: 1 -> 0")
    ap.add_argument("--fd", type=float, default=1e-5)
    args = ap.parse_args(argv)

    case = zoo.get(args.case)
    f0 = [float(x) for x in case.f]
    g0 = [float(x) for x in case.g]
    theta0 = np.array(f0 + g0, dtype=float)
    nf = len(f0)

    def unpack(th):
        return list(th[:nf]), list(th[nf:])

    m0 = build(f0, g0)
    e, sad = skeleton(m0)
    print(f"{case.name}: {len(sad)} saddles, levels "
          + ", ".join(f"{float(m0.L(p.a,p.b)):.5f}" for p in sad))

    # Candidate connections: adjacent in level, higher -> lower, both
    # branch directions tried, keep the ones whose separation evaluates.
    cands = []
    for i in range(len(sad) - 1, 0, -1):
        for sd in (+1, -1):
            for td in (+1, -1):
                d = separation(m0, float(sad[i].b), sd,
                               float(sad[i-1].b), td)
                if d is not None and math.isfinite(d):
                    cands.append((float(sad[i].b), sd,
                                  float(sad[i-1].b), td, d))
    if not cands:
        raise SystemExit("no evaluable separations on this case")
    cands.sort(key=lambda t: abs(t[4]))
    targets = cands[:args.pairs]
    print("targets:")
    for sb, sd, tb, td, d in targets:
        print(f"  b={sb:+.5f}({sd:+d}) -> b={tb:+.5f}({td:+d})   "
              f"delta0 = {d:+.6e}")
    print()

    def delta_vec(th):
        f, g = unpack(th)
        try:
            m = build(f, g)
        except Exception:
            return None
        out = []
        for sb, sd, tb, td, _ in targets:
            d = separation(m, sb, sd, tb, td)
            if d is None or not math.isfinite(d):
                return None
            out.append(d)
        return np.asarray(out)

    d0 = delta_vec(theta0)
    if d0 is None:
        raise SystemExit("baseline separation failed")
    th = theta0.copy()
    print(f"{'t':>7}{'||delta||':>14}   components")
    print("-" * 60)
    for stage in range(args.steps + 1):
        t = 1.0 - stage / args.steps
        for _ in range(3):                      # Newton iterations per stage
            d = delta_vec(th)
            if d is None:
                print("  (separation failed; backing off)"); break
            F = d - t * d0
            if np.linalg.norm(F) < 1e-12:
                break
            J = np.zeros((len(targets), len(th)))
            ok = True
            for j in range(len(th)):
                tp = th.copy(); tp[j] += args.fd
                dp = delta_vec(tp)
                if dp is None:
                    ok = False; break
                J[:, j] = (dp - d) / args.fd
            if not ok:
                print("  (Jacobian column failed)"); break
            step = -np.linalg.pinv(J) @ F      # minimum-norm
            # Backtrack.  A full Newton step readily leaves the region where
            # the separation evaluates at all -- the branch stops reaching
            # the level, or the skeleton reorders -- so shrink until the
            # step both evaluates and does not increase ||F||.
            lam, accepted = 1.0, False
            for _ in range(24):
                cand = th + lam * step
                dc = delta_vec(cand)
                if dc is not None and np.linalg.norm(dc - t * d0) <= \
                        np.linalg.norm(F) * (1.0 - 1e-4 * lam):
                    th, accepted = cand, True
                    break
                lam *= 0.5
            if not accepted:
                break
        d = delta_vec(th)
        if d is None:
            print(f"{t:>7.3f}   FAILED"); break
        print(f"{t:>7.3f}{np.linalg.norm(d):>14.6e}   "
              + "  ".join(f"{x:+.3e}" for x in d))

    f, g = unpack(th)
    print()
    print("f =", ", ".join(f"{x:.15g}" for x in f))
    print("g =", ", ".join(f"{x:.15g}" for x in g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
