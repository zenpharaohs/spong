#!/usr/bin/env python3
"""Where a descending branch crosses the saddle's own level, from both sides.

WHY THIS RATHER THAN THE LANDING FLIP.  Bisecting the wall on which minimum
a branch finally captures asks the tracer a question whose answer is settled
only after an arbitrarily long descent, and near a connection the lambda-lemma
makes that descent slow: the approach is where the integrators are weakest and
the answer is most chord-sensitive.  Measured on this family, bisecting the
landing put the wall at 2.17770956082838 under one plane-stepper setting and
2.17770956083137 under the production one -- a 3.0e-12 discrepancy that says
nothing about the wall and everything about which phase was ordered how.

THE CLEANER OBSERVABLE (Andrew's).  Take c* = the loss AT the target saddle.
On {L = c*} the level set is an exactly-known algebraic curve: the b-roots of
(C - c*)A - B^2, with a = a* +- sqrt((c* - u)/A).  Trace the branch only until
its loss first drops below c* -- a bracketed scalar condition, not a
destination -- and ask WHERE on that curve it crosses.  Then ask where the
target saddle's own unstable branches meet the same curve.  Both are points on
one exactly-known object, so the comparison is a distance along a known curve
rather than a difference of two integrated polylines.

AND IT EXTRAPOLATES.  Under constant-potential-rate descent the parameter IS
the loss, so both families can be read at matched levels c* - delta and the
separation extrapolated to delta -> 0.  That removes the arrival behaviour
near the saddle, which is exactly the part the tracer does worst.

WHAT IS MEASURED HERE
  crossing_b     b where the branch first reaches L <= c*, refined by
                 bisection on the traced polyline (the vertices bracket it)
  saddle_b       b where the target saddle's unstable branch meets the same
                 level -- computed from the branch, not assumed
  gap            |crossing_b - saddle_b| along the level curve's b parameter
  chord test     the same numbers at ds and ds/2: a quantity read at a FIXED
                 LEVEL should move far less under chord refinement than a
                 landing read after the whole descent.  That is the claim
                 this script exists to test, not to assume.

    python scripts/level_crossing.py
    python scripts/level_crossing.py --lam 2.17770956083137 --deltas 0,1e-9,1e-8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spong import model, portrait, sturm, zoo                    # noqa: E402


def rheostat(base, lam):
    root = math.sqrt(lam)
    f = [float(x)/root for x in base.f]
    g = [float(x)*root for x in base.g]
    n = 2*max(len(f), len(g)) - 1
    # Ordinary zoo cases name a distribution; ensemble cases are already
    # Models and carry the exact truncated moment functional.  Preserve that
    # prefix rather than selecting a representing distribution again.  This
    # is also the right abstraction for future arbitrary-moment inputs: the
    # polynomial loss depends only on the consumed prefix.
    if hasattr(base, "mu"):
        mu = tuple(base.mu[:n])
    else:
        mu = (model.moments_normal01 if base.moment_dist == "normal01"
              else model.moments_uniform01)(n)
    return model.build(f, g, mu)


def first_below(m, Y, level, exclude=None, radius=0.0):
    """(index, b, a) where the polyline first has L <= level, refined along
    the bracketing chord by bisection on L.  None when it never does.

    ``exclude``/``radius``: skip vertices still inside a ball around that
    point.  A branch LAUNCHED from the target saddle sits at c* already, so
    without this its mark would be the launch vertex itself rather than the
    place it meets the regular level after leaving.
    """
    L = np.array([float(m.L(float(a), float(b))) for a, b in Y])
    ok = L <= level
    if exclude is not None and radius > 0.0:
        d = np.hypot(Y[:, 0] - exclude[0], Y[:, 1] - exclude[1])
        ok &= d > radius
    below = np.flatnonzero(ok)
    if below.size == 0:
        return None
    k = int(below[0])
    if k == 0:
        return 0, float(Y[0, 1]), float(Y[0, 0])
    p0, p1 = np.asarray(Y[k-1], float), np.asarray(Y[k], float)
    for _ in range(200):
        mid = 0.5*(p0 + p1)
        if float(m.L(float(mid[0]), float(mid[1]))) > level:
            p0 = mid
        else:
            p1 = mid
        if float(np.hypot(*(p1 - p0))) <= 1e-15*(1.0 + float(np.hypot(*p1))):
            break
    q = 0.5*(p0 + p1)
    return k, float(q[1]), float(q[0])


def report(base, lam, branch_index, target_b, deltas, ds_scale):
    m = rheostat(base, lam)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    tgt = min((q for q in e.points if q.kind == "saddle"),
              key=lambda q: abs(float(q.b) - target_b))
    sa, sb = float(tgt.a), float(tgt.b)
    c_star = float(m.L(sa, sb))
    p = portrait.compute(m, _enumeration=e, _skip_audit=True)
    watched = p.branches[branch_index]
    Yw = np.asarray(watched.Y, float)

    print(f"\n lam={lam:.17g}  target saddle ({sa:.9f},{sb:.9f})  "
          f"c*={c_star:.12g}")
    targets = [(i, np.asarray(br.Y, float)) for i, br in enumerate(p.branches)
               if br.kind == "unstable"
               and abs(float(br.diag.get("saddle_b", 1e300)) - sb) <= 1e-9]
    if not targets:
        print("   no unstable branch launched from the target saddle")
        return c_star

    # SAME LEVEL ON BOTH SIDES.  At c* itself the target's unstable manifold
    # meets the critical level only AT the saddle -- the level set is
    # singular there and the two branches touch.  The marks exist only on the
    # regular level c* - delta, so the watched crossing and the target marks
    # must be read at the SAME positive delta; comparing a watched crossing
    # against marks computed once near c* compares different level sets.
    for delta in deltas:
        level = c_star - delta
        # leave the saddle's own neighbourhood before taking a mark: the
        # launch vertex is already at c*
        skip = max(4.0*math.sqrt(max(delta, 0.0)), 1e-9)
        marks = []
        for i, Yb in targets:
            hit = first_below(m, Yb, level, exclude=(sa, sb), radius=skip)
            if hit is not None:
                marks.append((i, hit[1]))
        hit = first_below(m, Yw, level)
        if hit is None:
            print(f"   delta={delta:<10.3g} watched branch never reaches "
                  f"L <= {level:.12g}")
            continue
        k, bx, ax = hit
        if not marks:
            print(f"   delta={delta:<10.3g} crossing b={bx:.12f} "
                  f"a={ax:.9g}   (no target mark outside r={skip:.2g})")
            continue
        gaps = "  ".join(f"br{i}: {bx - bm:+.6e}" for i, bm in marks)
        nearest = min(abs(bx - bm) for _i, bm in marks)
        print(f"   delta={delta:<10.3g} crossing vtx {k:>6} b={bx:.12f}  "
              f"marks {[round(bm, 9) for _i, bm in marks]}  "
              f"signed gap {gaps}  |min| {nearest:.3e}")
    return c_star


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--target-b", type=float, default=0.640274)
    ap.add_argument("--lams", default="2.1777095608313717,2.1777095608313752,"
                                      "2.17770956082838,2.0,4.0")
    ap.add_argument("--deltas", default="1e-12,1e-10,1e-8,1e-6,1e-4")
    args = ap.parse_args(argv)
    w = zoo.get_wall_family(args.family)
    base = zoo.get(w.base_case)
    deltas = [float(x) for x in args.deltas.split(",")]
    for lam in [float(x) for x in args.lams.split(",")]:
        report(base, lam, args.branch, args.target_b, deltas, 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
