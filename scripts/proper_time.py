#!/usr/bin/env python3
"""Find the wall by proper time: look for the tortoise among the hares.

THE IDEA (Andrew's).  Under the UNNORMALIZED gradient flow an orbit that
lands exactly on the target saddle takes infinite time to arrive.  The traces
are run at constant loss rate, so the physical time is recoverable from the
same vertices:

    dL/dtau = -||grad L||^2   ==>   tau = INT dL / ||grad L||^2 .

An orbit near the connection is the slow one -- the tortoise -- and its
arrival time diverges LOGARITHMICALLY as the parameter approaches the wall.
That is a smooth, well-resolved signal measurable far from the few ulps where
the landing flip is order-dependent, instead of asking binary64 which
outgoing branch wins at the last bit.

THE MODEL.  With Hessian eigenvalues -nu < 0 < kappa at the target saddle and
a generic transversal splitting y_in ~ M (Lambda - Lambda*),

    T(Lambda) = C - gamma log|Lambda - Lambda*| + o(1),   gamma = 1/(kappa+nu).

So each decade closer to the wall adds gamma*log(10) units of time -- an
immediately falsifiable slope.  A different slope means the family meets the
wall nontransversely, or the numerics have saturated.

THE EXTRAPOLATION.  Exponentiating linearizes it:

    r = exp(-T/gamma) ~ K_stop |Lambda - Lambda*| ,

and with the sector sign s from level_crossing.py,

    q = -sign(s) * exp(-T/gamma)

should run linearly through zero at Lambda*.  The two sides may have
different slopes but must extrapolate to the SAME intercept -- which is the
consistency check that a single-sided fit cannot give.

START ON A REGULAR SECTION.  By default the clock starts at L = c* + Delta.
With --source-delta it instead starts at

    L = c_source - source_delta,

below the originating saddle.  This fixes the tangential phase of departure:
an error that merely slides the numerical launch along the same unstable
orbit cannot move the time origin.  The remaining normal launch error should
contract away from the saddle and must be measured by refining the local
unstable-manifold construction.  Both clocks stop at the first crossing of
L = c*.

CLOCK.  The production-disabled diagnostic integrates d tau at the accepted
IRK collocation stages.  A full step and the accepted two-half composition
give an order-aware clock defect, independently of the spatial Richardson
test.  The same mechanism clocks normalized-arclength rescue steps using
1/||grad L|| at their stages.  Both boundary-crossing steps are shortened to
land on the requested loss sections.  --levels doubles the actual prefix
loss schedule; it is not the unrelated portrait chord refinement.

    python scripts/proper_time.py
    python scripts/proper_time.py --orders -4,-6 --delta 1e-6
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
sys.path.insert(0, str(REPO / "scripts"))

from spong import charts, portrait, sturm, zoo                   # noqa: E402
from level_crossing import first_below, rheostat                 # noqa: E402


def gamma_of(m, sa, sb):
    """1/(kappa+nu) from the exact Hessian at the target saddle."""
    h11, h12, h22 = m._native_kernel.hessian(sa, sb)
    ev = np.linalg.eigvalsh(np.array([[h11, h12], [h12, h22]], float))
    nu, kappa = -float(ev[0]), float(ev[1])
    return 1.0/(kappa + nu), kappa, nu


def trapezoidal_time(m, Y, c_hi, c_lo):
    """INT dL/||grad L||^2 from the first crossing of c_hi to that of c_lo.

    Trapezoid on the traced vertices in between, with both endpoints refined
    onto their levels so the interval is exactly [c_lo, c_hi].
    """
    K = m._native_kernel
    hi = first_below(m, Y, c_hi)
    lo = first_below(m, Y, c_lo)
    if hi is None or lo is None or lo[0] <= hi[0]:
        return None
    pts = [np.array([hi[2], hi[1]])]
    pts.extend(np.asarray(Y[k], float) for k in range(hi[0], lo[0]))
    pts.append(np.array([lo[2], lo[1]]))
    L = np.array([float(m.L(float(a), float(b))) for a, b in pts])
    g2 = np.array([float(np.dot(K.gradient(float(a), float(b)),
                                K.gradient(float(a), float(b))))
                   for a, b in pts])
    if np.any(g2 <= 0.0):
        return None
    # tau = INT dL/g2 along DECREASING L, so take the positive elapsed time
    dL = -np.diff(L)
    w = 0.5*(1.0/g2[:-1] + 1.0/g2[1:])
    return float(np.sum(dL*w))


def sector_sign(m, p, branch_index, sa, sb, c_star, delta):
    level = c_star - delta
    skip = max(4.0*math.sqrt(max(delta, 0.0)), 1e-9)
    marks = []
    for br in p.branches:
        if br.kind != "unstable":
            continue
        if abs(float(br.diag.get("saddle_b", 1e300)) - sb) > 1e-9:
            continue
        hit = first_below(m, np.asarray(br.Y, float), level,
                          exclude=(sa, sb), radius=skip)
        if hit is not None:
            marks.append(hit[1])
    if len(marks) != 2:
        return None
    hit = first_below(m, np.asarray(p.branches[branch_index].Y, float), level)
    if hit is None:
        return None
    return 2.0*hit[1] - marks[0] - marks[1]


def measure(base, lam, branch_index, target_b, delta, source_delta, order,
            launch_level,
            geometry_level, n_levels, clock_rtol, clock_atol):
    original = charts.GEOMETRIC_IRK_PRIMARY
    try:
        charts.GEOMETRIC_IRK_PRIMARY = order
        m = rheostat(base, lam)
        e = sturm.materialize_stubs(
            m, sturm.enumerate_critical_points(m),
            resolution_level=launch_level)
        tgt = min((q for q in e.points if q.kind == "saddle"),
                  key=lambda q: abs(float(q.b) - target_b))
        sa, sb = float(tgt.a), float(tgt.b)
        c_star = float(m.L(sa, sb))
        unstable_offset = branch_index-2*len(e.saddles)
        if not 0 <= unstable_offset < 2*len(e.saddles):
            raise ValueError("branch index does not select an unstable branch")
        source = e.saddles[unstable_offset//2]
        direction = 1 if unstable_offset % 2 == 0 else -1
        source_stub = next(
            stub for stub in source.stubs
            if stub.manifold == "unstable" and stub.b_direction == direction)
        launch_grid_error = float(
            dict(source_stub.certificates).get("grid_error", math.inf))
        if source_delta is None:
            clock_start = c_star+delta
        else:
            clock_start = float(m.L(source.a, source.b))-source_delta
        clock = {"start_level": clock_start, "stop_level": c_star,
                 "rtol": clock_rtol, "atol": clock_atol}
        p = portrait.compute(m, _enumeration=e, _skip_audit=True,
                             geometry_level=geometry_level,
                             _potential_clock=clock,
                             _potential_n_levels=n_levels)
    finally:
        charts.GEOMETRIC_IRK_PRIMARY = original
    Y = np.asarray(p.branches[branch_index].Y, float)
    g, kappa, nu = gamma_of(m, sa, sb)
    clock_diag = (p.branches[branch_index].diag
                  .get("potential_rate", {}).get("proper_time", {}))
    T = (float(clock_diag["tau"])
         if clock_diag.get("available", False) else None)
    T_trapezoid = trapezoidal_time(m, Y, clock_start, c_star)
    s = sector_sign(m, p, branch_index, sa, sb, c_star, delta)
    return (T, T_trapezoid, s, g, kappa, nu, clock_diag,
            launch_grid_error)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--target-b", type=float, default=0.640274)
    ap.add_argument("--center", type=float, default=2.17770956083,
                    help="nominal Lambda* about which offsets are taken")
    ap.add_argument("--exponents", default="-9,-10,-11,-12",
                    help="offsets +-10^e from the center")
    ap.add_argument("--delta", type=float, default=1e-6,
                    help="the incoming section sits at c* + delta")
    ap.add_argument("--source-delta", type=float,
                    help="instead start at c_source - source_delta")
    ap.add_argument("--orders", default="-6")
    ap.add_argument("--levels", default="0")
    ap.add_argument("--n-levels", type=int, default=12000,
                    help="base prefix loss-step count; level k multiplies by 2^k")
    ap.add_argument("--geometry-level", type=int, default=0)
    ap.add_argument("--launch-levels", default="0",
                    help="stub graph refinements; level k doubles grid intervals")
    ap.add_argument("--clock-rtol", type=float, default=1e-7)
    ap.add_argument("--clock-atol", type=float, default=1e-12)
    args = ap.parse_args(argv)

    w = zoo.get_wall_family(args.family)
    base = zoo.get(w.base_case)
    exps = [float(x) for x in args.exponents.split(",")]
    for order in [int(x) for x in args.orders.split(",")]:
        for launch_level in [int(x) for x in args.launch_levels.split(",")]:
          for level in [int(x) for x in args.levels.split(",")]:
            n_levels = args.n_levels*(2**level)
            section = (f"c_source - {args.source_delta:g}"
                       if args.source_delta is not None
                       else f"c* + {args.delta:g}")
            print(f"\n=== primary_order {order} (negative = pure), "
                  f"launch_level {launch_level}, n_levels {n_levels}, "
                  f"section at {section}")
            print(f"{'offset':>12} {'lambda':>22} {'T stage':>12} "
                  f"{'T trap':>12} {'err/tol':>10} {'sign s':>7} "
                  f"{'q = -sgn(s) e^(-T/g)':>22}")
            rows = []
            for e in exps:
                for sgn in (-1.0, +1.0):
                    off = sgn*10.0**e
                    lam = args.center + off
                    (T, T_trap, s, g, kappa, nu, clock_diag,
                     launch_grid_error) = measure(
                        base, lam, args.branch, args.target_b,
                        args.delta, args.source_delta, order,
                        launch_level,
                        args.geometry_level, n_levels,
                        args.clock_rtol, args.clock_atol)
                    if T is None or s is None:
                        print(f"{off:>+12.3g} {lam:>22.17g} "
                              f"{'--':>12} "
                              f"{('--' if T_trap is None else f'{T_trap:.6f}'):>12} "
                              f"{'--':>10} {'--':>7} {'no reading':>22}")
                        continue
                    q = -math.copysign(1.0, s)*math.exp(-T/g)
                    rows.append((off, T, s, q))
                    print(f"{off:>+12.3g} {lam:>22.17g} {T:>12.6f} "
                          f"{T_trap:>12.6f} "
                          f"{clock_diag.get('max_error_ratio', math.nan):>10.3g} "
                          f"{'+' if s > 0 else '-':>7} {q:>22.9e}")
            if rows:
                _T, kappa, nu = rows[0][1], kappa, nu
                print(f"   kappa={kappa:.9f} nu={nu:.9f}  "
                      f"gamma=1/(kappa+nu)={g:.9f}  "
                      f"predicted dT per decade = {g*math.log(10):.6f}\n"
                      f"   source-stub grid_error={launch_grid_error:.3e}")
                rows.sort(key=lambda r: r[0])
                side_roots = []
                for label, side in (("lower", [r for r in rows if r[0] < 0]),
                                    ("upper", [r for r in rows if r[0] > 0])):
                    if len(side) < 2:
                        continue
                    slope, intercept = np.polyfit(
                        [r[0] for r in side], [r[3] for r in side], 1)
                    root_offset = -intercept/slope
                    side_roots.append(root_offset)
                    print(f"   q-fit {label}: "
                          f"Lambda={args.center+root_offset:.17g} "
                          f"(center shift {root_offset:+.3e})")
                if len(side_roots) == 2:
                    print("   two-sided intercept spread: "
                          f"{abs(side_roots[1]-side_roots[0]):.3e}")
                for (o0, t0, _s0, _q0), (o1, t1, _s1, _q1) in zip(rows, rows[1:]):
                    if o0*o1 > 0 and abs(o1) < abs(o0):
                        print(f"   dT from {o0:+.2g} to {o1:+.2g}: "
                              f"{t1-t0:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
