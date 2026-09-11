#!/usr/bin/env python3
"""Basin area of each minimum, as an exact-boundary quadrature.

WHY THIS IS A BOUND AND NOT AN ESTIMATE.  Below the lowest saddle that
encloses it, the sublevel component containing a minimum holds no other
critical point, so every point in it descends to that minimum: the
component is a subset of the basin, and its area is a rigorous LOWER
bound on the basin's area.  The boundary is the level set L = c, whose
b-interval is a pair of roots of the level polynomial (isolated exactly by
Sturm elsewhere in spong; here they are refined numerically from the
enumeration's own bracketing), and the width in a is closed form:

    L = u(b) + A(b) w^2  ==>  |w| <= sqrt((c - u(b))/A(b)),

so the area is the one-dimensional integral

    area(c) = INT 2 sqrt((c - u(b))/A(b)) db

over that interval.  No sampling of the plane, no descent runs.

WHAT IT IS FOR.  The separated maximal-critical family (deg f = 1, uniform
moments, B prescribed with alternating separated roots) is a candidate
sequence of progressively untrainable nets: if the global minimum's basin
area falls geometrically in the separation while the rest of the landscape
does not, then a uniform initialisation finds the global minimum with
probability shrinking at that rate.  The ratio reported here --

    area(global minimum) / sum over all minima

-- is the trainability-relevant number, since it does not depend on the
arbitrary box one initialises in.

CAVEAT, stated because the number is a bound in ONE direction.  The true
basin extends past this component, along the corridors by which other
regions drain into the same minimum, so the ratio here can understate the
global minimum's share.  It is the right quantity for exhibiting
UNTRAINABILITY (a small bound is still small) and the wrong one for
claiming a net IS trainable.

    python scripts/basin_area.py                 # d = 2..8 at Lambda = 30
    python scripts/basin_area.py --degree 4 --separations 5,10,30,100,300
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spong import inverse, sturm                              # noqa: E402


def _u(m, b):
    return float(m.L(float(m.a_star(b)), b))


def _half_width(m, b, c):
    A = float(m.A(b))
    if not (A > 0.0):
        return 0.0
    return float(np.sqrt(max(c - _u(m, b), 0.0) / A))


def _bracket_root(m, c, lo, hi):
    """b in [lo, hi] with u(b) = c, by bisection on u - c (u is smooth and
    the bracket comes from the enumeration, so this only refines)."""
    flo = _u(m, lo) - c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = _u(m, mid) - c
        if (fm < 0.0) == (flo < 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def basin_open_to_infinity(m, b_min, c_merge, saddle_b):
    """Is the minimum's basin unbounded?

    The component below the merging saddle is always bounded in b -- it is
    cut off by the roots of u = c on either side -- so its area is finite
    whatever the basin does.  The BASIN is larger: above the merging level
    the component containing the minimum grows past that saddle, and if
    nothing bounds it out there, points arbitrarily far away drain into the
    region and the basin has infinite area.  Reporting the component's area
    as though it were the basin's would then be meaningless as a ratio,
    which is what it was doing for the low degrees.

    The test is exact and needs no sampling: at the merging level c the
    boundary is the root set of P = (C-c)A - B^2, so the component is open
    on a side exactly when P has no root beyond the merging saddle on that
    side.  A rational c slightly BELOW the true merging level is used, which
    can only add roots, so a "no root" answer is conservative.
    """
    from spong import _poly as Pmod
    c = Fraction(float(c_merge)) * (Fraction(1) - Fraction(1, 2**40))
    level = Pmod.sub(Pmod.scale(m.alpha, m.C - c), Pmod.mul(m.beta, m.beta))
    try:
        roots = [float(sturm.refine(level, iv, Fraction(1, 2**32)).mid)
                 for iv in sturm.isolate_roots(level)]
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        return None
    if saddle_b > b_min:
        return not any(r > saddle_b for r in roots)
    return not any(r < saddle_b for r in roots)


def basin_component(m, enumeration, b_min, saddle_bs, frac=1e-9):
    """(c, b_left, b_right) for the component around b_min just below its
    lowest enclosing saddle.  Returns None when no saddle encloses it."""
    u_min = _u(m, b_min)
    left = [b for b in saddle_bs if b < b_min]
    right = [b for b in saddle_bs if b > b_min]
    levels = [_u(m, b) for b in left[-1:]] + [_u(m, b) for b in right[:1]]
    if not levels:
        return None
    c = min(levels)
    c = u_min + (c - u_min) * (1.0 - frac)
    lo = _bracket_root(m, c, left[-1], b_min) if left else None
    hi = _bracket_root(m, c, b_min, right[0]) if right else None
    if lo is None or hi is None:
        return None
    merge_side = (left[-1] if (left and _u(m, left[-1]) <= (
        _u(m, right[0]) if right else float("inf"))) else
        (right[0] if right else left[-1]))
    return c, lo, hi, merge_side


def component_area(m, c, b_left, b_right, n=20001):
    """INT 2 sqrt((c-u)/A) db.  The integrand vanishes like a square root at
    both ends, so integrate in the variable that makes those ends regular:
    b = midpoint + halfspan*sin(theta) turns sqrt(endpoint) into a smooth
    factor and the trapezoid converges quickly."""
    mid = 0.5 * (b_left + b_right)
    half = 0.5 * (b_right - b_left)
    th = np.linspace(-np.pi/2, np.pi/2, n)
    bs = mid + half*np.sin(th)
    vals = np.array([2.0*_half_width(m, float(b), c)*half*np.cos(t)
                     for b, t in zip(bs, th)])
    return float(np.trapezoid(vals, th)) if hasattr(np, "trapezoid") \
        else float(np.trapz(vals, th))


def report(deg, separation):
    case = inverse.separated_linear_case(deg, Fraction(separation))
    m = case.model
    e = sturm.enumerate_critical_points(m)
    mins = sorted((float(q.b) for q in e.points if q.kind == "min"))
    sads = sorted((float(q.b) for q in e.points if q.kind == "saddle"))
    rows = []
    for b_min in mins:
        comp = basin_component(m, e, b_min, sads)
        if comp is None:
            rows.append((b_min, _u(m, b_min), None, None, True))
            continue
        c, lo, hi, merge_side = comp
        c_merge = _u(m, merge_side)
        open_out = basin_open_to_infinity(m, b_min, c_merge, merge_side)
        rows.append((b_min, _u(m, b_min), component_area(m, c, lo, hi),
                     (lo, hi), bool(open_out)))
    finite = [r for r in rows if r[2] is not None and not r[4]]
    areas = [r[2] for r in finite]
    total = sum(areas) if areas else 0.0
    gmin = min(rows, key=lambda r: r[1])
    share = (gmin[2]/total) if (gmin[2] and total > 0 and not gmin[4]) \
        else float("nan")
    print(f"\n=== deg g = {deg}, Lambda = {separation}: "
          f"{len(e.points)} critical points (ceiling {case.algebraic_ceiling}), "
          f"{len(mins)} minima")
    for b_min, u_min, area, span, open_out in rows:
        tag = " <-- global" if b_min == gmin[0] else ""
        if area is None:
            print(f"   b={b_min:14.6g} u={u_min:14.7g}  no enclosing saddle: "
                  f"BASIN UNBOUNDED{tag}")
        elif open_out:
            print(f"   b={b_min:14.6g} u={u_min:14.7g}  component>={area:12.6g}"
                  f"  but BASIN UNBOUNDED (open past the merging saddle){tag}")
        else:
            print(f"   b={b_min:14.6g} u={u_min:14.7g}  area>={area:12.6g}  "
                  f"b-span [{span[0]:.6g}, {span[1]:.6g}]{tag}")
    if np.isnan(share):
        print("   global-minimum share: NOT A RATIO -- an unbounded basin is "
              "in play")
    else:
        print(f"   global-minimum share of summed FINITE basin area: "
              f"{share:.6g}")
    return {"deg": deg, "separation": float(separation),
            "global_area": gmin[2], "global_unbounded": bool(gmin[4]),
            "total": total, "share": share}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--degrees", default="2,3,4,5,6,7,8")
    ap.add_argument("--separations", default=None)
    args = ap.parse_args(argv)

    out = []
    if args.separations:
        deg = args.degree or 4
        for s in args.separations.split(","):
            out.append(report(deg, Fraction(s)))
    else:
        degs = ([args.degree] if args.degree
                else [int(x) for x in args.degrees.split(",")])
        for d in degs:
            out.append(report(d, Fraction(30)))
    print("\n# summary")
    print(f"{'deg':>4} {'Lambda':>8} {'global area':>14} {'total':>14} "
          f"{'share':>12}  note")
    for r in out:
        ga = "-" if r["global_area"] is None else f"{r['global_area']:14.6g}"
        note = "global basin UNBOUNDED" if r["global_unbounded"] else ""
        print(f"{r['deg']:>4} {r['separation']:>8g} {ga:>14} "
              f"{r['total']:14.6g} {r['share']:12.6g}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
