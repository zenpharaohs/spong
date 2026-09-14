#!/usr/bin/env python3
"""Is sign(s) a stable bracketing test?  Vary the chord and the chart order.

THE OBSERVABLE.  At a level c* - delta just BELOW the target saddle, the
target's two unstable branches leave marks b_12 and b_13, and the watched
branch crosses at b_x.  The centered coordinate

    s = (b_x - b_12) + (b_x - b_13) = 2 b_x - b_12 - b_13

says which of the two outgoing sectors the watched branch entered.  Its SIGN
is a bracketing test that can be read after a short descent to a fixed level,
rather than after the full landing -- which is what makes it cheap and, the
claim to be tested here, far less chord-sensitive than the landing flip.

NOT A ROOT.  s is NOT continuous at the wall and must not be Brent-rooted.
At the connection the source orbit terminates AT the target saddle and never
reaches c* - delta at all, so for fixed delta > 0 the two sides approach
DIFFERENT marks and s jumps between them.  A sampled jump with a sign change
is indistinguishable from a zero crossing, and rooting it would converge to
something meaningless.  The continuous quantity lives on a level ABOVE the
saddle -- source W^u against target W^s, both of which exist at the
connection -- and is a separate measurement.

SCALE.  s/sqrt(delta) is the scale-free form (3.3827, 3.3821, 3.3805, 3.3652
measured at delta = 1e-12 ... 1e-6), as it must be: near a saddle the level
set sits at distance ~sqrt(delta) from the critical point.  s/delta is not
scale-free and an earlier reading that said so was arithmetic carelessness.

WHAT THIS RUNS.  sign(s) at one fixed delta for every combination of chord
(ds, ds/2) and primary chart order (GL4, GL6).  A bracket is trustworthy only
where all four agree; where they do not, the disagreement IS the result.

    python scripts/sign_s_stability.py
    python scripts/sign_s_stability.py --delta 1e-8 --lams 2.0,4.0
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

from spong import atlas, charts, model, portrait, sturm, zoo     # noqa: E402
from level_crossing import first_below, rheostat                 # noqa: E402


def s_value(m, e, p, branch_index, sa, sb, c_star, delta):
    """(s, b_x, marks) at the level c* - delta, or None."""
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
    b_x = hit[1]
    return 2.0*b_x - marks[0] - marks[1], b_x, marks


def run(base, lam, branch_index, target_b, delta, order, level):
    """s at one (chart order, geometry level).

    THE CHORD KNOB IS geometry_level: portrait sets
    resolution_divisor = 2**level and divides every ds by it, leaving the box
    and everything else alone.  Halving the chord by shrinking the view would
    have changed the box too, which is not a chord-only comparison.
    """
    m = rheostat(base, lam)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    tgt = min((q for q in e.points if q.kind == "saddle"),
              key=lambda q: abs(float(q.b) - target_b))
    sa, sb = float(tgt.a), float(tgt.b)
    c_star = float(m.L(sa, sb))
    original = charts.GEOMETRIC_IRK_PRIMARY
    try:
        charts.GEOMETRIC_IRK_PRIMARY = order
        p = portrait.compute(m, _enumeration=e, _skip_audit=True,
                             geometry_level=level)
    finally:
        charts.GEOMETRIC_IRK_PRIMARY = original
    out = s_value(m, e, p, branch_index, sa, sb, c_star, delta)
    if out is None:
        return None, "no reading"
    s, b_x, marks = out
    return s, f"b_x={b_x:.12f} marks=({marks[0]:.9f},{marks[1]:.9f})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--target-b", type=float, default=0.640274)
    ap.add_argument("--delta", type=float, default=1e-8)
    ap.add_argument("--lams", default="2.1777095608313717,2.1777095608313752,"
                                      "2.17770956082838,2.0,4.0")
    ap.add_argument("--orders", default="8,6")
    ap.add_argument("--levels", default="0,1",
                    help="geometry levels; ds is divided by 2**level")
    args = ap.parse_args(argv)

    w = zoo.get_wall_family(args.family)
    base = zoo.get(w.base_case)
    orders = [int(x) for x in args.orders.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    print(f"delta = {args.delta:g}   sign(s) per (chart order, ds/2**level)")
    for lam in [float(x) for x in args.lams.split(",")]:
        cells = []
        for order in orders:
            for level in levels:
                s, note = run(base, lam, args.branch, args.target_b,
                              args.delta, order, level)
                cells.append((order, level, s, note))
        # A cell that produced no reading is not a disagreement; it is a
        # missing datum, and counting it as one made every row read DISAGREE.
        signs = {"+" if c[2] > 0 else "-" for c in cells if c[2] is not None}
        missing = sum(1 for c in cells if c[2] is None)
        verdict = ("NO DATA" if not signs else
                   "AGREE" if len(signs) == 1 else "DISAGREE")
        if missing:
            verdict += f" ({missing} cell(s) missing)"
        print(f"\n lam={lam:.17g}   {verdict}")
        for order, level, s, note in cells:
            tag = f"GL{order} ds/{2**level:<3d}"
            if s is None:
                print(f"   {tag}  -- {note}")
            else:
                print(f"   {tag}  s={s:+.6e}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
