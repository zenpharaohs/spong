#!/usr/bin/env python3
"""Squeeze a saddle-connection wall until our own integrators give out.

THE OBSERVABLE.  Which minimum the watched branch captures is discrete and
exact, so a bracket needs no separation function: sample Lambda, read the
landing.  What makes the bracket TRUSTWORTHY is a second quantity -- at the
branch's closest approach to the target saddle, the polyline's own sagitta
bound must be smaller than the distance to that saddle.  Where it is not,
the curve cannot say which side of the connection it passed, and a landing
read there attests nothing.  Both endpoints of a quoted bracket must be
trusted or the bracket is not a bracket.

WHY REFINE UNTIL IT BREAKS.  The interesting number is not only where the
wall is but how finely our integrators can still assert a side.  Squeezing
in far enough, even IRK-GL4 and IRK-GL6 must give out -- just much later
than a non-anadromic method would, because they are symplectic and
time-symmetric and so do not damp the transverse behaviour that the whole
question turns on.  The stage at which the interior goes untrusted IS the
resolution limit, and it is reported rather than hidden.

ORDERS ARE RUN SEPARATELY.  GL4 and GL6 each trade truncation against
roundoff differently; agreement between them is evidence, and disagreement
is the honest width.  (GL8 is deliberately not included by default: at
these chords its truncation error is already under the evaluation floor, so
the extra stages contribute rounding -- measured on this same wall, GL8
returns a bracket as sharp as the others but offset by 3.0e-12.)

    python scripts/wall_squeeze.py
    python scripts/wall_squeeze.py --orders 4,6,8 --samples 12 --rounds 8
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

from spong import charts, model, portrait, sturm, topology, zoo   # noqa: E402


def rheostat(base, lam):
    root = math.sqrt(lam)
    f = [float(x)/root for x in base.f]
    g = [float(x)*root for x in base.g]
    n = 2*max(len(f), len(g)) - 1
    mu = (model.moments_normal01 if base.moment_dist == "normal01"
          else model.moments_uniform01)(n)
    return model.build(f, g, mu)


def probe(base, lam, branch_index, target_b):
    """(landing, distance, sagitta) at the closest approach, or None."""
    m = rheostat(base, lam)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    tgt = min((q for q in e.points if q.kind == "saddle"),
              key=lambda q: abs(float(q.b) - target_b))
    sa, sb = float(tgt.a), float(tgt.b)
    p = portrait.compute(m, _enumeration=e, _skip_audit=True)
    if branch_index >= len(p.branches):
        return None
    br = p.branches[branch_index]
    Y = np.asarray(br.Y, dtype=float)
    if len(Y) < 3:
        return None
    k = int(np.argmin(np.hypot(Y[:, 0]-sa, Y[:, 1]-sb)))
    dist = float(np.hypot(Y[k][0]-sa, Y[k][1]-sb))
    sag = np.asarray(topology._sagitta_bounds(Y), dtype=float)
    s = float(sag[min(k, len(sag)-1)]) if sag.size else float("inf")
    land = (round(float(Y[-1][0]), 6), round(float(Y[-1][1]), 6))
    return land, dist, s


def squeeze(base, lo, hi, branch_index, target_b, samples, rounds):
    """Refine while both ends stay trusted; report where that fails."""
    history = []
    for rnd in range(rounds):
        rows = []
        for i in range(samples + 1):
            lam = lo + (hi - lo)*i/samples
            r = probe(base, lam, branch_index, target_b)
            if r is None:
                continue
            land, dist, sag = r
            rows.append((lam, land, dist, sag, dist > sag))
        flips = [(rows[i], rows[i+1]) for i in range(len(rows)-1)
                 if rows[i][1] != rows[i+1][1]]
        if not flips:
            history.append((rnd, lo, hi, "no flip in the window", None))
            break
        a, b = flips[0]
        both = a[4] and b[4]
        history.append((rnd, a[0], b[0], "trusted" if both else "UNTRUSTED",
                        (a[2], a[3], b[2], b[3])))
        if not both:
            break
        if b[0] <= math.nextafter(a[0], math.inf):
            history.append((rnd+1, a[0], b[0], "adjacent floats", None))
            break
        lo, hi = a[0], b[0]
    return history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--target-b", type=float, default=0.640274)
    ap.add_argument("--lo", type=float, default=2.1777095608153787)
    ap.add_argument("--hi", type=float, default=2.1777095608403787)
    ap.add_argument("--orders", default="4,6")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args(argv)

    w = zoo.get_wall_family(args.family)
    base = zoo.get(w.base_case)
    original = charts.GEOMETRIC_IRK_PRIMARY
    finals = {}
    try:
        for order in [int(x) for x in args.orders.split(",")]:
            charts.GEOMETRIC_IRK_PRIMARY = order
            print(f"\n=== IRK-GL{order}")
            hist = squeeze(base, args.lo, args.hi, args.branch,
                           args.target_b, args.samples, args.rounds)
            for rnd, a, b, note, det in hist:
                width = b - a if (a is not None and b is not None) else 0.0
                extra = ""
                if det:
                    extra = (f"  dist {det[0]:.3e}/{det[2]:.3e}  "
                             f"sagitta {det[1]:.3e}/{det[3]:.3e}")
                print(f"   round {rnd}: [{a!r}, {b!r}]  width {width:.3e}  "
                      f"{note}{extra}")
            good = [h for h in hist if h[3] == "trusted"]
            if good:
                finals[order] = (good[-1][1], good[-1][2])
    finally:
        charts.GEOMETRIC_IRK_PRIMARY = original

    if len(finals) > 1:
        print("\n=== finest TRUSTED bracket per order")
        for order, (a, b) in finals.items():
            print(f"   GL{order}: [{a!r}, {b!r}]  width {b-a:.3e}")
        lo = min(a for a, _b in finals.values())
        hi = max(b for _a, b in finals.values())
        print(f"   union across orders: [{lo!r}, {hi!r}]  width {hi-lo:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
