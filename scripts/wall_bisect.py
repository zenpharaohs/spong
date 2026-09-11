#!/usr/bin/env python3
"""Bisect a wall by LANDING FATE, at several IRK orders, with our own tracer.

WHY NOT THE STORED BRACKET.  zoo's nonnearest-saddle-connection carries a
wall_bracket whose protocol cites scipy's Radau and DOP853.  Neither is
anadromic -- Radau is L-stable and dissipative by construction, DOP853 is
explicit and preserves nothing -- and scipy is not a dependency of this
project, so that protocol cannot be re-run here and should not be cited.
Measured 2026-09-07 with the current tracer: both endpoints of that bracket
and the stored wall_parameter give the SAME landing, so it does not straddle
a flip at all; the flip sits about 2.6e-9 lower, where a fresh Brent shoot
also puts delta = 0.

WHAT THIS DOES INSTEAD.  The landing fate is a discrete, exactly-observable
property of a portrait: which minimum the branch captures.  Bisecting on it
needs no separation function and no outside integrator.  Repeating the
bisection at IRK-GL orders 4, 6 and 8 gives the honest bracket: where the
orders agree, the wall is resolved; where they differ, THE SPREAD IS THE
BRACKET and should be quoted as such.  They are not guaranteed to agree to
a few ulps -- each order trades truncation against roundoff differently --
so the spread is the measurement, not a failure.

    python scripts/wall_bisect.py
    python scripts/wall_bisect.py --orders 4,6,8 --tol 1e-13
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

from spong import charts, model, portrait, sturm, zoo            # noqa: E402


def rheostat(base, lam):
    root = math.sqrt(lam)
    f = [float(x)/root for x in base.f]
    g = [float(x)*root for x in base.g]
    n = 2*max(len(f), len(g)) - 1
    mu = (model.moments_normal01 if base.moment_dist == "normal01"
          else model.moments_uniform01)(n)
    return model.build(f, g, mu)


def landing(base, lam, branch_index):
    """The captured endpoint of the watched branch, or None."""
    m = rheostat(base, lam)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    p = portrait.compute(m, _enumeration=e, _skip_audit=True)
    if branch_index >= len(p.branches):
        return None
    br = p.branches[branch_index]
    if not br.term.startswith("capture"):
        return None
    Y = np.asarray(br.Y, dtype=float)
    return (round(float(Y[-1][0]), 6), round(float(Y[-1][1]), 6))


def bisect(base, lo, hi, branch_index, tol):
    flo, fhi = landing(base, lo, branch_index), landing(base, hi, branch_index)
    if flo is None or fhi is None or flo == fhi:
        return None, flo, fhi, 0
    n = 0
    while hi - lo > tol*max(abs(lo), 1.0):
        mid = 0.5*(lo + hi)
        if mid <= lo or mid >= hi:               # binary64 floor reached
            break
        fm = landing(base, mid, branch_index)
        n += 1
        if fm is None:
            return None, flo, fhi, n
        if fm == flo:
            lo = mid
        else:
            hi = mid
    return (lo, hi), flo, fhi, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--lo", type=float, default=2.17770956)
    ap.add_argument("--hi", type=float, default=2.17770957)
    ap.add_argument("--orders", default="4,6,8")
    ap.add_argument("--tol", type=float, default=1e-15)
    args = ap.parse_args(argv)

    w = zoo.get_wall_family(args.family)
    base = zoo.get(w.base_case)
    print(f"{args.family}: stored wall {w.wall_parameter!r}")
    print(f"   stored bracket {w.wall_bracket!r}")
    original = charts.GEOMETRIC_IRK_PRIMARY
    results = {}
    try:
        for order in [int(x) for x in args.orders.split(",")]:
            charts.GEOMETRIC_IRK_PRIMARY = order
            bracket, flo, fhi, n = bisect(base, args.lo, args.hi,
                                          args.branch, args.tol)
            results[order] = bracket
            if bracket is None:
                print(f"   GL{order}: no flip in [{args.lo}, {args.hi}] "
                      f"(lo -> {flo}, hi -> {fhi})")
                continue
            lo, hi = bracket
            print(f"   GL{order}: flip in [{lo!r}, {hi!r}]  width "
                  f"{hi-lo:.3e}  ({n} portraits)  {flo} | {fhi}")
    finally:
        charts.GEOMETRIC_IRK_PRIMARY = original

    good = [b for b in results.values() if b]
    if len(good) > 1:
        lo = min(b[0] for b in good); hi = max(b[1] for b in good)
        print(f"\n   ORDER-INDEPENDENT BRACKET [{lo!r}, {hi!r}]  "
              f"width {hi-lo:.3e}")
        print("   (the union over orders: where they agree the wall is "
              "resolved, where they differ the spread IS the bracket)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
