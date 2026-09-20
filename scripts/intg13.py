#!/usr/bin/env python3
"""Andrew's deg f = 1, deg g = 13 small-integer cases (2026-09-19).

Two separate symptoms to separate: seed 16 traces but the allocator runs
into its time limit, and OTHER seeds of the same family fail to trace at
all, which is a defect regardless of what the allocator does.

    python scripts/intg13.py            # the reported case, seed 16
    python scripts/intg13.py --scan 40  # look for the ones that error
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spong import atlas, model, portrait, sturm                  # noqa: E402

F = [3, 3]
G16 = [4, -3, 0, 3, -1, 2, -1, -3, 3, -1, 1, -3, -1, 1]

# The UnboundLocalError case: B and N share a factor (or one is not
# squarefree), so enumeration takes the H branch, while the backbone
# reduction is trivial so the cheap route was taken above it.
F11 = [0, -1]
G11 = [3, 0, 0, -2, 0, 1, -4, 2, 1, -1, 1, 0, 4, 1]


def build(g, f=F):
    n = 2*max(len(f), len(g)) - 1
    return model.build([Fraction(x) for x in f], [Fraction(x) for x in g],
                       model.moments_uniform01(n))


def look(tag, g, f=F):
    print(f"\n=== {tag}: f = {f}  g = {g}")
    m = build(g, f)
    t0 = time.perf_counter()
    try:
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    except Exception:
        print("  ENUMERATION/STUBS FAILED")
        traceback.print_exc(limit=6)
        return
    t_enum = time.perf_counter() - t0
    kinds = {}
    for q in e.points:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    print(f"  enumerate+stubs {t_enum:6.2f}s  {kinds}  "
          f"legal_max_b={atlas.legal_max_b(m):.4g}  "
          f"gauge={atlas.genericity(m)}")
    for q in sorted(e.points, key=lambda q: float(q.b)):
        u = float(m.L(float(q.a), float(q.b)))
        print(f"     {q.kind:>7} b={float(q.b):14.6g} a={float(q.a):14.6g} "
              f"u={u:14.8g}")
    t0 = time.perf_counter()
    try:
        p = portrait.certified_compute(m, _enumeration=e)
    except Exception:
        print(f"  CERTIFIED_COMPUTE FAILED after "
              f"{time.perf_counter()-t0:.1f}s")
        traceback.print_exc(limit=8)
        return
    dt = time.perf_counter() - t0
    terms = {}
    for br in p.branches:
        terms[br.term] = terms.get(br.term, 0) + 1
    top = p.ledger.get("topology", {})
    print(f"  portrait {dt:7.2f}s  {top.get('status')} / "
          f"{top.get('resolution_reason')}  terms {terms}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0,
                    help="also try this many seeds of the same family")
    ap.add_argument("--degree", type=int, default=13)
    ap.add_argument("--span", type=int, default=4,
                    help="coefficients drawn from -span..span")
    args = ap.parse_args(argv)

    look("seed 16 (reported)", G16)
    look("seed 11 (UnboundLocalError)", G11, F11)
    for seed in range(args.scan):
        rng = np.random.default_rng(seed)
        g = [int(x) for x in rng.integers(-args.span, args.span + 1,
                                          size=args.degree + 1)]
        if g[-1] == 0:
            g[-1] = 1
        look(f"scan seed {seed}", g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
