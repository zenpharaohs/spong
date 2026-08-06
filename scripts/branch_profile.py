#!/usr/bin/env python3
"""Where the geometry time goes, per branch.

    python scripts/branch_profile.py [zoo-name ...]

Parallel speedup over independent branches is bounded by

    sum(branch_sec) / max(branch_sec)

no matter how many workers there are, because one branch cannot be split:
arc-length continuation is sequential in its own parameter.  Measuring that
ratio per case is the difference between knowing the ceiling and inferring it
from a flat scaling curve.

Also reported per branch: the fraction of its vertices produced by native
segments, and stiff_frac -- how much of the trace was shallow-water zone work,
which is NumPy over sounding grids and never went to C.  Together they say
whether a slow branch is slow because of work the port already covers, or
because of work still in Python.

Runs single-threaded on purpose: branch_sec under contention measures the
scheduler, not the branch.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    from spong import engine, model, portrait, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import engine, model, portrait, zoo


def profile(name: str) -> None:
    z = zoo.get(name)
    f, g = list(z.f), list(z.g)
    mu = (model.moments_normal01 if z.moment_dist == "normal01"
          else model.moments_uniform01)(2 * max(len(f), len(g)) - 1)
    m = model.build(f, g, mu)

    os.environ["SPONG_WORKERS"] = "1"
    t0 = time.perf_counter()
    p = portrait.certified_compute(m, view=z.default_view)
    wall = time.perf_counter() - t0

    rows = []
    for br in p.branches:
        d = br.diag
        rows.append((
            float(d.get("branch_sec", 0.0)),
            br.kind,
            float(d.get("saddle_b", float("nan"))),
            br.term,
            len(br.Y),
            float(d.get("stiff_frac", float("nan"))),
            int(d.get("native_steps", 0)),
        ))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows) or 1.0
    longest = rows[0][0] if rows else 1.0

    print(f"\n=== {name} — engine {engine.active_name()}, "
          f"wall {wall:.1f}s, {len(rows)} branches ===")
    print(f"{'sec':>8s} {'%tot':>6s} {'kind':>9s} {'saddle b':>10s} "
          f"{'term':>12s} {'pts':>8s} {'stiff':>7s} {'nat.steps':>10s}")
    for sec, kind, sb, term, npts, stiff, nsteps in rows:
        print(f"{sec:8.2f} {100*sec/total:6.1f} {kind:>9s} {sb:10.4f} "
              f"{term:>12s} {npts:8d} {stiff:7.3f} {nsteps:10d}")
    print(f"\n  sum {total:8.2f}s   longest {longest:8.2f}s"
          f"   CEILING {total/longest:5.2f}x")
    top2 = sum(r[0] for r in rows[:2])
    print(f"  two slowest branches are {100*top2/total:.0f}% of branch time")


def main() -> int:
    names = sys.argv[1:] or ["tricky-d11"]
    for name in names:
        profile(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
