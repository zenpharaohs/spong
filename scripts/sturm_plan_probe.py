#!/usr/bin/env python3
"""Which polynomials get a persistent GMP Sturm plan during one portrait.

    python scripts/sturm_plan_probe.py linear-target-d17-thrash

The plan (squarefree Sturm chain) is the single most expensive exact object
the pipeline builds -- 37 of them on d17-thrash, 7 s -- while the model has
three polynomials of interest.  This logs every plan construction: degree,
peak coefficient bits, seconds, and the nearest non-sturm spong frame that
asked for it, then groups by (caller, degree).  If most plans are derived
polynomials built per critical point or per level, that is where a shift or
a cache would pay; if they are the model's own B, N, H and their
derivatives, the cost is the chain and only precision policy moves it.
"""

from __future__ import annotations

import collections
import os
import sys
import time
import traceback
from functools import lru_cache

if "--workers" in sys.argv:
    i = sys.argv.index("--workers")
    os.environ["SPONG_WORKERS"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
os.environ.setdefault("SPONG_WORKERS", "1")
os.environ.setdefault("SPONG_ENGINE", "native")

sys.path.insert(0, os.path.dirname(__file__))
import potential_corpus as pc                                # noqa: E402
from potential_corpus import portrait, sturm                 # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "linear-target-d17-thrash"
m, e, z = pc.context(name)
sturm._native_sturm_plan.cache_clear()
rows = []
original = sturm._native_sturm_plan.__wrapped__


@lru_cache(maxsize=sturm._CACHE)
def logged(integers):
    t = time.perf_counter()
    plan = original(integers)
    dt = time.perf_counter() - t
    bits = max(abs(x).bit_length() for x in integers)
    frames = [f for f in traceback.extract_stack(limit=14)
              if "/spong/" in f.filename and "sturm.py" not in f.filename
              and "sturm_plan_probe" not in f.filename]
    who = (f"{frames[-1].filename.split('/')[-1]}:{frames[-1].name}"
           if frames else "?")
    rows.append((dt, len(integers) - 1, bits, who))
    return plan


sturm._native_sturm_plan = logged
t = time.perf_counter()
portrait.certified_compute(m, view=z.default_view)
wall = time.perf_counter() - t
print(f"{name}: wall {wall:.1f}s, {len(rows)} plans, "
      f"{sum(r[0] for r in rows):.1f}s building them\n")
print(f"{'seconds':>8} {'plans':>5} {'degree':>6} {'max bits':>8}  caller")
groups = collections.defaultdict(lambda: [0, 0.0, 0])
for dt, deg, bits, who in rows:
    g = groups[(who, deg)]
    g[0] += 1
    g[1] += dt
    g[2] = max(g[2], bits)
for (who, deg), (n, s, bits) in sorted(groups.items(),
                                        key=lambda kv: -kv[1][1]):
    print(f"{s:8.2f} {n:5d} {deg:6d} {bits:8d}  {who}")
