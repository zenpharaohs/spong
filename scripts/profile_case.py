#!/usr/bin/env python3
"""Profile one zoo portrait under the native engine.

    python scripts/profile_case.py linear-target-d17-thrash
    python scripts/profile_case.py linear-target-d17-thrash 40

Prints the top N entries by cumulative time, then by internal time,
restricted to spong's own modules so the view is what remains in Python.
"""

import cProfile
import os
import pstats
import sys
import time

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import potential_corpus as pc                                # noqa: E402
from potential_corpus import engine, portrait                # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "linear-target-d17-thrash"
args = [a for a in sys.argv[2:] if not a.startswith("--")]
top = int(args[0]) if args else 30

m, e, z = pc.context(name)
print(f"engine: {engine.active_name()}  case: {name}  workers: "
      f"{os.environ['SPONG_WORKERS']}")
t0 = time.perf_counter()
prof = cProfile.Profile()
prof.enable()
portrait.certified_compute(m, view=z.default_view)
prof.disable()
print(f"wall: {time.perf_counter()-t0:.1f}s\n")

stats = pstats.Stats(prof)
stats.sort_stats("cumulative").print_stats(r"spong", top)
stats.sort_stats("tottime").print_stats(top)

if "--callers" in sys.argv:
    # Who is still doing exact rational arithmetic in Python?  The
    # cumulative table cannot say: Fraction work is spread thin under
    # many spong frames.  Callers of the primitives can.
    print("\n=== callers of the residual Fraction/bigint primitives ===")
    stats.sort_stats("tottime").print_callers(
        r"math\.gcd|fractions\.py.*(_mul|_add|_sub|_richcmp|__pow__)"
        r"|_sign_int|_decimal", 12)
