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

# Either a zoo case by name, or an ensemble case by seed:
#     python scripts/profile_case.py --seed 1194542352 --mode random
argv = sys.argv[1:]
seed = mode = None
if "--seed" in argv:
    i = argv.index("--seed")
    seed = int(argv[i + 1])
    del argv[i:i + 2]
if "--mode" in argv:
    i = argv.index("--mode")
    mode = argv[i + 1]
    del argv[i:i + 2]
positional = [a for a in argv if not a.startswith("--")]
name = positional[0] if positional else "linear-target-d17-thrash"
top = int(positional[1]) if len(positional) > 1 else 30

if seed is not None:
    import random
    from qualify import directed_model, random_model
    from spong import sturm
    build = random_model if (mode or "random") == "random" else directed_model
    m, spec = build(random.Random(seed), 5)
    if m is None:
        raise SystemExit(f"generator declined seed {seed}")
    view = None
    name = f"{spec} (seed {seed})"
else:
    m, e, z = pc.context(name)
    view = z.default_view

print(f"engine: {engine.active_name()}  case: {name}  workers: "
      f"{os.environ['SPONG_WORKERS']}")
t0 = time.perf_counter()
prof = cProfile.Profile()
prof.enable()
portrait.certified_compute(m, view=view)
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
