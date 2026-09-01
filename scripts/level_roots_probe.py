#!/usr/bin/env python3
"""Validate spong.level_roots against the Sturm path.

    python scripts/level_roots_probe.py                  # whole zoo
    python scripts/level_roots_probe.py tricky-d11 --random 200

For each case: the levels the certifier actually uses (captured by
hooking merge_tree.level_polynomial during certified_compute) plus random
levels and levels a hair away from each critical value.  At every level
the two paths must agree on

    the number of real roots of R_c and their order
      (every monotone-path interval contains exactly one Sturm interval's
       root: checked by Sturm counts on the monotone intervals)
    count(c, lo, hi) on random and root-straddling (lo, hi]
    gap_index for every critical point below the level

LevelTie from the monotone path must coincide with the Sturm path's
inability (a double root / sign alternation failure / value_sign None).
Timings are reported for both.  Nothing here changes production.
"""

from __future__ import annotations

import os
import random
import sys
import time
from fractions import Fraction

# Synthetic levels can produce level polynomials whose primitive integer
# coefficients exceed CPython's default int->str limit on the way into the
# native plan; production levels are dyadic (proposed from floats) and do
# not, but the probe should not die on its own inputs.
sys.set_int_max_str_digits(0)

os.environ.setdefault("SPONG_WORKERS", "1")
os.environ.setdefault("SPONG_ENGINE", "native")

sys.path.insert(0, os.path.dirname(__file__))
import potential_corpus as pc                                # noqa: E402
from potential_corpus import portrait, sturm, zoo            # noqa: E402
from spong import _poly as P                                 # noqa: E402
from spong import level_roots, merge_tree                    # noqa: E402

argv = sys.argv[1:]
n_random = 60
if "--random" in argv:
    i = argv.index("--random")
    n_random = int(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if not a.startswith("--")]
names = args or list(zoo.names())
rng = random.Random(20260901)


def production_levels(m, z):
    seen = []
    orig = merge_tree.level_polynomial

    def rec(mm, c):
        seen.append(Fraction(c))
        return orig(mm, c)
    merge_tree.level_polynomial = rec
    try:
        portrait.certified_compute(m, view=z.default_view)
    finally:
        merge_tree.level_polynomial = orig
    return sorted(set(seen))


def sturm_roots(R):
    return sorted(sturm.isolate_roots(R), key=lambda iv: (iv.lo, iv.hi))


grand = {"levels": 0, "agree": 0, "ties": 0, "disagree": 0,
         "t_sturm": 0.0, "t_mono": 0.0}
for name in names:
    m, e, z = pc.context(name)
    try:
        LR = level_roots.LevelRoots(m, e)
    except ValueError as ex:
        print(f"{name:<28s} SKIP ({ex})")
        continue
    levels = production_levels(m, z)
    n_prod = len(levels)
    # Critical values as production proposes levels: from floats, so the
    # rationals are dyadic and the level polynomials stay small.
    crit = [Fraction.from_float(float(
        Fraction(m.C) - P.eval_at(m.beta, q.interval.mid) ** 2
        / P.eval_at(m.alpha, q.interval.mid))) for q in e.points]
    lo_c, hi_c = min(crit), max(crit)
    span = hi_c - lo_c if hi_c > lo_c else Fraction(1)
    for _ in range(n_random):
        levels.append(Fraction.from_float(
            float(lo_c - span / 2 + Fraction(rng.random()) * 2 * span)))
    for v in crit:
        for eps in (1e-6, -1e-6, 1e-14):
            levels.append(Fraction.from_float(float(v) + eps * (1 + abs(float(v)))))
    # the exact tie every model has: u = C at every B-root saddle
    levels.append(Fraction(m.C))
    stats = {"levels": 0, "agree": 0, "ties": 0, "disagree": 0,
             "t_sturm": 0.0, "t_mono": 0.0}
    for c in levels:
        stats["levels"] += 1
        R = merge_tree.level_polynomial(m, c)
        # Sturm path
        t = time.perf_counter()
        s_roots = sturm_roots(R)
        s_gaps = {}
        s_ok = True
        for i, q in enumerate(e.points):
            if q.kind == "degenerate":
                continue
            vs = merge_tree.value_sign(m, q, c)
            if vs is None:
                s_ok = False
                break
            s_gaps[i] = merge_tree._gap_index(m, R, q)
        t_s = time.perf_counter() - t
        # monotone path
        t = time.perf_counter()
        try:
            m_roots = LR.roots(c)
            m_gaps = {i: LR.gap_index(c, q) for i, q in enumerate(e.points)
                      if q.kind != "degenerate"}
            m_ok = True
        except level_roots.LevelTie:
            m_ok = False
        t_m = time.perf_counter() - t
        stats["t_sturm"] += t_s
        stats["t_mono"] += t_m
        if not s_ok or not m_ok:
            stats["ties"] += 1
            if s_ok != m_ok:
                stats["disagree"] += 1
                print(f"  {name} level {float(c):.12g}: tie disagreement "
                      f"sturm_ok={s_ok} mono_ok={m_ok}")
            continue
        bad = []
        if len(s_roots) != len(m_roots):
            bad.append(f"root count {len(s_roots)} vs {len(m_roots)}")
        else:
            for iv in m_roots:
                k = sturm.count_roots(R, iv.lo, iv.hi) if not iv.exact \
                    else int(P.eval_at(R, iv.lo) == 0)
                if k != 1:
                    bad.append(f"monotone interval [{float(iv.lo)},"
                               f"{float(iv.hi)}] holds {k} Sturm roots")
        if s_gaps != m_gaps:
            bad.append(f"gap indices {s_gaps} vs {m_gaps}")
        # counts on random and straddling intervals
        for _ in range(4):
            a, b = sorted(Fraction(rng.uniform(-6, 6)) for _ in range(2))
            if LR.count(c, a, b) != sturm.count_roots(R, a, b):
                bad.append(f"count on ({float(a)},{float(b)}]")
        for iv in s_roots:
            a = iv.lo if iv.exact else iv.mid
            if LR.count(c, None, a) != sturm.count_roots(R, None, a):
                bad.append(f"count below {float(a)}")
        if bad:
            stats["disagree"] += 1
            print(f"  {name} level {float(c):.12g}: " + "; ".join(bad))
        else:
            stats["agree"] += 1
    print(f"{name:<28s} levels {stats['levels']:4d} (prod {n_prod:3d}) "
          f"agree {stats['agree']:4d} ties {stats['ties']:3d} "
          f"DISAGREE {stats['disagree']:3d}   "
          f"sturm {stats['t_sturm']:6.2f}s  monotone {stats['t_mono']:6.2f}s")
    for k in grand:
        grand[k] += stats[k]
print(f"\nTOTAL levels {grand['levels']} agree {grand['agree']} ties "
      f"{grand['ties']} DISAGREE {grand['disagree']}   "
      f"sturm {grand['t_sturm']:.2f}s  monotone {grand['t_mono']:.2f}s")
