#!/usr/bin/env python3
"""Diff two normalize_retry outputs per seed: statuses, reasons, times.

    python scripts/engine_diff.py out/engine-random-sturm.jsonl \\
                                  out/engine-random-monotone.jsonl

Written for the level-root engine comparison (SPONG_LEVEL_ROOTS), but it
compares any two runs of the same ensemble: every seed present in both
files is compared on after.status and after.reason; differing seeds are
listed with both verdicts and both times; totals and the wall-time ratio
follow.  Seeds present in only one file are reported, not compared.
"""

import json
import sys
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        after = r.get("after") or {}
        rows[int(r["seed"])] = (
            after.get("status", r.get("error", "?")), after.get("reason"),
            float(after.get("seconds") or 0.0), r.get("case"))
    return rows


a_path, b_path = sys.argv[1], sys.argv[2]
A, B = load(a_path), load(b_path)
common = sorted(set(A) & set(B))
only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
diff = [(s, A[s], B[s]) for s in common
        if (A[s][0], A[s][1]) != (B[s][0], B[s][1])]
print(f"{a_path}: {len(A)} rows   {b_path}: {len(B)} rows   "
      f"common {len(common)}")
if only_a or only_b:
    print(f"  only in first: {only_a[:10]}{'...' if len(only_a) > 10 else ''}")
    print(f"  only in second: {only_b[:10]}{'...' if len(only_b) > 10 else ''}")
print(f"\n{len(diff)} seeds differ in status/reason")
for s, x, y in diff:
    print(f"  seed {s:>11} case {x[3]}: {x[0]}:{x[1]} ({x[2]:.1f}s)  ->  "
          f"{y[0]}:{y[1]} ({y[2]:.1f}s)")
cert_a = sum(1 for s in common if A[s][0] == "certified")
cert_b = sum(1 for s in common if B[s][0] == "certified")
t_a = sum(A[s][2] for s in common)
t_b = sum(B[s][2] for s in common)
print(f"\ncertified: {cert_a} vs {cert_b} of {len(common)}")
print(f"wall: {t_a:.0f}s vs {t_b:.0f}s  (ratio {t_b / t_a if t_a else 0:.2f})")
