#!/usr/bin/env python3
"""Measure existing zoo portraits independently; does not change their verdicts.

python scripts/credibility_probe.py minimal-quartet tricky-d11 > evidence.jsonl
Every chord is measured. Regions are reported separately, not silently skipped.
"""
import collections
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from spong import credibility, model, portrait, zoo


def summarize(rows):
    measured = [r for r in rows if r["status"] == "measured"]
    return {"segments": len(rows),
            "statuses": dict(collections.Counter(r["status"] for r in rows)),
            "max_sin_squared": max((r["sin_squared"] for r in measured), default=None),
            "opposed_midpoint_flow": sum(r["flow_alignment"] == -1 for r in measured),
            "orthogonal_midpoint_flow": sum(r["flow_alignment"] == 0 for r in measured),
            "reversed_loss": sum(r["loss_direction"] == -1 for r in rows),
            "constant_loss": sum(r["loss_direction"] == 0 for r in rows)}


def probe(name):
    case = zoo.get(name)
    moments = model.moments_normal01 if case.moment_dist == "normal01" else model.moments_uniform01
    m = model.build(list(case.f), list(case.g), moments(2*max(len(case.f), len(case.g))-1))
    start = time.perf_counter()
    p = portrait.certified_compute(m, view=case.default_view)
    construction_seconds = time.perf_counter()-start
    branches = []
    suffixes = p.ledger["topology"].get("terminal_suffixes", [])
    for i, br in enumerate(p.branches):
        start = time.perf_counter()
        rows = credibility.level_normal_reference(m, br.Y, 1 if br.kind == "stable" else -1)
        elapsed = time.perf_counter()-start
        germ = int(br.diag.get("critical_steps", 0))
        terminal = suffixes[i].get("start") if i < len(suffixes) else None
        terminal = len(rows) if terminal is None else max(germ, int(terminal))
        branches.append({"branch": i, "kind": br.kind, "term": br.term,
                         "seconds": elapsed, "all": summarize(rows),
                         "local_launch": summarize(rows[:germ]),
                         "continuation": summarize(rows[germ:terminal]),
                         "terminal_representation": summarize(rows[terminal:])})
    return {"case": name, "status": p.ledger["topology"]["status"],
            "construction_seconds": construction_seconds,
            "reference_seconds": sum(b["seconds"] for b in branches),
            "branches": branches}


if __name__ == "__main__":
    os.environ["SPONG_WORKERS"] = "1"
    for name in sys.argv[1:] or ["minimal-quartet"]:
        print(json.dumps(probe(name)), flush=True)
