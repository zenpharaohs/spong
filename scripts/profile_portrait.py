#!/usr/bin/env python3
"""Profile the certified portrait pipeline, per zoo case.

    python scripts/profile_portrait.py                       # default cases
    python scripts/profile_portrait.py tricky-d11 --top 30
    python scripts/profile_portrait.py --all                 # incl. d17-thrash
    python scripts/profile_portrait.py --wall nonnearest-saddle-connection

Two views of the same run, read together:

  1. cProfile of portrait.certified_compute, attributed to spong functions
     (sorted by cumulative and by own time).  This says WHERE the seconds go.
  2. The branches' own diagnostics: per-phase accepted/rejected/capped/
     arclength/turn-rejected step counts, vertex totals, and the audit's
     event counts.  This says WHY -- which controls fired, how often, and
     whether a phase spent its steps on progress or on resolution.

Both are written to out/profile-<case>.txt, and a one-line summary per
case is printed.  --wall profiles the explorer's near-wall configuration
(legal-size trace box, native engine) for a wall family at its stored
coordinate, which is the case that drove this weekend's tracer changes.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import os
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    from spong import atlas, model, portrait, wall_shoot, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import atlas, model, portrait, wall_shoot, zoo

DEFAULT_CASES = ("quadratic-stiff", "nonnearest-attachment", "tricky-d11")
PHASE_KEYS = ("potential_rate", "potential_rate_ascent", "centered_arrival",
              "candidate_level_events")


def _model_for_case(name):
    z = zoo.get(name)
    f, g = list(z.f), list(z.g)
    n = 2*max(len(f), len(g))-1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(f, g, mu), z.default_view


def _phase_rows(branches):
    """Aggregate the per-phase step counters across branches."""
    totals = {}
    vertices = 0
    terms = {}
    for br in branches:
        vertices += len(br.Y)
        terms[br.term] = terms.get(br.term, 0)+1
        for key in PHASE_KEYS:
            entries = br.diag.get(key)
            if entries is None:
                continue
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                row = totals.setdefault(key, {})
                for field, value in entry.items():
                    if isinstance(value, (int, float)) and "steps" in field:
                        row[field] = row.get(field, 0)+value
                row["phases"] = row.get("phases", 0)+1
    return vertices, terms, totals


def _spong_stats(profile: cProfile.Profile, top: int) -> str:
    buffer = io.StringIO()
    stats = pstats.Stats(profile, stream=buffer)
    stats.sort_stats("cumulative")
    buffer.write("=== cumulative (spong only) ===\n")
    stats.print_stats(r"spong", top)
    stats.sort_stats("tottime")
    buffer.write("\n=== own time (all) ===\n")
    stats.print_stats(top)
    return buffer.getvalue()


def profile_case(name: str, top: int, out_dir: Path) -> str:
    m, view = _model_for_case(name)
    profile = cProfile.Profile()
    t0 = time.perf_counter()
    profile.enable()
    p = portrait.certified_compute(m, view=view)
    profile.disable()
    elapsed = time.perf_counter()-t0
    return _report(name, p, profile, elapsed, top, out_dir)


def profile_wall(name: str, top: int, out_dir: Path) -> str:
    family = zoo.get_wall_family(name)
    m = wall_shoot.rheostat_model(family, family.wall_parameter)
    bmax = atlas.legal_max_b(m)
    amax = bmax/max(1.0, math.sqrt(max(1, atlas.effective_degree(m))))
    view = (-amax, amax, -bmax, bmax)
    profile = cProfile.Profile()
    t0 = time.perf_counter()
    profile.enable()
    p = portrait.certified_compute(m, view=view)
    profile.disable()
    elapsed = time.perf_counter()-t0
    return _report(f"wall-{name}", p, profile, elapsed, top, out_dir)


def _report(label, p, profile, elapsed, top, out_dir) -> str:
    topo = p.ledger["topology"]
    vertices, terms, phases = _phase_rows(p.branches)
    lines = [
        f"{label}: {elapsed:.2f}s  status {topo['status']}"
        f" ({topo.get('resolution_reason')})  level {topo.get('geometry_level')}"
        f"  branches {len(p.branches)}  vertices {vertices}"
        f"  unattested {topo.get('unattested_turn_count', 0)}"
        f"  ambiguous {topo.get('ambiguous_count', 0)}"
        f"  forbidden {topo.get('forbidden_count', 0)}",
        f"  terms: {terms}",
        f"  engine {os.environ.get('SPONG_ENGINE', 'python')}  "
        f"workers {os.environ.get('SPONG_WORKERS', '1')}",
    ]
    for attempt in topo.get("attempts", []):
        lines.append(f"  attempt level {attempt['geometry_level']}: "
                     f"{attempt['status']} {attempt['reason']} "
                     f"{attempt['elapsed_sec']:.2f}s")
    for key, row in phases.items():
        fields = ", ".join(f"{k} {v}" for k, v in sorted(row.items()))
        lines.append(f"  {key}: {fields}")
    summary = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir/f"profile-{label}.txt"
    path.write_text(summary+"\n\n"+_spong_stats(profile, top))
    print(summary)
    print(f"  -> {path}")
    return summary


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--all", action="store_true",
                        help="every zoo case, including d17-thrash")
    parser.add_argument("--wall", action="append", default=[],
                        help="profile a wall family at its stored coordinate "
                             "in the explorer's legal-size box")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", default=str(REPO/"out"))
    args = parser.parse_args(argv)
    cases = (list(zoo.names()) if args.all else
             args.cases or list(DEFAULT_CASES))
    out_dir = Path(args.out)
    for name in cases:
        profile_case(name, args.top, out_dir)
    for name in args.wall:
        profile_wall(name, args.top, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
