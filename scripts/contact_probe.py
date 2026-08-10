"""Why are contact events not being discharged?

    python scripts/contact_probe.py quadratic-stiff
    python scripts/contact_probe.py near-slide-d2

Runs the audit at geometry_level 0 ONLY -- the census reports the escalated
level 2, which is a consequence of the level-0 refusal, not its cause.

For every branch it prints length, terminal-suffix kind and start, and the
endpoint certificate's method.  Then, for each forbidden intersection, it
prints the exact reason the discharge test failed:

    both branches' suffix kinds and starts,
    whether each segment index is at or past its own suffix start,
    and, for two captures, the distance between their terminal minima.

`same_sublevel_end` requires ALL of: both suffix kinds `minimum_sublevel`,
both starts present, si >= start_i, sj >= start_j, and the two terminal
minima within allowed_radius.  One column will be False; that column is
the defect.  No inference -- just the failing conjunct.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

import numpy as np                                          # noqa: E402
from spong import model, portrait, zoo                      # noqa: E402


def build_model(name):
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu), z


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "quadratic-stiff"
    m, z = build_model(name)
    p = portrait.certified_compute(m, view=z.default_view,
                                   max_geometry_level=0)
    top = p.ledger["topology"]
    print(f"{name}  status={top.get('status')}  "
          f"reason={top.get('resolution_reason')}  "
          f"forbidden={top.get('forbidden_count')}  "
          f"ambiguous={top.get('ambiguous_count')}")

    suffixes = top.get("terminal_suffixes", [])
    ends = {x["branch"]: x for x in top.get("unstable_ends", ())}
    print("\nbranches:")
    for i, br in enumerate(p.branches):
        s = suffixes[i] if i < len(suffixes) else {}
        e = ends.get(i, {})
        print(f"  br{i:<3d} {br.kind:<9s} {br.term:<10s} n={len(br.Y):<7d} "
              f"suffix={str(s.get('kind')):<18s} start={s.get('start')} "
              f"method={e.get('method')} "
              f"certified={e.get('certified')}")

    forbidden = top.get("forbidden_intersections", [])
    print(f"\nforbidden intersections (showing up to 12 of "
          f"{top.get('forbidden_count')}):")
    for item in forbidden[:12]:
        i, j = item["branches"]
        si, sj = item["segments"]
        ti = suffixes[i] if i < len(suffixes) else {}
        tj = suffixes[j] if j < len(suffixes) else {}
        gi, gj = ti.get("start"), tj.get("start")
        past_i = gi is not None and si >= gi
        past_j = gj is not None and sj >= gj
        same_kind = ti.get("kind") == tj.get("kind")
        distance = None
        if ti.get("kind") == tj.get("kind") == "minimum_sublevel":
            distance = float(np.hypot(
                ti["terminal"][0] - tj["terminal"][0],
                ti["terminal"][1] - tj["terminal"][1]))
        print(f"  br{i}xbr{j} seg=({si},{sj}) "
              f"kinds=({ti.get('kind')},{tj.get('kind')}) same={same_kind} "
              f"starts=({gi},{gj}) past=({past_i},{past_j}) "
              f"minima_distance={distance}")

    print("\nlegend: discharge needs same kind, both past, and for two "
          "captures a minima distance within allowed_radius "
          f"(~{max(1024*np.finfo(float).eps*max(1.0, 1.0), 1e-11):.2e} "
          "scaled by the box).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
