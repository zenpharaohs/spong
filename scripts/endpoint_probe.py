"""Which unstable endpoint is uncertified, and what does the audit say?

    python scripts/endpoint_probe.py

unstable_endpoint_unresolved does NOT mean the far-field funnel refused.
funnel_probe showed the corridor accepting at width 2^-43 on a refusing model,
with all four sign tests holding at the endpoint and on the ray.

topology.audit records a per-branch verdict in unstable_ends, each with a kind
(finite_capture / infinity_escape / incomplete), certified, and a reason.  This
prints them for every refusing directed model, so the failing endpoint kind is
named rather than assumed.  Capture certificates go through exact_tube_at and
the sublevel inventory -- a different mechanism from the corridor, and the one
that dominates the audit's cost.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait                                # noqa: E402
from qualify import directed_model                         # noqa: E402


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    rng = random.Random(20260806)
    tally: dict = {}

    for case in range(20):
        seed = rng.randrange(2 ** 31)
        sub = random.Random(seed)
        m, spec = directed_model(sub, 5)
        if m is None:
            continue
        p = portrait.certified_compute(m)
        top = p.ledger["topology"]
        if top.get("resolution_reason") != "unstable_endpoint_unresolved":
            continue
        print(f"\n[{case}] seed {seed}  {spec}")
        for end in top.get("unstable_ends", []):
            i = end.get("branch")
            br = p.branches[i] if i is not None and i < len(p.branches) else None
            mark = "ok " if end.get("certified") else "NO "
            key = (end.get("kind"), end.get("reason") or "-")
            if not end.get("certified"):
                tally[key] = tally.get(key, 0) + 1
            print(f"    {mark}br{i:<3d} kind={str(end.get('kind')):<16s} "
                  f"term={str(br.term) if br else '?':<28s} "
                  f"reason={end.get('reason')}")
            for k in ("method", "entry_index", "minimum",
                      "scaled_relative_half_width"):
                if k in end:
                    print(f"          {k} = {end[k]}")

    print("\nuncertified unstable endpoints by (kind, reason):")
    for k, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  {k[0]} / {k[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
