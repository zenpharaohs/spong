"""What does _sublevel_component_inventory actually return on a failing capture?

    python scripts/inventory_probe.py

capture_probe ruled out the merging-level gate: all 9 failing branches are
BELOW it, and their targets match an enumerated minimum exactly.  Every failing
target is a prescribed dead-neuron minimum at large |b|, and L at the last
measured point equals L(minimum) to ten digits while the point is up to 123
units away -- the valley floor is flat to machine precision over that stretch.

exact_tube_at needs four things at once:

    inventory["certified"]           the exact work resolved
    inventory["bounded"]             the component is bounded
    not inventory["saddles"]         no saddle inside it
    len(inventory["minima"]) == 1    exactly one minimum

and declines silently if any fails.  In case 0 the lowest saddle above the
minimum is at 0.1238295 against a level of 0.1238243, so the component ought to
isolate -- which points at `certified` or `bounded` rather than at the
geometry.  This prints the whole dict at several indices along the failing
branch, so the declining condition is named rather than inferred.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait, topology                       # noqa: E402
from qualify import directed_model                          # noqa: E402


def show(m, enumeration, Y, index):
    inv = topology._sublevel_component_inventory(m, enumeration, Y[index])
    keys = ("certified", "reason", "bounded", "level_upper",
            "left_boundary", "right_boundary")
    parts = []
    for k in keys:
        if k in inv:
            v = inv[k]
            parts.append(f"{k}={v:.8g}" if isinstance(v, float)
                         else f"{k}={v}")
    parts.append(f"minima={len(inv.get('minima', []))}")
    parts.append(f"saddles={len(inv.get('saddles', []))}")
    print(f"        i={index:<7d} " + "  ".join(parts))


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rng = random.Random(20260806)

    for case in range(20):
        seed = rng.randrange(2 ** 31)
        if only is not None and case != only:
            continue
        sub = random.Random(seed)
        m, spec = directed_model(sub, 5)
        if m is None:
            continue
        p = portrait.certified_compute(m)
        top = p.ledger["topology"]
        if top.get("resolution_reason") != "unstable_endpoint_unresolved":
            continue
        bad = [e for e in top.get("unstable_ends", [])
               if not e.get("certified")]
        if not bad:
            continue
        print(f"\n[{case}] seed {seed}  {spec}")
        for end in bad:
            i = end["branch"]
            br = p.branches[i]
            last = len(br.Y) - 2
            lower = min(last, max(0, int(br.diag.get("critical_steps", 0))))
            print(f"    br{i}  n={len(br.Y)}  critical_steps={lower}")
            probes = sorted({lower, last // 4, last // 2,
                             3 * last // 4, last - 1, last})
            for idx in probes:
                if 0 <= idx <= last:
                    show(m, p.enumeration, br.Y, idx)
        if only is not None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
