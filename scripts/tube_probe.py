"""Log every exact_tube_at decision on the failing capture branches.

    python scripts/tube_probe.py [case]

The open contradiction: capture_probe reports L at the last measured point of
a failing branch as 0.1238242752 -- the target minimum's level -- while
inventory_probe at the same index reports level_upper = 1.5e16, RISING along
the branch.  A descent orbit's loss cannot rise, so either level_upper is not
the orbit's level, or the indices are not ordered as assumed, or one of the
probes reads the wrong thing.

This settles it by measurement, not inference.  SPONG_TUBE_LOG makes
topology._capture_certificate write, for every index exact_tube_at is
ACTUALLY called with:

    L_fp64        binary64 L at the point (what capture_probe measured)
    L_exact       the exact rational loss at the same dyadic point
    slack         the strictness slack scale/2^48 added to form the level
    level_upper   L_exact + slack, the level the inventory tests (what
                  inventory_probe reported)
    certified / reason / bounded / saddles / minima   the four conditions,
                  separately, with the inventory's own decline reason

plus one capture_search line per branch naming the threshold in force
(lowest merging saddle level, u_infinity, and the gate result at `last`).
Certifying sibling branches log too -- their lines are the control, showing
what a succeeding tube looks like on the same model.

If the code reads as suspected, the log will show level_upper - L_exact ==
slack exactly, with L_exact tracking L_fp64 near the minimum's level while
slack grows like the polynomial scale |A|(1+a^2)+2|aB|+|B| over 2^48 as |b|
grows -- i.e. the rising sequence was the SLACK, not the orbit's level, and
the level jumps the merging saddle by many orders while the min-saddle gap
is ~5e-6.  But that is the hypothesis the log exists to confirm or kill.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait                                 # noqa: E402
from qualify import directed_model                          # noqa: E402


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    log = REPO / "out" / "tube_probe.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log.unlink()
    os.environ["SPONG_TUBE_LOG"] = str(log)

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
        marker = f"[{case}] seed {seed}  {spec}"
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"# {marker}\n")
        p = portrait.certified_compute(m)
        top = p.ledger["topology"]
        outcome = top.get("resolution_reason") or top.get("status")
        print(f"{marker}  ->  {outcome}")
        if only is not None:
            break

    print(f"\nlog written to {log}")
    print("case 0 alone reproduces the contradiction's numbers:")
    print("    python scripts/tube_probe.py 0")
    print("failing branches are the target= tags whose lines never reach")
    print("bounded=True; grep one tag to read a single branch's search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
