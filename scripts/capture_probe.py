"""Why does _capture_certificate decline?

    python scripts/capture_probe.py

endpoint_probe showed all 9 unstable_endpoint_unresolved refusals are the same
thing: finite_capture / no_level_tube_or_convex_capture_ball.  Exactly one
branch per model fails while its siblings certify by exact_level_tube, often
into the SAME minimum -- so it is one defect appearing nine times, not nine
problems.

_capture_certificate skips its whole level-tube search unless the last
MEASURED point is below the lowest saddle level above the target minimum
(`below_merging_level(last)`).  But capture itself is declared GEOMETRICALLY:
_segment_capture fires when a chord passes within cap_r = 4*ds of the target.
On a directed model the box is enormous, so ds and hence cap_r are large, and
capture can be declared while the trajectory is still well above the merging
level -- where no bounded one-minimum sublevel component exists yet.

This prints, for each failing branch: the loss at the last measured point, the
target minimum's level, the merging threshold, and the gap.  If L(last) sits
ABOVE the threshold the gate is the cause and the remedy is upstream -- either
descend further before declaring capture, or scale cap_r to the basin rather
than to the chord.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait                                # noqa: E402
from qualify import directed_model                         # noqa: E402


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    rng = random.Random(20260806)
    above = below = 0

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
        bad = [e for e in top.get("unstable_ends", [])
               if not e.get("certified")]
        if not bad:
            continue
        print(f"\n[{case}] seed {seed}  {spec}")
        for end in bad:
            i = end["branch"]
            br = p.branches[i]
            target = br.diag.get("target")
            print(f"    br{i}  term={br.term}  n={len(br.Y)}  target={target}")
            if target is None:
                print("        no target recorded")
                continue
            # nearest enumerated minimum to the recorded target
            q = min(p.enumeration.minima,
                    key=lambda z: (float(z.a) - target[0]) ** 2
                    + (float(z.b) - target[1]) ** 2)
            dist = float(np.hypot(float(q.a) - target[0],
                                  float(q.b) - target[1]))
            lmin = float(m.L(q.a, q.b))
            saddle_levels = sorted(
                float(m.L(s.a, s.b)) for s in p.enumeration.saddles
                if float(m.L(s.a, s.b)) > lmin)
            last = len(br.Y) - 2
            llast = float(m.L(float(br.Y[last, 0]), float(br.Y[last, 1])))
            gap = float(np.hypot(float(br.Y[last, 0]) - float(q.a),
                                 float(br.Y[last, 1]) - float(q.b)))
            print(f"        minimum ({float(q.a):.6g},{float(q.b):.6g}) "
                  f"target-match dist {dist:.3e}")
            print(f"        L(min) {lmin:.10g}   L(last measured) "
                  f"{llast:.10g}   distance to min {gap:.3e}")
            if saddle_levels:
                thr = saddle_levels[0]
                ok = llast < thr
                above += (not ok)
                below += ok
                print(f"        lowest saddle above min {thr:.10g}   "
                      + ("BELOW threshold (gate passes)" if ok
                         else "ABOVE threshold -- gate skips the search"))
            else:
                print("        no saddle above the minimum")

    print(f"\nfailing branches with last measured point above the "
          f"merging level: {above}")
    print(f"                                       below it:        {below}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
