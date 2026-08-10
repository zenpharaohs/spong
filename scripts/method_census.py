"""Which certificate route actually decides each branch.

    python scripts/method_census.py                    # cheap zoo cases
    python scripts/method_census.py tricky-d11
    python scripts/method_census.py --all              # includes the beast

The capture path now decides from the exact merge tree, with the slack
ladder and the strictly-convex ball left behind it as fallbacks.  The
argument says both are dead -- separating levels sit between consecutive
distinct critical values, so any point below its target's merging saddle
already lies in a component holding that minimum alone.  This measures it.

A `exact_level_tube` or `strictly_convex_ball` in the capture column means
the tree declined on a branch it should have decided: investigate before
concluding anything.  An all-`exact_merge_tree` census across the zoo and
the directed run is what licenses deleting the ladder from the capture
path -- code goes on measurements here, not on arguments.

The escape column is expected to stay mixed: the tree cannot force escape
until a trace descends below the outermost critical value, and a box_exit
may leave the box first, which is what the funnel is still for.
"""
from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

from spong import model, portrait, zoo                     # noqa: E402

CHEAP = ["near-slide-d2", "dead-neuron-far-saddle-d3", "quadratic-stiff",
         "minimal-quartet", "nonnearest-attachment"]
BEAST = ["tricky-d11", "linear-target-d17-thrash"]


def build_model(name):
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu), z


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--all"]
    names = args or (CHEAP + BEAST if "--all" in sys.argv else CHEAP)
    totals = collections.Counter()
    fallbacks = []
    for name in names:
        m, z = build_model(name)
        t0 = time.perf_counter()
        p = portrait.certified_compute(m, view=z.default_view)
        elapsed = time.perf_counter() - t0
        top = p.ledger["topology"]
        tally = collections.Counter()
        for end in top.get("unstable_ends", ()):
            key = (end["kind"],
                   end.get("method") if end.get("certified")
                   else f"UNCERTIFIED:{end.get('reason')}")
            tally[key] += 1
            totals[key] += 1
            if (end["kind"] == "finite_capture" and end.get("certified")
                    and end.get("method") != "exact_merge_tree"):
                fallbacks.append((name, end["branch"], end["method"]))
        print(f"\n{name}   status={top.get('status')}   ({elapsed:.1f}s)")
        if top.get("status") != "certified":
            # The census is about ROUTES; when a case refuses, the reason is
            # almost never in the capture column, so print what actually
            # decided the status rather than leaving it to inference.
            stable_bad = [x for x in top.get("stable_tails", ())
                          if not x.get("certified")]
            print(f"    reason={top.get('resolution_reason')}   "
                  f"forbidden={top.get('forbidden_count')}   "
                  f"ambiguous={top.get('ambiguous_count')}   "
                  f"uncertified stable tails={len(stable_bad)}   "
                  f"geometry_level={top.get('geometry_level')}")
            starts = [s.get("start") for s in top.get("terminal_suffixes", ())]
            print(f"    terminal suffix starts={starts}")
            for x in stable_bad[:4]:
                print(f"      stable br{x['branch']} "
                      f"reason={x.get('reason')}")
        for (kind, method), count in sorted(tally.items()):
            print(f"    {kind:<16s} {method:<34s} {count}")
        sys.stdout.flush()

    print("\n=== totals")
    for (kind, method), count in sorted(totals.items()):
        print(f"    {kind:<16s} {method:<34s} {count}")
    if fallbacks:
        print("\nCAPTURE FALLBACKS FIRED -- the tree declined here:")
        for name, branch, method in fallbacks:
            print(f"    {name} br{branch} -> {method}")
        print("Investigate before deleting anything.")
    else:
        print("\nNo capture fallback fired: the slack ladder and the convex "
              "ball are dead on these cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
