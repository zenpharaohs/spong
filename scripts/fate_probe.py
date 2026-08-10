"""Launch- and terminal-time fate discovery -- the Lehmer filter.

    python scripts/fate_probe.py                        # 20 directed models
    python scripts/fate_probe.py 0 3                    # directed cases 0, 3
    python scripts/fate_probe.py dead-neuron-far-saddle-d3 tricky-d11
                                                        # zoo cases by name

Start with the simple zoo cases before the nasty directed ones.

For every unstable branch: the certified candidate set (minima + open
ends) at the first post-graph sample; for captures, the same filter at
the last measured sample (``terminal``), which is where launch-unforced
captures become forced; and whether the traced target lies inside its
candidates.  A target OUTSIDE its candidate set is a soundness alarm for
either the filter or the trace.  The summary counts launch-forced,
terminal-forced, unforced, declined, and alarms -- the data for how far
endpoint-closeness-only qualification reaches.

Runs geometry only (audit skipped where the signature allows): the filter
needs the enumeration and samples, not a verdict.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

from spong import fates, model, portrait, zoo               # noqa: E402
from qualify import directed_model                           # noqa: E402


def _zoo_model(name):
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu), f"zoo:{name}"


def _directed(wanted):
    rng = random.Random(20260806)
    for case in range(20):
        seed = rng.randrange(2 ** 31)
        if wanted is not None and case not in wanted:
            continue
        sub = random.Random(seed)
        m, spec = directed_model(sub, 5)
        if m is None:
            continue
        yield m, f"[{case}] seed {seed}  {spec}"


def main() -> int:
    args = sys.argv[1:]
    names = [a for a in args if not a.isdigit()]
    numbers = {int(a) for a in args if a.isdigit()}
    cases = ([_zoo_model(nm) for nm in names] if names
             else _directed(numbers or None))

    launch_forced = terminal_forced = escape_forced = 0
    unforced = declined = alarms = 0
    for m, label in cases:
        try:
            p = portrait.compute(m, _skip_audit=True)
        except TypeError:                        # older compute signature
            p = portrait.compute(m)
        print(f"\n{label}", flush=True)
        for e in fates.fate_report(m, p.enumeration, p.branches):
            if e["kind"] != "unstable":
                continue
            i = e["branch"]
            if not e.get("certified"):
                declined += 1
                print(f"    br{i:<3d} {e['term']:<10s} "
                      f"filter declined: {e.get('reason')}")
                continue
            ends = ",".join(e["unbounded_ends"]) or "-"
            terminal = e.get("terminal_fates") or {}
            t_forced = bool(terminal.get("forced"))
            # An unbounded component with no critical point and exactly
            # one open end forces the ESCAPE fate: the orbit has nowhere
            # else to go (zero ends with zero minima cannot certify, by
            # the Euler equality).
            e_forced = (not e["minima"] and not e["saddles"]
                        and len(e["unbounded_ends"]) == 1)
            mark = ("FORCED@launch" if e["forced"]
                    else "ESCAPE-forced" if e_forced
                    else "forced@terminal" if t_forced
                    else f"{len(e['minima'])} minima")
            bad = (e.get("target_in_candidates") is False
                   or terminal.get("target_in_candidates") is False)
            launch_forced += e["forced"]
            escape_forced += (not e["forced"]) and e_forced
            terminal_forced += (not (e["forced"] or e_forced)) and t_forced
            unforced += not (e["forced"] or e_forced or t_forced)
            alarms += bad
            note = "  TARGET OUTSIDE CANDIDATES" if bad else ""
            print(f"    br{i:<3d} {e['term']:<10s} "
                  f"shift={e['slack_shift']:<3d} {mark:<16s} "
                  f"ends={ends}{note}")

    print(f"\nforced at launch: {launch_forced}   "
          f"escape-forced: {escape_forced}   "
          f"forced only at terminal: {terminal_forced}   "
          f"unforced: {unforced}   declined: {declined}   "
          f"target-outside alarms: {alarms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
