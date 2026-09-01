#!/usr/bin/env python3
"""Record and check a corpus of centered raw arrivals — the parity tier.

    python scripts/arrival_corpus.py record
    python scripts/arrival_corpus.py check
    python scripts/arrival_corpus.py check tricky-d11

The centered raw arrival (charts._centered_raw_arrival) finishes a known
connection with the regular target-centered gradient flow.  It has migrated
to one C entry point, spong_centered_arrival, over the relocated jet kernel
(spong_jet).  As for the potential-rate segments (scripts/potential_corpus.py,
whose helpers this shares), the goldens are too coarse to develop against:
this corpus records what the arrival was ASKED during ordinary zoo portraits
and what the PYTHON loop answered, and the check demands that the Python
oracle and the C entry point both reproduce every answer to the last bit.

Each entry identifies its jet by the target's coordinates, which re-find the
enumerated critical point (and its LocalJet) on replay.  Recording always
runs the Python oracle; checking replays through both.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import potential_corpus as pc                                # noqa: E402
from potential_corpus import charts, engine, portrait, zoo  # noqa: E402


def jet_for(e, at: float, bt: float):
    for q in e.points:
        if float(q.a) == at and float(q.b) == bt:
            return q.local
    raise KeyError(f"no enumerated critical point at ({at!r}, {bt!r})")


def run_oracle(m, e, i: dict):
    diag: dict = {}
    pts, term = charts._centered_raw_arrival_python(
        tuple(i["start"]), tuple(i["target"]),
        jet_for(e, *i["target"]), i["cap_r"], diag,
        max_steps=i["max_steps"])
    return pts, term, {}, diag.get("centered_arrival")


def run_dispatch(m, e, i: dict):
    diag: dict = {}
    pts, term = charts._centered_raw_arrival(
        tuple(i["start"]), tuple(i["target"]),
        jet_for(e, *i["target"]), i["cap_r"], diag,
        max_steps=i["max_steps"])
    return pts, term, {}, diag.get("centered_arrival")


def run_native(m, e, i: dict):
    saved = engine._active
    engine._active = engine._ENGINES["native"]
    try:
        return run_dispatch(m, e, i)
    finally:
        engine._active = saved


def native_present() -> bool:
    try:
        from spong import _native
    except ImportError:
        return False
    return hasattr(_native, "centered_arrival")


def record_zoo(names) -> list[dict]:
    entries: list[dict] = []
    active = {"case": None, "e": None}
    original = charts._centered_raw_arrival

    @functools.wraps(original)
    def rec(start, target, arrival_local, cap_r, engine_diag,
            max_steps=4096):
        i = {"start": [float(start[0]), float(start[1])],
             "target": [float(target[0]), float(target[1])],
             "cap_r": float(cap_r), "max_steps": int(max_steps)}
        result = run_oracle(None, active["e"], i)
        pts, term, _extra, diag_entry = result
        entries.append({
            "case": active["case"], "index": len(entries),
            "input": i, "output": pc.output_of(pts, term, {}, diag_entry),
        })
        if diag_entry is not None:
            engine_diag["centered_arrival"] = diag_entry
        return pts, term

    charts._centered_raw_arrival = rec
    try:
        for name in names:
            m, e, z = pc.context(name)
            active["case"], active["e"] = name, e
            portrait.certified_compute(m, view=z.default_view)
            print(f"recorded {name:<38s} {len(entries)} arrivals so far")
    finally:
        charts._centered_raw_arrival = original
    return entries


def path_for() -> Path:
    return pc.CORPUS / "centered_arrival.json"


def do_record(argv) -> int:
    names = list(argv) if argv else list(zoo.names())
    entries = record_zoo(names)
    pc.CORPUS.mkdir(parents=True, exist_ok=True)
    path_for().write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
    terms: dict = {}
    for entry in entries:
        t = entry["output"]["term"]
        terms[t] = terms.get(t, 0) + 1
    print(f"\n{len(entries)} arrivals -> {path_for()}")
    for t, n in sorted(terms.items()):
        print(f"   {n:5d}  {t}")
    return 0


def do_check(argv) -> int:
    if not path_for().exists():
        print("no corpus — run record first")
        return 1
    entries = json.loads(path_for().read_text())
    if argv:
        entries = [e for e in entries if e["case"] in set(argv)]
    with_native = native_present()
    if not with_native:
        print("native centered_arrival ABSENT — "
              "checking the oracle against the recording only")
    bad = 0
    for entry in entries:
        m, e, _z = pc.context(entry["case"])
        i = entry["input"]
        legs = [("oracle", run_oracle(m, e, i))]
        if with_native:
            legs.append(("native", run_native(m, e, i)))
        for leg, result in legs:
            got = pc.output_of(*result)
            deltas = pc.compare(entry["output"], got)
            if deltas:
                bad += 1
                print(f"{entry['case']} arrival {entry['index']} [{leg}]:")
                for line in deltas:
                    print(f"    {line}")
    print(f"\n{len(entries)} arrivals, {bad} disagreeing replays")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "check"):
        print(__doc__)
        return 2
    print(f"engine: {engine.active_name()} "
          f"(recording always uses the Python oracle)")
    return (do_record if sys.argv[1] == "record" else do_check)(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
