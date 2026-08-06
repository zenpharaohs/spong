#!/usr/bin/env python3
"""Record and check a corpus of _continue_curve calls — the segment tier.

    python scripts/segment_corpus.py record
    python scripts/segment_corpus.py check
    python scripts/segment_corpus.py check tricky-d11

WHY A SEGMENT TIER
------------------
Three levels of pinning now exist, at increasing granularity:

    tests/test_native_parity.py   one Gauss STEP, bit-comparable
    this corpus                   one ENGINE SEGMENT, bit-comparable
    tests/golden/                 one PORTRAIT, assertions + tolerances

The goldens are too coarse to develop a C port of _continue_curve against.
That loop carries a lot of policy — R_SWITCH chart switching under
MAX_SWITCHES, the shallow handoff with its 5%-slaved test and its stall trim,
eight halvings to continuation_floor, the descent-realization test that picks
the flow-connected root of a multi-root stage system, and the
chart×method floor-fallback ladder ending in normalized arclength.  A port
can get one of those clauses wrong and still produce a portrait whose skeleton
and certificates pass.  This records what each call was ASKED and what it
ANSWERED, so the port is developed against exact expected output.

It also measures which paths matter.  The recorded diag deltas count how often
each rung of the floor-fallback ladder fires and how often the normalized
rescue is reached; a rung that never fires on the zoo can be ported last and
left delegating to Python meanwhile.

WHAT IS STORED
--------------
Inputs are stored in full — they are all scalars, short tuples, or target
lists.  Outputs are stored as term, switch count, final state, vertex count, a
SHA-256 of the raw point bytes, and a few sampled vertices.  Full polylines
would be hundreds of megabytes; the hash detects any difference at all and the
samples make a failure legible.

The jet argument is not serializable (it holds a live LocalKernel).  Its
critical point is recorded instead, and replay rebuilds the model and
enumeration from the zoo case and looks the jet up again — so the replayed
call receives the same object graph, not a reconstruction of it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "tests" / "corpus"

try:
    from spong import charts, model, portrait, sturm, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import charts, model, portrait, sturm, zoo


# --------------------------------------------------------------------------
# model context, shared by record and check
# --------------------------------------------------------------------------

_CONTEXT: dict = {}


def context(name: str):
    """(model, enumeration) for a zoo case, built once."""
    if name not in _CONTEXT:
        z = zoo.get(name)
        f, g = list(z.f), list(z.g)
        mu = (model.moments_normal01 if z.moment_dist == "normal01"
              else model.moments_uniform01)(2 * max(len(f), len(g)) - 1)
        m = model.build(f, g, mu)
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
        _CONTEXT[name] = (m, e, z)
    return _CONTEXT[name]


def find_local(e, a: float, b: float):
    """The jet whose critical point matches a recorded one."""
    for q in e.points:
        if (abs(float(q.a) - a) <= 1e-12 * (1 + abs(a))
                and abs(float(q.b) - b) <= 1e-12 * (1 + abs(b))):
            return q.local
    return None


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def _digest(points) -> str:
    arr = np.asarray(points, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sample(points, k: int = 8):
    arr = np.asarray(points, dtype=float)
    if len(arr) <= 2 * k:
        idx = range(len(arr))
    else:
        idx = sorted({0, 1, 2, *np.linspace(0, len(arr) - 1, k).astype(int),
                      len(arr) - 3, len(arr) - 2, len(arr) - 1})
    return [[int(i), float(arr[i, 0]), float(arr[i, 1])] for i in idx]


def _scalars(x):
    if x is None:
        return None
    return [float(v) for v in x]


def record_zoo(names) -> list[dict]:
    entries: list[dict] = []
    original = charts._continue_curve
    active = {"case": None}

    @functools.wraps(original)
    def recording(m, b, w, flow, targets, box, ds, **kw):
        diag = kw.get("engine_diag")
        before = dict(diag) if isinstance(diag, dict) else {}
        local = kw.get("centered_local")
        result = original(m, b, w, flow, targets, box, ds, **kw)
        pts, term, switches, (b_end, w_end) = result
        after = dict(diag) if isinstance(diag, dict) else {}
        delta = {k: v for k, v in after.items()
                 if k not in before or before[k] != v}
        entries.append({
            "case": active["case"],
            "index": len(entries),
            "input": {
                "b": float(b), "w": float(w), "flow": int(flow),
                "targets": [[float(t[0]), float(t[1])] for t in targets],
                "box": _scalars(box), "ds": float(ds),
                "cap_r": (None if kw.get("cap_r") is None
                          else float(kw["cap_r"])),
                "ds0": (None if kw.get("ds0") is None
                        else float(kw["ds0"])),
                "shallow_gate": _scalars(kw.get("shallow_gate")),
                "centered_local_at": (None if local is None else
                                      [float(local.a), float(local.b)]),
            },
            "output": {
                "term": term,
                "switches": int(switches),
                "b_end": float(b_end), "w_end": float(w_end),
                "n_points": int(len(pts)),
                "sha256": _digest(pts),
                "sample": _sample(pts),
            },
            # Which exotic paths this call actually exercised.  Diag is
            # mutated in place and shared across a branch, so record the
            # delta rather than the accumulated state.
            "diag_delta": {k: (v if isinstance(v, (int, float, str, bool))
                               else str(v))
                           for k, v in delta.items()},
        })
        return result

    charts._continue_curve = recording
    try:
        for name in names:
            active["case"] = name
            m, e, z = context(name)
            portrait.certified_compute(m, view=z.default_view)
            print(f"recorded {name:<38s} {len(entries)} segments so far")
    finally:
        charts._continue_curve = original
    return entries


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def replay(entry: dict):
    m, e, _z = context(entry["case"])
    i = entry["input"]
    local = (None if i["centered_local_at"] is None
             else find_local(e, *i["centered_local_at"]))
    diag: dict = {}
    pts, term, switches, (b_end, w_end) = charts._continue_curve(
        m, i["b"], i["w"], i["flow"],
        [tuple(t) for t in i["targets"]], tuple(i["box"]), i["ds"],
        cap_r=i["cap_r"], ds0=i["ds0"],
        shallow_gate=(None if i["shallow_gate"] is None
                      else tuple(i["shallow_gate"])),
        engine_diag=diag, centered_local=local)
    return {"term": term, "switches": int(switches),
            "b_end": float(b_end), "w_end": float(w_end),
            "n_points": int(len(pts)), "sha256": _digest(pts),
            "sample": _sample(pts)}, diag


def compare(expected: dict, got: dict) -> list[str]:
    bad = []
    for key in ("term", "switches", "n_points", "sha256"):
        if expected[key] != got[key]:
            bad.append(f"{key}: was {expected[key]!r}, now {got[key]!r}")
    for key in ("b_end", "w_end"):
        if expected[key] != got[key]:
            bad.append(f"{key}: was {expected[key]!r}, now {got[key]!r}")
    if bad and expected["sample"] != got["sample"]:
        for (i0, a0, b0), (i1, a1, b1) in zip(expected["sample"],
                                              got["sample"]):
            if (i0, a0, b0) != (i1, a1, b1):
                bad.append(f"  first differing sample: vertex {i0} "
                           f"({a0!r},{b0!r}) -> ({a1!r},{b1!r})")
                break
    return bad


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def path_for() -> Path:
    return CORPUS / "continue_curve.json"


def do_record(argv) -> int:
    names = list(argv) if argv else list(zoo.names())
    entries = record_zoo(names)
    CORPUS.mkdir(parents=True, exist_ok=True)
    path_for().write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")

    paths: dict = {}
    for entry in entries:
        for key in entry["diag_delta"]:
            paths[key] = paths.get(key, 0) + 1
    print(f"\n{len(entries)} segments -> {path_for()}")
    print("diag keys observed (how often a call touched each path):")
    for key, n in sorted(paths.items(), key=lambda kv: -kv[1]):
        print(f"   {n:5d}  {key}")
    return 0


def do_check(argv) -> int:
    if not path_for().exists():
        print("no corpus — run record first")
        return 1
    entries = json.loads(path_for().read_text())
    if argv:
        entries = [e for e in entries if e["case"] in set(argv)]
    bad = 0
    for entry in entries:
        got, _diag = replay(entry)
        deltas = compare(entry["output"], got)
        if deltas:
            bad += 1
            print(f"{entry['case']} segment {entry['index']}:")
            for line in deltas:
                print(f"    {line}")
    print(f"\n{len(entries) - bad}/{len(entries)} segments identical")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "check"):
        print(__doc__)
        return 2
    try:
        from spong import engine
        print(f"engine: {engine.active_name()}")
    except ImportError:
        pass
    return (do_record if sys.argv[1] == "record" else do_check)(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
