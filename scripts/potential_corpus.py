#!/usr/bin/env python3
"""Record and check a corpus of potential-rate segments — the parity tier.

    python scripts/potential_corpus.py record
    python scripts/potential_corpus.py check
    python scripts/potential_corpus.py check tricky-d11

WHY
---
The three constant-potential-rate phases (prefix, level event, ascent) are
migrating to one C entry point, spong_potential_rate_segment.  Like
_continue_curve before them (scripts/segment_corpus.py), the goldens are too
coarse to develop the port against: an off-by-an-ulp accept/reject produces a
different — usually still plausible — polyline.  This corpus records what each
phase was ASKED during ordinary zoo portraits and what the PYTHON loop
answered, and the check demands that the Python oracle and the C entry point
both reproduce every answer to the last bit: vertices, endpoint, term, and
every counter.

Recording always runs the Python loops (the executable specification),
whatever SPONG_ENGINE says — the corpus is spec truth, not engine output.
Checking replays each entry through BOTH the oracle and, when the binding is
present, the native entry point via the production dispatch wrapper.

The shared-arithmetic doctrine makes this meaningful: the oracle loops
evaluate L, grad L and H through Kernel.loss/gradient/hessian — the library's
Horner kernels, the same arithmetic the C uses — so the comparison judges
loop logic, not evaluators (charts._shared_field).
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
    from spong import charts, engine, model, portrait, sturm, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import charts, engine, model, portrait, sturm, zoo


# --------------------------------------------------------------------------
# model context, shared by record and check
# --------------------------------------------------------------------------

_CONTEXT: dict = {}


def context(name: str):
    """(model, enumeration, case) for a zoo case, built once."""
    if name not in _CONTEXT:
        z = zoo.get(name)
        f, g = list(z.f), list(z.g)
        mu = (model.moments_normal01 if z.moment_dist == "normal01"
              else model.moments_uniform01)(2 * max(len(f), len(g)) - 1)
        m = model.build(f, g, mu)
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
        _CONTEXT[name] = (m, e, z)
    return _CONTEXT[name]


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


def _critical_list(critical):
    if critical is None:
        return None
    return [[float(a), float(b)]
            for a, b in np.asarray(critical, dtype=float).reshape(-1, 2)]


# --------------------------------------------------------------------------
# the three phases, one uniform run interface
# --------------------------------------------------------------------------
#
# run_oracle / run_native both return
#     (pts, term, extra, diag_entry)
# where extra carries the phase's other return values (prefix: b_end, w_end;
# level event: captured) and diag_entry is the phase's own diagnostics dict —
# every field of which is part of the parity contract.

def run_oracle(kind: str, m, i: dict):
    diag: dict = {}
    critical = (None if i["critical"] is None
                else np.asarray(i["critical"], dtype=float))
    if kind == "prefix":
        pts, b_end, w_end, term = charts._potential_rate_prefix_python(
            m, i["a0"], i["b0"], tuple(i["target"]), tuple(i["box"]),
            i["cap_r"], diag, n_levels=i["n_levels"], critical=critical)
        return pts, term, {"b_end": b_end, "w_end": w_end}, \
            diag.get("potential_rate")
    if kind == "level_event":
        pts, term, captured = charts._potential_rate_level_event_python(
            m, i["a0"], i["b0"], [tuple(t) for t in i["targets"]],
            tuple(i["box"]), i["cap_r"], diag,
            n_levels=i["n_levels"], critical=critical)
        events = diag.get("candidate_level_events")
        return pts, term, \
            {"captured": None if captured is None else list(captured)}, \
            (events[-1] if events else None)
    pts, term = charts._potential_rate_box_exit_python(
        m, tuple(i["start"]), tuple(i["box"]), i["ds"], diag,
        max_steps=i["max_steps"], critical=critical)
    return pts, term, {}, diag.get("potential_rate_ascent")


def run_native(kind: str, m, i: dict):
    """The production dispatch wrapper, with the native engine forced."""
    saved = engine._active
    engine._active = engine._ENGINES["native"]
    try:
        return run_dispatch(kind, m, i)
    finally:
        engine._active = saved


def run_dispatch(kind: str, m, i: dict):
    diag: dict = {}
    critical = (None if i["critical"] is None
                else np.asarray(i["critical"], dtype=float))
    if kind == "prefix":
        pts, b_end, w_end, term = charts._potential_rate_prefix(
            m, i["a0"], i["b0"], tuple(i["target"]), tuple(i["box"]),
            i["cap_r"], diag, n_levels=i["n_levels"], critical=critical)
        return pts, term, {"b_end": b_end, "w_end": w_end}, \
            diag.get("potential_rate")
    if kind == "level_event":
        pts, term, captured = charts._potential_rate_level_event(
            m, i["a0"], i["b0"], [tuple(t) for t in i["targets"]],
            tuple(i["box"]), i["cap_r"], diag,
            n_levels=i["n_levels"], critical=critical)
        events = diag.get("candidate_level_events")
        return pts, term, \
            {"captured": None if captured is None else list(captured)}, \
            (events[-1] if events else None)
    pts, term = charts._potential_rate_box_exit(
        m, tuple(i["start"]), tuple(i["box"]), i["ds"], diag,
        max_steps=i["max_steps"], critical=critical)
    return pts, term, {}, diag.get("potential_rate_ascent")


def native_present() -> bool:
    try:
        from spong import _native
    except ImportError:
        return False
    return hasattr(_native, "potential_rate_segment")


def output_of(pts, term, extra, diag_entry) -> dict:
    out = {"term": term, "n_points": int(len(pts)),
           "sha256": _digest(pts), "sample": _sample(pts),
           "diag": diag_entry}
    out.update(extra)
    return out


def compare(expected: dict, got: dict) -> list[str]:
    bad = []
    for key in sorted(set(expected) | set(got)):
        if key == "sample":
            continue
        if expected.get(key) != got.get(key):
            bad.append(f"{key}: was {expected.get(key)!r}, "
                       f"now {got.get(key)!r}")
    if bad and expected.get("sample") != got.get("sample"):
        for e, g in zip(expected["sample"], got["sample"]):
            if e != g:
                bad.append(f"  first differing sample: vertex {e[0]} "
                           f"({e[1]!r},{e[2]!r}) -> ({g[1]!r},{g[2]!r})")
                break
    return bad


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------

def record_zoo(names) -> list[dict]:
    entries: list[dict] = []
    active = {"case": None}
    originals = {
        "prefix": charts._potential_rate_prefix,
        "level_event": charts._potential_rate_level_event,
        "ascent": charts._potential_rate_box_exit,
    }

    def note(kind, i, result):
        pts, term, extra, diag_entry = result
        entries.append({
            "kind": kind, "case": active["case"], "index": len(entries),
            "input": i, "output": output_of(pts, term, extra, diag_entry),
        })

    @functools.wraps(originals["prefix"])
    def rec_prefix(m, a0, b0, target, box, cap_r, engine_diag,
                   n_levels=12000, critical=None):
        i = {"a0": float(a0), "b0": float(b0),
             "target": [float(target[0]), float(target[1])],
             "box": [float(x) for x in box], "cap_r": float(cap_r),
             "n_levels": int(n_levels), "critical": _critical_list(critical)}
        result = run_oracle("prefix", m, i)
        note("prefix", i, result)
        pts, term, extra, diag_entry = result
        if diag_entry is not None:
            engine_diag["potential_rate"] = diag_entry
        return pts, extra["b_end"], extra["w_end"], term

    @functools.wraps(originals["level_event"])
    def rec_level_event(m, a0, b0, targets, box, cap_r, engine_diag,
                        n_levels=2048, critical=None):
        i = {"a0": float(a0), "b0": float(b0),
             "targets": [[float(t[0]), float(t[1])] for t in targets],
             "box": [float(x) for x in box], "cap_r": float(cap_r),
             "n_levels": int(n_levels), "critical": _critical_list(critical)}
        result = run_oracle("level_event", m, i)
        note("level_event", i, result)
        pts, term, extra, diag_entry = result
        if diag_entry is not None:
            engine_diag.setdefault(
                "candidate_level_events", []).append(diag_entry)
        captured = (None if extra["captured"] is None
                    else tuple(extra["captured"]))
        return pts, term, captured

    @functools.wraps(originals["ascent"])
    def rec_ascent(m, start, box, ds, engine_diag,
                   max_steps=100000, critical=None):
        i = {"start": [float(start[0]), float(start[1])],
             "box": [float(x) for x in box], "ds": float(ds),
             "max_steps": int(max_steps),
             "critical": _critical_list(critical)}
        result = run_oracle("ascent", m, i)
        note("ascent", i, result)
        pts, term, _extra, diag_entry = result
        if diag_entry is not None:
            engine_diag["potential_rate_ascent"] = diag_entry
        return pts, term

    charts._potential_rate_prefix = rec_prefix
    charts._potential_rate_level_event = rec_level_event
    charts._potential_rate_box_exit = rec_ascent
    try:
        for name in names:
            active["case"] = name
            m, e, z = context(name)
            portrait.certified_compute(m, view=z.default_view)
            print(f"recorded {name:<38s} {len(entries)} segments so far")
    finally:
        charts._potential_rate_prefix = originals["prefix"]
        charts._potential_rate_level_event = originals["level_event"]
        charts._potential_rate_box_exit = originals["ascent"]
    return entries


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def path_for() -> Path:
    return CORPUS / "potential_rate.json"


# --------------------------------------------------------------------------
# platform tags
# --------------------------------------------------------------------------
#
# A parity corpus records BIT-IDENTICAL arithmetic, and the shared fused
# kernels (spong_gauss2, spong_jet) compile with the platform's contraction
# defaults -- FMA on arm64/clang -- by design: the arithmetic at that level
# is allowed to be what it is, and useful optimizations that move a few
# ulps are not forbidden.  The consequence is that bit-parity is defined
# PER PLATFORM: a corpus recorded on one machine class is not a parity
# oracle on another (the Python oracle leg itself replays the recording
# platform's kernel arithmetic).  Each corpus therefore carries a sidecar
# <name>.platform.json naming where it was recorded; checks elsewhere skip
# with that reason rather than fail.

def platform_tag() -> dict:
    import platform as _pf
    return {"machine": _pf.machine(), "system": _pf.system()}


def sidecar_for(corpus_path: Path) -> Path:
    return corpus_path.with_suffix(".platform.json")


def write_platform_tag(corpus_path: Path) -> None:
    sidecar_for(corpus_path).write_text(
        json.dumps(platform_tag(), indent=1, sort_keys=True) + "\n")


def platform_mismatch(corpus_path: Path) -> str | None:
    """A skip reason when this machine cannot replay the corpus, else None."""
    sidecar = sidecar_for(corpus_path)
    if not sidecar.exists():
        return None                      # untagged: assume replayable
    tag = json.loads(sidecar.read_text())
    here = platform_tag()
    if (tag.get("machine"), tag.get("system")) == \
            (here["machine"], here["system"]):
        return None
    return (f"corpus recorded on {tag.get('machine')}-{tag.get('system')}, "
            f"this is {here['machine']}-{here['system']}: the fused kernels "
            f"round differently across platforms, so bit-parity holds per "
            f"platform (re-record here to check this platform)")


def do_record(argv) -> int:
    names = list(argv) if argv else list(zoo.names())
    entries = record_zoo(names)
    CORPUS.mkdir(parents=True, exist_ok=True)
    path_for().write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
    write_platform_tag(path_for())
    kinds: dict = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    print(f"\n{len(entries)} segments -> {path_for()}")
    for kind, n in sorted(kinds.items()):
        print(f"   {n:5d}  {kind}")
    return 0


def do_check(argv) -> int:
    if not path_for().exists():
        print("no corpus — run record first")
        return 1
    mismatch = platform_mismatch(path_for())
    if mismatch:
        print(f"NOTE: {mismatch}\n      (differences below are expected "
              f"platform arithmetic, not regressions)")
    entries = json.loads(path_for().read_text())
    if argv:
        entries = [e for e in entries if e["case"] in set(argv)]
    with_native = native_present()
    if not with_native:
        print("native potential_rate_segment ABSENT — "
              "checking the oracle against the recording only")
    bad = 0
    for entry in entries:
        m, _e, _z = context(entry["case"])
        kind, i = entry["kind"], entry["input"]
        legs = [("oracle", run_oracle(kind, m, i))]
        if with_native:
            legs.append(("native", run_native(kind, m, i)))
        for leg, result in legs:
            got = output_of(*result)
            deltas = compare(entry["output"], got)
            if deltas:
                bad += 1
                print(f"{entry['case']} {kind} segment "
                      f"{entry['index']} [{leg}]:")
                for line in deltas:
                    print(f"    {line}")
    print(f"\n{len(entries)} segments, {bad} disagreeing replays")
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
