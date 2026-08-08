#!/usr/bin/env python3
"""Replay the segment corpus through the native _continue_curve port.

    python scripts/check_continue_native.py
    python scripts/check_continue_native.py tricky-d11

Compiles src/spong_continue.c into a temporary shared library and drives it
with ctypes, replaying every entry in tests/corpus/continue_curve.json.  No
rebuild of the extension module, no binding, no change to charts.py: this
answers the only question that matters at this stage -- does the C reproduce
the reference implementation exactly -- before any of that plumbing exists.

Comparison is on term, switch count, vertex count, and the SHA-256 of the
packed point array, so a single differing double anywhere in a 14,000-vertex
polyline fails.  DELEGATE is reported separately: it is not a failure, it is
the port declining a path the corpus does not cover, and the caller is
specified to re-run those segments in Python.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "src" / "c" / "spong_continue.c"
INC = REPO / "include"
CORPUS = REPO / "tests" / "corpus" / "continue_curve.json"

sys.path.insert(0, str(HERE))
from segment_corpus import context, find_local          # noqa: E402

TERMS = {
    0: "capture", 1: "box_exit", 2: "enter_shallow",
    3: "abort_stationary", 4: "abort_switch_limit", 5: "abort_nonfinite",
    6: "abort_step_failure", 7: "abort_max_steps",
    100: "DELEGATE", 101: "NEED_CAPACITY",
}
REASONS = {0: "none", 1: "floor_ladder", 2: "stall_trim", 3: "centered_chart"}


class Field(ctypes.Structure):
    _fields_ = [(n, ctypes.POINTER(ctypes.c_double))
                for n in ("A", "Ap", "App", "B", "Bp", "Bpp", "N", "Np")] + \
               [(n, ctypes.c_size_t)
                for n in ("nA", "nAp", "nApp", "nB", "nBp", "nBpp",
                          "nN", "nNp")] + \
               [("C", ctypes.c_double)]


class Result(ctypes.Structure):
    _fields_ = [
        ("term", ctypes.c_int), ("delegate_reason", ctypes.c_int),
        ("switches", ctypes.c_int),
        ("b_end", ctypes.c_double), ("w_end", ctypes.c_double),
        ("n_points", ctypes.c_size_t),
        ("steps_taken", ctypes.c_uint64), ("steps_rejected", ctypes.c_uint64),
    ]


def build() -> ctypes.CDLL:
    out = Path(tempfile.mkdtemp()) / "libspong_continue.dylib"
    # setup.py builds _native.c with -O3 and no float flags, so match that by
    # default.  SPONG_CFLAGS overrides it -- worth trying
    # "-O3 -ffp-contract=off" to rule contraction in or out as a cause of
    # last-bit divergence.
    extra = os.environ.get("SPONG_CFLAGS", "-O3").split()
    cmd = ["cc", *extra, "-shared", "-fPIC", f"-I{INC}",
           str(SRC), "-o", str(out), "-lm"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(out))
    lib.spong_continue_curve.restype = ctypes.c_int
    lib.spong_continue_curve.argtypes = [
        ctypes.POINTER(Field),
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(Result),
    ]
    return lib


def _arr(t):
    a = (ctypes.c_double * len(t))(*[float(x) for x in t])
    return a


def field_for(m):
    """spong_continue_field from the model's pure-float coefficient tuples.

    These are exactly the arrays Model hands to _native.Kernel, plus C, which
    the Kernel does not carry.
    """
    names = ("_fa", "_fap", "_fapp", "_fb", "_fbp", "_fbpp", "_fn", "_fnp")
    keep = [_arr(getattr(m, n)) for n in names]
    f = Field()
    for slot, buf in zip(("A", "Ap", "App", "B", "Bp", "Bpp", "N", "Np"), keep):
        setattr(f, slot, ctypes.cast(buf, ctypes.POINTER(ctypes.c_double)))
    for slot, buf in zip(("nA", "nAp", "nApp", "nB", "nBp", "nBpp",
                          "nN", "nNp"), keep):
        setattr(f, slot, len(buf))
    f.C = float(m.C)
    return f, keep                      # keep buffers alive


def run(lib, entry):
    m, e, _z = context(entry["case"])
    i = entry["input"]
    f, _keep = field_for(m)

    tgt = [c for t in i["targets"] for c in t]
    tgt_buf = (ctypes.c_double * max(len(tgt), 1))(*(tgt or [0.0]))
    box_buf = (ctypes.c_double * 4)(*i["box"])
    gate = i["shallow_gate"]
    gate_buf = ((ctypes.c_double * 2)(*gate)) if gate else None

    res = Result()
    cap = 0
    # max_steps is DERIVED in charts._continue_curve -- a runaway guard at
    # max(200000, 128*diagonal/ds), now that the operative limit is the
    # arclength budget inside the engine.  Hard-coding 200000 here made the C
    # abort where the reference did not, which reads as a parity failure and
    # is not one.
    diagonal = ((i["box"][1] - i["box"][0]) ** 2
                + (i["box"][3] - i["box"][2]) ** 2) ** 0.5
    max_steps = int(max(200000.0, 128.0 * diagonal / max(i["ds"], 1e-300)))
    for _attempt in range(4):
        pts = (ctypes.c_double * max(2 * cap, 2))()
        lib.spong_continue_curve(
            ctypes.byref(f), i["b"], i["w"], i["flow"],
            tgt_buf, len(i["targets"]),
            0.0 if i["cap_r"] is None else i["cap_r"],
            box_buf, i["ds"], -1.0 if i["ds0"] is None else i["ds0"],
            gate_buf, max_steps, pts, cap, ctypes.byref(res))
        if res.term != 101:
            break
        cap = res.n_points + 64
    arr = np.ctypeslib.as_array(pts, shape=(max(cap, 1) * 2,))[:res.n_points * 2]
    arr = arr.reshape(-1, 2).copy()
    return res, arr


def main() -> int:
    if not CORPUS.exists():
        print("no corpus — run scripts/segment_corpus.py record first")
        return 1
    entries = json.loads(CORPUS.read_text())
    if len(sys.argv) > 1:
        entries = [x for x in entries if x["case"] in set(sys.argv[1:])]

    lib = build()
    print()
    ok = delegated = bad = 0
    reasons: dict = {}
    for entry in entries:
        want = entry["output"]
        res, arr = run(lib, entry)
        term = TERMS.get(res.term, str(res.term))
        if term == "DELEGATE":
            delegated += 1
            key = REASONS.get(res.delegate_reason, str(res.delegate_reason))
            reasons[key] = reasons.get(key, 0) + 1
            continue
        got_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        deltas = []
        if term != want["term"]:
            deltas.append(f"term: {want['term']} -> {term}")
        if res.switches != want["switches"]:
            deltas.append(f"switches: {want['switches']} -> {res.switches}")
        if res.n_points != want["n_points"]:
            deltas.append(f"n_points: {want['n_points']} -> {res.n_points}")
        if got_sha != want["sha256"]:
            deltas.append("points differ")
            for idx, a0, b0 in want["sample"]:
                if idx < len(arr):
                    a1, b1 = float(arr[idx, 0]), float(arr[idx, 1])
                    if (a1, b1) != (a0, b0):
                        deltas.append(f"  vertex {idx}: ({a0!r},{b0!r})"
                                      f" -> ({a1!r},{b1!r})")
                        break
        if deltas:
            bad += 1
            print(f"{entry['case']} segment {entry['index']}:")
            for line in deltas:
                print(f"    {line}")
        else:
            ok += 1

    print(f"\n{ok} identical, {delegated} delegated, {bad} DIFFER"
          f"   ({len(entries)} segments)")
    if reasons:
        print("delegation reasons:")
        for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"   {n:5d}  {key}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
