#!/usr/bin/env python3
"""Where does a stalled case spend its hour?  Instrument the continuation
dispatch on one ensemble seed.

    python scripts/delegate_probe.py --seed 1414065525 [--mode directed]
                                     [--degree 5] [--cap 4000]

Wraps _native.continue_curve to log every DELEGATE (reason, steps the C
port took before handing back) and charts._continue_curve_python to run the
re-run with max_steps capped at --cap, timing each capped replay.  The
branch terms reported are therefore NOT the production verdict -- a capped
replay ends in abort_max_steps -- but the cost per Python step times the
uncapped max_steps is the hour, or is not.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import charts, portrait, _native            # noqa: E402
from qualify import directed_model, random_model       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=("directed", "random"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--cap", type=int, default=4000,
                    help="max_steps for each delegated Python replay")
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    m, spec = generate(random.Random(args.seed), args.degree)
    print(f"seed {args.seed}: {spec}")

    native_calls = []
    orig_native = _native.continue_curve

    def logged_native(*a, **k):
        t = time.perf_counter()
        out = orig_native(*a, **k)
        term_code, reason, switches, b_end, w_end, taken, rejected, _ = out
        native_calls.append({
            "term_code": term_code, "reason": reason, "taken": taken,
            "rejected": rejected, "max_steps": a[-1],
            "b0": a[2], "w0": a[3], "flow": a[4], "b_end": b_end,
            "sec": time.perf_counter()-t})
        return out
    _native.continue_curve = logged_native

    python_calls = []
    orig_python = charts._continue_curve_python

    def capped_python(m_, b0, w0, flow, targets, box, ds, max_steps=200000,
                      **kw):
        t = time.perf_counter()
        out = orig_python(m_, b0, w0, flow, targets, box, ds,
                          max_steps=min(max_steps, args.cap), **kw)
        pts, term, switches, (b_end, w_end) = out
        sec = time.perf_counter()-t
        python_calls.append({
            "b0": b0, "w0": w0, "flow": flow, "ds": ds,
            "requested_max_steps": max_steps, "cap": args.cap,
            "term": term, "n_pts": len(pts), "b_end": b_end,
            "sec": sec, "sec_per_step": sec/max(1, min(max_steps, args.cap)),
            "projected_full_sec": sec/max(1, min(max_steps, args.cap))
            * max_steps})
        return out
    charts._continue_curve_python = capped_python

    t0 = time.perf_counter()
    p = portrait.compute(m, _skip_audit=True)
    print(f"geometry (capped replays) {time.perf_counter()-t0:.1f}s")
    for i, br in enumerate(p.branches):
        delegates = {k: v for k, v in br.diag.items()
                     if k.startswith("native_delegate_")}
        print(f"  br{i:<2} {br.kind:<8} {br.term:<24} n={len(br.Y):<7}"
              f" saddle_b={br.diag.get('saddle_b')!s:<22} {delegates}")
    print(f"\n{len(native_calls)} native continue_curve calls; "
          f"DELEGATEs:")
    for c in native_calls:
        if c["term_code"] == charts._NATIVE_DELEGATE:
            print(f"  reason={charts._DELEGATE_REASON.get(c['reason'], c['reason'])}"
                  f" taken={c['taken']} rejected={c['rejected']}"
                  f" max_steps={c['max_steps']} flow={c['flow']}"
                  f" b0={c['b0']:.6g} -> b_end={c['b_end']:.6g}"
                  f" ({c['sec']:.2f}s in C)")
    print(f"\n{len(python_calls)} Python replays (capped at {args.cap}):")
    for c in python_calls:
        print(f"  flow={c['flow']} b0={c['b0']:.6g} -> b_end={c['b_end']:.6g}"
              f" term={c['term']} n_pts={c['n_pts']} {c['sec']:.1f}s"
              f" = {1e3*c['sec_per_step']:.2f} ms/step;"
              f" requested max_steps={c['requested_max_steps']}"
              f" -> projected {c['projected_full_sec']/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
