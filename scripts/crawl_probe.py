#!/usr/bin/env python3
"""Anatomy of a crawling continuation segment.

    python scripts/crawl_probe.py                          # directed:1414065525, br9
    python scripts/crawl_probe.py --case directed:1414065525 --flow 1
    python scripts/crawl_probe.py --steps 200000 --at 10,100,1000,10000,100000

The port gave the crawl a number: on directed seed 1414065525 the unstable
branch off the a*=0 saddle takes 3.5M native steps with 294k halvings, and
the two stable branches at the same saddle are carried to the box by
~155k fast-GL4 floor rescues -- GL6 fails at the floor chord on essentially
every step and the rung meant for isolated singular points propagates the
branch one floor-chord at a time.  This probe looks at that.

Part 1 replays the recorded segment through the C (capped) and reports the
state along the crawl: b, w = a - a*(b), |w| relative to |a*|, the chord,
the chart velocities and their ratio, |grad L|, the depth gauge.

Part 2 stands at a few crawl vertices and takes ONE step attempt the way
the loop does -- GL6 on the active chart at the observed chord and its
doublings up to ds -- printing the stage result and the descent test's
expected / actual / slack, with GL4 beside it.  That is the mechanism: why
GL6 refuses there and GL4 does not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import charts, _native                      # noqa: E402
from segment_corpus import context, find_local         # noqa: E402

CORPUS = REPO / "tests" / "corpus" / "continue_curve.json"


def entry_for(case: str, flow: int, index: int | None):
    entries = [e for e in json.loads(CORPUS.read_text())
               if e["case"] == case and e["input"]["flow"] == flow]
    if not entries:
        raise SystemExit(f"no corpus segment for {case} flow={flow}; "
                         f"record it: python scripts/segment_corpus.py "
                         f"record {case}")
    if index is not None:
        entries = [e for e in entries if e["index"] == index]
    # the longest recorded one is the crawl
    return max(entries, key=lambda e: e["output"]["n_points"])


def replay_native(m, i, max_steps):
    kernel = m._native_kernel
    flat = [c for t in i["targets"] for c in t]
    out = _native.continue_curve(
        kernel, float(m.C), float(i["b"]), float(i["w"]), int(i["flow"]),
        flat, 2e-3 if i["cap_r"] is None else float(i["cap_r"]),
        [float(x) for x in i["box"]], float(i["ds"]),
        -1.0 if i["ds0"] is None else float(i["ds0"]),
        None if i["shallow_gate"] is None else [float(x) for x in i["shallow_gate"]],
        int(max_steps), int(i["centered_local_at"] is not None))
    term_code, reason, switches, b_end, w_end, taken, rejected, blob, diag = out
    pts = np.frombuffer(blob, dtype=float).reshape(-1, 2)
    return pts, charts._NATIVE_TERM.get(term_code, term_code), taken, rejected, diag


def state_row(m, a, b, flow, chord):
    w = a - m.s_a_star(b)
    vb, vw = charts._s_velocities(m, b, w)
    vb, vw = flow*vb, flow*vw
    chart = "slow" if abs(vw) <= charts.R_SWITCH*abs(vb) else "fast"
    g = m.gradL(a, b)
    kappa = charts._s_depth_gauge_floor(m, b)
    return (f"b={b:+.10e} a={a:+.3e} w={w:+.3e} |w/a*|={abs(w)/max(abs(m.s_a_star(b)),1e-300):.1e}"
            f" chord={chord:.2e} vb={vb:+.3e} vw={vw:+.3e} |vw/vb|={abs(vw)/max(abs(vb),1e-300):.2e}"
            f" {chart} |gradL|={float(np.hypot(*g)):.2e} L={float(m.L(a,b)):.10e} kappa={kappa:.2e}")


def one_attempt(m, b, w, flow, cur, chart):
    """Mirror one loop attempt at chord `cur` on `chart`; return dict."""
    native = m._native_kernel
    vb, vw = charts._s_velocities(m, b, w)
    vb, vw = flow*vb, flow*vw
    res = {}
    for method in ("gl6", "gl4"):
        try:
            if chart == "slow":
                h = cur/(1.0 + (vw/vb)**2)**0.5*(1.0 if vb > 0 else -1.0)
                step = native.slow_step if method == "gl6" else native.slow_step_gl4
                w_new = step(b, w, h); b_new = b + h
            else:
                h = cur/(1.0 + (vb/vw)**2)**0.5*(1.0 if vw > 0 else -1.0)
                step = native.fast_step if method == "gl6" else native.fast_step_gl4
                b_new = step(w, b, h); w_new = w + h
        except (ZeroDivisionError, FloatingPointError, OverflowError) as ex:
            res[method] = f"raised {type(ex).__name__}"
            continue
        if not (np.isfinite(b_new) and np.isfinite(w_new)):
            res[method] = f"stage NaN (h={h:.2e})"
            continue
        a_prev = m.s_a_star(b) + w
        a_new = m.s_a_star(b_new) + w_new
        da, db = a_new - a_prev, b_new - b
        expected = flow*float(m.gradL(a_prev, b) @ np.array([da, db]))
        actual = flow*float(m.L(a_new, b_new) - m.L(a_prev, b))
        slack = 64.0*np.finfo(float).eps*(1.0 + abs(float(m.L(a_prev, b))))
        ok = not (expected >= 0.0 or actual > 1e-4*expected + slack)
        res[method] = (f"{'PASS' if ok else 'FAIL'} h={h:+.2e} da={da:+.2e} db={db:+.2e}"
                       f" expected={expected:+.3e} actual={actual:+.3e}"
                       f" ratio={actual/expected if expected else float('nan'):+.3f}"
                       f" slack={slack:.1e}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="directed:1414065525")
    ap.add_argument("--flow", type=int, default=1)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--at", default="1,2,3,5,10,30,100,300,1000,3000,10000,"
                                    "30000,100000,199999")
    args = ap.parse_args(argv)

    m, e, _z = context(args.case)
    entry = entry_for(args.case, args.flow, args.index)
    i = entry["input"]
    print(f"{args.case} segment {entry['index']}: flow={i['flow']} "
          f"b0={i['b']:.10g} w0={i['w']:.3e} ds={i['ds']:.3e} ds0={i['ds0']}"
          f" box={i['box']} targets={len(i['targets'])}"
          f" centered_local={i['centered_local_at']}")
    cur0 = i["ds"] if i["ds0"] is None else min(i["ds"], max(i["ds0"], 1e-12))
    floor = cur0/128.0
    print(f"launch chord cur0={cur0:.3e}  continuation_floor={floor:.3e}"
          f"  recorded: term={entry['output']['term']} "
          f"n_points={entry['output']['n_points']}")

    saddle = min(e.saddles, key=lambda q: abs(float(q.b) - i["b"]))
    tag = " (a*=0 saddle: B has a root here)" if abs(float(saddle.a)) < 1e-8 else ""
    print(f"nearest saddle: b={float(saddle.b):.10g} a={float(saddle.a):.3e}"
          f" a*(b)={m.s_a_star(float(saddle.b)):.3e}{tag}")

    pts, term, taken, rejected, diag = replay_native(m, i, args.steps)
    print(f"\nnative replay capped at {args.steps}: term={term} taken={taken}"
          f" rejected={rejected} n_points={len(pts)}")
    print(f"rescues: { {k: v for k, v in diag.items() if k != 'step_failure'} }")
    if "step_failure" in diag:
        print(f"step_failure: {diag['step_failure']}")

    print("\nstate along the crawl:")
    idx = sorted({int(k) for k in args.at.split(",") if int(k) < len(pts)})
    for k in idx:
        chord = float(np.hypot(*(pts[k] - pts[k-1]))) if k > 0 else 0.0
        print(f"  [{k:>7d}] " + state_row(m, float(pts[k, 0]), float(pts[k, 1]),
                                         i["flow"], chord))
    if len(pts) > 2:
        chords = np.hypot(*(pts[1:] - pts[:-1]).T)
        print(f"  chords: min={chords.min():.2e} median={np.median(chords):.2e}"
              f" max={chords.max():.2e}  floor={floor:.2e}"
              f"  b: {pts[0,1]:.6g} -> {pts[-1,1]:.6g}  a: {pts[0,0]:.3e} -> {pts[-1,0]:.3e}")

    print("\none step attempt at crawl vertices (GL6 as the loop takes it; GL4 beside):")
    for k in [j for j in idx if j > 0][-4:]:
        a, b = float(pts[k, 0]), float(pts[k, 1])
        w = a - m.s_a_star(b)
        chord = float(np.hypot(*(pts[k] - pts[k-1])))
        vb, vw = charts._s_velocities(m, b, w)
        vb, vw = i["flow"]*vb, i["flow"]*vw
        chart = "slow" if abs(vw) <= charts.R_SWITCH*abs(vb) else "fast"
        other = "fast" if chart == "slow" else "slow"
        print(f"  vertex {k}: b={b:.10g} w={w:.3e} chart={chart} observed chord={chord:.2e}")
        cur = max(chord, floor)
        while cur <= i["ds"]*1.0000001:
            r = one_attempt(m, b, w, i["flow"], cur, chart)
            print(f"    cur={cur:.2e} {chart}: gl6 {r['gl6']}")
            print(f"    {'':>12} {chart}: gl4 {r['gl4']}")
            if cur == max(chord, floor):
                r2 = one_attempt(m, b, w, i["flow"], cur, other)
                print(f"    {'':>12} {other}: gl6 {r2['gl6']}")
                print(f"    {'':>12} {other}: gl4 {r2['gl4']}")
            cur *= 8.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
