"""What exactly stopped this branch?

Not a search for a pattern -- a readout of the obstacle, per branch, for one
case.  Every abort site already records why it gave up; `charts.py` stashes a
`conditioning_refusal` block in `branch.diag` whenever the global handoff will
not condition, with the margins that decided it.  Nothing was reading it.

    python scripts/case_profile.py 555999196 --mode directed
    python scripts/case_profile.py 555999196 --mode directed --full-diag

Seeds are the ones printed in the ensemble JSONL, so any case in a run is
reproducible on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

from spong import atlas, portrait                      # noqa: E402
from qualify import directed_model, random_model       # noqa: E402

FINISHED = ("capture", "box_exit")


def _num(x, width=13):
    if x is None:
        return f"{'-':>{width}}"
    try:
        return f"{float(x):>{width}.5g}"
    except (TypeError, ValueError):
        return f"{str(x):>{width}}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--full-diag", action="store_true",
                    help="dump every diag key for each unfinished branch")
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    built = generate(random.Random(args.seed), args.degree)
    if built is None or built[0] is None:
        raise SystemExit("generator declined this seed")
    m, spec = built
    print(f"seed {args.seed}   {spec}")
    try:
        print(f"  d_eff {atlas.effective_degree(m)}   "
              f"legal_max_b {atlas.legal_max_b(m):.6g}")
    except Exception as exc:                            # noqa: BLE001
        print(f"  (atlas: {type(exc).__name__}: {exc})")

    t0 = time.perf_counter()
    p = portrait.certified_compute(m)
    wall = time.perf_counter() - t0
    top = p.ledger.get("topology", {})
    print(f"  {top.get('status')}   {top.get('resolution_reason')}   "
          f"level {top.get('geometry_level')}   {wall:.2f}s")

    # Every saddle, so an aborting branch can be placed among its neighbours
    # rather than described in isolation.
    print(f"\n  saddles:  " + "  ".join(
        f"b={float(q.b):.5g} (a={float(q.a):.4g})" for q in p.enumeration.saddles))

    unfinished = [(i, br) for i, br in enumerate(p.branches)
                  if br.term not in FINISHED]
    print(f"\n  {len(p.branches)} branches, {len(unfinished)} unfinished")
    for i, br in enumerate(p.branches):
        sb = br.diag.get("saddle_b")
        mark = "   " if br.term in FINISHED else "-> "
        print(f"  {mark}{i:>3} {br.kind:<9}{br.term:<28}"
              f"n={len(br.Y):>7}  b*={_num(sb, 11)}")

    if not unfinished:
        print("\n  nothing to explain: every branch finished")
        return 0

    for i, br in unfinished:
        print(f"\n  === branch {i}: {br.kind}/{br.term} "
              f"at b* = {_num(br.diag.get('saddle_b'), 1).strip()} ===")
        # The stub is what the abort sites talk about: how far it reached,
        # whether the global field was ready at its endpoint, how many steps
        # the critical chart managed.
        for key in ("materialized_stub", "stub_reach", "stub_physical_reach",
                    "stub_handoff_chord", "stub_global_field_ready",
                    "stub_endpoint_evaluator", "critical_chart",
                    "critical_order", "critical_steps", "handoff_certified",
                    "unstable_direction", "stable_sign"):
            if key in br.diag:
                print(f"      {key:<28s}{br.diag[key]}")

        refusal = br.diag.get("conditioning_refusal")
        if refusal:
            print("      conditioning_refusal:")
            for key, value in refusal.items():
                print(f"        {key:<32s}{_num(value)}")
        else:
            print("      (no conditioning_refusal block -- this branch "
                  "stopped somewhere else)")

        # WHICH REGIME?  A regime-2 branch is pinned at a root of A' with
        # A'' < 0 (an attracting trap, a local max of A) while |a| runs out;
        # its validity gate 2|B'/A'| DIVERGES there, so no a-threshold exists
        # and a shrunken box cannot help it -- it wants the vertical tail
        # b* + 2B'(b*)/(A''(b*) a) instead.  A regime-1 branch has a finite
        # gate and a box bound would stop it at the lock point.
        Y = np.asarray(br.Y, dtype=float)
        a_end, b_end = float(Y[-1, 0]), float(Y[-1, 1])
        Ac = np.asarray([float(c) for c in m.alpha])[::-1]
        Bc = np.asarray([float(c) for c in m.beta])[::-1]
        Apc, Appc, Bpc = (np.polyder(Ac), np.polyder(np.polyder(Ac)),
                          np.polyder(Bc))
        rr = np.roots(Apc)
        rr = np.sort(rr[np.abs(rr.imag) < 1e-9*np.maximum(1.0, np.abs(rr))].real)
        traps = [(float(r), float(np.polyval(Appc, r))) for r in rr
                 if float(np.polyval(Appc, r)) < 0.0]
        Av, Apv, Bpv = (np.polyval(Ac, b_end), np.polyval(Apc, b_end),
                        np.polyval(Bpc, b_end))
        astar = np.polyval(Bc, b_end)/Av if Av else float("nan")
        gate = abs(2.0*Bpv/Apv) if Apv else float("inf")
        print(f"      end (a, b)                  ({a_end:.6g}, {b_end:.6g})")
        print(f"      |a*| there                  {abs(astar):.6g}")
        print(f"      2|B'/A'| there              {gate:.6g}")
        print(f"      validity gate max(...)      "
              f"{max(abs(astar), gate):.6g}"
              f"   |a| = {abs(a_end):.6g}"
              f"   {'CLEARS' if abs(a_end) > 3*max(abs(astar), gate) else 'DOES NOT CLEAR'}")
        if traps:
            near = min(traps, key=lambda t: abs(t[0]-b_end))
            print(f"      nearest attracting trap     b* = {near[0]:.8g}"
                  f"  (A'' = {near[1]:.4g})"
                  f"   |b_end - b*| = {abs(b_end-near[0]):.4g}")
            print(f"      all traps (A'=0, A''<0):    "
                  + ", ".join(f"{t[0]:.6g}" for t in traps))
            verdict = ("REGIME 2 (pinned at a trap -- box bound cannot help)"
                       if abs(b_end-near[0]) < 1e-3*max(1.0, abs(near[0]))
                       else "regime 1 (away from every trap)")
            print(f"      -> {verdict}")
        else:
            print("      no attracting traps in this model -> regime 1")

        if args.full_diag:
            rest = {k: v for k, v in br.diag.items()
                    if k not in ("conditioning_refusal",)}
            print("      full diag:")
            print("        " + json.dumps(rest, indent=2,
                                          default=str).replace("\n", "\n        "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
