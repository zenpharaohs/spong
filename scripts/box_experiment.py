"""Does the chord scale explain the stall?  Inner cluster versus whole skeleton.

Seed 953953598's skeleton is BIMODAL: three critical points near |b| ~ 0.2
and one minimum out at b = -40960.  legal_max_b = 61441 is correctly sized to
contain all of them -- but the target chord ds = (box span)/30000 is then set
by the outer cluster, so the inner branches are traced with a chord roughly
1e5 times their own geometry.

This asks one question and does not pretend to be a fix: if the box is cut to
the INNER cluster alone -- accepting that the remote minimum falls outside it,
so the result is NOT a certification -- do the inner branches trace cleanly?

  * If yes, the chord/box coupling is the cost driver and the fix is to
    decouple the tracing scale from the box that must span the skeleton.
  * If no, the stiffness is intrinsic to those separatrices and the
    Sundman/collocation machinery in mse-bundle is what is actually needed.

    python scripts/box_experiment.py 953953598 --mode directed
"""

from __future__ import annotations

import argparse
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

from spong import atlas, engine, portrait, sturm        # noqa: E402
from qualify import directed_model, random_model        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--gap", type=float, default=10.0,
                    help="a b-gap this many times the inner span starts a "
                         "new cluster")
    ap.add_argument("--pad", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=200000,
                    help="hard cap so a stalled run reports instead of hanging")
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    built = generate(random.Random(args.seed), args.degree)
    if built is None or built[0] is None:
        raise SystemExit("generator declined this seed")
    m, spec = built
    print(f"seed {args.seed}   {spec}")
    print(f"  legal_max_b = {atlas.legal_max_b(m):.6g}")

    e = sturm.enumerate_critical_points(m)
    pts = sorted(e.points, key=lambda q: float(q.b))
    print("  skeleton:")
    for q in pts:
        print(f"    {q.kind:<7} b = {float(q.b):>16.8g}   a = {float(q.a):>14.6g}")

    # Cluster in b: a gap far larger than the running span starts a new group.
    groups, cur = [], [pts[0]]
    for prev, q in zip(pts, pts[1:]):
        span = max(float(cur[-1].b) - float(cur[0].b), 1e-12)
        if float(q.b) - float(prev.b) > args.gap * max(span, 1e-6):
            groups.append(cur); cur = [q]
        else:
            cur.append(q)
    groups.append(cur)
    print(f"\n  {len(groups)} cluster(s) in b:")
    for gi, grp in enumerate(groups):
        print(f"    cluster {gi}: {len(grp)} points, "
              f"b in [{float(grp[0].b):.6g}, {float(grp[-1].b):.6g}]")

    inner = max(groups, key=len)
    bs = [float(q.b) for q in inner]
    as_ = [float(q.a) for q in inner]
    bspan = max(max(bs)-min(bs), 1e-3)
    aspan = max(max(as_)-min(as_), 1e-3)
    bc, ac = 0.5*(max(bs)+min(bs)), 0.5*(max(as_)+min(as_))
    box = (ac - args.pad*aspan, ac + args.pad*aspan,
           bc - args.pad*bspan, bc + args.pad*bspan)
    ds = (abs(box[1]-box[0]) + abs(box[3]-box[2])) / 30000.0
    print(f"\n  inner-cluster box {tuple(round(x, 8) for x in box)}")
    print(f"    target chord ds = {ds:.6g}"
          f"   (against {(2*atlas.legal_max_b(m)*2)/30000.0:.6g}"
          f" from legal_max_b)")

    # Trace the inner cluster's branches DIRECTLY -- no audit, no certificate.
    # This is a tracing experiment, not a portrait: the remote minimum is
    # outside this box, so nothing here is a certification.
    print(f"\n  {'saddle b':>14}{'dir':>5}{'kind':>10}{'term':>26}{'n':>10}"
          f"{'secs':>9}")
    print("  " + "-"*74)
    stub_e = sturm.materialize_stubs(m, e)
    for q in stub_e.points:
        if q.kind != "saddle" or not any(
                abs(float(q.b) - float(r.b)) < 1e-12 for r in inner):
            continue
        for sign in (+1, -1):
            for kind in ("stable", "unstable"):
                t0 = time.perf_counter()
                try:
                    if kind == "stable":
                        br = engine.trace_stable(
                            m, float(q.b), sign, box=box,
                            critical_local=None, critical_stub=None)
                    else:
                        br = engine.trace_unstable(
                            m, float(q.b), sign, box=box)
                except Exception as exc:                # noqa: BLE001
                    print(f"  {float(q.b):>14.6g}{sign:>5}{kind:>10}"
                          f"{'ERROR ' + type(exc).__name__:>26}")
                    continue
                print(f"  {float(q.b):>14.6g}{sign:>5}{kind:>10}"
                      f"{br.term:>26}{len(br.Y):>10}"
                      f"{time.perf_counter()-t0:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
