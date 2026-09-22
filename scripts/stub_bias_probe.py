#!/usr/bin/env python3
"""How far is a materialized stub from the true invariant manifold?

WHY.  The production launch (sturm.materialize_stubs) carries a systematic
bias: the native shooter's root stays at 2.17770956136537 as its step shrinks,
2.6e-9 from the converged wall 2.177709563954816, and swapping ONLY the launch
for the high-order germ removes six orders of it (docs/wall_continuation.md).
Yet the stub's own grid_error falls 6.2e-7 -> 1.55e-7 -> 3.88e-8 under
refinement -- ratios of exactly 4.0, textbook second order.  A discretization
converging at its design order to a wrong answer is solving the WRONG
continuous problem accurately.  This measures which.

WHAT.  The native germ K(t) solves F(K(t)) = rho t K'(t) to order 10 at 192
bits, so a germ point is on the true manifold to far better than the stub.
Its DISTANCE TO THE STUB POLYLINE is therefore the stub's transverse error at
that place -- parameterization-free, which matters because the germ's t and
the stub's arclength are different variables.  Varying the launch radius r
and fitting err ~ r^p reads the defect off the exponent:

    p ~ 0          an offset: saddle location or eigen-direction origin
    p ~ 1          a wrong tangent: the eigenvector itself is off
    p ~ k+1        truncation: the local graph is only order-k accurate
    err grows as the stub is refined     something in how the grid is built

Run at Lambda = 1 so base (alpha, b) and physical (a, b) coincide.

    python scripts/stub_bias_probe.py
    python scripts/stub_bias_probe.py --radii 0.0005,0.001,0.002,0.004,0.008
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spong import model as model_mod, sturm, zoo                    # noqa: E402
from spong.wall_native import NativeSectionContinuation            # noqa: E402


def _polyline_distance(p, curve):
    """Euclidean distance from p to the polyline through curve."""
    p = np.asarray(p, float)
    pts = np.asarray(curve, float)
    best = math.inf
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i+1]
        d = b - a
        denom = float(d @ d)
        t = 0.0 if denom == 0.0 else min(1.0, max(0.0, float((p-a) @ d)/denom))
        best = min(best, float(np.hypot(*(p - (a + t*d)))))
    return best


def _crossings(evaluation):
    """The two Crossing records of an Evaluation, by position.

    Evaluation(lam, gap, level, source_crossing, target_crossing) per
    wall_native._decode; read by field order so a rename does not silently
    pick the wrong attribute.
    """
    fields = dataclasses.fields(evaluation)
    return getattr(evaluation, fields[3].name), getattr(evaluation, fields[4].name)


def _point(crossing):
    """First field of a Crossing is the (alpha, b) point."""
    first = getattr(crossing, dataclasses.fields(crossing)[0].name)
    return (float(first[0]), float(first[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nonnearest-saddle-connection")
    ap.add_argument("--radii", default="0.0005,0.001,0.002,0.004,0.008")
    ap.add_argument("--launch-order", type=int, default=10)
    args = ap.parse_args(argv)

    family = zoo.get_wall_family(args.family)
    base = zoo.get(family.base_case)
    n = 2*max(len(base.f), len(base.g)) - 1
    moments = (model_mod.moments_normal01 if base.moment_dist == "normal01"
               else model_mod.moments_uniform01)(n)
    m = model_mod.build(list(base.f), list(base.g), moments)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    saddles = [i for i, q in enumerate(e.points) if q.kind == "saddle"]
    radii = [float(x) for x in args.radii.split(",")]
    print(f"{args.family} base model at Lambda = 1; saddles at inventory "
          f"indices {saddles}")

    for i in saddles:
        q = e.points[i]
        sa, sb = float(q.a), float(q.b)
        level_i = float(m.L(sa, sb))
        # The native context needs source loss STRICTLY above target loss.
        # Every root of B is a saddle at exactly C, so two B-saddles tie and
        # cannot be paired; the lowest saddle has no lower partner at all.
        lower = [j for j in saddles if j != i
                 and float(m.L(float(e.points[j].a), float(e.points[j].b)))
                 < level_i]
        if not lower:
            print(f"\n saddle {i} at ({sa:.6g},{sb:.6g}), level {level_i:.6g}: "
                  f"no strictly lower saddle to pair with -- measured below "
                  f"as a TARGET instead")
            continue
        partner = lower[0]
        tq = e.points[partner]
        ta, tb = float(tq.a), float(tq.b)
        stubs = list(getattr(q, "stubs", ()) or ())
        tstubs = list(getattr(tq, "stubs", ()) or ())
        if not stubs:
            print(f"\n saddle {i} at ({sa:.6g},{sb:.6g}): no stubs, skipped")
            continue
        grid = [float(dict(s.certificates).get("grid_error", math.nan))
                for s in stubs]
        print(f"\n saddle {i} at ({sa:.6g},{sb:.6g}) level {level_i:.6g} -> "
              f"target saddle {partner}: {len(stubs)} stubs, grid_error "
              f"{', '.join(f'{g:.2e}' for g in grid)}")
        for direction in (+1, -1):
            rows, trows = [], []
            for r in radii:
                try:
                    ctx = NativeSectionContinuation(
                        m, source_index=i, target_index=partner,
                        source_direction=direction,
                        launch_order=args.launch_order,
                        launch_radius=repr(r))
                    source, target = _crossings(ctx.launch("1"))
                except Exception as exc:
                    print(f"   dir {direction:+d} r={r:g}: "
                          f"{type(exc).__name__}: {str(exc)[:90]}")
                    continue
                gp = _point(source)
                dist_saddle = math.hypot(gp[0] - sa, gp[1] - sb)
                err = min(_polyline_distance(gp, s.curve) for s in stubs)
                rows.append((dist_saddle, err))
                line = (f"   dir {direction:+d}  r={r:<8g} unstable germ "
                        f"|z-s|={dist_saddle:.3e} dist to stub={err:.3e}")
                if tstubs:
                    tp = _point(target)
                    tdist = math.hypot(tp[0] - ta, tp[1] - tb)
                    terr = min(_polyline_distance(tp, s.curve)
                               for s in tstubs)
                    trows.append((tdist, terr))
                    line += (f"   | stable germ at {partner}: "
                             f"|z-s|={tdist:.3e} dist={terr:.3e}")
                print(line)
            for tag, data in (("unstable", rows), ("stable  ", trows)):
                usable = [(d, x) for d, x in data if d > 0 and x > 0]
                if len(usable) >= 3:
                    slope = float(np.polyfit(
                        np.log([d for d, _x in usable]),
                        np.log([x for _d, x in usable]), 1)[0])
                    print(f"   dir {direction:+d}  {tag} fitted err ~ "
                          f"r^{slope:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
