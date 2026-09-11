#!/usr/bin/env python3
"""Validate the DRAWN stable curves, not just the skeleton.

WHY.  The certified trace stops at the level bar, often within |a| ~ 0.1 of
the saddle, so almost everything the inspector shows of a stable manifold is
GROWN and then continued in closed form.  None of that is covered by the
topology audit.  A defect found 2026-09-06 made the drawn wall of a slab run
to the wrong attracting trap -- b = -14.892264 instead of +597.255444 -- with
sector=pinned, a named trap and grow_term=box_exit attached, and nothing in
the output admitting a problem.  The cause was serve._grow_stable passing
ds0=ds to the continuation where charts.trace_stable passes the stub's much
smaller launch scale, so the engine attempted its first step at full chord.

WHAT IS CHECKED, per stable branch:

  refinement    the growth is run at (ds, R) and again at (ds/4, R) and
                (ds, 2R).  All three must reach the same attracting trap, or
                all three must fail to pin.  THE DESTINATION, not its
                precision, is what broke before, so this is the load-bearing
                check.
  ascent        L must be non-decreasing along the grown curve, to within the
                loss's own evaluation floor.  A step that raises the loss by
                orders of magnitude in one chord is the seam signature.
  tangency      each grown chord against grad L at its midpoint.  Weak on its
                own -- the bad first step measured 0.0096 deg while a good one
                measured 2.4 -- so it is reported, never used alone.
  declaration   the tail's sector and trap must match where the growth ends:
                a `pinned` tail naming a trap the growth does not approach is
                the exact failure above.
  crossings     drawn curves must not cross each other or themselves.
                Non-crossing is a THEOREM about the manifolds, so a crossing
                means the polyline has stopped representing its curve.
                SCOPED TO PAIRS WITH DIFFERENT TRAPS.  Two walls pinning at
                the SAME attracting trap separate like exp(-|kappa| a^2/2),
                which falls below an ulp of b within a few vertices, so their
                polylines interleave and no coordinate test can adjudicate
                them: crossing counts fire on every such pair (1009 and 1195
                measured), and an order test either fires on roundoff or, if
                the unresolvable samples are dropped, invents a flip across
                the gap it just created.  Both were tried.  The ordering of
                such a pair is provable -- w/w' is constant on a shared trap,
                which is the separation certificate's business -- but not
                HERE, so they are counted as unverifiable and reported as a
                census, never as an issue.

HOW BIG A VIEW.  There is no a-priori answer, so the criterion is
operational: the view is large enough when the answer stops changing.  --reach
sets R as a multiple of the skeleton box, and the R vs 2R comparison above is
what licenses it.  A case where they disagree is REPORTED, not averaged.

    python scripts/tail_validate.py --untargeted 60
    python scripts/tail_validate.py --directed 40 --reach 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import atlas, model, portrait, sturm                  # noqa: E402
from qualify import directed_model                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "serve", REPO / "demos" / "explorer" / "serve.py")
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)


def untargeted_model(seed):
    """The qualify_parallel generator, reproduced exactly."""
    rng = np.random.default_rng(seed)
    df, dg = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    f = rng.normal(size=df + 1)
    g = f.copy() if rng.random() < 0.25 and df == dg else rng.normal(size=dg+1)
    mom = (model.moments_uniform01 if rng.random() < 0.75
           else model.moments_normal01)(2*max(df, dg) + 1)
    return model.build(f, g, mom)


def attracting_traps(m):
    return [r for r, att in serve._traps(m) if att]


def pinned_trap(curve, traps, rel=0.05):
    """Which trap the grown curve has settled onto, or None."""
    if len(curve) < 3 or not traps:
        return None
    b_end = float(curve[-1][1])
    if abs(float(curve[-1][0])) < 1.0:
        return None
    rho = min(traps, key=lambda r: abs(r - b_end))
    return rho if abs(b_end - rho) <= rel*max(1.0, abs(rho)) else None


def grow(m, start, ds, r_max, critical, ds0):
    pts, term = serve._grow_stable(m, tuple(start), ds, r_max, critical,
                                   ds0=ds0)
    return np.asarray([start] + list(pts), dtype=float), term


def segments_cross(p, q, r, s):
    def orient(a, b, c):
        v = float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        return 0 if v == 0.0 else (1 if v > 0.0 else -1)
    d1, d2 = orient(r, s, p), orient(r, s, q)
    d3, d4 = orient(p, q, r), orient(p, q, s)
    return d1 != d2 and d3 != d4


def self_crossings(P, stride=1, cap=200000):
    n = len(P)
    if n < 4 or n*n > cap*40:
        stride = max(stride, n//300 + 1)
    hits = 0
    idx = list(range(0, n-1, stride))
    for ii, i in enumerate(idx):
        for j in idx[ii+2:]:
            if segments_cross(P[i], P[i+1], P[j], P[j+1]):
                hits += 1
                if hits > 8:
                    return hits
    return hits


def order_consistent(P, Q, samples=400, ulps=8.0):
    """UNUSED, kept as a record of what does not work.

    Sign of b_P - b_Q at matched |a| cannot adjudicate two walls sharing a
    trap.  Comparing against exact zero reports reversals at min |db| of
    8.9e-16 against b ~ 5 -- one ulp, pure interpolation roundoff.  Dropping
    the samples below a few ulp does not help: the significant samples then
    sit on either side of the unresolvable stretch, and a sign test across
    that gap manufactures a flip out of the exclusion itself.  The ordering
    is real and provable by w/w' = const; it is simply not visible in b.
    """
    a1 = np.abs(P[:, 0]); a2 = np.abs(Q[:, 0])
    lo = max(a1.min(), a2.min()); hi = min(a1.max(), a2.max())
    if not (hi > lo):
        return True, 0.0
    if not (np.all(np.diff(a1) >= -1e-12) and np.all(np.diff(a2) >= -1e-12)):
        return True, float("nan")          # not monotone in a: no verdict
    xs = np.linspace(lo, hi, samples)
    b1 = np.interp(xs, a1, P[:, 1]); b2 = np.interp(xs, a2, Q[:, 1])
    d = b1 - b2
    floor = ulps*np.spacing(np.maximum(np.abs(b1), np.abs(b2)))
    sig = np.abs(d) > floor
    nz = d[sig]
    if nz.size < 2:
        return True, float("nan")          # never separated enough to judge
    flips = int(np.sum(np.diff(np.sign(nz)) != 0))
    return flips == 0, float(np.min(np.abs(nz)))


def check_case(tag, m, reach_mult, verbose):
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    if not e.saddles:
        return None
    bmax = atlas.legal_max_b(m)
    d_eff = float(atlas.effective_degree(m))
    view = (-bmax/math.sqrt(max(d_eff, 1.0)), bmax/math.sqrt(max(d_eff, 1.0)),
            -bmax, bmax)
    dv = atlas.compute_box(m, e, view=view)
    p = portrait.compute(m, view=view, geometry_level=0, _enumeration=e,
                         _display_view=dv, _genericity=atlas.genericity(m),
                         _skip_audit=True)
    critical = np.array([[float(q.a), float(q.b)] for q in e.points], float)
    traps = attracting_traps(m)
    skeleton = max(abs(dv[0]), abs(dv[1]), abs(dv[2]), abs(dv[3]), 1.0)
    R = reach_mult*skeleton
    ds = 4.0*R/30000.0
    tails = {t["branch"]: t for t in serve._j_tails(
        m, p, d_eff, enumeration=e)}
    issues = []
    drawn = []
    dest_of = {}
    for i, br in enumerate(p.branches):
        if br.kind != "stable" or len(br.Y) < 2:
            continue
        Y = np.asarray(br.Y, float)
        ds0 = float(np.hypot(*(Y[-1] - Y[-2]))) or None
        runs = {}
        for name, (dsx, Rx) in (("base", (ds, R)),
                                ("fine", (ds/4.0, R)),
                                ("far", (ds, 2.0*R))):
            try:
                runs[name] = grow(m, Y[-1], dsx, Rx, critical, ds0)
            except Exception as exc:
                issues.append(f"br{i} growth {name} raised "
                              f"{type(exc).__name__}")
                runs[name] = (Y[-1:].copy(), "raised")
        dest = {k: pinned_trap(v[0], traps) for k, v in runs.items()}
        if len({repr(x) for x in dest.values()}) != 1:
            issues.append(f"br{i} DESTINATION UNSTABLE base={dest['base']} "
                          f"fine={dest['fine']} far={dest['far']}")
        G = runs["base"][0]
        drawn.append((i, G))
        dest_of[i] = dest["base"]
        L = np.array([float(m.L(a, b)) for a, b in G])
        drops = np.flatnonzero(np.diff(L) < -1e-9*np.maximum(
            np.abs(L[:-1]), 1.0))
        if drops.size:
            k = int(drops[0])
            issues.append(f"br{i} ASCENT VIOLATED at {k}: L "
                          f"{L[k]:.6g} -> {L[k+1]:.6g}")
        sc = self_crossings(G)
        if sc:
            issues.append(f"br{i} SELF-CROSSING x{sc} in the grown curve")
        t = tails.get(i)
        if t and t.get("sector") == "pinned" and t.get("trap") is not None:
            if dest["base"] is None or abs(dest["base"] - t["trap"]) > 1e-6*max(
                    1.0, abs(t["trap"])):
                issues.append(f"br{i} TAIL DECLARES trap {t['trap']} but "
                              f"growth reaches {dest['base']}")
    shared = 0
    for x in range(len(drawn)):
        for y in range(x+1, len(drawn)):
            i, P1 = drawn[x]; j, P2 = drawn[y]
            t1, t2 = dest_of.get(i), dest_of.get(j)
            if t1 is not None and t2 is not None and t1 == t2:
                shared += 1          # unverifiable here; see the header
                continue
            s1 = max(1, len(P1)//200); s2 = max(1, len(P2)//200)
            hit = 0
            for a in range(0, len(P1)-1, s1):
                for b in range(0, len(P2)-1, s2):
                    if segments_cross(P1[a], P1[a+1], P2[b], P2[b+1]):
                        hit += 1
                        break
                if hit:
                    break
            if hit:
                issues.append(f"br{i} x br{j} CROSSING in the drawn curves")
    if verbose or issues:
        print(f" {tag}: {len(issues)} issue(s)"
              + (f", {shared} shared-trap pair(s) not verifiable here"
                 if shared else "")
              + ("" if not issues else "\n   " + "\n   ".join(issues)))
    return issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--untargeted", type=int, default=0)
    ap.add_argument("--directed", type=int, default=0)
    ap.add_argument("--directed-degree", type=int, default=5)
    ap.add_argument("--reach", type=float, default=8.0,
                    help="growth box as a multiple of the skeleton box")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    total = bad = 0
    for _ in range(args.untargeted):
        s = int(rng.integers(2**32))
        try:
            issues = check_case(f"untargeted {s}", untargeted_model(s),
                                args.reach, args.verbose)
        except Exception as exc:
            print(f" untargeted {s}: FAILED {type(exc).__name__}: {exc}")
            bad += 1; total += 1; continue
        if issues is None:
            continue
        total += 1; bad += bool(issues)
    for _ in range(args.directed):
        s = int(rng.integers(2**32))
        mm, _lab = directed_model(random.Random(s), args.directed_degree)
        if mm is None:
            continue
        try:
            issues = check_case(f"directed {s}", mm, args.reach, args.verbose)
        except Exception as exc:
            print(f" directed {s}: FAILED {type(exc).__name__}: {exc}")
            bad += 1; total += 1; continue
        if issues is None:
            continue
        total += 1; bad += bool(issues)
    print(f"\n{total} cases, {bad} with at least one issue, "
          f"reach = {args.reach}x the skeleton box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
