#!/usr/bin/env python3
"""Measure the SWITCHOVER budget: how close is the full flow to the
asymptotic flow, geometrically, as a function of radius?

    python scripts/switchover_probe.py --case linear-target-d17-thrash
    python scripts/switchover_probe.py --custom "1,1,1" "1,1,1" uniform01

MEASUREMENT ONLY.  No acceptance rule, no thresholds: this exists so the
switchover contract can be written from data instead of from taste.  Two
attempts at that contract have already failed for reasons this probe would
have exposed in a minute -- an unrestricted best-angle search put the
switchover AT a B-saddle (where the asymptotic direction (b, d a)
degenerates to the b-axis at a = 0 and any vertical launch matches it to
zero angle), and a K-drift test scaled by r^2 accepted drift of order 100%
OF K.

WHAT IS MEASURED.  The transition is a change of REPRESENTATION of one
flow, so the comparison is between geometries, not between raw fields:
F = -grad L and c(x) F have identical phase curves for any positive c, and
the asymptotic field's overall scale is exactly such a factor.  So compare
the unit tangent and the arclength curvature, both dimensionless:

    T      = F / |F|,                    F     = -grad L
    T_inf  = G / |G|,                    G     = (b, d_eff a)
    kappa      = (I - T T^t) DF T / |F|
    kappa_inf  = (I - T_inf T_inf^t) DG T_inf / |G|,   DG = [[0,1],[d,0]]

    eps0 = |T - s T_inf|                 (s = +-1, whichever aligns)
    eps1 = r |kappa - kappa_inf|

eps0 is the tangent mismatch -- what the eye sees as a corner at the seam.
eps1 is the curvature mismatch, scaled by r to make it dimensionless, and
it is the one that DISTINGUISHES MEMBERS OF THE HYPERBOLA FAMILY: every
curve b^2 - d a^2 = K has the same far-field direction, so tangent
agreement alone cannot tell whether the trajectory is on the right K.  A
switchover rule built on eps0 alone is underdetermined; that was the flaw
in the previous attempt.

The remainder terms are degree m-1 against the leading degree m, so both
epsilons are expected to fall like 1/r.  The columns therefore also report
eps0*r and eps1*r: if those level off, the 1/r law is in force and a
majorant argument can promise the agreement keeps improving after the
switch, which is what turns a local match into a tail statement.  If they
do not level off, the branch is not in the asymptotic regime and no
switchover is licensed there however small the raw angle looks.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "demos" / "explorer"))

from spong import atlas, model, portrait, zoo                # noqa: E402


def unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (v/n if n > 0 else None), n


def geometry(m, d_eff, a, b):
    """(T, kappa, T_inf, kappa_inf) at (a, b), or None where undefined."""
    F = -np.asarray(m.gradL(a, b), dtype=float)
    T, nF = unit(F)
    if T is None or not np.all(np.isfinite(T)):
        return None
    DF = -np.asarray(m.hessL(a, b), dtype=float)
    P = np.eye(2) - np.outer(T, T)
    kap = P @ (DF @ T) / nF

    G = np.array([b, d_eff*a], dtype=float)
    Ti, nG = unit(G)
    if Ti is None:
        return None
    DG = np.array([[0.0, 1.0], [d_eff, 0.0]])
    kapi = (np.eye(2) - np.outer(Ti, Ti)) @ (DG @ Ti) / nG
    return T, kap, Ti, kapi


def eps(m, d_eff, a, b):
    g = geometry(m, d_eff, a, b)
    if g is None:
        return None
    T, kap, Ti, kapi = g
    s = 1.0 if float(T @ Ti) >= 0 else -1.0      # trace may run either way
    e0 = float(np.hypot(*(T - s*Ti)))
    e1 = math.hypot(a, b) * float(np.hypot(*(kap - s*kapi)))
    return e0, e1


def build(args):
    if args.custom:
        f = [float(x) for x in args.custom[0].split(",")]
        g = [float(x) for x in args.custom[1].split(",")]
        dist = args.custom[2] if len(args.custom) > 2 else "uniform01"
        mu = (model.moments_normal01 if dist == "normal01"
              else model.moments_uniform01)(2*max(len(f), len(g)) - 1)
        return model.build(f, g, mu), None
    z = zoo.get(args.case)
    f, g = list(z.f), list(z.g)
    mu = (model.moments_normal01 if z.moment_dist == "normal01"
          else model.moments_uniform01)(2*max(len(f), len(g)) - 1)
    return model.build(f, g, mu), getattr(z, "default_view", None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="linear-target-d17-thrash")
    ap.add_argument("--custom", nargs="*", default=None,
                    help='f g [dist], e.g. --custom "1,1,1" "1,1,1" uniform01')
    ap.add_argument("--extend", action="store_true", default=True,
                    help="include the display extension (default on)")
    ap.add_argument("--no-extend", dest="extend", action="store_false")
    args = ap.parse_args(argv)

    m, view = build(args)
    p = portrait.certified_compute(m, view=view)
    e = p.enumeration
    d_eff = float(atlas.effective_degree(m))
    r_crit = max([math.hypot(float(q.a), float(q.b)) for q in e.points] or [1.0])
    print(f"{args.case if not args.custom else 'custom'}: d_eff={d_eff:g}  "
          f"r_crit={r_crit:.4g}  {len(e.saddles)} saddles")

    ext = {}
    if args.extend:
        import serve
        try:
            ext = {x["branch"]: x
                   for x in serve._stable_extensions(m, p, e, d_eff=d_eff)}
        except Exception as exc:                      # probe must not die
            print(f"  (no extensions: {type(exc).__name__}: {exc})")

    for i, br in enumerate(p.branches):
        if br.kind != "stable":
            continue
        pts = [(float(q[0]), float(q[1])) for q in br.Y]
        x = ext.get(i)
        if x and x.get("points"):
            pts += [(float(q[0]), float(q[1])) for q in x["points"][1:]]
        rows = []
        for a, b in pts:
            r = math.hypot(a, b)
            v = eps(m, d_eff, a, b)
            if v is not None:
                rows.append((r, v[0], v[1], b*b - d_eff*a*a))
        if not rows:
            print(f"br{i:<2} (no usable points)")
            continue
        print(f"br{i:<2} b*={br.diag.get('saddle_b')!s:<22} term={br.term:<11}"
              f" n={len(pts)} r[{min(r for r,_,_,_ in rows):.3g},"
              f"{max(r for r,_,_,_ in rows):.3g}]")
        print(f"     {'r':>10} {'eps0':>10} {'eps1':>10} {'eps0*r':>10} "
              f"{'eps1*r':>10} {'K':>12} {'K/r^2':>9}")
        # Report at the LAST point in each radius octave -- the epsilons are
        # functions of position, and the octave is the natural x-axis for a
        # 1/r law.
        seen = set()
        for r, e0, e1, K in rows:
            oct_ = math.floor(math.log2(max(r, 1e-300))*2)/2
            if oct_ in seen:
                continue
            seen.add(oct_)
            print(f"     {r:>10.4g} {e0:>10.3e} {e1:>10.3e} {e0*r:>10.3e} "
                  f"{e1*r:>10.3e} {K:>12.4g} {K/max(r*r,1e-300):>9.4f}")
        r, e0, e1, K = rows[-1]
        print(f"     {'end':>10} {e0:>10.3e} {e1:>10.3e} {e0*r:>10.3e} "
              f"{e1*r:>10.3e} {K:>12.4g} {K/max(r*r,1e-300):>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
