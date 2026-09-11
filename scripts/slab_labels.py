#!/usr/bin/env python3
"""Is a basin slab bounded?  Ask the J labels of the walls.

THE STRUCTURE (Andrew's, for the separated maximal-critical family).  At
large separation every saddle is a B-saddle, so the skeleton alternates
minimum / saddle along b and the unstable branches cut the plane into
horizontal SLABS, one per minimum.  A slab's walls are the stable
manifolds of the two adjacent saddles.  The slab -- and therefore the
basin inside it -- is bounded exactly when those two walls approach each
other, and open when they diverge.

WHY J DECIDES IT.  Dropping B, db/da = (a/2)A'/A is separable and

    J = a^2/2 - INT^b 2A/A' ds

is conserved EXACTLY along a stable escape.  Two walls escaping along the
same ray differ only in their labels, and the separation between them
goes like

    db ~ (J_2 - J_1) / (2 sqrt(d) a),

a power law: distinct labels diverge in b as a grows (the wall separation
in the plane does not close), equal labels stay together.  So the
boundedness of the slab is a comparison of two scalars computed at the
launch, with no tracing -- J is conserved, and the branch locks onto its
value within a few vertices of the stub (measured elsewhere: within
20 steps on tricky-d11, often at step 0).

WHAT THIS PROBE CHECKS.  For each minimum, take the saddles adjacent in b,
compute J for the stable branch of each on the side facing the minimum,
and compare.  Then set that against the finite/infinite classification
that basin_area.py derives from the level sets.  If the J labels agree
exactly on the finite slabs and differ on the open ones, the label IS the
criterion and the area computation can be built on it.  If they do not
line up, the slab picture needs revisiting before anything is built.

    python scripts/slab_labels.py --degrees 2,3,4 --separation 30
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import atlas, inverse, portrait, sturm                # noqa: E402


def j_parts(m, d_eff):
    """(c1, rhos, residues) for J = a^2/2 - b^2/(2d) - 2 c1 b
    - 2 Re SUM r_j log(b - rho_j); see demos/explorer/serve.py."""
    A = np.asarray(m._fa, dtype=float)
    Ap = np.polyder(A[::-1])
    App = np.polyder(Ap)
    q, _r = np.polydiv(A[::-1], Ap)
    c1 = float(q[-1]) if len(q) else 0.0
    rhos = np.roots(Ap)
    res = np.array([np.polyval(A[::-1], z)/np.polyval(App, z) for z in rhos],
                   dtype=complex)
    return c1, rhos, res


def j_value(a, b, d_eff, c1, rhos, res):
    v = 0.5*a*a - b*b/(2.0*d_eff) - 2.0*c1*b
    v -= 2.0*float(np.real(np.sum(res*np.log(complex(b) - rhos))))
    return v


def report(deg, separation):
    case = inverse.separated_linear_case(deg, Fraction(separation))
    m = case.model
    d_eff = float(atlas.effective_degree(m))
    c1, rhos, res = j_parts(m, d_eff)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    bmax = atlas.legal_max_b(m)
    amax = bmax/np.sqrt(max(1.0, d_eff))
    view = (-amax, amax, -bmax, bmax)
    p = portrait.compute(m, view=view, geometry_level=0, _enumeration=e,
                         _display_view=atlas.compute_box(m, e, view=view),
                         _genericity=atlas.genericity(m), _skip_audit=True)
    print(f"\n=== deg g = {deg}, Lambda = {separation}  "
          f"({len(e.points)} critical points, d_eff = {d_eff:g})")

    # J per stable branch, evaluated a few vertices past the launch where the
    # label has locked but the trace is still short.
    label = {}
    for i, br in enumerate(p.branches):
        if br.kind != "stable" or len(br.Y) < 8:
            continue
        b_star = float(br.diag.get("saddle_b", float("nan")))
        vals = []
        for k in (len(br.Y)//4, len(br.Y)//2, len(br.Y)-1):
            a, b = float(br.Y[k][0]), float(br.Y[k][1])
            v = j_value(a, b, d_eff, c1, rhos, res)
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        drift = (max(vals) - min(vals))/max(abs(vals[-1]), 1.0)
        label.setdefault(b_star, []).append((vals[-1], drift, i, br.term))

    for b_star in sorted(label):
        for v, drift, i, term in label[b_star]:
            print(f"   saddle b*={b_star:14.6g}  br{i:<3d} J={v:16.9g}  "
                  f"drift={drift:8.2e}  {term}")

    saddles = sorted(float(q.b) for q in e.points if q.kind == "saddle")
    mins = sorted(float(q.b) for q in e.points if q.kind == "min")
    print(f"\n   slab walls per minimum "
          f"(J of each adjacent saddle's stable branches):")
    for b_min in mins:
        left = [s for s in saddles if s < b_min]
        right = [s for s in saddles if s > b_min]
        if not left or not right:
            print(f"   min b={b_min:14.6g}: only one adjacent saddle -- "
                  f"slab open on one side")
            continue
        lo, hi = left[-1], right[0]
        jl = [v for v, _dr, _i, _t in label.get(lo, [])]
        jr = [v for v, _dr, _i, _t in label.get(hi, [])]
        if not jl or not jr:
            print(f"   min b={b_min:14.6g}: no stable label on one wall")
            continue
        gap = min(abs(x - y) for x in jl for y in jr)
        scale = max(max(map(abs, jl)), max(map(abs, jr)), 1.0)
        print(f"   min b={b_min:14.6g}: walls b*={lo:.6g} / {hi:.6g}  "
              f"min |dJ| = {gap:12.6g}  relative {gap/scale:10.3e}")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", default="2,3,4")
    ap.add_argument("--separation", default="30")
    args = ap.parse_args(argv)
    for d in [int(x) for x in args.degrees.split(",")]:
        report(d, Fraction(args.separation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
