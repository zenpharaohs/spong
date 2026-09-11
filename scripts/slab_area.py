#!/usr/bin/env python3
"""Basin slabs, their areas, and the chance a Gaussian initialisation lands
in one.

THE PICTURE (Andrew's).  In the separated maximal-critical family every
saddle is a B-saddle, so the skeleton alternates minimum / saddle in b and
the STABLE manifolds of the saddles cut the plane into horizontal slabs,
one per minimum: the slab is that minimum's basin.  A slab is bounded when
its two walls pin at the SAME attracting trap -- a real root of A' with
A'' < 0 -- and open when there is no trap between them, since then nothing
draws the walls together.  The traps are Sturm-exact from A' and A''
alone, so the classification needs no tracing at all.

THE WALL, AND WHY THE AREA IS A GAUSSIAN INTEGRAL.  Near a trap the
transverse coordinate w = b - rho obeys dw/da = kappa a w with
kappa = A''(rho)/(2A(rho)) < 0, so a wall launched at the saddle b* runs

    b(a) = rho + (b* - rho) exp(kappa a^2 / 2),

and two walls sharing a trap keep width (b*_R - b*_L) exp(kappa a^2/2).
The slab area is then exact:

    area = (b*_R - b*_L) * sqrt(2 pi / |kappa|).

NOT THE SUBLEVEL COMPONENT.  An earlier version of this measurement used
the sublevel component below the merging saddle as a lower bound on the
basin.  It is a bound, but a hopeless one here: A(rho) ~ 1e19 makes the
component razor thin in a (1.7e-9 at deg 4) while the basin is wide (219).
The component is the region below a level; the basin is the whole slab.

WHAT IT IS FOR.  Xavier initialisation draws the weights near the origin
with order-one scale, so what matters for trainability is not the area but
the PROBABILITY that a draw lands in the global minimum's slab.  That is
computed here for an isotropic Gaussian of the given sigma by integrating
the wall geometry against the density.  If the probability falls
geometrically in d or Lambda, the family is a sequence of progressively
untrainable nets in the only sense that matters to someone running SGD.

CAVEAT.  The wall model is the trap's linearisation: exact in the limit,
and the walls are certified only to where the level bar stops them (|a|
about 0.1 on the deg-4 case), so the slab geometry rests on the asymptotic
form rather than on traced vertices.  It is stated here rather than hidden
because the numbers below inherit it.

    python scripts/slab_area.py --degrees 2,3,4,5,6 --separation 30
    python scripts/slab_area.py --degree 4 --separations 3,10,30,100,300
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

from spong import inverse, sturm, _poly as P                     # noqa: E402


def traps(m):
    """(rho, attracting) for every real root of A', exactly classified."""
    Ap = P.deriv(m.alpha)
    App = P.deriv(Ap)
    out = []
    for iv in sturm.isolate_roots(Ap):
        iv = sturm.refine(Ap, iv, Fraction(1, 2**48))
        s = sturm.interval_sign(App, iv)
        out.append((float(iv.mid), s is not None and s < 0))
    return sorted(out)


def kappa_at(m, rho):
    A = float(m.A(rho))
    Ad = np.asarray(m._fa, dtype=float)[::-1]
    App = float(np.polyval(np.polyder(np.polyder(Ad)), rho))
    return App/(2.0*A)


def slab_of(b_min, saddles, trap_list):
    """Deprecated: a trap strictly between the bounding saddles.

    WRONG, kept only as a warning.  The walls do not stay inside the saddle
    pair: at deg 4, Lambda 30 the global minimum sits between saddles at
    -6.143 and 1, while both facing walls pin at rho = -14.892, BELOW the
    left saddle.  This rule reported that slab OPEN and gave a nan
    probability for exactly the case the study is about.  Use wall_traps().
    """
    left = [s for s in saddles if s < b_min]
    right = [s for s in saddles if s > b_min]
    if not left or not right:
        return None
    lo, hi = left[-1], right[0]
    inside = [r for r, att in trap_list if att and lo < r < hi]
    return (lo, hi, inside[0]) if inside else None


def wall_traps(m, enumeration, trap_list, reach=2.0e3):
    """Where each saddle's stable branches actually pin, by growing them.

    The certified trace stops at the level bar almost at once (|a| ~ 0.1 on
    the deg-4 case), far too early to show an asymptote, so the wall's trap
    is read by continuing the same ascent the tracer uses until b settles.
    Returns {saddle_b: {+1: rho or None, -1: rho or None}} keyed by the sign
    of a the branch runs into.
    """
    from spong import charts
    critical = np.array([[float(q.a), float(q.b)] for q in enumeration.points],
                        dtype=float)
    attracting = [r for r, att in trap_list if att]
    out = {}
    for q in enumeration.points:
        if q.kind != "saddle":
            continue
        b_star = float(q.b)
        out[b_star] = {}
        for sign in (+1, -1):
            start = (float(q.a) + sign*1e-6, b_star)
            box = (-reach, reach, -reach, reach)
            ds = reach/30000.0
            try:
                pts, _term = charts._potential_rate_box_exit(
                    m, start, box, ds, {}, max_steps=200000,
                    critical=critical)
            except Exception:
                out[b_star][sign] = None
                continue
            end = pts[-1]
            b_end = float(end[1])
            if not attracting or abs(float(end[0])) < 1.0:
                out[b_star][sign] = None
                continue
            rho = min(attracting, key=lambda r: abs(r - b_end))
            # Only call it pinned when b really has settled onto the trap:
            # a diagonal escape leaves b growing without bound.
            out[b_star][sign] = rho if abs(b_end - rho) < 0.05*max(
                1.0, abs(rho)) else None
    return out


def closed_slab(b_min, saddles, wall_map):
    """(b_left, b_right, rho) when the two facing walls share a trap."""
    left = [s for s in saddles if s < b_min]
    right = [s for s in saddles if s > b_min]
    if not left or not right:
        return None
    lo, hi = left[-1], right[0]
    for sign in (+1, -1):
        rl = (wall_map.get(lo) or {}).get(sign)
        rr = (wall_map.get(hi) or {}).get(sign)
        if rl is not None and rr is not None and rl == rr:
            return lo, hi, rl
    return None


def gaussian_share(lo, hi, rho, kappa, sigma, n=20001):
    """P[(a,b) in the slab] for an isotropic N(0, sigma^2) draw.

    The slab at height a spans b in [rho + (lo-rho)e^{ka^2/2},
    rho + (hi-rho)e^{ka^2/2}], so the probability is a one-dimensional
    integral of the b-interval's Gaussian mass against the a-density.
    """
    from math import erf, sqrt
    amax = 8.0*sigma
    a = np.linspace(-amax, amax, n)
    damp = np.exp(kappa*a*a/2.0)
    b_lo = rho + (lo - rho)*damp
    b_hi = rho + (hi - rho)*damp
    cdf = lambda x: 0.5*(1.0 + np.array([erf(v/(sigma*sqrt(2.0))) for v in x]))
    mass_b = cdf(b_hi) - cdf(b_lo)
    dens_a = np.exp(-a*a/(2.0*sigma*sigma))/(sigma*np.sqrt(2.0*np.pi))
    return float(np.trapezoid(mass_b*dens_a, a)) if hasattr(np, "trapezoid") \
        else float(np.trapz(mass_b*dens_a, a))


def report(deg, separation, sigma):
    case = inverse.separated_linear_case(deg, Fraction(separation))
    m = case.model
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    tr = traps(m)
    mins = sorted(float(q.b) for q in e.points if q.kind == "min")
    sads = sorted(float(q.b) for q in e.points if q.kind == "saddle")
    wall_map = wall_traps(m, e, tr)
    u = lambda b: float(m.L(float(m.a_star(b)), b))
    print(f"\n=== deg g = {deg}, Lambda = {separation}: "
          f"{len(e.points)} critical points, "
          f"{sum(1 for _r, att in tr if att)} attracting traps")
    for b_star in sorted(wall_map):
        w = wall_map[b_star]
        print(f"   saddle b*={b_star:14.6g}  walls pin at "
              f"a>0: {w.get(1)}  a<0: {w.get(-1)}")
    rows = []
    for b_min in mins:
        slab = closed_slab(b_min, sads, wall_map)
        if slab is None:
            rows.append((b_min, u(b_min), None, None, None))
            continue
        lo, hi, rho = slab
        k = kappa_at(m, rho)
        area = (hi - lo)*np.sqrt(2.0*np.pi/abs(k))
        share = gaussian_share(lo, hi, rho, k, sigma)
        rows.append((b_min, u(b_min), area, share, (lo, hi, rho, k)))
    gmin = min(rows, key=lambda r: r[1])
    for b_min, u_min, area, share, det in rows:
        tag = " <-- global" if b_min == gmin[0] else ""
        if area is None:
            print(f"   min b={b_min:14.6g} u={u_min:12.6g}  OPEN "
                  f"(no shared trap){tag}")
        else:
            lo, hi, rho, k = det
            print(f"   min b={b_min:14.6g} u={u_min:12.6g}  walls "
                  f"[{lo:.6g}, {hi:.6g}] -> rho={rho:.6g}  kappa={k:.4g}  "
                  f"area={area:12.6g}  P_init={share:11.4e}{tag}")
    tot = sum(r[3] for r in rows if r[3] is not None)
    print(f"   global minimum: P_init = "
          f"{gmin[3] if gmin[3] is not None else float('nan'):.6e}"
          f"   (all closed slabs together: {tot:.6e})")
    return {"deg": deg, "separation": float(separation),
            "p_global": gmin[3], "p_all": tot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--degrees", default="2,3,4,5,6")
    ap.add_argument("--separations", default=None)
    ap.add_argument("--separation", default=None,
                    help="single separation for the --degrees sweep")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="isotropic init scale; Xavier on a 1-in/1-out "
                         "neuron is about 1")
    args = ap.parse_args(argv)
    out = []
    if args.separations:
        deg = args.degree or 4
        for s in args.separations.split(","):
            out.append(report(deg, Fraction(s), args.sigma))
    else:
        degs = ([args.degree] if args.degree
                else [int(x) for x in args.degrees.split(",")])
        sep = Fraction(args.separation) if args.separation else Fraction(30)
        for d in degs:
            out.append(report(d, sep, args.sigma))
    print(f"\n# summary (sigma = {args.sigma})")
    print(f"{'deg':>4} {'Lambda':>8} {'P(global basin)':>18} "
          f"{'P(any closed slab)':>20}")
    for r in out:
        pg = float("nan") if r["p_global"] is None else r["p_global"]
        print(f"{r['deg']:>4} {r['separation']:>8g} {pg:>18.6e} "
              f"{r['p_all']:>20.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
