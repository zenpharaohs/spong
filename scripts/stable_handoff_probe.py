"""Where a stable branch COULD have stopped, versus where it did.

Pure observation -- computes no certificate and changes no verdict.  For each
stable branch it reports three indices along the traced polyline:

  bar      first vertex with L > U*, U* = max_b u(b) = the highest saddle
           loss.  Above it the superlevel set is two critical-point-free
           components, so the ascent orbit cannot reach a critical point and
           the branch is certainly unbounded.  Sharp: at or below U* some
           fiber is full, the superlevel set is connected and holds the
           saddle attaining U*, so level information alone cannot decide.

  half     first vertex outside the backbone half-plane, a <= a0 < min a*
           or a >= a1 > max a*.  Critical-point-free and forward-invariant
           for the ascent flow regardless of level, so this can fire below
           the bar -- it is a complement to it, not a weaker version.

  sagitta  first vertex where the matched asymptotic agrees with the traced
           curve to within the accuracy the curve was obtained to, sustained
           over a window.  delta = chord * turn / 8 per segment, the same
           yardstick contact attestation uses.

The gap between those and `n-1` is the waste.  If `sagitta` lands well before
a stall, an early handoff is worth building; if it sits on top of the stall,
the criterion is too conservative to buy anything.

    python stable_handoff_probe.py tricky-d11
    python stable_handoff_probe.py --seed 499170635 --mode directed --degree 5
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from spong import atlas, model, portrait, sturm, zoo


# ---------------------------------------------------------------- the bar

def highest_saddle_loss(m, e):
    """U* = max_b u(b), attained at a saddle.

    u' = -B*N/A^2, so u's critical points are the roots of B (level C) and of
    N; the maxima are the u''<0 ones, which are exactly the saddles.  Taken
    from the enumeration rather than re-derived so this agrees with whatever
    the certifier would use.
    """
    losses = [float(m.L(p.a, p.b)) for p in e.points if p.kind == "saddle"]
    return max(losses) if losses else None


def backbone_halfplane(m, samples=200001):
    """Dyadic a0 < min a* and a1 > max a* over the reals.

    a* = B/A is bounded: A > 0 kills the poles and deg A = 2d > d >= deg B
    sends a* -> 0 at both ends, so the extremes are finite.  Sampled here for
    a MEASUREMENT; a certificate would fix a0 by the exact positivity of
    B - a0*A on R, which is the psi_positive machinery at degree 2d.
    """
    bmax = atlas.legal_max_b(m)
    bs = np.linspace(-bmax, bmax, samples)
    A = np.polyval(list(m.alpha)[::-1], bs)
    B = np.polyval(list(m.beta)[::-1], bs)
    good = np.abs(A) > 0.0
    star = B[good] / A[good]
    lo, hi = float(star.min()), float(star.max())
    span = max(hi - lo, 1e-12)
    return lo - 0.02 * span, hi + 0.02 * span, lo, hi


def _dyadic_below(x):
    y = -1.0
    while y > x:
        y *= 2.0
    return y


def _dyadic_above(x):
    y = 1.0
    while y < x:
        y *= 2.0
    return y


# ------------------------------------------------------- asymptotic tail

def tail_coefficients(m):
    """Offset diagonal b = sqrt(d)*a + c0 + K0/(2 sqrt(d) a).

    From db/da = a(aA' - 2B')/2(aA - B): for d >= 2 the B terms are O(b^-d),
    below the 1/b corrections, so db/da = (d a / b)(1 - a1/(2 d b) + ...)
    with a1 = A_{2d-1}/A_{2d}.  Integrating gives b^2 = d a^2 - a1 a/sqrt(d)
    + K0.  c0 = -a1/(2d) is exact from the top two coefficients of A alone;
    K0 is the single free constant, matched at the handoff.
    """
    d = atlas.effective_degree(m)
    A = [float(c) for c in m.alpha]          # ascending powers
    if len(A) < 2 or d < 1:
        return None
    a1 = A[-2] / A[-1]
    return d, a1, -a1 / (2.0 * d)


def tail_b(a, d, a1, K0, sign_b):
    """b(a) on the matched asymptotic, or None where it does not apply."""
    disc = d * a * a - a1 * a / math.sqrt(d) + K0
    if disc < 0.0:
        return None
    return math.copysign(math.sqrt(disc), sign_b)


def match_K0(a_p, b_p, d, a1):
    return b_p * b_p - d * a_p * a_p + a1 * a_p / math.sqrt(d)


# ------------------------------------------------------------- sagitta

def sagitta_bounds(Y):
    """Per-segment delta = chord * turn / 8, read off the polyline.

    Local by construction: large where the curve turns, tiny in a flat tail.
    Mirrors topology._sagitta_bounds; recomputed here so the probe runs
    against a checkout whose topology module may differ.
    """
    n = len(Y)
    if n < 3:
        return np.zeros(max(n - 1, 0))
    d = np.diff(Y, axis=0)
    chord = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(np.diff(ang))
    turn = np.minimum(turn, 2.0 * np.pi - turn)
    turn = np.concatenate(([turn[0]], turn))
    return chord * turn / 8.0


# --------------------------------------------------------------- scan

def tangency_residual(m, a, b, ta, tb):
    """|T x grad L| / (|T| |grad L|) -- the house perpendicularity metric.

    Zero iff the direction (ta, tb) is parallel to the field, i.e. iff it is
    the direction of a genuine integral curve there.
    """
    Ac = list(m.alpha)[::-1]
    Bc = list(m.beta)[::-1]
    Ap = np.polyder(np.asarray(Ac, dtype=float))
    Bp = np.polyder(np.asarray(Bc, dtype=float))
    A = np.polyval(Ac, b); B = np.polyval(Bc, b)
    ga = 2.0 * (a * A - B)
    gb = a * (a * np.polyval(Ap, b) - 2.0 * np.polyval(Bp, b))
    ng = math.hypot(ga, gb); nt = math.hypot(ta, tb)
    if ng == 0.0 or nt == 0.0:
        return float("inf")
    return abs(ta * gb - tb * ga) / (ng * nt)


def first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def scan_branch(m, br, ustar, a0, a1_hp, coeffs, window=8):
    Y = np.asarray(br.Y, dtype=float)
    n = len(Y)
    out = {"n": n, "bar": None, "half": None, "sagitta": None,
           "tangent": None}
    if n == 0:
        return out

    a, b = Y[:, 0], Y[:, 1]
    Acoef = list(m.alpha)[::-1]
    Bcoef = list(m.beta)[::-1]
    L = float(m.C) - 2.0 * a * np.polyval(Bcoef, b) + a * a * np.polyval(Acoef, b)

    if ustar is not None:
        out["bar"] = first_true(L > ustar)
    out["half"] = first_true((a <= a0) | (a >= a1_hp))

    if coeffs is not None and n >= window + 2:
        d, a1c, _ = coeffs
        delta = sagitta_bounds(Y)
        ok = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if abs(a[i]) < 1e-12:
                continue
            K0 = match_K0(a[i], b[i], d, a1c)
            # Agreement is tested FORWARD of the candidate handoff: the tail
            # has to predict the continuation, not merely touch it.
            good = True
            for j in range(i + 1, min(i + 1 + window, n)):
                pred = tail_b(a[j], d, a1c, K0, b[i])
                if pred is None:
                    good = False
                    break
                tol = float(delta[min(j, len(delta) - 1)])
                if abs(pred - b[j]) > max(tol, 0.0):
                    good = False
                    break
            ok[i] = good
        out["sagitta"] = first_true(ok)

    # Apples-to-apples: is the ASYMPTOTIC direction as faithful to the field
    # as the CHORD direction already is?  Both measured by the same tangency
    # residual, so this asks whether substituting the tail makes the curve a
    # worse representative -- which is the actual question, unlike the
    # sagitta, whose tolerance collapses precisely in a straight tail.
    if coeffs is not None and n >= window + 2:
        d, a1c, _ = coeffs
        ok2 = np.zeros(n, dtype=bool)
        for i in range(1, n - 1):
            if abs(a[i]) < 1e-12:
                continue
            K0 = match_K0(a[i], b[i], d, a1c)
            good = True
            for j in range(i, min(i + window, n - 1)):
                bj = tail_b(a[j], d, a1c, K0, b[i])
                if bj is None or abs(bj) < 1e-300:
                    good = False; break
                # db/da on the matched tail
                dbda = (d * a[j] - a1c / (2.0 * math.sqrt(d))) / bj
                r_as = tangency_residual(m, a[j], b[j], 1.0, dbda)
                r_ch = tangency_residual(m, a[j], b[j],
                                         a[j+1] - a[j], b[j+1] - b[j])
                if not (r_as <= max(r_ch, 1e-15)):
                    good = False; break
            ok2[i] = good
        out["tangent"] = first_true(ok2)
    return out


def fmt(i, n):
    if i is None:
        return "     —"
    return f"{i:6d}" + (f" ({100.0*i/max(n-1,1):4.1f}%)" if n > 1 else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", help="zoo case name")
    ap.add_argument("--window", type=int, default=8,
                    help="vertices the tail must predict before handoff")
    args = ap.parse_args(argv)
    if not args.case:
        ap.error("give a zoo case name, e.g. tricky-d11")

    case = zoo.get(args.case)
    d = max(len(case.f) - 1, len(case.g) - 1)
    mu = (model.moments_uniform01 if case.moment_dist == "uniform01"
          else model.moments_normal01)(2 * d + 1)
    m = model.build(case.f, case.g, mu)

    e = sturm.enumerate_critical_points(m)
    ustar = highest_saddle_loss(m, e)
    a0, a1_hp, star_lo, star_hi = backbone_halfplane(m)
    coeffs = tail_coefficients(m)

    print(f"{case.name}   psi_positive={e.psi_positive}  morse={e.morse}")
    print(f"  C  = {float(m.C):.12g}")
    print(f"  U* = {ustar:.12g}   (highest saddle loss)"
          if ustar is not None else "  U* = —")
    print(f"  backbone a* range: [{star_lo:.6g}, {star_hi:.6g}]")
    print(f"  half-plane: a <= {a0:.6g}  or  a >= {a1_hp:.6g}")
    if coeffs:
        dd, a1c, c0 = coeffs
        print(f"  tail: b = sqrt({dd})*a + {c0:.6g} + K0/(2 sqrt({dd}) a)")
    print()

    p = portrait.certified_compute(m, view=case.default_view)
    status = (p.ledger or {}).get("topology", {}).get("status")
    print(f"  portrait status = {status}")
    print()

    hdr = (f"{'br':>4}  {'term':<16} {'n':>7} {'a_end':>10}  {'bar':>13}"
           f" {'half':>13} {'sagitta':>13} {'tangent':>13}")
    print(hdr)
    print("-" * len(hdr))
    for i, br in enumerate(p.branches):
        if br.kind != "stable":
            continue
        r = scan_branch(m, br, ustar, a0, a1_hp, coeffs, window=args.window)
        aend = float(np.asarray(br.Y)[-1, 0])
        print(f"{i:>4}  {br.term:<16} {r['n']:>7} {aend:>10.1f}  "
              f"{fmt(r['bar'], r['n']):>13} {fmt(r['half'], r['n']):>13} "
              f"{fmt(r['sagitta'], r['n']):>13} {fmt(r['tangent'], r['n']):>13}")
    print()
    print("bar/half/sagitta are vertex indices; the gap to n-1 is the waste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
