"""Complex portrait of the backbone rational function, per seed.

Everything this month's diagnostics measured was a REAL-root instrument.
This probe computes the complex structure of the exact polynomials behind
u = C - B^2/A and reports the three signatures the real instruments cannot
see:

  POLES        complex zeros of the reduced denominator of B^2/A.  The
               psi strip is the
               near-real pair: Re = strip location, Im = strip depth,
               |Res u| = -B(z)^2/A'(z) calibrates the a* spike.

  GHOST PAIRS  near-real conjugate roots of the reduced numerator of u':
               saddle-node just off the axis gives u' ~ c((b-b0)^2+eps^2)
               -- a plateau of width eps and depth eps^2 on the real
               backbone, crawl cost ~1/eps, while every real margin reads
               healthy.  Candidate explanation for the healthy-axes trio.

  PENCIL       level sets are real slices of y^2 = S_l(b),
               S_l = B^2 + (l - C) A: a pencil of hyperelliptic curves
               with a fixed number of branch points moving algebraically
               in l.  Critical values are exactly the l where a branch
               point lands on the real axis; the minimum distance of the
               off-axis branch points to R over the working l-range is
               the level machinery's true conditioning number.

Coefficients are assembled exactly and common factors are removed before
classification.  Floating roots only propose disks; exact Lehmer-Schur
(Schur-Cohn) counts or exact linear Rouche witnesses certify each disk and
completeness of the divisor.

    python scripts/pole_portrait.py 202251424 1198854733 1283395251 \
        953953598 555999196 1785201004 1143710268
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import sturm                                   # noqa: E402
from spong.complex_structure import certify_backbone      # noqa: E402
from qualify import directed_model, random_model          # noqa: E402


# ------------------------------------------------------------------ #
# exact polynomial assembly (ascending Fraction tuples)
# ------------------------------------------------------------------ #

def _trim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return tuple(p) if p else (Fraction(0),)


def pmul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, ci in enumerate(p):
        if ci:
            for j, cj in enumerate(q):
                out[i + j] += ci * cj
    return _trim(out)


def pderiv(p):
    return _trim(tuple(k * c for k, c in enumerate(p)))[1:] or (Fraction(0),)


def plin(alpha, p, beta, q):
    n = max(len(p), len(q))
    return _trim(tuple(
        alpha * (p[k] if k < len(p) else 0)
        + beta * (q[k] if k < len(q) else 0) for k in range(n)))


def exact_ABCN(m):
    f, g, mu = tuple(m.f), tuple(m.g), tuple(m.mu)
    A = _trim([mu[n] * sum(g[j] * g[n - j]
                           for j in range(max(0, n - len(g) + 1),
                                          min(n, len(g) - 1) + 1))
               for n in range(2 * len(g) - 1)])
    B = _trim([g[j] * sum(f[i] * mu[i + j] for i in range(len(f)))
               for j in range(len(g))])
    C = sum(f[i] * f[k] * mu[i + k]
            for i in range(len(f)) for k in range(len(f)))
    N = plin(Fraction(1), pmul(pderiv(A), B),
             Fraction(-2), pmul(pderiv(B), A))
    return A, B, C, N


def croots(p):
    """Complex roots of an ascending Fraction polynomial via numpy."""
    arr = np.array([float(c) for c in p], dtype=float)
    scale = np.max(np.abs(arr))
    if scale == 0 or len(arr) < 2:
        return np.array([])
    return np.roots((arr / scale)[::-1])


def peval(p, z):
    v = 0.0 + 0.0j if isinstance(z, complex) else 0.0
    for c in reversed(p):
        v = v * z + float(c)
    return v


# ------------------------------------------------------------------ #

def pair_rows(roots, real_tol=1e-9):
    """Conjugate pairs (Im > 0 representative) sorted by |Im| ascending."""
    ups = [z for z in roots if z.imag > real_tol * (1 + abs(z))]
    return sorted(ups, key=lambda z: abs(z.imag))


def real_of(roots, real_tol=1e-9):
    return sorted(z.real for z in roots
                  if abs(z.imag) <= real_tol * (1 + abs(z)))


def portrait(seed, mode, degree):
    generate = directed_model if mode == "directed" else random_model
    built = generate(random.Random(seed), degree)
    if built is None or built[0] is None:
        return {"seed": seed, "error": "generator declined"}
    m, spec = built
    D = m.backbone_den
    Pnum = m.backbone_num
    H = m.critical_reduced
    Dp = pderiv(D)
    certificate = certify_backbone(m)
    out = {"seed": seed, "spec": str(spec),
           "complex_status": ("validated" if certificate.complete
                              else "partial"),
           "complex_certificate": certificate.as_dict(),
           "deg_transverse_A": len(m.alpha)-1,
           "deg_valley_den": certificate.valley_denominator.degree,
           "deg_backbone_den": len(D)-1,
           "deg_critical": len(H)-1}

    # ---- validation against the Sturm skeleton --------------------
    e = sturm.enumerate_critical_points(m)
    skel = sorted(float(q.b) for q in e.points)
    matched = (certificate.critical.complete
               and all(any(disk.distance_from_real_interval(
                   q.interval.lo, q.interval.hi) == 0
                           for disk in certificate.critical.disks)
                       for q in e.points))
    out["skeleton_matches_critical_disks[VALIDATED]"] = bool(matched)

    # ---- poles: complex zeros of the reduced backbone denominator --
    poles = []
    for disk in certificate.denominator.disks:
        if disk.centre_im <= 0:
            continue
        z = complex(float(disk.centre_re), float(disk.centre_im))
        # u=C-P/D, hence Res(u,z)=-P(z)/D'(z).
        residue = -peval(Pnum, z)/peval(Dp, z)
        poles.append({"re": z.real, "im": z.imag,
                      "disk_radius[VALIDATED]": float(disk.radius),
                      "res_u_re[PROPOSAL]": residue.real,
                      "res_u_im[PROPOSAL]": residue.imag,
                      "abs_res_u[PROPOSAL]": abs(residue)})
    poles.sort(key=lambda row: row["im"])
    poles = poles[:4]
    out["poles"] = poles

    # ---- ghost pairs: nonreal roots of the reduced critical divisor --
    critical_pairs = []
    for disk in certificate.critical.disks:
        if disk.centre_im <= 0 or disk.real_axis_clearance() <= 0:
            continue
        b0 = float(disk.centre_re)
        critical_pairs.append({
            "re": b0, "eps_centre": float(disk.centre_im),
            "real_axis_clearance_lower[VALIDATED]": float(
                disk.real_axis_clearance()),
            "disk_radius[VALIDATED]": float(disk.radius),
            "uprime_at_re[PROPOSAL]": float(
                peval(H, b0)/peval(D, b0)**2),
        })
    critical_pairs.sort(key=lambda row: row[
        "real_axis_clearance_lower[VALIDATED]"])
    out["critical_pairs"] = critical_pairs[:4]

    valley_poles = []
    for disk in certificate.valley_denominator.disks:
        if disk.centre_im > 0:
            valley_poles.append({
                "re": float(disk.centre_re),
                "im": float(disk.centre_im),
                "real_axis_clearance_lower[VALIDATED]": float(
                    disk.real_axis_clearance()),
                "disk_radius[VALIDATED]": float(disk.radius),
            })
    valley_poles.sort(key=lambda row: row[
        "real_axis_clearance_lower[VALIDATED]"])
    out["valley_chart_poles"] = valley_poles[:4]

    # ---- the level pencil -----------------------------------------
    # critical values u(b*) over the real skeleton, then the off-axis
    # branch-point distance to R over the working l-range
    Df = np.array([float(c) for c in D])
    Pf = np.array([float(c) for c in Pnum])
    Cf = float(m.C)

    def u_of(b):
        return Cf - peval(Pnum, b) / peval(D, b)

    crit_vals = sorted(u_of(b) for b in skel)
    out["critical_values"] = crit_vals
    if crit_vals:
        lo, hi = crit_vals[0], crit_vals[-1]
        pad = 0.1 * (hi - lo) if hi > lo else max(1.0, abs(hi)) * 0.1
        grid = np.linspace(lo - pad, hi + pad, 400)
        best = None
        for lv in grid:
            n = max(len(Pf), len(Df))
            S = np.zeros(n)
            S[:len(Pf)] += Pf
            S[:len(Df)] += (lv - Cf) * Df
            sc = np.max(np.abs(S))
            if sc == 0:
                continue
            for z in np.roots((S / sc)[::-1]):
                im = abs(z.imag)
                if im > 1e-9 * (1 + abs(z)):
                    if best is None or im < best["min_im"]:
                        best = {"min_im": im, "at_level": float(lv),
                                "re": float(z.real)}
        out["pencil"] = best
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--mode", choices=("directed", "random"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "out" / "pole_portrait.json"))
    args = ap.parse_args(argv)

    results = []
    for seed in args.seeds:
        r = portrait(seed, args.mode, args.degree)
        results.append(r)
        if "error" in r:
            print(f"\nseed {seed}: {r['error']}")
            continue
        print(f"\nseed {seed}   {r['spec']}   "
              f"degA {r['deg_transverse_A']}  "
              f"degD {r['deg_backbone_den']}  "
              f"degH {r['deg_critical']}   "
              f"exact complex divisor {r['complex_status']}   "
              f"skeleton-vs-H-disks "
              f"{'OK' if r['skeleton_matches_critical_disks[VALIDATED]'] else 'MISMATCH'}")
        for p in r["poles"]:
            print(f"  pole pair   re {p['re']:>14.6g}   im {p['im']:>12.6g}"
                  f"   disk r {p['disk_radius[VALIDATED]']:>10.3g}"
                  f"   |Res u| {p['abs_res_u[PROPOSAL]']:>12.6g}")
        for gp in r["critical_pairs"]:
            print(f"  H pair      re {gp['re']:>14.6g}   "
                  f"clearance "
                  f"{gp['real_axis_clearance_lower[VALIDATED]']:>12.6g}"
                  f"   u' there {gp['uprime_at_re[PROPOSAL]']:>12.6g}")
        for vp in r["valley_chart_poles"]:
            print(f"  a* pole     re {vp['re']:>14.6g}   "
                  f"clearance "
                  f"{vp['real_axis_clearance_lower[VALIDATED]']:>12.6g}")
        if r.get("pencil"):
            pc = r["pencil"]
            print(f"  pencil      min off-axis |Im| {pc['min_im']:.6g}"
                  f"   at level {pc['at_level']:.6g}   re {pc['re']:.6g}")

    # comparison table
    print(f"\n{'seed':>12} {'min u-pole clr':>16} {'min H-pair clr':>17}"
          f" {'min a*-pole clr':>17} {'pencil min Im':>14}")
    print("-" * 82)
    for r in results:
        if "error" in r:
            continue
        min_im = min((p["im"]-p["disk_radius[VALIDATED]"]
                      for p in r["poles"]), default=float("nan"))
        min_eps = min((p["real_axis_clearance_lower[VALIDATED]"]
                       for p in r["critical_pairs"]), default=float("nan"))
        min_valley = min((p["real_axis_clearance_lower[VALIDATED]"]
                          for p in r["valley_chart_poles"]),
                         default=float("nan"))
        pmin = r["pencil"]["min_im"] if r.get("pencil") else float("nan")
        print(f"{r['seed']:>12} {min_im:>16.6g} {min_eps:>17.6g}"
              f" {min_valley:>17.6g} {pmin:>14.6g}")

    Path(args.out).write_text(json.dumps(results, indent=2,
                                         default=str) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
