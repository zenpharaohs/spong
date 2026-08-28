"""Does the hyperbolic asymptote become admissible more often at low degree?

Random cases per degree; for every STABLE branch record whether the matched
asymptotic ever becomes as faithful to the field as the traced chord, and how
far out in b the branch actually got relative to the scale where the expansion
should start working.

The expansion is in 1/b about b = infinity: A = alpha b^(2d) (1 + a1/b + ...),
so it is trustworthy once |b| is past the outermost root of A -- and A has
degree 2d, so at higher degree that radius grows while the box does not.
`b_end / rootA` is the number that should decide it, not the degree itself.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "src")
from spong import atlas, model, portrait, sturm            # noqa: E402

sys.path.insert(0, "scripts")
from stable_handoff_probe import (                          # noqa: E402
    highest_saddle_loss, tail_coefficients, match_K0, tail_b,
    tangency_residual, first_true)


def mulberry32(seed):
    t = seed & 0xFFFFFFFF

    def rnd():
        nonlocal t
        t = (t + 0x6D2B79F5) & 0xFFFFFFFF
        x = t
        x = ((x ^ (x >> 15)) * (x | 1)) & 0xFFFFFFFF
        x ^= (x + ((x ^ (x >> 7)) * (x | 61))) & 0xFFFFFFFF
        return (((x ^ (x >> 14)) & 0xFFFFFFFF)) / 4294967296.0
    return rnd


def random_case(seed, deg):
    rnd = mulberry32(seed * 2654435761 + deg * 10007)
    def coeffs():
        c = [2.0 * rnd() - 1.0 for _ in range(deg + 1)]
        if abs(c[deg]) < 0.05:
            c[deg] = -1.0 if c[deg] < 0 else 1.0
        return c
    g = coeffs()
    # g(0) = 0 makes A(0) = 0, so psi-positivity fails at the origin for every
    # distribution and the case can never certify.  Same guard the explorer uses.
    if abs(g[0]) < 0.05:
        g[0] = 0.5
    return coeffs(), g


def outer_root_radius(m):
    """Largest |real root| of A, the radius past which A ~ alpha b^(2d)."""
    A = np.asarray([float(c) for c in m.alpha], dtype=float)
    if len(A) < 2:
        return 0.0
    r = np.roots(A[::-1])
    # A > 0 everywhere (psi_positive) means A has NO real roots, so the
    # expansion radius is set by the largest COMPLEX root modulus.
    return float(np.max(np.abs(r))) if r.size else 0.0


def scan(m, br, coeffs, window=8):
    Y = np.asarray(br.Y, dtype=float)
    n = len(Y)
    if n < window + 2 or coeffs is None:
        return None
    a, b = Y[:, 0], Y[:, 1]
    d, a1c, _ = coeffs
    for i in range(1, n - 1):
        if abs(a[i]) < 1e-12:
            continue
        K0 = match_K0(a[i], b[i], d, a1c)
        good = True
        for j in range(i, min(i + window, n - 1)):
            bj = tail_b(a[j], d, a1c, K0, b[i])
            if bj is None or abs(bj) < 1e-300:
                good = False
                break
            dbda = (d * a[j] - a1c / (2.0 * math.sqrt(d))) / bj
            r_as = tangency_residual(m, a[j], b[j], 1.0, dbda)
            r_ch = tangency_residual(m, a[j], b[j],
                                     a[j + 1] - a[j], b[j + 1] - b[j])
            if not (r_as <= max(r_ch, 1e-15)):
                good = False
                break
        if good:
            return i
    return None


def scan_loose(m, br, coeffs, tol=1e-3, window=8):
    """Looser: asymptotic tangency residual below an absolute tolerance.

    The strict test asks the tail to match a high-order chord, a demanding
    standard in a smooth region.  This asks only that it be a good integral
    curve in its own right.
    """
    Y = np.asarray(br.Y, dtype=float)
    n = len(Y)
    if n < window + 2 or coeffs is None:
        return None
    a, b = Y[:, 0], Y[:, 1]
    d, a1c, _ = coeffs
    for i in range(1, n - 1):
        if abs(a[i]) < 1e-12:
            continue
        K0 = match_K0(a[i], b[i], d, a1c)
        good = True
        for j in range(i, min(i + window, n - 1)):
            bj = tail_b(a[j], d, a1c, K0, b[i])
            if bj is None or abs(bj) < 1e-300:
                good = False; break
            dbda = (d * a[j] - a1c / (2.0 * math.sqrt(d))) / bj
            if tangency_residual(m, a[j], b[j], 1.0, dbda) > tol:
                good = False; break
        if good:
            return i
    return None


def main():
    import csv
    rows = []
    degrees = [1, 2, 3, 4, 5, 6]
    seeds = [1, 2, 3, 4, 5]
    print(f"{'deg':>4}{'cases':>7}{'stable':>8}{'tangent':>9}{'rate':>8}"
          f"{'med |b_end|':>13}{'med rootA':>11}{'med ratio':>11}")
    for deg in degrees:
        nb = nt = nl = ncase = 0
        bends, roots, ratios = [], [], []
        for sd in seeds:
            f, g = random_case(sd, deg)
            try:
                D = max(len(f) - 1, len(g) - 1)
                mu = model.moments_uniform01(2 * D + 1)
                m = model.build(f, g, mu)
                e = sturm.enumerate_critical_points(m)
                if not e.psi_positive or not e.morse:
                    continue
                p = portrait.certified_compute(m)
            except Exception:
                continue
            ncase += 1
            coeffs = tail_coefficients(m)
            rA = outer_root_radius(m)
            for br in p.branches:
                if br.kind != "stable":
                    continue
                Y = np.asarray(br.Y, dtype=float)
                if len(Y) < 12:
                    continue
                nb += 1
                be = abs(float(Y[-1, 1]))
                bends.append(be)
                roots.append(rA)
                ratios.append(be / rA if rA > 1e-12 else float("inf"))
                st = scan(m, br, coeffs) is not None
                lo = scan_loose(m, br, coeffs) is not None
                if st:
                    nt += 1
                if lo:
                    nl += 1
                rows.append({
                    "deg": deg, "seed": sd, "n": len(Y),
                    "b_end": be, "rootA": rA,
                    "ratio": (be / rA if rA > 1e-12 else float("inf")),
                    "strict": int(st), "loose": int(lo)})
        med = lambda v: float(np.median(v)) if v else float("nan")
        rate = (100.0 * nl / nb) if nb else float("nan")
        print(f"{deg:>4}{ncase:>7}{nb:>8}{nt:>9}{nl:>7}{rate:>7.1f}%"
              f"{med(bends):>13.2f}{med(roots):>11.2f}{med(ratios):>11.2f}",
              flush=True)
    with open("degree_sweep_branches.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # Does an INDIVIDUAL branch's ratio predict firing?  The median over all
    # branches cannot see this: if only the high-ratio tail fires, a flat
    # median is perfectly compatible with a sharp per-branch threshold.
    fired = [r["ratio"] for r in rows if r["loose"] and math.isfinite(r["ratio"])]
    quiet = [r["ratio"] for r in rows if not r["loose"] and math.isfinite(r["ratio"])]
    nan = float("nan")
    print()
    print(f"per-branch ratio |  fired n={len(fired):3d}  "
          f"median {np.median(fired) if fired else nan:.2f}  "
          f"min {min(fired) if fired else nan:.2f}")
    print(f"                 |  quiet n={len(quiet):3d}  "
          f"median {np.median(quiet) if quiet else nan:.2f}  "
          f"max {max(quiet) if quiet else nan:.2f}")
    print("wrote degree_sweep_branches.csv")


if __name__ == "__main__":
    main()
