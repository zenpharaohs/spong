"""Frobenius launch vs Poincare stubs: accuracy, reach, and speed.

The unstable separatrix at a saddle b1 is, in the valley graph chart
w = a - a*(b), the unique ANALYTIC solution of the orbit equation at its
regular singular point (docs/wall_theory.md).  This script computes its
Taylor jet directly and races it against the engine's Poincare-transform
stubs.

JET.  Clearing denominators, the orbit equation is polynomial:

    (BN + A'A^2 w^2 - 2AGw)(A^2 w' + G) = 2A^5 w,    G = B'A - A'B.

Truncate w = sum_{k>=1} c_k s^k, s = b - b1.  The Hessian's unstable
eigenvector fixes c_1 (the order-zero balance has two invariant directions),
then Newton solves only for c_2..c_n.  Pinning c_1 prevents an
ill-conditioned high-order solve from drifting to the other invariant germ.
One analytic curve passes THROUGH the saddle, so a single jet serves
both departure directions; stubs are per-direction.

CHECKS.  Both compare against fine RK4 on the orbit ODE (the Poincare
technology's own trajectory), but they have different evidential force:

  * STUB-RELATIVE: launch from the stub endpoint, compare at 2x/5x/10x
    the stub's own offset.  Requires a non-degenerate stub.
  * SERIES SELF-CONSISTENCY (absolute radii): seed RK from the truncated
    series and compare farther out at 10x/100x/1000x.  This remains useful
    when a stub is degenerate, but is circular near its seed and is NOT an
    independent proof that the selected germ is correct.

The measurement domain is capped conservatively by the smaller of

    R_fixed = dist(b1, nearest other fixed zero of H, D, V, or A)
    R_emp   = a coefficient-growth estimate.

R_fixed is only a fixed-obstruction heuristic: nonlinear invariant graphs
can have movable complex singularities before any coefficient pole.  The
exact Lehmer-Schur certificate separately reports a validated pole-free
distance for the reduced backbone.  A true convergence radius still needs
a nonlinear majorant.  Also reported: wall-clock for stub materialisation
versus jet Newton and series evaluation.

    python scripts/frobenius_launch.py 1785201004
    python scripts/frobenius_launch.py 953953598 --pow2
    python scripts/frobenius_launch.py 555999196 --pow2 --order 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import _poly as P, sturm                       # noqa: E402
from spong.complex_structure import certify_backbone      # noqa: E402
from qualify import directed_model, random_model          # noqa: E402
from pole_portrait import exact_ABCN, croots                # noqa: E402
from box_experiment_arms import normalised                # noqa: E402


def fl(p):
    return np.array([float(c) for c in p], dtype=float)


def pv(p, x):
    return np.polyval(p[::-1], x)


class Chart:
    """Float coefficient arrays for the orbit equation at one model."""

    def __init__(self, m):
        A, B, C, N = exact_ABCN(m)
        A, B, N = tuple(A), tuple(B), tuple(N)
        self.A, self.B, self.N = fl(A), fl(B), fl(N)
        self.D = fl(m.backbone_den)
        valley_common = P.gcd_poly(m.alpha, m.beta)
        valley_denominator, remainder = P.divmod_exact(
            m.alpha, valley_common)
        if remainder:
            raise ArithmeticError("failed to reduce exact valley chart")
        self.V = fl(valley_denominator)
        self.H = fl(m.critical_reduced)
        self.C = float(C)
        Ap, Bp = P.deriv(A), P.deriv(B)
        self.Ap, self.Bp = fl(Ap), fl(Bp)
        self.G = np.polysub((np.polymul(self.Bp[::-1], self.A[::-1])),
                            (np.polymul(self.Ap[::-1], self.B[::-1])))[::-1]
        self.BN = np.polymul(self.B[::-1], self.N[::-1])[::-1]
        self.A2 = np.polymul(self.A[::-1], self.A[::-1])[::-1]
        self.A5 = np.polymul(np.polymul(self.A2[::-1], self.A2[::-1]),
                             self.A[::-1])[::-1]
        self.ApA2 = np.polymul(self.Ap[::-1], self.A2[::-1])[::-1]
        self.AG = np.polymul(self.A[::-1], self.G[::-1])[::-1]
        exact_G = P.sub(P.mul(Bp, A), P.mul(Ap, B))
        exact_A2 = P.mul(A, A)
        self._exact_series = {
            "BN": P.mul(B, N),
            "ApA2": P.mul(Ap, exact_A2),
            "AG": P.mul(A, exact_G),
            "A2": exact_A2,
            "G": exact_G,
            "A5": P.mul(P.mul(exact_A2, exact_A2), A),
        }

    def astar(self, b):
        return pv(self.B, b) / pv(self.A, b)

    def astar_prime(self, b):
        return pv(self.G, b) / pv(self.A2, b)

    def wprime(self, b, w):
        """Orbit ODE dw/db, direct form (for the RK referee)."""
        Lb = (pv(self.BN, b) / pv(self.A2, b)
              + pv(self.Ap, b) * w * w
              - 2.0 * pv(self.A, b) * self.astar_prime(b) * w)
        return 2.0 * pv(self.A, b) * w / Lb - self.astar_prime(b)

    @staticmethod
    def _shift_exact(p, centre: Fraction, n):
        """p(centre+s), translated exactly before the one float rounding."""
        out = []
        for k in range(n+1):
            out.append(float(sum((
                p[j]*comb(j, k)*centre**(j-k)
                for j in range(k, len(p))
            ), Fraction(0))))
        return np.asarray(out)

    def series_at(self, q, n):
        centre = q.local.center_interval.mid if q.local is not None \
            else Fraction(float(q.b))
        out = {name: self._shift_exact(p, centre, n)
               for name, p in self._exact_series.items()}
        # The centre represents the isolated algebraic critical point to
        # 2^-160.  As in LocalJet, impose the structural critical identity
        # rather than retaining the midpoint's meaningless residual.
        out["BN"][0] = 0.0
        return out


def smul(x, y, n):
    return np.convolve(x[:n + 1], y[:n + 1])[:n + 1]


def residual(local, c, n, with_scale=False):
    """First n+1 series coefficients of the cleared orbit equation."""
    w = np.zeros(n + 1)
    w[1:1 + len(c)] = c
    wp = np.array([(k + 1) * w[k + 1] if k + 1 <= n else 0.0
                   for k in range(n + 1)])
    BN, ApA2, AG = local["BN"], local["ApA2"], local["AG"]
    A2, G, A5 = local["A2"], local["G"], local["A5"]
    ww = smul(w, w, n)
    first_terms = (BN, smul(ApA2, ww, n), -2.0*smul(AG, w, n))
    second_terms = (smul(A2, wp, n), G)
    first = sum(first_terms, start=np.zeros(n+1))
    second = sum(second_terms, start=np.zeros(n+1))
    right = 2.0*smul(A5, w, n)
    value = smul(first, second, n)-right
    if not with_scale:
        return value
    first_scale = sum((np.abs(x) for x in first_terms),
                      start=np.zeros(n+1))
    second_scale = sum((np.abs(x) for x in second_terms),
                       start=np.zeros(n+1))
    scale = smul(first_scale, second_scale, n) + np.abs(right)
    return value, scale


def unstable_slope(m, ch, q, verbose=False):
    """c1 from the Hessian's unstable eigenvector, in (b, w)."""
    a, b = float(q.a), float(q.b)
    if q.local is not None:
        vals = np.asarray(q.local.spectral.eigenvalues)
        frame = np.asarray(q.local.spectral.frame)
        da, db = float(frame[0, 0]), float(frame[1, 0])
        H00 = q.local.exact_hessian[0][0]
        H01 = q.local.exact_hessian[0][1]
        astar_prime = float(-H01/H00)
    else:
        A, Ap = pv(ch.A, b), pv(ch.Ap, b)
        d2A = np.polyval(np.polyder(np.polyder(ch.A[::-1])), b)
        d2B = np.polyval(np.polyder(np.polyder(ch.B[::-1])), b)
        Laa = 2.0 * A
        Lab = 2.0 * (a * Ap - pv(ch.Bp, b))
        Lbb = -2.0 * a * d2B + a * a * d2A
        H = np.array([[Laa, Lab], [Lab, Lbb]])
        vals, vecs = np.linalg.eigh(H)
        v = vecs[:, int(np.argmin(vals))]
        da, db = float(v[0]), float(v[1])
        astar_prime = ch.astar_prime(b)
    if verbose:
        print(f"      [diag] b={b:+.6g}  a={a:+.6g}  eigvals={vals}  "
              f"eigvec(unstable)=({da:+.6g},{db:+.6g})  "
              f"|da/db|={abs(da/db) if db else float('inf'):.6g}")
    if abs(db) < 1e-14 * abs(da):
        return None                          # transverse: not graphable
    return da/db-astar_prime


def frobenius_jet(m, ch, q, n, verbose=False):
    if n < 1:
        return None, "order must be at least one"
    c1 = unstable_slope(m, ch, q, verbose=verbose)
    if c1 is None:
        return None, "transverse eigenvector (chart cannot represent it)"
    b1 = float(q.b)
    local = ch.series_at(q, n)
    c = np.zeros(n)
    c[0] = c1
    cond = float("nan")
    # c1 selects the unstable eigendirection and is not a free Newton
    # variable.  Solve the triangular tail equations without allowing a
    # high-order ill-conditioned solve to jump to the other invariant germ.
    for _ in range(60):
        full = residual(local, c, n)
        r = full[2:n+1]
        if len(r) == 0 or np.max(np.abs(r)) == 0.0:
            break
        J = np.zeros((n-1, n-1))
        for j in range(1, n):
            h = 1e-7 * max(1.0, abs(c[j]))
            cp = c.copy()
            cp[j] += h
            J[:, j-1] = (residual(local, cp, n)[2:n+1]-r)/h
        cond = np.linalg.cond(J)
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None, f"singular tail Jacobian (cond={cond:.3e})"
        c[1:] += step
        if np.max(np.abs(step)) <= 1e-14 * max(1.0, np.max(np.abs(c[1:]))):
            break
    r_final, r_scale = residual(local, c, n, with_scale=True)
    active = slice(1, n+1)
    relative = np.max(np.abs(r_final[active])
                      / np.maximum(r_scale[active], 1e-300))
    if not np.isfinite(relative) or relative > 1e-9:
        return None, (f"Newton stalled, not converged: "
                     f"relative backward error={relative:.3e}, "
                     f"|residual|={np.max(np.abs(r_final[active])):.3e}, "
                     f"cond={cond:.3e}")
    if verbose:
        print(f"      [diag] Newton converged, backward error={relative:.3e}, "
              f"|residual|={np.max(np.abs(r_final[active])):.3e}, "
              f"cond(J)={cond:.3e}, c1={c1:+.6g}")
    return c, None


def fixed_obstruction_distance(ch, b1):
    """Heuristic distance to other fixed divisors, not a radius proof."""
    others = []
    for p in (ch.H, ch.D, ch.V, ch.A):
        for z in croots(tuple(p)):
            if abs(z - b1) > 1e-9 * (1.0 + abs(b1)):
                others.append(abs(z - b1))
    return min(others) if others else float("inf")


def rk_reference(ch, b0, w0, b_targets):
    out = {}
    b, w = float(b0), float(w0)
    for bt in b_targets:
        steps = 20000
        h = (bt - b) / steps
        for _ in range(steps):
            k1 = ch.wprime(b, w)
            k2 = ch.wprime(b + h / 2, w + h * k1 / 2)
            k3 = ch.wprime(b + h / 2, w + h * k2 / 2)
            k4 = ch.wprime(b + h, w + h * k3)
            w += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            b += h
        out[bt] = w
    return out


def series_w(c, s):
    return float(np.polyval(c[::-1], s) * s)


def series_self_consistency_test(ch, c, b1, test_cap, direction,
                                 radii=(10.0, 100.0, 1000.0)):
    """Compare series and RK after initializing RK from that same series.

    This measures persistence of the truncated series along the ODE.  It is
    deliberately labelled self-consistency, not an independent launch or
    convergence certificate.
    """
    if not np.isfinite(test_cap) or test_cap <= 0.0:
        return []
    scale_floor = 1e-10 * max(1.0, abs(b1))
    eps_abs = min(test_cap * 1e-3,
                  max(test_cap * 1e-6, scale_floor))
    if not np.isfinite(eps_abs) or eps_abs <= 0.0:
        return []
    eps0 = direction * eps_abs
    w0 = series_w(c, eps0)
    targets_s = [r * eps0 for r in radii
                 if abs(r * eps0) < 0.25 * test_cap]
    if not targets_s:
        return []
    targets_b = [b1 + s for s in targets_s]
    ref = rk_reference(ch, b1 + eps0, w0, targets_b)
    rows = []
    for s, bt in zip(targets_s, targets_b):
        wv = series_w(c, s)
        wr = ref[bt]
        rows.append({"s": s, "b": bt, "w_series": wv, "w_rk": wr,
                     "rel_err": abs(wv - wr) / max(abs(wr), 1e-300)})
    return rows


def run(seed, mode, degree, pow2, order):
    generate = directed_model if mode == "directed" else random_model
    built = generate(__import__("random").Random(seed), degree)
    if built is None or built[0] is None:
        return {"seed": seed, "error": "generator declined"}
    m, spec = built
    if pow2:
        m = normalised(m, pow2=True)
    ch = Chart(m)
    t0 = time.perf_counter()
    backbone_certificate = certify_backbone(m)
    complex_time = time.perf_counter() - t0
    e = sturm.enumerate_critical_points(m)
    t0 = time.perf_counter()
    stub_e = sturm.materialize_stubs(m, e)
    stub_time = time.perf_counter() - t0
    report = {"seed": seed, "spec": str(spec), "pow2": bool(pow2),
              "complex_backbone_status": (
                  "validated" if backbone_certificate.complete else "partial"),
              "complex_backbone_secs": round(complex_time, 4),
              "stub_materialise_secs": round(stub_time, 4), "saddles": []}
    print(f"\nseed {seed}   {spec}   stub materialisation "
          f"{stub_time*1e3:.1f} ms (all saddles); exact complex divisor "
          f"{'validated' if backbone_certificate.complete else 'partial'} "
          f"in {complex_time:.2f} s")
    for q in stub_e.points:
        if q.kind != "saddle":
            continue
        b1 = float(q.b)
        t0 = time.perf_counter()
        c, fail_reason = frobenius_jet(m, ch, q, order, verbose=True)
        jet_time = time.perf_counter() - t0
        pole_clearance = backbone_certificate.pole_clearance(q.interval)
        valley_clearance = backbone_certificate.valley_clearance(q.interval)
        if c is None:
            print(f"  saddle b={b1:+.6g}: jet failed -- {fail_reason}")
            report["saddles"].append({
                "b": b1, "status": "failed", "reason": fail_reason,
                "jet_secs": round(jet_time, 4),
                "backbone_pole_clearance_lower[VALIDATED]": (
                    float(pole_clearance)
                    if pole_clearance is not None else None),
                "valley_chart_pole_clearance_lower[VALIDATED]": (
                    float(valley_clearance)
                    if valley_clearance is not None else None),
            })
            continue
        R_fixed = fixed_obstruction_distance(ch, b1)
        tail = [abs(x) ** (1.0 / (k + 1)) for k, x in enumerate(c)
                if x != 0.0]
        R_emp = 1.0 / max(tail[-5:]) if len(tail) >= 5 else float("nan")
        caps = [x for x in (R_fixed, R_emp)
                if np.isfinite(x) and x > 0.0]
        test_cap = min(caps) if caps else float("nan")
        row = {"b": b1, "status": "computed",
               "jet_secs": round(jet_time, 4),
               "fixed_obstruction_distance[HEURISTIC]": (
                   R_fixed if np.isfinite(R_fixed) else None),
               "coefficient_radius_estimate[EMPIRICAL]": (
                   R_emp if np.isfinite(R_emp) else None),
               "measurement_cap[HEURISTIC]": (
                   test_cap if np.isfinite(test_cap) else None),
               "backbone_pole_clearance_lower[VALIDATED]": (
                   float(pole_clearance)
                   if pole_clearance is not None else None),
               "valley_chart_pole_clearance_lower[VALIDATED]": (
                   float(valley_clearance)
                   if valley_clearance is not None else None),
               "dirs": []}
        print(f"  saddle b={b1:+.6g}   jet({order}) {jet_time*1e3:.1f} ms   "
              f"fixed obstruction {R_fixed:.4g} [heuristic]   "
              f"coefficient radius {R_emp:.4g} [empirical]   "
              f"pole clearance "
              f"{float(pole_clearance) if pole_clearance is not None else float('nan'):.4g} "
              f"and valley-chart clearance "
              f"{float(valley_clearance) if valley_clearance is not None else float('nan'):.4g} "
              "[validated]")

        # Available without a stub, but circular at its seed.  It is a useful
        # regression/continuation check, not a proof of the selected germ.
        for direction in (-1, +1):
            rows = series_self_consistency_test(
                ch, c, b1, test_cap, direction)
            if not rows:
                continue
            print(f"    dir {direction:+d} (series self-consistency):")
            for cr in rows:
                print(f"      s={cr['s']:+.3e}  series {cr['w_series']:+.9e}"
                      f"   rk {cr['w_rk']:+.9e}   rel {cr['rel_err']:.2e}")
            row["dirs"].append({"direction": direction,
                                "mode": "series_self_consistency",
                                "independent": False,
                                "compare": rows})

        for s in q.stubs:
            if s.manifold != "unstable":
                continue
            curve = np.asarray(s.curve, dtype=float)
            if len(curve) < 2:
                print(f"    dir {s.b_direction:+d}: stub is degenerate "
                      "(zero-length curve) -- this is the launch failure "
                      "the horizon class exhibits; see self-consistency rows "
                      "above instead")
                continue
            bs, as_ = float(curve[-1, 1]), float(curve[-1, 0])
            s_stub = bs - b1
            if abs(s_stub) < 1e-13 * max(1.0, abs(b1)):
                print(f"    dir {s.b_direction:+d}: stub offset ~0 "
                      "(degenerate) -- see self-consistency rows above instead")
                continue
            w_stub = as_ - ch.astar(bs)
            w_ser = float(np.polyval(c[::-1], s_stub) * s_stub)
            radii = [r for r in (2.0, 5.0, 10.0)
                     if (not np.isfinite(test_cap)
                         or abs(s_stub) * r < 0.25 * test_cap)]
            targets = [b1 + s_stub * r for r in radii]
            t0 = time.perf_counter()
            ref = rk_reference(ch, bs, w_stub, targets)
            rk_time = time.perf_counter() - t0
            cmp_rows = []
            for r, bt in zip(radii, targets):
                sv = bt - b1
                wv = float(np.polyval(c[::-1], sv) * sv)
                wr = ref[bt]
                err = abs(wv - wr) / max(abs(wr), 1e-300)
                cmp_rows.append({"radius_x_stub": r, "b": bt,
                                 "w_series": wv, "w_rk": wr,
                                 "rel_err": err})
            d = {"direction": int(s.b_direction), "mode": "stub",
                 "stub_offset": s_stub,
                 "agree_at_stub": abs(w_ser - w_stub)
                 / max(abs(w_stub), 1e-300),
                 "rk_secs": round(rk_time, 4), "compare": cmp_rows}
            row["dirs"].append(d)
            print(f"    dir {s.b_direction:+d} (stub-relative): stub offset "
                  f"{s_stub:+.3e}   series-vs-stub "
                  f"{d['agree_at_stub']:.2e}")
            for cr in cmp_rows:
                print(f"      x{cr['radius_x_stub']:>4.0f} stub radius:  "
                      f"series {cr['w_series']:+.9e}   "
                      f"rk {cr['w_rk']:+.9e}   rel {cr['rel_err']:.2e}")
        report["saddles"].append(row)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--mode", choices=("directed", "random"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--pow2", action="store_true")
    ap.add_argument("--order", type=int, default=24)
    ap.add_argument("--out",
                    default=str(REPO / "out" / "frobenius_launch.json"))
    args = ap.parse_args(argv)
    results = [run(s, args.mode, args.degree, args.pow2, args.order)
               for s in args.seeds]
    Path(args.out).write_text(json.dumps(results, indent=2,
                                         default=str) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
