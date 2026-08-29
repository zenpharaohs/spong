"""Frobenius launch vs Poincare stubs: accuracy, reach, and speed.

The unstable separatrix at a saddle b1 is, in the valley graph chart
w = a - a*(b), the unique ANALYTIC solution of the orbit equation at its
regular singular point (docs/wall_theory.md).  This script computes its
Taylor jet directly and races it against the engine's Poincare-transform
stubs.

JET.  Clearing denominators, the orbit equation is polynomial:

    (BN + A'A^2 w^2 - 2AGw)(A^2 w' + G) = 2A^5 w,    G = B'A - A'B.

Truncate w = sum_{k>=1} c_k s^k, s = b - b1, and solve the first n
residual coefficients for c_1..c_n by Newton; c_1 is seeded from the
Hessian's unstable eigenvector (the order-zero balance is quadratic --
two invariant directions -- and the eigenvector selects the branch).
One analytic curve passes THROUGH the saddle, so a single jet serves
both departure directions; stubs are per-direction.

REFEREE.  The incumbent judges itself: launch from the stub endpoint and
continue by fine RK4 on the orbit ODE (the Poincare technology's own
trajectory), then measure |w_series - w_RK| at increasing radii, up to a
fraction of the certified convergence radius

    R = dist(b1, nearest other complex root of B*N or zero of A),

which the complex portrait supplies.  Also reported: the empirical
radius 1/limsup|c_k|^{1/k}, and wall-clock for stub materialisation vs
jet Newton vs series evaluation.

    python scripts/frobenius_launch.py 1785201004
    python scripts/frobenius_launch.py 953953598 --pow2
    python scripts/frobenius_launch.py 555999196 --pow2 --order 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import sturm                                   # noqa: E402
from qualify import directed_model, random_model          # noqa: E402
from pole_portrait import exact_ABCN, pmul, pderiv, croots  # noqa: E402
from box_experiment_arms import normalised                # noqa: E402


def fl(p):
    return np.array([float(c) for c in p], dtype=float)


def pv(p, x):
    return np.polyval(p[::-1], x)


class Chart:
    """Float coefficient arrays for the orbit equation at one model."""

    def __init__(self, m):
        A, B, C, N = exact_ABCN(m)
        self.A, self.B, self.N = fl(A), fl(B), fl(N)
        self.C = float(C)
        self.Ap, self.Bp = fl(pderiv(A)), fl(pderiv(B))
        self.G = np.polysub((np.polymul(self.Bp[::-1], self.A[::-1])),
                            (np.polymul(self.Ap[::-1], self.B[::-1])))[::-1]
        self.BN = np.polymul(self.B[::-1], self.N[::-1])[::-1]
        self.A2 = np.polymul(self.A[::-1], self.A[::-1])[::-1]
        self.A5 = np.polymul(np.polymul(self.A2[::-1], self.A2[::-1]),
                             self.A[::-1])[::-1]
        self.ApA2 = np.polymul(self.Ap[::-1], self.A2[::-1])[::-1]
        self.AG = np.polymul(self.A[::-1], self.G[::-1])[::-1]

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


def taylor_at(p, b1, n):
    """First n+1 Taylor coefficients of polynomial p about b1."""
    out = np.zeros(n + 1)
    q = p[::-1].copy()
    for k in range(n + 1):
        out[k] = np.polyval(q, b1)
        q = np.polyder(q)
        out[k] /= 1.0 if k == 0 else 1.0  # divided below
    fact = 1.0
    for k in range(1, n + 1):
        fact *= k
        out[k] /= fact
    return out


def smul(x, y, n):
    return np.convolve(x[:n + 1], y[:n + 1])[:n + 1]


def residual(ch, b1, c, n):
    """First n+1 series coefficients of the cleared orbit equation."""
    w = np.zeros(n + 1)
    w[1:1 + len(c)] = c
    wp = np.array([(k + 1) * w[k + 1] if k + 1 <= n else 0.0
                   for k in range(n + 1)])
    BN = taylor_at(ch.BN, b1, n)
    ApA2 = taylor_at(ch.ApA2, b1, n)
    AG = taylor_at(ch.AG, b1, n)
    A2 = taylor_at(ch.A2, b1, n)
    G = taylor_at(ch.G, b1, n)
    A5 = taylor_at(ch.A5, b1, n)
    left = smul(BN + smul(ApA2, smul(w, w, n), n) - 2.0 * smul(AG, w, n),
                smul(A2, wp, n) + G, n)
    return left - 2.0 * smul(A5, w, n)


def unstable_slope(m, ch, q):
    """c1 from the Hessian's unstable eigenvector, in (b, w)."""
    a, b = float(q.a), float(q.b)
    A, Ap = pv(ch.A, b), pv(ch.Ap, b)
    App = pv(fl(pderiv(tuple(ch.A))), b) if False else None
    # exact second derivatives of L = C - 2aB + a^2 A
    d2A = np.polyval(np.polyder(np.polyder(ch.A[::-1])), b)
    d2B = np.polyval(np.polyder(np.polyder(ch.B[::-1])), b)
    Laa = 2.0 * A
    Lab = 2.0 * (a * Ap - pv(ch.Bp, b))
    Lbb = -2.0 * a * d2B + a * a * d2A
    H = np.array([[Laa, Lab], [Lab, Lbb]])
    vals, vecs = np.linalg.eigh(H)
    v = vecs[:, int(np.argmin(vals))]        # negative eigenvalue: unstable
    da, db = float(v[0]), float(v[1])
    if abs(db) < 1e-14 * abs(da):
        return None                          # transverse: not graphable
    return da / db - ch.astar_prime(b)


def frobenius_jet(m, ch, q, n):
    c1 = unstable_slope(m, ch, q)
    if c1 is None:
        return None
    b1 = float(q.b)
    c = np.zeros(n)
    c[0] = c1
    for _ in range(60):                      # Newton on residual coeffs 1..n
        r = residual(ch, b1, c, n)[1:n + 1]
        if np.max(np.abs(r)) == 0.0:
            break
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(c[j]))
            cp = c.copy()
            cp[j] += h
            J[:, j] = (residual(ch, b1, cp, n)[1:n + 1] - r) / h
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None
        c += step
        if np.max(np.abs(step)) <= 1e-14 * max(1.0, np.max(np.abs(c))):
            break
    return c


def convergence_radius(ch, b1):
    others = []
    for p in (ch.BN, ch.A):
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


def run(seed, mode, degree, pow2, order):
    generate = directed_model if mode == "directed" else random_model
    built = generate(__import__("random").Random(seed), degree)
    if built is None or built[0] is None:
        return {"seed": seed, "error": "generator declined"}
    m, spec = built
    if pow2:
        m = normalised(m, pow2=True)
    ch = Chart(m)
    e = sturm.enumerate_critical_points(m)
    t0 = time.perf_counter()
    stub_e = sturm.materialize_stubs(m, e)
    stub_time = time.perf_counter() - t0
    report = {"seed": seed, "spec": str(spec), "pow2": bool(pow2),
              "stub_materialise_secs": round(stub_time, 4), "saddles": []}
    print(f"\nseed {seed}   {spec}   stub materialisation "
          f"{stub_time*1e3:.1f} ms (all saddles)")
    for q in stub_e.points:
        if q.kind != "saddle":
            continue
        b1 = float(q.b)
        t0 = time.perf_counter()
        c = frobenius_jet(m, ch, q, order)
        jet_time = time.perf_counter() - t0
        if c is None:
            print(f"  saddle b={b1:+.6g}: jet failed (transverse or "
                  "singular Newton)")
            continue
        R = convergence_radius(ch, b1)
        tail = [abs(x) ** (1.0 / (k + 1)) for k, x in enumerate(c)
                if x != 0.0]
        R_emp = 1.0 / max(tail[-5:]) if len(tail) >= 5 else float("nan")
        row = {"b": b1, "jet_secs": round(jet_time, 4),
               "R_complex": R, "R_empirical": R_emp, "dirs": []}
        print(f"  saddle b={b1:+.6g}   jet({order}) {jet_time*1e3:.1f} ms   "
              f"R_complex {R:.4g}   R_empirical {R_emp:.4g}")
        for s in q.stubs:
            if s.manifold != "unstable":
                continue
            curve = np.asarray(s.curve, dtype=float)
            bs, as_ = float(curve[-1, 1]), float(curve[-1, 0])
            s_stub = bs - b1
            w_stub = as_ - ch.astar(bs)
            w_ser = float(np.polyval(c[::-1], s_stub) * s_stub)
            radii = [r for r in (2.0, 5.0, 10.0)
                     if abs(s_stub) * r < 0.6 * R]
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
            d = {"direction": int(s.b_direction),
                 "stub_offset": s_stub,
                 "agree_at_stub": abs(w_ser - w_stub)
                 / max(abs(w_stub), 1e-300),
                 "rk_secs": round(rk_time, 4), "compare": cmp_rows}
            row["dirs"].append(d)
            print(f"    dir {s.b_direction:+d}: stub offset "
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
