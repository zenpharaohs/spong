"""Reproducible artifact for theorems.md Theorem 8.4/8.5 (minimal-quartet).

Reproduces, from the zoo case alone:
  1. the certified skeleton (4 criticals S m S m, positions and Q values);
  2. the landing-fate transition under the Lambda-rheostat
     (m2 below the transition, m1 above);
  3. the bisected transition bracket;
  4. the hug-scaling table (closest approach to S' vs bracket offset) that
     supports -- at EMPIRICAL grade only -- the saddle-connection type.

Evidence-grade tooling by design: scipy integrators (Radau default, DOP853
cross-check), the same grade as the Former-Theorem-1 refutation.  Nothing
here is a certificate; see docs/theorems.md Theorem 8 for what is claimed.

Usage:
  PYTHONPATH=src python demos/minimal_quartet_wall.py            # fates + skeleton
  PYTHONPATH=src python demos/minimal_quartet_wall.py --bisect   # + bracket
  PYTHONPATH=src python demos/minimal_quartet_wall.py --hug      # + hug table
Writes out/minimal_quartet_wall.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.integrate import solve_ivp

from spong import zoo

pv = P.polyval


def build():
    case = zoo.get("minimal-quartet")
    g = np.array(case.g)
    f = np.array(case.f)
    d = len(g) - 1
    mu = np.array([1.0 / (k + 1) for k in range(2 * d + 1)])
    A = mu * P.polymul(g, g)
    B = np.array([g[j] * sum(f[i] * mu[i + j] for i in range(len(f)))
                  for j in range(d + 1)])
    N = np.trim_zeros(P.polysub(P.polymul(P.polyder(A), B),
                                2.0 * P.polymul(P.polyder(B), A)), 'b')
    rb = np.sort(np.roots(N[::-1]).real)
    Nd = P.polyder(N)
    for _ in range(50):
        rb = rb - pv(rb, N) / pv(rb, Nd)
    return A, B, np.sort(rb)


def skeleton(A, B, rb):
    Q = lambda x: pv(x, B) ** 2 / pv(x, A)
    q = Q(rb)
    eps = 0.01 * np.diff(rb).min()
    types = ['S' if Q(b) < min(Q(b - eps), Q(b + eps)) else 'm' for b in rb]
    return q, types, float(B[-1] ** 2 / A[-1])


def branch_fate(A, B, rb, lam, method="Radau"):
    """Fate of the +b branch of the high saddle: 'm1', 'm2', or None."""
    Ad, Bd = P.polyder(A), P.polyder(B)
    b_S, b_m1, b_Sp, b_m2 = rb
    def rhs(t, z):
        a, b = z
        return (-2.0 * (a * lam * pv(b, A) - pv(b, B)),
                -(a * a * lam * pv(b, Ad) - 2.0 * a * pv(b, Bd)))
    a_s = pv(b_S, B) / (lam * pv(b_S, A))
    h12 = 2 * (a_s * lam * pv(b_S, Ad) - pv(b_S, Bd))
    H = np.array([[2 * lam * pv(b_S, A), h12],
                  [h12, a_s ** 2 * lam * pv(b_S, P.polyder(Ad))
                   - 2 * a_s * pv(b_S, P.polyder(Bd))]])
    w, V = np.linalg.eigh(H)
    v = V[:, np.argmin(w)]
    evs = []
    for bm in (b_m1, b_m2):
        am = pv(bm, B) / (lam * pv(bm, A))
        def ev(t, z, am=am, bm=bm):
            return np.hypot(z[0] - am, z[1] - bm) - 1e-8 * (1 + abs(am))
        ev.terminal, ev.direction = True, -1
        evs.append(ev)
    sc = max(1.0, abs(a_s))
    for sgn in (+1, -1):
        sol = solve_ivp(rhs, (0, 1e6), np.array([a_s, b_S]) + sgn * 1e-9 * sc * v,
                        method=method, rtol=1e-12, atol=1e-14 * sc, events=evs)
        hits = [i for i, te in enumerate(sol.t_events) if len(te)]
        if hits and sol.y[1].max() > b_S + 0.5:   # the right-going branch
            return 'm2' if hits[0] == 1 else 'm1'
    return None


def bisect(A, B, rb, lo=1.0, hi=20.0, method="Radau"):
    assert branch_fate(A, B, rb, lo, method) == 'm2'
    assert branch_fate(A, B, rb, hi, method) == 'm1'
    log = []
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        r = branch_fate(A, B, rb, mid, method)
        log.append((mid, r))
        if r == 'm2':
            lo = mid
        elif r == 'm1':
            hi = mid
        else:
            break                                  # unresolved: report as-is
        if hi - lo < 4e-15 * mid:
            break
    return lo, hi, log


def hug_table(A, B, rb, lam_star, offsets=(1e-4, 1e-7, 1e-10, 1e-13)):
    Ad, Bd = P.polyder(A), P.polyder(B)
    b_S, _, b_Sp, _ = rb
    out = []
    for off in offsets:
        lam = lam_star * (1 + off)
        def rhs(t, z):
            a, b = z
            return (-2.0 * (a * lam * pv(b, A) - pv(b, B)),
                    -(a * a * lam * pv(b, Ad) - 2.0 * a * pv(b, Bd)))
        a_s = pv(b_S, B) / (lam * pv(b_S, A))
        h12 = 2 * (a_s * lam * pv(b_S, Ad) - pv(b_S, Bd))
        H = np.array([[2 * lam * pv(b_S, A), h12],
                      [h12, a_s ** 2 * lam * pv(b_S, P.polyder(Ad))
                       - 2 * a_s * pv(b_S, P.polyder(Bd))]])
        w, V = np.linalg.eigh(H)
        v = V[:, np.argmin(w)]
        zS = (pv(b_Sp, B) / (lam * pv(b_Sp, A)), b_Sp)
        best = np.inf
        for sgn in (+1, -1):
            sol = solve_ivp(rhs, (0, 300.0), np.array([a_s, b_S]) + sgn * 1e-9 * v,
                            method="Radau", rtol=1e-12, atol=1e-14, max_step=0.5)
            best = min(best, float(np.hypot(sol.y[0] - zS[0],
                                            sol.y[1] - zS[1]).min()))
        out.append((off, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--hug", action="store_true")
    ap.add_argument("--method", default="Radau", choices=["Radau", "DOP853"])
    args = ap.parse_args()

    A, B, rb = build()
    q, types, qinf = skeleton(A, B, rb)
    report = {
        "case": "minimal-quartet",
        "criticals_b": [float(x) for x in rb],
        "types": types,
        "Q_at_criticals": [float(x) for x in q],
        "Q_infinity": qinf,
        "integrator": args.method,
    }
    print("skeleton:", list(zip(np.round(rb, 10), types, np.round(q, 4))),
          " Q_inf=%.4f" % qinf)
    assert types == ['S', 'm', 'S', 'm'], "SmSm skeleton expected"

    fates = {lam: branch_fate(A, B, rb, lam, args.method)
             for lam in (1.0, 7.6, 7.7, 20.0)}
    report["fates"] = {str(k): v for k, v in fates.items()}
    print("landing fates:", fates)
    assert fates[7.6] == 'm2' and fates[7.7] == 'm1', "transition in (7.6, 7.7)"

    if args.bisect:
        lo, hi, _ = bisect(A, B, rb, method=args.method)
        report["bracket"] = [lo, hi]
        print(f"transition bracket: [{lo!r}, {hi!r}]  width {hi - lo:.3g}")

    if args.hug:
        lam_star = report.get("bracket", [7.651823524762018])[0]
        tab = hug_table(A, B, rb, lam_star)
        report["hug_scaling"] = tab
        for off, dist in tab:
            print(f"offset {off:8.1e}: closest approach to S' = {dist:.3e}")

    os.makedirs("out", exist_ok=True)
    with open("out/minimal_quartet_wall.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote out/minimal_quartet_wall.json")


if __name__ == "__main__":
    main()
