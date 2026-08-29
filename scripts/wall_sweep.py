"""Phase-1 wall sweep: pointwise complex portrait along a certified
central path to a saddle connection.

Reuses the machinery of connect_saddles.py: the same zoo case, the same
traced separation function delta (ground truth), the same central path
F(theta, t) = delta(theta) - t*delta(theta_0), t: 1 -> 0, arriving at the
wall at t = 0.  At every stage it records, alongside delta:

  dK        K(s_src) - K(s_tgt) with K = b^2 - m a^2 (far-field invariant;
            out of regime at O(1) b -- its residual at the wall calibrates
            eps, per docs/wall_theory.md)
  minImA    nearest pole pair of u to the real axis (psi-strip depth)
  minEpsN   nearest complex N-pair (ghost-pair depth)
  nu_src,   the per-critical-point invariant nu = 2A^3/(BN)' at the two
  nu_tgt    saddles of the target connection (Frobenius index = Hessian
            eigenvalue ratio)

Predictions on record: delta -> 0 at t = 0 by construction; every
pointwise algebraic column varies SMOOTHLY, with no signature at the
wall (wall-blindness theorem); dK trends without vanishing.

    python scripts/wall_sweep.py --case nonnearest-attachment --pairs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("SPONG_ENGINE", "native")

import connect_saddles as cs                              # noqa: E402
from spong import sturm, zoo                              # noqa: E402
from pole_portrait import (exact_ABCN, croots, pair_rows,  # noqa: E402
                           pmul, pderiv, peval as fpeval)


def diagnostics(m, src_b, tgt_b):
    A, B, C, N = exact_ABCN(m)
    m_deg = len(m.g) - 1
    BNp_B, BNp_N = pderiv(B), pderiv(N)

    def bn_prime(b):
        return (fpeval(BNp_B, b) * fpeval(N, b)
                + fpeval(B, b) * fpeval(BNp_N, b))

    e = sturm.enumerate_critical_points(m)

    def near(b0):
        return min(e.points, key=lambda q: abs(float(q.b) - b0))

    def nu(b0):
        q = near(b0)
        b = float(q.b)
        return 2.0 * fpeval(A, b) ** 3 / bn_prime(b)

    def K(b0):
        q = near(b0)
        return float(q.b) ** 2 - m_deg * float(q.a) ** 2

    def levels(b0):
        q = near(b0)
        return float(m.L(q.a, q.b))

    a_pairs = pair_rows(croots(A))
    n_pairs = pair_rows(croots(N))
    return {
        "dK": K(src_b) - K(tgt_b),
        "minImA": min((z.imag for z in a_pairs), default=float("nan")),
        "minEpsN": min((z.imag for z in n_pairs), default=float("nan")),
        "nu_src": nu(src_b), "nu_tgt": nu(tgt_b),
        "u_src": levels(src_b), "u_tgt": levels(tgt_b),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="nonnearest-attachment")
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--fd", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    case = zoo.get(args.case)
    f0 = [float(x) for x in case.f]
    g0 = [float(x) for x in case.g]
    theta0 = np.array(f0 + g0, dtype=float)
    nf = len(f0)

    def unpack(th):
        return list(th[:nf]), list(th[nf:])

    m0 = cs.build(f0, g0)
    e, sad = cs.skeleton(m0)
    cands = []
    for i in range(len(sad) - 1, 0, -1):
        for sd in (+1, -1):
            for td in (+1, -1):
                d = cs.separation(m0, float(sad[i].b), sd,
                                  float(sad[i - 1].b), td)
                if d is not None and math.isfinite(d):
                    cands.append((float(sad[i].b), sd,
                                  float(sad[i - 1].b), td, d))
    if not cands:
        raise SystemExit("no evaluable separations on this case")
    cands.sort(key=lambda t: abs(t[4]))
    targets = cands[:args.pairs]
    src_b, _, tgt_b, _, _ = targets[0]
    print(f"{case.name}: target connection b={src_b:+.5f} -> b={tgt_b:+.5f}")

    def delta_vec(th):
        f, g = unpack(th)
        try:
            m = cs.build(f, g)
        except Exception:                                  # noqa: BLE001
            return None, None
        out = []
        for sb, sd, tb, td, _ in targets:
            d = cs.separation(m, sb, sd, tb, td)
            if d is None or not math.isfinite(d):
                return None, None
            out.append(d)
        return np.asarray(out), m

    d0, _ = delta_vec(theta0)
    if d0 is None:
        raise SystemExit("baseline separation failed")
    th = theta0.copy()
    rows = []
    print(f"{'t':>7}{'delta':>13}{'dK':>12}{'minImA':>10}{'minEpsN':>10}"
          f"{'nu_src':>11}{'nu_tgt':>11}")
    print("-" * 76)
    for stage in range(args.steps + 1):
        t = 1.0 - stage / args.steps
        for _ in range(3):
            d, _ = delta_vec(th)
            if d is None:
                break
            F = d - t * d0
            if np.linalg.norm(F) < 1e-12:
                break
            J = np.zeros((len(targets), len(th)))
            ok = True
            for j in range(len(th)):
                tp = th.copy()
                tp[j] += args.fd
                dp, _ = delta_vec(tp)
                if dp is None:
                    ok = False
                    break
                J[:, j] = (dp - d) / args.fd
            if not ok:
                break
            step = -np.linalg.pinv(J) @ F
            lam = 1.0
            for _ in range(24):
                cand = th + lam * step
                dc, _ = delta_vec(cand)
                if dc is not None and np.linalg.norm(dc - t * d0) <= \
                        np.linalg.norm(F) * (1.0 - 1e-4 * lam):
                    th = cand
                    break
                lam *= 0.5
        d, m = delta_vec(th)
        if d is None:
            print(f"{t:>7.3f}   FAILED")
            break
        diag = diagnostics(m, src_b, tgt_b)
        rows.append({"t": t, "theta": [float(x) for x in th],
                     "delta": [float(x) for x in d], **diag})
        print(f"{t:>7.3f}{d[0]:>13.4e}{diag['dK']:>12.5g}"
              f"{diag['minImA']:>10.4g}{diag['minEpsN']:>10.4g}"
              f"{diag['nu_src']:>11.4g}{diag['nu_tgt']:>11.4g}")

    out_path = Path(args.out) if args.out else (
        REPO / "out" / f"wall_sweep-{args.case}.json")
    out_path.write_text(json.dumps(
        {"case": args.case, "targets": targets, "rows": rows},
        indent=2, default=str) + "\n")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
