"""Phase-1 wall sweep: pointwise complex portrait along a certified
central path to a saddle connection.

Reuses the machinery of connect_saddles.py: the same zoo case, the same
traced separation function delta (ground truth), the same central path
F(theta, t) = delta(theta) - t*delta(theta_0), t: 1 -> 0, arriving at the
wall at t = 0.  At every stage it records, alongside delta:

  dK        K(s_src) - K(s_tgt) with K = b^2 - m a^2 (far-field invariant;
            out of regime at O(1) b -- its residual at the wall calibrates
            eps, per docs/wall_theory.md)
  poleClr   exact lower clearance from the real axis to the reduced
            backbone-pole disks
  critClr   exact lower clearance to the nonreal reduced critical disks
  rho_src,  actual Hessian spectral ratios at the two saddles of the target
  rho_tgt   connection (unlike 2A/u'', these include the valley shear)

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
from spong.complex_structure import certify_backbone      # noqa: E402


def diagnostics(m, src_b, tgt_b):
    m_deg = len(m.g) - 1
    e = sturm.enumerate_critical_points(m)
    complex_certificate = certify_backbone(m)

    def near(b0):
        return min(e.points, key=lambda q: abs(float(q.b) - b0))

    def spectral_ratio(b0):
        q = near(b0)
        lm, lp = q.local.spectral.eigenvalues
        return lp/lm

    def K(b0):
        q = near(b0)
        return float(q.b) ** 2 - m_deg * float(q.a) ** 2

    def levels(b0):
        q = near(b0)
        return float(m.L(q.a, q.b))

    def nonreal_clearance(divisor):
        if not divisor.complete:
            return float("nan")
        values = [disk.real_axis_clearance() for disk in divisor.disks
                  if disk.real_axis_clearance() > 0]
        return float(min(values)) if values else float("nan")

    return {
        "dK": K(src_b) - K(tgt_b),
        "complex_status": ("validated" if complex_certificate.complete
                           else "partial"),
        "min_backbone_pole_clearance[VALIDATED]": nonreal_clearance(
            complex_certificate.denominator),
        "min_valley_pole_clearance[VALIDATED]": nonreal_clearance(
            complex_certificate.valley_denominator),
        "min_critical_complex_clearance[VALIDATED]": nonreal_clearance(
            complex_certificate.critical),
        "spectral_ratio_src[HIGH_PRECISION]": spectral_ratio(src_b),
        "spectral_ratio_tgt[HIGH_PRECISION]": spectral_ratio(tgt_b),
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
    print(f"{'t':>7}{'delta':>13}{'dK':>12}{'poleClr':>11}{'critClr':>11}"
          f"{'rho_src':>11}{'rho_tgt':>11}")
    print("-" * 77)
    for stage in range(args.steps + 1):
        t = 1.0 - stage / args.steps
        for _ in range(8):
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
            accepted = False
            for _ in range(24):
                cand = th + lam * step
                dc, _ = delta_vec(cand)
                if dc is not None and np.linalg.norm(dc - t * d0) <= \
                        np.linalg.norm(F) * (1.0 - 1e-4 * lam):
                    th = cand
                    accepted = True
                    break
                lam *= 0.5
            if not accepted:
                break
        d, m = delta_vec(th)
        if d is None:
            print(f"{t:>7.3f}   FAILED")
            break
        path_residual = float(np.linalg.norm(d-t*d0))
        if path_residual >= 1e-10:
            print(f"{t:>7.3f}   PATH UNCONVERGED "
                  f"(residual {path_residual:.3e}); not recorded")
            break
        diag = diagnostics(m, src_b, tgt_b)
        rows.append({"t": t, "theta": [float(x) for x in th],
                     "delta": [float(x) for x in d],
                     "path_residual[RESIDUAL]": path_residual, **diag})
        print(f"{t:>7.3f}{d[0]:>13.4e}{diag['dK']:>12.5g}"
              f"{diag['min_backbone_pole_clearance[VALIDATED]']:>11.4g}"
              f"{diag['min_critical_complex_clearance[VALIDATED]']:>11.4g}"
              f"{diag['spectral_ratio_src[HIGH_PRECISION]']:>11.4g}"
              f"{diag['spectral_ratio_tgt[HIGH_PRECISION]']:>11.4g}")

    out_path = Path(args.out) if args.out else (
        REPO / "out" / f"wall_sweep-{args.case}.json")
    out_path.write_text(json.dumps(
        {"case": args.case, "targets": targets, "rows": rows},
        indent=2, default=str) + "\n")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
