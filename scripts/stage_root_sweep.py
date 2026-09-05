#!/usr/bin/env python3
"""Do the parasitic stage roots acquire stability, and where?

    python scripts/stage_root_sweep.py --case directed:202251424 \
        --state=-1.008882821,808.3
    python scripts/stage_root_sweep.py --case directed:99912409 \
        --state=-1.0798824489,2.471 --field potential

The plane steppers (spong_gauss2's IRK on the normalized-gradient and
potential-rate fields) have twice returned a "converged" step of exactly
|h|/9 -- on different models at different states.  With GL6 weights
(5/18, 4/9, 5/18) and unit stages, |h|*(5/18 - 4/9 + 5/18) = |h|/9 is
what the configuration K1 = K3 = -K2 realises: an alternating stage root,
a genuine solution of the stage equations that is not the flow.  Its
relatives are familiar -- implicit midpoint's false fixed point is the
2-stage member of the same family -- which suggests they are not
accidents but aliasing images of the true configuration that BECOME
ATTRACTING once h exceeds the flow's turning scale.

This sweeps h and, at each h, runs the stage Newton from
  * the true configuration   K_i = f(z)            (the production start)
  * the alternating one      K_1 = K_3 = -K_2 = f(z)
and reports, for each: whether it converges, to what configuration
(cos of stage 1 against stage 2), the realised chord ratio
|sum b_i K_i| / |h| (1.0 for a true short arc, 1/9 for the alternating
root at order 6, 0 for order 4's (1/2, 1/2) with K_1 = -K_2), the
residual, and the spectral radius of the fixed-point map's Jacobian
I - (I - h A x J)^-1 at the root, which is what "attracting" means.

REVERSAL.  Each converged root is then re-solved from -h starting at the
step's endpoint, and the sweep reports whether the backward step returns
to the original point (anadromy) for the true root and for the parasitic
one.  Any future admissibility test must be a symmetric function -- the
mirror c <-> 1-c with the palindromic weights is what time-reversibility
rests on -- so this column is the check that a candidate test does not
quietly cost us the reversal we rely on.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import charts                                # noqa: E402
from segment_corpus import context                      # noqa: E402

# Gauss-Legendre tableaux, in the same form spong_gauss2.c holds them.
S15 = np.sqrt(15.0)
GL = {
    4: (np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6]),
        np.array([[0.25, 0.25 - np.sqrt(3)/6],
                  [0.25 + np.sqrt(3)/6, 0.25]]),
        np.array([0.5, 0.5])),
    6: (np.array([0.5 - S15/10, 0.5, 0.5 + S15/10]),
        np.array([[5/36, 2/9 - S15/15, 5/36 - S15/30],
                  [5/36 + S15/24, 2/9, 5/36 - S15/24],
                  [5/36 + S15/30, 2/9 + S15/15, 5/36]]),
        np.array([5/18, 4/9, 5/18])),
    # Four-stage Gauss (order 8), the tableau _native.c carries.  Note the
    # nodes: Gauss nodes are symmetric about 1/2, so ODD s puts one node
    # EXACTLY at the midpoint -- GL6 has c2 = 1/2, GL4 and GL8 do not.
    # Under a hemstitching step in a narrow canyon the midpoint is the
    # canyon floor, where the Hessian is worst conditioned and the
    # normalized field's direction is least determined; the alternating
    # configuration K1 = K3 = -K2 realises |h|*(5/18 - 4/9 + 5/18) =
    # |h|/9, which is the step we twice saw accepted.  Whether the even-s
    # methods survive where GL6 cannot is what this sweep now tests.
    8: (np.array([0.06943184420297371, 0.33000947820757187,
                  0.66999052179242813, 0.93056815579702629]),
        np.array([
            [0.08696371128436346, -0.02660418008499879,
             0.012627462689404725, -0.003555149685795683],
            [0.18811811749986807, 0.16303628871563654,
             -0.027880428602470895, 0.006735500594538155],
            [0.16719192197418877, 0.35395300603374397,
             0.16303628871563654, -0.014190694931141142],
            [0.17748257225452260, 0.31344511474186835,
             0.35267675751627190, 0.08696371128436346]]),
        np.array([0.17392742256872693, 0.32607257743127307,
                  0.32607257743127307, 0.17392742256872693])),
}


def make_field(m, kind: str):
    """(f, J) of the plane field at z, as spong_gauss2 computes them."""
    native = m._native_kernel

    def fj(z):
        a, b = float(z[0]), float(z[1])
        g = np.array(native.gradient(a, b), dtype=float)
        h11, h12, h22 = native.hessian(a, b)
        H = np.array([[h11, h12], [h12, h22]], dtype=float)
        if kind == "normalized":
            n = float(np.hypot(*g))
            f = g/n
            # d(g/|g|) = (I - f f^T) H / |g|
            J = (np.eye(2) - np.outer(f, f)) @ H / n
        else:
            q = float(g @ g)
            f = g/q
            # d(g/|g|^2) = H/q - 2 g (H g)^T / q^2
            J = H/q - 2.0*np.outer(g, H @ g)/(q*q)
        return f, J
    return fj


def stage_newton(fj, z, h, order, K0, maxit=60, tol=1e-14):
    """Plain (undamped) stage Newton from a given start; no line search.

    Returns (K, iterations, residual, converged).  Undamped on purpose:
    this asks which roots EXIST and attract, not what production does.
    """
    c, A, _b = GL[order]
    s = len(c)
    K = K0.copy()
    for it in range(1, maxit+1):
        Y = z + h*(A @ K)
        F = np.array([fj(Y[i])[0] for i in range(s)])
        R = K - F
        r = float(np.max(np.abs(R)))
        if r < tol*max(1.0, float(np.max(np.abs(K)))):
            return K, it, r, True
        # Full 2s x 2s stage Jacobian: I - h (A kron J_i)
        M = np.eye(2*s)
        for i in range(s):
            Ji = fj(Y[i])[1]
            for j in range(s):
                M[2*i:2*i+2, 2*j:2*j+2] -= h*A[i, j]*Ji
        try:
            d = np.linalg.solve(M, R.reshape(-1))
        except np.linalg.LinAlgError:
            return K, it, r, False
        K = K - d.reshape(s, 2)
        if not np.all(np.isfinite(K)):
            return K, it, r, False
    return K, maxit, r, False


def spectral_radius(fj, z, h, order, K):
    """rho of the fixed-point iteration's Jacobian at the root.

    The stage map is K <- F(z + h A K); its Jacobian is h (A kron J), and
    the root attracts the SIMPLE iteration when rho < 1.  Newton converges
    from a neighbourhood regardless, so this measures how the root
    presents itself to a fixed-point-flavoured solve; the Newton basin is
    reported separately by the two-start experiment.
    """
    c, A, _b = GL[order]
    s = len(c)
    Y = z + h*(A @ K)
    M = np.zeros((2*s, 2*s))
    for i in range(s):
        Ji = fj(Y[i])[1]
        for j in range(s):
            M[2*i:2*i+2, 2*j:2*j+2] = h*A[i, j]*Ji
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def describe(fj, z, h, order, K, converged):
    c, A, b = GL[order]
    step = h*(b @ K)
    realised = float(np.hypot(*step))/abs(h)
    n0 = float(np.hypot(*K[0])) or 1.0
    n1 = float(np.hypot(*K[1])) or 1.0
    cos01 = float(K[0] @ K[1])/(n0*n1)
    rho = spectral_radius(fj, z, h, order, K) if converged else float("nan")
    return step, realised, cos01, rho


def reversal(fj, z, h, order, K, step):
    """Re-solve from the endpoint with -h; does it come back?"""
    z1 = z + step
    f1 = fj(z1)[0]
    K1 = np.array([f1]*len(GL[order][0]))
    Kb, _it, _r, ok = stage_newton(fj, z1, -h, order, K1)
    if not ok:
        return None
    c, A, b = GL[order]
    back = z1 - h*(b @ Kb)      # -h step: z1 + (-h) * sum b K
    return float(np.hypot(*(back - z)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="directed:202251424")
    ap.add_argument("--state", default="-1.008882821,808.3",
                    help="b,w of the state to stand at")
    ap.add_argument("--field", default="normalized",
                    choices=("normalized", "potential"))
    ap.add_argument("--order", type=int, default=6, choices=(4, 6, 8))
    ap.add_argument("--hs", default="0.01,0.03,0.1,0.3,1,2,3,10,35,100")
    ap.add_argument("--flow", type=int, default=1)
    args = ap.parse_args(argv)

    m, _e, _z = context(args.case)
    b, w = (float(s) for s in args.state.split(","))
    a = m.s_a_star(b) + w
    z = np.array([a, b])
    fj = make_field(m, args.field)
    f0, J0 = fj(z)
    nJ = float(np.max(np.abs(np.linalg.eigvals(J0))))
    print(f"{args.case} at a={a:.6e} b={b:.10g} ({args.field} field, "
          f"GL{args.order})")
    print(f"  |f|={float(np.hypot(*f0)):.6f}  rho(J)={nJ:.4e}  "
          f"turning scale 1/rho(J) = {1.0/nJ:.4e}")
    print(f"  {'h':>10} {'h*rho':>9}  start  {'conv':>5} {'it':>3} "
          f"{'|R|':>9} {'chord/|h|':>10} {'cos(K1,K2)':>11} "
          f"{'rho(hAJ)':>9}  {'reversal':>9}")

    s = len(GL[args.order][0])
    for hval in [float(x) for x in args.hs.split(",")]:
        h = args.flow*hval
        starts = {"true": np.array([f0]*s)}
        alt = np.array([f0]*s)
        # The alternating configuration: for odd s the middle stage (the
        # one sitting at c = 1/2) reverses; for even s there is no middle
        # stage and the reversal is taken on the second.
        alt[s//2 if s % 2 else 1] = -f0
        starts["alt"] = alt
        for name, K0 in starts.items():
            K, it, r, ok = stage_newton(fj, z, h, args.order, K0)
            step, realised, cos01, rho = describe(
                fj, z, h, args.order, K, ok)
            rev = reversal(fj, z, h, args.order, K, step) if ok else None
            rev_s = "-" if rev is None else f"{rev:.2e}"
            print(f"  {hval:>10.4g} {hval*nJ:>9.2e}  {name:<5}  "
                  f"{'yes' if ok else 'NO':>5} {it:>3} {r:>9.2e} "
                  f"{realised:>10.4f} {cos01:>11.4f} {rho:>9.2e}  "
                  f"{rev_s:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
