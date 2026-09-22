#!/usr/bin/env python3
"""Manifold jet vs native germ vs production grid stub.

THE CLAIM UNDER TEST.  The production stub comes from a centered
second-order GRID (the Hadamard graph transform's fixed point, discretized),
and it carries a curvature error the grid's own refinement cannot see:
measured on nonnearest-saddle-connection, the stable stub at saddle 4 sits
1.5e-8 from the true manifold at r = 8e-3, growing as r^2 -- a wrong
quadratic coefficient, not a truncation.

The replacement is the invariant manifold's TAYLOR SERIES, computed by the
parameterization recursion F(K(t)) = lambda t K'(t), which at order k reads

    (DF(z*) - k lambda I) K_k = -R_k,

R_k being the order-k coefficient of F(K) built from K_1..K_{k-1}.  At a
saddle this is never singular -- k lambda is positive and DF's other
eigenvalue is negative -- so every order is solvable: no resonance, no
small divisor, as many terms as wanted.  The only input is the field's
Taylor data at the saddle, which is exact because L is a polynomial.

THE CHECK.  The native GMP germ (order 10, 192 bits) is an independent
high-order reference.  If this jet lands on it to near roundoff where the
grid stub misses by 1e-8, the grid is the defect and the jet is the fix.
Distances are to the jet CURVE, found by Newton on the foot of the
perpendicular, so the jet's parameter and the germ's need not agree.

    python scripts/manifold_jet_probe.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import model as model_mod, sturm, zoo                    # noqa: E402
from spong.wall_native import NativeSectionContinuation            # noqa: E402
from stub_bias_probe import _crossings, _point, _polyline_distance  # noqa: E402


def pmul(p, q, n):
    r = np.convolve(p, q)[:n + 1]
    out = np.zeros(n + 1)
    out[:len(r)] = r
    return out


def compose(coeffs, x, n):
    """sum_j coeffs[j] x(t)^j, truncated at order n (Horner on series)."""
    res = np.zeros(n + 1)
    for c in reversed(coeffs):
        res = pmul(res, x, n)
        res[0] += c
    return res


def deriv(c):
    return [k*c[k] for k in range(1, len(c))] or [0.0]


def peval(c, x):
    return float(np.polyval(list(reversed(c)), x))


def field_series(A, B, a, b, n, sign):
    """sign * grad L along the series (a(t), b(t)), to order n.

    L = C - 2aB + a^2 A, so L_a = 2(aA - B), L_b = a^2 A' - 2a B'.
    """
    Ab, Bb = compose(A, b, n), compose(B, b, n)
    Apb, Bpb = compose(deriv(A), b, n), compose(deriv(B), b, n)
    La = 2.0*(pmul(a, Ab, n) - Bb)
    Lb = pmul(pmul(a, a, n), Apb, n) - 2.0*pmul(a, Bpb, n)
    return sign*La, sign*Lb


def grad_hess(A, B, a, b):
    Av, Bv = peval(A, b), peval(B, b)
    A1, B1 = peval(deriv(A), b), peval(deriv(B), b)
    A2, B2 = peval(deriv(deriv(A)), b), peval(deriv(deriv(B)), b)
    g = np.array([2*(a*Av - Bv), a*a*A1 - 2*a*B1])
    H = np.array([[2*Av, 2*(a*A1 - B1)],
                  [2*(a*A1 - B1), a*a*A2 - 2*a*B2]])
    return g, H


def polish(A, B, a, b, iters=8):
    """Newton on grad L = 0, from the certified isolating-interval point."""
    z = np.array([a, b], float)
    for _ in range(iters):
        g, H = grad_hess(A, B, z[0], z[1])
        z = z - np.linalg.solve(H, g)
    return z


def manifold_jet(A, B, z0, sign, n):
    """Taylor coefficients of the unstable manifold of F = sign*grad L.

    sign = -1: unstable manifold of DESCENT.  sign = +1: unstable manifold
    of ascent, i.e. the STABLE manifold of descent.
    """
    _g, H = grad_hess(A, B, z0[0], z0[1])
    DF = sign*H
    w, V = np.linalg.eig(DF)
    i = int(np.argmax(w.real))
    lam = float(w[i].real)
    v = np.real(V[:, i])
    v = v/np.hypot(*v)
    Ka = np.zeros(n + 1)
    Kb = np.zeros(n + 1)
    Ka[0], Kb[0] = z0
    Ka[1], Kb[1] = v
    for k in range(2, n + 1):
        Ka[k] = Kb[k] = 0.0
        Fa, Fb = field_series(A, B, Ka, Kb, n, sign)
        R = np.array([Fa[k], Fb[k]])
        Kk = np.linalg.solve(DF - k*lam*np.eye(2), -R)
        Ka[k], Kb[k] = Kk
    return Ka, Kb, lam


def distance_to_jet(Ka, Kb, p, T):
    """Distance from p to the jet curve over t in [-T, T]."""
    ts = np.linspace(-T, T, 20001)
    xa = np.polyval(Ka[::-1], ts)
    xb = np.polyval(Kb[::-1], ts)
    j = int(np.argmin((xa - p[0])**2 + (xb - p[1])**2))
    t = float(ts[j])
    da, db = np.polyder(Ka[::-1]), np.polyder(Kb[::-1])
    dda, ddb = np.polyder(da), np.polyder(db)
    for _ in range(40):
        ka, kb = np.polyval(Ka[::-1], t), np.polyval(Kb[::-1], t)
        k1a, k1b = np.polyval(da, t), np.polyval(db, t)
        k2a, k2b = np.polyval(dda, t), np.polyval(ddb, t)
        ga = (ka - p[0])*k1a + (kb - p[1])*k1b
        gp = k1a*k1a + k1b*k1b + (ka - p[0])*k2a + (kb - p[1])*k2b
        if gp == 0.0:
            break
        step = ga/gp
        t -= step
        if abs(step) < 1e-18:
            break
    ka, kb = np.polyval(Ka[::-1], t), np.polyval(Kb[::-1], t)
    return math.hypot(ka - p[0], kb - p[1])


def main() -> int:
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    base = zoo.get(family.base_case)
    n_mom = 2*max(len(base.f), len(base.g)) - 1
    moments = (model_mod.moments_normal01 if base.moment_dist == "normal01"
               else model_mod.moments_uniform01)(n_mom)
    m = model_mod.build(list(base.f), list(base.g), moments)
    A = [float(c) for c in m.alpha]
    B = [float(c) for c in m.beta]
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))

    source, target = 2, 4
    radii = [0.0005, 0.001, 0.002, 0.004, 0.008]
    cases = ((source, -1, "unstable of descent at saddle 2 (source)", True),
             (target, +1, "stable of descent at saddle 4 (target)", False))

    for index, sign, label, is_source in cases:
        q = e.points[index]
        z0 = polish(A, B, float(q.a), float(q.b))
        stubs = list(getattr(q, "stubs", ()) or ())
        jets = {n: manifold_jet(A, B, z0, sign, n) for n in (6, 12)}
        lam = jets[12][2]
        print(f"\n {label}: z* = ({z0[0]:.12f}, {z0[1]:.12f}), "
              f"expanding rate {lam:.6g}")
        print(f"   {'dir':>4} {'|z-s|':>10} {'germ->jet12':>12} "
              f"{'germ->jet6':>11} {'germ->stub':>11}")
        for direction in (+1, -1):
            for r in radii:
                try:
                    ctx = NativeSectionContinuation(
                        m, source_index=source, target_index=target,
                        source_direction=direction,
                        target_direction=direction,
                        launch_radius=repr(r))
                    src, tgt = _crossings(ctx.launch("1"))
                except Exception as exc:
                    print(f"   {direction:+4d} r={r:g}: {type(exc).__name__}: "
                          f"{str(exc)[:70]}")
                    continue
                gp = _point(src if is_source else tgt)
                dist = math.hypot(gp[0] - z0[0], gp[1] - z0[1])
                T = 2.5*dist
                j12 = distance_to_jet(jets[12][0], jets[12][1], gp, T)
                j6 = distance_to_jet(jets[6][0], jets[6][1], gp, T)
                stub = min(_polyline_distance(gp, s.curve) for s in stubs)
                print(f"   {direction:+4d} {dist:>10.3e} {j12:>12.3e} "
                      f"{j6:>11.3e} {stub:>11.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
