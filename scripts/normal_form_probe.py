#!/usr/bin/env python3
"""Prototype: the global normal form L = u(b) + v^2, tested before migration.

With v = sqrt(A(b)) (a - a*(b)), a* = B/A, the map (a, b) -> (b, v) is a
global diffeomorphism under hypothesis (A) -- for each b, a -> v is affine
with slope sqrt(A) > 0 -- and it puts the loss in exact normal form

    L = u(b) + v^2,        u = C - B^2/A.

All model-specific difficulty moves into the pulled-back metric.  With
p = a*' - v A'/(2 A^{3/2}), the Euclidean gradient flow becomes

    b' = -u' + 2 p sqrt(A) v,      v' = -2 A (1 + p^2) v + p sqrt(A) u'.

Orbits are EXACTLY those of the original flow: this is a change of
coordinates with the metric carried along, not a change of metric (which is
what the rheostat is, and which moves orbits).

FOUR CLAIMS, each tested here, before anything migrates toward portrait.py:

  1. exactness       L(a,b) == u(b) + v^2 to roundoff at arbitrary points;
  2. case 165        directed seed 953953598 has a* running from 0 at its two
                     B-saddles (b = +-0.21) to -4.45e7 at b = 0.  The (a,b)
                     tracer exhausted 304,706 steps on the inner unstable
                     branch having covered ~3% of the way.  In (b,v) the whole
                     journey should be a short slide to the minimum;
  3. agreement       on near-slide-d2, repatriated (b,v) traces must sit on
                     the production traces to their own accuracy -- a new
                     coordinate system that disagrees on an easy case is
                     wrong, not clever;
  4. conditioning    the round trip's normal amplification should be
                     |det J| / |J t| = 1/(sqrt(A) |J t|): below 1 along the
                     slow manifold, 1 in the fast phase.  MEASURED here by
                     perturbing actual trace points, not argued.

INTEGRATOR.  Two-stage Radau IIA (order 3, L-stable).  Not Gauss-Legendre:
the v-rate 2A(1+p^2) reaches ~1e13 in case 165, and GL's amplification
tends to +1 at large h*lambda, so a stiff decaying mode would not damp.
Newton's Jacobian is by COMPLEX STEP -- exact to roundoff, no subtraction --
since the field is rational in (b, v).  Step control by step doubling.

    python scripts/normal_form_probe.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as NP

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ["SPONG_STUB_MODE"] = "jet"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import model as mm, portrait, sturm, zoo                  # noqa: E402
from qualify import directed_model                                  # noqa: E402

RADAU_A = np.array([[5/12, -1/12], [3/4, 1/4]])
RADAU_B = np.array([3/4, 1/4])


class NormalForm:
    def __init__(self, m):
        self.A = np.array([float(c) for c in m.alpha])      # ascending
        self.B = np.array([float(c) for c in m.beta])
        self.dA = NP.polyder(self.A)
        self.dB = NP.polyder(self.B)
        self.C = float(m.C)

    def _ev(self, c, x):
        return NP.polyval(x, c)

    def u(self, b):
        A, B = self._ev(self.A, b), self._ev(self.B, b)
        return self.C - B*B/A

    def L_direct(self, a, b):
        return self.C - 2*a*self._ev(self.B, b) + a*a*self._ev(self.A, b)

    def to_bv(self, a, b):
        A, B = self._ev(self.A, b), self._ev(self.B, b)
        return np.array([b, np.sqrt(A)*(a - B/A)])

    def to_ab(self, y):
        b, v = y[0], y[1]
        A, B = self._ev(self.A, b), self._ev(self.B, b)
        return np.array([B/A + v/np.sqrt(A), b])

    def field(self, y):
        b, v = y[0], y[1]
        A, A1 = self._ev(self.A, b), self._ev(self.dA, b)
        B, B1 = self._ev(self.B, b), self._ev(self.dB, b)
        sq = np.sqrt(A)
        astar1 = (B1*A - B*A1)/(A*A)
        p = astar1 - v*A1/(2*A*sq)
        up = B*(A1*B - 2*B1*A)/(A*A)
        return np.array([-up + 2*p*sq*v,
                         -2*A*(1 + p*p)*v + p*sq*up])

    def jac(self, y):
        """Complex-step Jacobian: exact to roundoff, no subtraction."""
        h = 1e-40
        J = np.empty((2, 2))
        for k in range(2):
            yc = np.asarray(y, complex).copy()
            yc[k] += 1j*h
            J[:, k] = np.imag(self.field(yc))/h
        return J

    def slope(self, b):
        """p on the backbone, and A: the two numbers the conditioning uses."""
        A, A1 = self._ev(self.A, b), self._ev(self.dA, b)
        B, B1 = self._ev(self.B, b), self._ev(self.dB, b)
        return (B1*A - B*A1)/(A*A), A


def radau_step(nf, y, h):
    K = np.array([nf.field(y), nf.field(y)])
    for _ in range(40):
        Y = [y + h*(RADAU_A[i, 0]*K[0] + RADAU_A[i, 1]*K[1]) for i in range(2)]
        F = [nf.field(Yi) for Yi in Y]
        R = np.concatenate([K[0] - F[0], K[1] - F[1]])
        Js = [nf.jac(Yi) for Yi in Y]
        M = np.eye(4)
        for i in range(2):
            for j in range(2):
                M[2*i:2*i+2, 2*j:2*j+2] -= h*RADAU_A[i, j]*Js[i]
        dK = np.linalg.solve(M, -R).reshape(2, 2)
        K = K + dK
        if not np.all(np.isfinite(K)):
            raise FloatingPointError("Newton diverged")
        if np.max(np.abs(dK)) <= 1e-15*(1 + np.max(np.abs(K))):
            break
    return y + h*(RADAU_B[0]*K[0] + RADAU_B[1]*K[1])


def trace(nf, y0, stop, rtol=1e-10, max_steps=20000):
    """Adaptive Radau IIA in time; step doubling, order 3."""
    y = np.asarray(y0, float)
    vscale = math.sqrt(max(1.0, abs(nf.C)))
    atol = np.array([1e-13, 1e-13*vscale])
    h, steps, rejected, path = 1e-12, 0, 0, [y.copy()]
    while steps < max_steps and h > 1e-300:
        try:
            y1 = radau_step(nf, y, h)
            y2 = radau_step(nf, radau_step(nf, y, h/2), h/2)
        except (FloatingPointError, np.linalg.LinAlgError):
            h *= 0.25
            rejected += 1
            continue
        if not np.all(np.isfinite(y2)):
            h *= 0.25
            rejected += 1
            continue
        scale = atol + rtol*np.maximum(np.abs(y), np.abs(y2))
        err = float(np.max(np.abs(y2 - y1)/7.0/scale))
        if err <= 1.0:
            y = y2
            steps += 1
            path.append(y.copy())
            if stop(y):
                break
        else:
            rejected += 1
        h *= min(4.0, max(0.2, 0.9*err**-0.25 if err > 0 else 4.0))
    return np.array(path), steps, rejected


def polyline_distance(P, Y):
    A = Y[:-1]
    D = Y[1:] - A
    dd = (D*D).sum(1)
    dd[dd == 0] = 1e-300
    out = []
    for p in P:
        t = np.clip(((p - A)*D).sum(1)/dd, 0, 1)
        out.append(float(np.sqrt((((A + t[:, None]*D) - p)**2).sum(1)).min()))
    return np.array(out)


def conditioning(nf, path, label, samples=6):
    """Measured normal amplification of the round trip vs |det J|/|J t|."""
    print(f"\n  conditioning along {label}:")
    print(f"    {'b':>10} {'A':>10} {'|p|':>10} {'measured':>11} "
          f"{'predicted':>11} {'tangential':>11}")
    idx = np.linspace(1, len(path) - 2, samples).astype(int)
    for k in idx:
        y = path[k]
        F = nf.field(y)
        if not np.all(np.isfinite(F)) or np.hypot(*F) == 0:
            continue
        t = F/np.hypot(*F)
        n = np.array([-t[1], t[0]])
        eps = 1e-8*(1 + np.max(np.abs(y)))
        base = nf.to_ab(y)
        Jt = (nf.to_ab(y + eps*t) - base)/eps
        d = (nf.to_ab(y + eps*n) - base)/eps
        T = Jt/np.hypot(*Jt)
        normal = abs(d[0]*T[1] - d[1]*T[0])
        tangential = abs(d @ T)
        p, A = nf.slope(y[0])
        predicted = (1/math.sqrt(A))/np.hypot(*Jt)
        print(f"    {y[0]:10.4g} {A:10.3g} {abs(p):10.3g} {normal:11.3e} "
              f"{predicted:11.3e} {tangential:11.3e}")


def claim_exactness(nf, label, rng):
    worst = 0.0
    for _ in range(2000):
        b = rng.uniform(-3, 3)
        a = rng.uniform(-10, 10)
        y = nf.to_bv(a, b)
        direct = nf.L_direct(a, b)
        normal = nf.u(b) + y[1]**2
        worst = max(worst, abs(direct - normal)/max(1.0, abs(direct)))
    print(f"  1. exactness on {label}: max rel |L - (u + v^2)| = {worst:.2e}")


def main() -> int:
    rng = random.Random(20260922)

    # ---- case 165 -------------------------------------------------------
    print("=== case 165 (directed seed 953953598)")
    m = directed_model(random.Random(953953598), 5)[0]
    nf = NormalForm(m)
    claim_exactness(nf, "case 165", rng)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    saddle = min((q for q in e.points if q.kind == "saddle"),
                 key=lambda q: abs(float(q.b) + 0.2103))
    inner = [q for q in e.points if q.kind == "min"
             and abs(float(q.b)) < abs(float(saddle.b))]
    target = min(inner, key=lambda q: abs(float(q.b)))
    b_min = float(target.b)
    L_min = nf.u(b_min)
    print(f"  saddle b={float(saddle.b):.6g}; inner minimum b={b_min:.6g}, "
          f"a*={float(target.a):.6g}, u={L_min:.6g}")
    stub = max((s for s in saddle.stubs if s.manifold == "unstable"),
               key=lambda s: float(s.curve[-1][1]))
    a0, b0 = map(float, stub.curve[-1])
    y0 = nf.to_bv(a0, b0)
    tol = 1e-10*max(1.0, abs(nf.C))
    t0 = time.perf_counter()
    path, steps, rej = trace(
        nf, y0, lambda y: nf.u(y[0]) + y[1]**2 - L_min < tol)
    dt = time.perf_counter() - t0
    end = nf.to_ab(path[-1])
    losses = np.array([nf.u(y[0]) + y[1]**2 for y in path])
    print(f"  2. inner branch in (b,v): {steps} accepted, {rej} rejected, "
          f"{dt:.1f}s")
    print(f"     from b={b0:.6g} to b={path[-1][0]:.8g} (min at {b_min:.8g}), "
          f"v={path[-1][1]:.3g}")
    print(f"     repatriated end a={end[0]:.8g} (a* of minimum "
          f"{float(target.a):.8g}); loss above minimum "
          f"{losses[-1]-L_min:.3g}")
    print(f"     loss monotone: {bool(np.all(np.diff(losses) <= 1e-12*abs(nf.C)))}"
          f"  (production: abort_max_steps at 304,706 steps, ~3% of the way)")
    conditioning(nf, path, "case 165 inner branch")

    # ---- near-slide-d2 --------------------------------------------------
    print("\n=== near-slide-d2")
    case = zoo.get("near-slide-d2")
    n = 2*max(len(case.f), len(case.g)) - 1
    m = mm.build(list(case.f), list(case.g), mm.moments_uniform01(n))
    nf = NormalForm(m)
    claim_exactness(nf, "near-slide-d2", rng)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    p = portrait.compute(m, _enumeration=e, _skip_audit=True)
    near = lambda z: min(e.points, key=lambda q: float(
        np.hypot(float(q.a) - z[0], float(q.b) - z[1])))
    shown = False
    for i, br in enumerate(p.branches):
        if br.kind != "unstable" or br.term != "capture":
            continue
        Y = np.asarray(br.Y, float)
        src, dst = near(Y[0]), near(Y[-1])
        best = None
        for s in src.stubs:
            L = len(s.curve)
            if s.manifold == "unstable" and L <= len(Y):
                gap = float(np.hypot(*(Y[L-1] - np.asarray(s.curve[-1], float))))
                if best is None or gap < best[0]:
                    best = (gap, s, L)
        if best is None or best[0] > 1e-12:
            continue
        _, s, L = best
        a0, b0 = map(float, s.curve[-1])
        L_min = nf.u(float(dst.b))
        tol = 1e-10*max(1.0, abs(nf.C))
        path, steps, rej = trace(
            nf, nf.to_bv(a0, b0),
            lambda y: nf.u(y[0]) + y[1]**2 - L_min < tol)
        AB = np.array([nf.to_ab(y) for y in path])
        sub = AB[np.linspace(0, len(AB) - 1, min(60, len(AB))).astype(int)]
        d = polyline_distance(sub, Y[L-1:])
        print(f"  3. br{i}: {steps} steps (production {len(Y)} points); "
              f"repatriated trace -> production: max {d.max():.2e}, "
              f"median {np.median(d):.2e}")
        if not shown:
            conditioning(nf, path, f"near-slide-d2 br{i}")
            shown = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
