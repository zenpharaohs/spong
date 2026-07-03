"""The only integrators: the Gauss collocation pair.

SPONG_FOUNDING Part II, section 10.  IMM (implicit midpoint = 1-stage
Gauss, order 2) and IRK4-GL (2-stage Gauss, order 4).  Symmetric
(anadromic: Φ₋ₕ = Φₕ⁻¹), symplectic, A-stable, portable.  Anadromicity is
REQUIRED (manifold duality; level-set conservation), not preferred.

Dense output is the method's own collocation polynomial — the quadrature
that defines the step also defines the interpolant — and event location
root-finds on it.  Step control is the house doctrine: coarse step, halve,
halve again, Aitken-extrapolate (`richardson3`), stop when successive
extrapolants agree within the caller's tolerance (for portraits:
tol_plot = span / (Z_max · pixels)).  The reversal gap — integrate
forward, integrate back, measure the return error — is exported as a
per-span RESIDUAL certificate.

No black boxes: everything here is a few dozen lines of NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

_S3 = np.sqrt(3.0)

# 2-stage Gauss (IRK4-GL) Butcher data
_GL2_C = (0.5 - _S3 / 6.0, 0.5 + _S3 / 6.0)
_GL2_A = ((0.25, 0.25 - _S3 / 6.0),
          (0.25 + _S3 / 6.0, 0.25))

_NEWTON_TOL = 1e-13
_NEWTON_MAX = 30


def _fd_jac(F, x, y, f0):
    n = y.size
    J = np.empty((n, n))
    for k in range(n):
        h = 1e-7 * (1.0 + abs(y[k]))
        yp = y.copy()
        yp[k] += h
        J[:, k] = (np.atleast_1d(F(x, yp)) - f0) / h
    return J


def _newton_stages(F, jac, x, y, h, c, A):
    """Solve the Gauss stage equations Y_i = y + h Σ_j A_ij F(x + c_j h, Y_j).

    Returns the stage derivatives k_i = F(x + c_i h, Y_i).
    """
    s, n = len(c), y.size
    xs = [x + ci * h for ci in c]
    # warm start: explicit Euler stage guesses
    f0 = np.atleast_1d(F(x, y))
    K = np.tile(f0, (s, 1))

    for _ in range(_NEWTON_MAX):
        Y = [y + h * sum(A[i][j] * K[j] for j in range(s)) for i in range(s)]
        Fv = [np.atleast_1d(F(xs[i], Y[i])) for i in range(s)]
        R = np.concatenate([K[i] - Fv[i] for i in range(s)])
        if np.max(np.abs(R)) < _NEWTON_TOL * (1.0 + np.max(np.abs(K))):
            break
        # block Newton on K: dR_i/dK_j = δ_ij I − h A_ij J(x_j, Y_j)... with
        # J evaluated at the stage values (chain rule through Y_i).
        Js = [np.atleast_2d(jac(xs[i], Y[i])) if jac is not None
              else _fd_jac(F, xs[i], Y[i], Fv[i]) for i in range(s)]
        M = np.zeros((s * n, s * n))
        for i in range(s):
            for j in range(s):
                blk = -h * A[i][j] * (Js[i] @ np.eye(n))
                if i == j:
                    blk += np.eye(n)
                M[i * n:(i + 1) * n, j * n:(j + 1) * n] = blk
        dK = np.linalg.solve(M, -R).reshape(s, n)
        K = K + dK
    return K


@dataclass(frozen=True)
class Step:
    """One accepted step with its collocation dense output."""
    x: float
    h: float
    y0: np.ndarray
    y1: np.ndarray
    K: np.ndarray                 # stage derivatives, shape (s, n)

    def dense(self, theta):
        """Collocation polynomial at x + theta*h, theta in [0, 1].

        s = 1: u = y0 + h·θ·k1 (linear).
        s = 2: u = y0 + h[(θ²/2 − c2θ)k1/(c1−c2) + (θ²/2 − c1θ)k2/(c2−c1)].
        """
        th = np.asarray(theta, dtype=float)
        if self.K.shape[0] == 1:
            return self.y0 + self.h * np.multiply.outer(th, self.K[0])
        c1, c2 = _GL2_C
        w1 = (th**2 / 2 - c2 * th) / (c1 - c2)
        w2 = (th**2 / 2 - c1 * th) / (c2 - c1)
        return (self.y0 + self.h *
                (np.multiply.outer(w1, self.K[0])
                 + np.multiply.outer(w2, self.K[1])))


def step(F, x, y, h, method="gl4", jac=None) -> Step:
    """One anadromic Gauss step.  method: 'imm' (order 2) or 'gl4' (order 4)."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if method == "imm":
        c, A = (0.5,), ((0.5,),)
    elif method == "gl4":
        c, A = _GL2_C, _GL2_A
    else:
        raise ValueError(f"unknown method {method!r}")
    K = _newton_stages(F, jac, x, y, h, c, A)
    y1 = y + h * K.mean(axis=0) if len(c) == 2 else y + h * K[0]
    return Step(x, h, y, y1, K)


def gl4_scalar(f, j, x: float, y: float, h: float,
               tol: float = 1e-13, maxit: int = 30) -> float:
    """Tier-0 scalar GL4 step: pure floats, closed-form 2x2 stage Newton.

    Identical mathematics to step(..., method='gl4') for scalar systems;
    ~20x less interpreter overhead (no numpy objects in the loop).
    """
    c1, c2 = _GL2_C
    (a11, a12), (a21, a22) = _GL2_A
    x1, x2 = x + c1 * h, x + c2 * h
    k = f(x, y)
    K1 = K2 = k
    for _ in range(maxit):
        Y1 = y + h * (a11 * K1 + a12 * K2)
        Y2 = y + h * (a21 * K1 + a22 * K2)
        r1 = K1 - f(x1, Y1)
        r2 = K2 - f(x2, Y2)
        m_ = abs(K1)
        if abs(K2) > m_:
            m_ = abs(K2)
        if (abs(r1) if abs(r1) > abs(r2) else abs(r2)) < tol * (1.0 + m_):
            break
        J1 = j(x1, Y1)
        J2 = j(x2, Y2)
        m11 = 1.0 - h * a11 * J1
        m12 = -h * a12 * J1
        m21 = -h * a21 * J2
        m22 = 1.0 - h * a22 * J2
        det = m11 * m22 - m12 * m21
        K1 += (-m22 * r1 + m12 * r2) / det
        K2 += (m21 * r1 - m11 * r2) / det
    return y + h * 0.5 * (K1 + K2)


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[Step, ...]

    @property
    def xs(self):
        return np.array([s.x for s in self.steps] + [self.steps[-1].x
                                                     + self.steps[-1].h])

    @property
    def ys(self):
        return np.vstack([s.y0 for s in self.steps] + [self.steps[-1].y1])

    @property
    def y_end(self):
        return self.steps[-1].y1


def solve(F, x0, x1, y0, n, method="gl4", jac=None) -> Trajectory:
    """Fixed-step integration of y' = F(x, y) over [x0, x1] with n steps."""
    h = (x1 - x0) / n
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    out = []
    x = x0
    for _ in range(n):
        st = step(F, x, y, h, method=method, jac=jac)
        out.append(st)
        y = st.y1
        x += h
    return Trajectory(tuple(out))


def reversal_gap(F, x0, x1, y0, n, method="gl4", jac=None) -> float:
    """Anadromy certificate: forward n steps, backward n steps, return error.

    For a symmetric method this is Newton-tolerance + roundoff, NOT a
    method-order quantity; a non-symmetric scheme cannot pass this test.
    """
    fwd = solve(F, x0, x1, y0, n, method=method, jac=jac)
    back = solve(F, x1, x0, fwd.y_end, n, method=method, jac=jac)
    y0v = np.atleast_1d(np.asarray(y0, dtype=float))
    return float(np.max(np.abs(back.y_end - y0v)))


# --------------------------------------------------------------------- #
# richardson3 — the house extrapolation primitive (user's MATLAB original) #
# --------------------------------------------------------------------- #


def richardson3(x, x_old, x_old_old):
    """Aitken Δ² extrapolation from the latest three terms of a linearly
    converging sequence.  Elementwise; entries already converged (relative
    change below sqrt(eps)) are passed through unextrapolated — the
    denominator is cancellation-prone exactly there.
    """
    x = np.asarray(x, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    x_old_old = np.asarray(x_old_old, dtype=float)
    y = x.copy()

    num = x * x_old_old - x_old**2
    den = x + x_old_old - 2.0 * x_old

    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.abs(1.0 - x / x_old)
        r2 = np.abs(1.0 - x_old / x_old_old)
    todo = np.minimum(r1, r2) > np.sqrt(np.finfo(float).eps)
    todo &= den != 0.0
    y = np.where(todo, np.divide(num, den, out=np.zeros_like(y), where=den != 0),
                 y)
    return y


@dataclass(frozen=True)
class RichardsonResult:
    y_end: np.ndarray             # best extrapolant at x1
    err_est: float                # |last two extrapolants| disagreement
    n_steps: int                  # finest step count used
    trajectory: Trajectory        # finest trajectory (for dense output)
    converged: bool


def solve_richardson(F, x0, x1, y0, tol, method="gl4", jac=None,
                     n0=8, max_doublings=14) -> RichardsonResult:
    """House step-size doctrine: n, 2n, 4n, ... with Aitken extrapolation at
    the endpoint; stop when successive extrapolants agree within tol.

    tol is the DELIVERABLE tolerance (for portraits: tol_plot); the stop
    rule and the accuracy guarantee are the same inequality (RESIDUAL).
    """
    ns = [n0, 2 * n0, 4 * n0]
    trajs = [solve(F, x0, x1, y0, n, method=method, jac=jac) for n in ns]
    ends = [t.y_end for t in trajs]
    extrap_prev = None
    for _ in range(max_doublings):
        extrap = richardson3(ends[-1], ends[-2], ends[-3])
        if extrap_prev is not None:
            err = float(np.max(np.abs(extrap - extrap_prev)))
            if err < tol:
                return RichardsonResult(extrap, err, ns[-1], trajs[-1], True)
        extrap_prev = extrap
        ns.append(2 * ns[-1])
        trajs.append(solve(F, x0, x1, y0, ns[-1], method=method, jac=jac))
        trajs = trajs[-3:]
        ends = [t.y_end for t in trajs]
    err = float("inf") if extrap_prev is None else \
        float(np.max(np.abs(richardson3(ends[-1], ends[-2], ends[-3])
                            - extrap_prev)))
    return RichardsonResult(ends[-1], err, ns[-1], trajs[-1], False)


# --------------------------------------------------------------------- #
# Event location on the collocation polynomial                           #
# --------------------------------------------------------------------- #


def find_event(traj: Trajectory, event: Callable[[float, np.ndarray], float],
               tol_theta: float = 1e-14):
    """First zero-crossing of event(x, y) along the trajectory.

    Bisects the event function composed with each step's collocation
    polynomial.  Returns (x*, y*) or None.
    """
    for st in traj.steps:
        e0 = event(st.x, st.y0)
        e1 = event(st.x + st.h, st.y1)
        if e0 == 0.0:
            return st.x, st.y0
        if e0 * e1 > 0:
            continue
        lo, hi, elo = 0.0, 1.0, e0
        while hi - lo > tol_theta:
            mid = 0.5 * (lo + hi)
            em = event(st.x + mid * st.h, st.dense(mid))
            if em == 0.0:
                lo = hi = mid
                break
            if (em > 0) == (elo > 0):
                lo, elo = mid, em
            else:
                hi = mid
        th = 0.5 * (lo + hi)
        return st.x + th * st.h, np.atleast_1d(st.dense(th))
    return None
