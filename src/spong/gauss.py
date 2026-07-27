"""The only integrators: the Gauss collocation family.

SPONG_FOUNDING Part II, section 10.  IMM (implicit midpoint = 1-stage
Gauss, order 2), IRK4-GL (2-stage, order 4) and IRK6-GL (3-stage,
order 6).  Symmetric
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
_S15 = np.sqrt(15.0)

# 2-stage Gauss (IRK4-GL) Butcher data
_GL2_C = (0.5 - _S3 / 6.0, 0.5 + _S3 / 6.0)
_GL2_A = ((0.25, 0.25 - _S3 / 6.0),
          (0.25 + _S3 / 6.0, 0.25))
_GL2_B = (0.5, 0.5)

# 3-stage Gauss (IRK6-GL) Butcher data
_GL3_C = (0.5 - _S15 / 10.0, 0.5, 0.5 + _S15 / 10.0)
_GL3_A = ((5/36,               2/9 - _S15/15,  5/36 - _S15/30),
          (5/36 + _S15/24,     2/9,            5/36 - _S15/24),
          (5/36 + _S15/30,     2/9 + _S15/15,  5/36))
_GL3_B = (5/18, 4/9, 5/18)

# (c, A, b) by name — the family, in increasing order
_TABLEAU = {
    "imm": ((0.5,), ((0.5,),), (1.0,)),
    "gl4": (_GL2_C, _GL2_A, _GL2_B),
    "gl6": (_GL3_C, _GL3_A, _GL3_B),
}

# stage nodes by stage count, for the dense-output collocation polynomial
_STAGES = {len(c): c for c, _, _ in _TABLEAU.values()}

_NEWTON_TOL = 1e-13
_NEWTON_MAX = 30

# Ill-conditioning trip for the closed-form stage solve, as a Hadamard-style
# ratio |det M| / prod(row inf-norms).  Calibrated on REAL traces, not on a
# synthetic dissipative sweep -- that mistake set this to 1e-3 and aborted a
# legitimate d=17 branch (test_linear_vs_d17_horizontal_branch...) whose
# marginal region reaches 9.64e-04.  The three empirical points:
#
#   0.4159  saturation for ANY dissipative stiffness, = 1/(120*prod_i
#           max_j|a_ij|) exactly; also the minimum over 9173 stage matrices
#           sampled from real straddling-suite traces
#   9.6e-04 hardest real trace seen (d=17 at |b| ~ 10.4): cond ~ 5e3, i.e.
#           still ~1e-12 accurate -- ELEVATED, not broken; must not trip
#   2.9e-08 a genuine Pade-root singularity: cond ~ 2.7e+08
#
# The guard must separate breakdown from mere elevation, so it belongs between
# the last two, not above them.  cond ~ 4-8/ratio, so 1e-6 trips at cond ~5e6.
_STAGE_GUARD = 1e-6


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

        u(x + θh) = y0 + h · Σ_i ℓ_i(θ) k_i,  ℓ_i(θ) = ∫₀^θ L_i(τ) dτ

        with L_i the Lagrange basis on the stage nodes — the quadrature that
        defines the step also defines the interpolant.  Written for ANY stage
        count: s = 1 gives the linear form, s = 2 the quadratic one, and s = 3
        (GL6) its own cubic.  Special-casing s ≤ 2 here would silently hand a
        3-stage step the 2-stage polynomial, which uses the wrong nodes AND
        drops k₃ — and `find_event` root-finds on this.
        """
        th = np.asarray(theta, dtype=float)
        s = self.K.shape[0]
        c = _STAGES[s]
        acc = None
        for i in range(s):
            # ∏_{j≠i}(τ − c_j) / ∏_{j≠i}(c_i − c_j), integrated from 0 to θ
            roots = [c[j] for j in range(s) if j != i]
            num = np.poly(roots) if roots else np.array([1.0])
            den = np.prod([c[i] - c[j] for j in range(s) if j != i]) \
                if roots else 1.0
            integ = np.polyint(num)                    # ℓ_i, constant term 0
            wi = np.polyval(integ, th) / den
            term = np.multiply.outer(wi, self.K[i])
            acc = term if acc is None else acc + term
        return self.y0 + self.h * acc


def step(F, x, y, h, method="gl6", jac=None) -> Step:
    """One anadromic Gauss step.  method: 'imm' (2), 'gl4' (4), 'gl6' (6)."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    try:
        c, A, b = _TABLEAU[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}") from None
    K = _newton_stages(F, jac, x, y, h, c, A)
    y1 = y + h * sum(bi * K[i] for i, bi in enumerate(b))
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


def gl6_scalar(f, j, x: float, y: float, h: float,
               tol: float = 1e-13, maxit: int = 30) -> float:
    """Tier-0 scalar GL6 step: pure floats, closed-form 3x3 stage Newton.

    Identical mathematics to step(..., method='gl6') for scalar systems;
    ~11x faster than that path and still ~7.7x faster than GL4 through
    NumPy, at 1.93x the cost of gl4_scalar per step (against ~10x fewer
    steps for equal accuracy at order 6 vs 4).

    Why the closed form is safe rather than merely convenient.  The stage
    matrix is M = I − h·diag(J_i)·A with A the CONSTANT Gauss tableau, and

        det(I − zA) = Q_s(z),  the (s,s) Pade denominator of exp

    (s=3: 1 − z/2 + z²/10 − z³/120; verified to 3e−15 relative).  A-stability
    IS the statement that every root of Q_s lies in Re z > 0 — measured here
    at 4.6444 and 3.6778 ± 3.5088i — so for dissipative h·λ the matrix cannot
    be singular, and |det M| grows like |z|³/120 rather than collapsing.
    Measured cond₂(M) saturates at 10.4436, flat from z = −1e4 to −1e14: the
    conditioning is bounded INDEPENDENTLY of stiffness.  (A is not normal —
    ‖AAᵀ−AᵀA‖_F = 0.4655 — and that departure is exactly the gap between
    cond 10.44 and the eigenvalue spread 1.0945.)

    That bound is proved for the frozen Jacobian D = λI; here the J_i differ
    by O(h) across the stage points, so `tests/test_gauss_gl6.py` checks the
    varying-D case against a pivoted LU directly (measured cond ≤ 24.6, i.e.
    ~2.4x the frozen-D figure but still O(10) uniformly).

    A second, independent guarantee covers the other end: for |h·J| below the
    diagonal-dominance threshold, unpivoted LU is stable by the classical
    result.  Per-row thresholds here are ∞, 9.9476 and 1.6406, so the binding
    one is |h·J| < 1.64 — the mild regime, which is exactly where order 6
    earns its keep.  Pade/conditioning covers the stiff end; dominance covers
    the mild end.

    Iterative refinement is therefore NOT applied.  cond·u ≈ 5e−15 makes it
    convergent, but measurement against an exact rational solve shows the raw
    adjugate is already at machine precision (4.15e−16 … 1.27e−15 across |h·J|
    from 1e−3 to 1e12) and one refinement step gains only 1.1x–2.0x — nothing,
    for ~45% more work per Newton iteration.  It is the right fallback if a
    future variant (larger systems, non-scalar J) loses the bound.
    """
    c1, c2, c3 = _GL3_C
    (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = _GL3_A
    b1, b2, b3 = _GL3_B
    x1, x2, x3 = x + c1 * h, x + c2 * h, x + c3 * h
    k = f(x, y)
    K1 = K2 = K3 = k
    for _ in range(maxit):
        Y1 = y + h * (a11 * K1 + a12 * K2 + a13 * K3)
        Y2 = y + h * (a21 * K1 + a22 * K2 + a23 * K3)
        Y3 = y + h * (a31 * K1 + a32 * K2 + a33 * K3)
        r1 = K1 - f(x1, Y1)
        r2 = K2 - f(x2, Y2)
        r3 = K3 - f(x3, Y3)
        m_ = abs(K1)
        if abs(K2) > m_:
            m_ = abs(K2)
        if abs(K3) > m_:
            m_ = abs(K3)
        rm = abs(r1)
        if abs(r2) > rm:
            rm = abs(r2)
        if abs(r3) > rm:
            rm = abs(r3)
        if rm < tol * (1.0 + m_):
            break
        J1, J2, J3 = j(x1, Y1), j(x2, Y2), j(x3, Y3)
        m11 = 1.0 - h * a11 * J1
        m12 = -h * a12 * J1
        m13 = -h * a13 * J1
        m21 = -h * a21 * J2
        m22 = 1.0 - h * a22 * J2
        m23 = -h * a23 * J2
        m31 = -h * a31 * J3
        m32 = -h * a32 * J3
        m33 = 1.0 - h * a33 * J3
        # ROW-SCALE first.  The adjugate forms TRIPLE products where a 2x2
        # determinant forms double ones, so unscaled it overflows ~1e150
        # against GL4's ~1e300 — a 150-order loss of dynamic range that showed
        # up out of sample as a NaN step (abort_nonfinite) on a random
        # portrait GL4 handled.  Dividing row i and r_i by the row's inf-norm
        # leaves dK unchanged, puts every entry in [-1,1], and makes the
        # Hadamard ratio simply |det| of the scaled matrix.
        n1 = abs(m11)
        if abs(m12) > n1: n1 = abs(m12)
        if abs(m13) > n1: n1 = abs(m13)
        n2 = abs(m21)
        if abs(m22) > n2: n2 = abs(m22)
        if abs(m23) > n2: n2 = abs(m23)
        n3 = abs(m31)
        if abs(m32) > n3: n3 = abs(m32)
        if abs(m33) > n3: n3 = abs(m33)
        if n1 == 0.0 or n2 == 0.0 or n3 == 0.0:
            raise FloatingPointError("GL6 stage matrix has a zero row")
        # Row-equilibrate, then LU with PARTIAL PIVOTING.  Half the flops of an
        # adjugate (~13 mults vs ~30), the determinant is just as free (product
        # of pivots), and pivoting bounds every multiplier by 1 — which removes
        # the overflow hazard structurally rather than by scaling around it
        # (the unscaled adjugate died at |h·J| ~ 1e150 on triple products).
        # Both regimes are covered: below the diagonal-dominance threshold
        # |h·J| < 1.64 LU is stable even unpivoted, and pivoting handles above
        # it, with a 3x3 growth factor of at most 4.
        a = [[m11 / n1, m12 / n1, m13 / n1],
             [m21 / n2, m22 / n2, m23 / n2],
             [m31 / n3, m32 / n3, m33 / n3]]
        v = [r1 / n1, r2 / n2, r3 / n3]
        det = 1.0
        for col in range(3):
            p = col
            big = abs(a[col][col])
            for row in range(col + 1, 3):
                if abs(a[row][col]) > big:
                    big, p = abs(a[row][col]), row
            if p != col:
                a[col], a[p] = a[p], a[col]
                v[col], v[p] = v[p], v[col]
                det = -det
            piv = a[col][col]
            det *= piv
            if piv == 0.0:
                break
            for row in range(col + 1, 3):
                f_ = a[row][col] / piv
                for k_ in range(col, 3):
                    a[row][k_] -= f_ * a[col][k_]
                v[row] -= f_ * v[col]
        # Ill-conditioning guard (the ONLY guard needed: entries are
        # essentially exact, so small backward error IS small forward error
        # via forward <= cond * backward, and any backward-stable solve does).
        # |det| of the equilibrated matrix IS the Hadamard ratio.
        if abs(det) < _STAGE_GUARD:
            raise FloatingPointError(
                "GL6 stage matrix is ill-conditioned "
                f"(Hadamard ratio {abs(det):.2e} < {_STAGE_GUARD:.0e}); "
                "h*lambda is not dissipative, so the Pade/A-stability bound "
                "does not apply here")
        d3 = v[2] / a[2][2]
        d2 = (v[1] - a[1][2] * d3) / a[1][1]
        d1 = (v[0] - a[0][1] * d2 - a[0][2] * d3) / a[0][0]
        K1 -= d1
        K2 -= d2
        K3 -= d3
    return y + h * (b1 * K1 + b2 * K2 + b3 * K3)


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


def solve(F, x0, x1, y0, n, method="gl6", jac=None) -> Trajectory:
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


def reversal_gap(F, x0, x1, y0, n, method="gl6", jac=None) -> float:
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

    with np.errstate(divide="ignore", invalid="ignore"):
        num = x * x_old_old - x_old**2
        den = x + x_old_old - 2.0 * x_old
        r1 = np.abs(1.0 - x / x_old)
        r2 = np.abs(1.0 - x_old / x_old_old)
        todo = np.minimum(r1, r2) > np.sqrt(np.finfo(float).eps)
        todo &= den != 0.0
        y = np.where(
            todo, np.divide(num, den, out=np.zeros_like(y), where=den != 0),
            y)
    return y


@dataclass(frozen=True)
class RichardsonResult:
    y_end: np.ndarray             # best extrapolant at x1
    err_est: float                # |last two extrapolants| disagreement
    n_steps: int                  # finest step count used
    trajectory: Trajectory        # finest trajectory (for dense output)
    converged: bool


def solve_richardson(F, x0, x1, y0, tol, method="gl6", jac=None,
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
