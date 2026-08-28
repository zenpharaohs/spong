"""Exact model coefficients and the (b, w) chart.

SPONG_FOUNDING Part II, sections 1-3:

    L(a,b) = C - 2a·B(b) + a²·A(b) = u(b) + A(b)·w²,   w = a - a*(b)
    a* = B/A,   u = C - B²/A,   ∇L = (2Aw, P),
    P(b,w) = u' + A'w² - 2Aw·a*'

Critical structure (all critical points lie on the backbone):

    u' = B·N / A²  with  N = A'B - 2B'A
    ⇒ critical b-values = real roots of N  UNION  real roots of B
    H11 = 2A, H12 = -2A·a*', H22 = 2A·a*'² + u'',  det H = 2A·u''

Coefficient constructions are EXACT (rational arithmetic over the dyadic
inputs); the floating evaluators are the NumPy kernels traces use.

The loss and its gradient are linear in the raw-moment vector.  For
``D = max(deg(f), deg(g))`` and ``mu = (mu_0, ..., mu_2D)``, the exact
moment response is exposed by ``moment_loss_weights_exact`` and
``moment_jacobian_exact``; ``moment_jacobian`` is their binary64 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import _poly as P
from ._poly import Poly


def _npc(p: Poly) -> np.ndarray:
    """Ascending float coefficient array for numpy.polynomial evaluation."""
    return np.array([float(c) for c in p] or [0.0], dtype=float)


def _ev(c: np.ndarray, x):
    return np.polynomial.polynomial.polyval(x, c)


def _ordinary_rational_input(x) -> bool:
    """Whether established direct-Horner rounding is safely in range."""
    if np.isscalar(x):
        value = float(x)
        return np.isfinite(value) and abs(value) <= 32.0
    values = np.asarray(x, dtype=float)
    return bool(np.all(np.isfinite(values) & (np.abs(values) <= 32.0)))


def _rational_product(x, numerators, denominators):
    """Evaluate a product of polynomials divided by another, projectively.

    Direct far-field formulas such as ``B*N/A**2`` are badly scaled: every
    polynomial and the squared denominator can overflow even though the ratio
    is the small, finite quantity ``u' ~ C_inf/b**2``.  Once a conservative
    coefficient/degree bound puts any direct product near binary64 overflow,
    use ``t = 1/b`` and ``p(b) = b**deg(p) * reverse(p)(t)``.  Only the net
    power is formed.  Ordinary coordinates retain their established Horner
    rounding, which is also the native/Python parity contract.  Scalars remain
    scalars and arrays retain their shape.
    """
    xx = np.asarray(x, dtype=float)
    flat = xx.reshape(-1)
    out = np.empty_like(flat)

    def log_bound(coefficients):
        degree = sum(len(c) - 1 for c in coefficients)
        bound = 0.0
        for coeff in coefficients:
            largest = float(np.max(np.abs(coeff)))
            if largest == 0.0:
                return -np.inf, degree
            bound += np.log(largest) + np.log(len(coeff))
        return bound, degree

    numerator_bound, numerator_degree = log_bound(numerators)
    denominator_bound, denominator_degree = log_bound(denominators)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_b = np.log(np.maximum(np.abs(flat), 1.0))
        direct_bound = np.maximum(
            numerator_bound + numerator_degree * log_b,
            denominator_bound + denominator_degree * log_b)
    # exp(600) is about 1e260: enough headroom for Horner roundoff and for the
    # subsequent products, while staying far above every ordinary portrait
    # coordinate.  This is a routing threshold, not an acceptance tolerance.
    reciprocal = (np.abs(flat) > 1.0) & (direct_bound > 600.0)

    def product(coefficients, values, *, reverse=False):
        result = np.ones_like(values)
        for coeff in coefficients:
            result *= _ev(coeff[::-1] if reverse else coeff, values)
        return result

    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        ordinary = ~reciprocal
        if np.any(ordinary):
            values = flat[ordinary]
            out[ordinary] = (
                product(numerators, values)
                / product(denominators, values))
        if np.any(reciprocal):
            values = flat[reciprocal]
            inverse = 1.0 / values
            degree = numerator_degree - denominator_degree
            out[reciprocal] = (
                np.power(values, degree)
                * product(numerators, inverse, reverse=True)
                / product(denominators, inverse, reverse=True))

    shaped = out.reshape(xx.shape)
    return shaped.item() if xx.ndim == 0 else shaped


def horner(c: tuple, x: float) -> float:
    """Pure-float Horner — the Tier-0 scalar fast path (0.33 µs vs 1.6 µs
    through numpy's polyval on scalars; the engine loop lives here)."""
    acc = 0.0
    for v in reversed(c):
        acc = acc * x + v
    return acc


@dataclass(frozen=True)
class Model:
    """Single polynomial neuron: f target, g activation, mu moments."""

    f: Poly
    g: Poly
    mu: tuple[Fraction, ...]    # moment sequence mu_0..mu_m

    # exact derived coefficients (ascending):
    alpha: Poly = field(init=False)      # A(b) = E[g(bX)^2]
    beta: Poly = field(init=False)       # B(b) = E[f(X) g(bX)]
    C: Fraction = field(init=False)      # E[f(X)^2]
    N: Poly = field(init=False)          # A'B - 2B'A
    backbone_common: Poly = field(init=False)  # gcd(B², A)
    backbone_num: Poly = field(init=False)     # reduced B² numerator
    backbone_den: Poly = field(init=False)     # reduced A denominator
    critical_reduced: Poly = field(init=False) # numerator of u'
    moment_A_weights: tuple[Poly, ...] = field(init=False)
    moment_B_weights: tuple[Poly, ...] = field(init=False)
    moment_C_weights: tuple[Fraction, ...] = field(init=False)

    def __post_init__(self):
        f, g, mu = self.f, self.g, self.mu
        df, dg = P.degree(f), P.degree(g)
        need = 2 * max(df, dg) + 1
        if len(mu) < need:
            raise ValueError(f"need moments mu_0..mu_{need - 1}, got {len(mu)}")

        # Coefficients of each raw moment in A(b), B(b), and C.  These are
        # properties of (f, g), independent of the particular moment vector.
        # Keeping them exact makes the affine dependence on mu inspectable and
        # gives an exact oracle for the binary64 field evaluator below.
        gg = P.mul(g, g)
        ff = P.mul(f, f)
        moment_A = []
        moment_B = []
        moment_C = []
        for m in range(need):
            ca = gg[m] if m < len(gg) else Fraction(0)
            moment_A.append(P.trim((Fraction(0),) * m + (ca,)))

            bm = [Fraction(0)] * (dg + 1)
            for j in range(dg + 1):
                i = m - j
                if 0 <= i <= df:
                    bm[j] = g[j] * f[i]
            moment_B.append(P.trim(tuple(bm)))
            moment_C.append(ff[m] if m < len(ff) else Fraction(0))

        alpha = [Fraction(0)] * (2 * dg + 1)
        for i in range(dg + 1):
            for j in range(dg + 1):
                alpha[i + j] += g[i] * g[j] * mu[i + j]

        beta = [Fraction(0)] * (dg + 1)
        for j in range(dg + 1):
            beta[j] = g[j] * sum((f[i] * mu[i + j] for i in range(df + 1)),
                                 Fraction(0))

        c = sum((f[i] * f[k] * mu[i + k]
                 for i in range(df + 1) for k in range(df + 1)), Fraction(0))

        a = P.trim(tuple(alpha))
        b = P.trim(tuple(beta))
        n = P.sub(P.mul(P.deriv(a), b),
                  P.scale(P.mul(P.deriv(b), a), Fraction(2)))
        b2 = P.mul(b, b)
        common = P.gcd_poly(b2, a)
        reduced_num, rem_num = P.divmod_exact(b2, common)
        reduced_den, rem_den = P.divmod_exact(a, common)
        if rem_num or rem_den:
            raise ArithmeticError("failed to reduce exact backbone loss")
        critical_reduced = P.sub(
            P.mul(reduced_num, P.deriv(reduced_den)),
            P.mul(P.deriv(reduced_num), reduced_den))

        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "N", n)
        object.__setattr__(self, "backbone_common", common)
        object.__setattr__(self, "backbone_num", reduced_num)
        object.__setattr__(self, "backbone_den", reduced_den)
        object.__setattr__(self, "critical_reduced", critical_reduced)
        object.__setattr__(self, "moment_A_weights", tuple(moment_A))
        object.__setattr__(self, "moment_B_weights", tuple(moment_B))
        object.__setattr__(self, "moment_C_weights", tuple(moment_C))

        # floating coefficient caches
        object.__setattr__(self, "_ca", _npc(a))
        object.__setattr__(self, "_cb", _npc(b))
        object.__setattr__(self, "_cap", _npc(P.deriv(a)))
        object.__setattr__(self, "_cbp", _npc(P.deriv(b)))
        object.__setattr__(self, "_capp", _npc(P.deriv(P.deriv(a))))
        object.__setattr__(self, "_cbpp", _npc(P.deriv(P.deriv(b))))
        object.__setattr__(self, "_cn", _npc(n))
        object.__setattr__(self, "_cnp", _npc(P.deriv(n)))
        object.__setattr__(self, "_moment_A_float",
                           tuple(tuple(float(x) for x in p) or (0.0,)
                                 for p in moment_A))
        object.__setattr__(self, "_moment_B_float",
                           tuple(tuple(float(x) for x in p) or (0.0,)
                                 for p in moment_B))

        # pure-float coefficient tuples for the scalar fast path
        fl = lambda p: tuple(float(x) for x in p) or (0.0,)
        object.__setattr__(self, "_fa", fl(a))
        object.__setattr__(self, "_fb", fl(b))
        object.__setattr__(self, "_fap", fl(P.deriv(a)))
        object.__setattr__(self, "_fbp", fl(P.deriv(b)))
        object.__setattr__(self, "_fapp", fl(P.deriv(P.deriv(a))))
        object.__setattr__(self, "_fbpp", fl(P.deriv(P.deriv(b))))
        object.__setattr__(self, "_fn", fl(n))
        object.__setattr__(self, "_fnp", fl(P.deriv(n)))
        self._attach_native_kernel()

    def _attach_native_kernel(self) -> None:
        """(Re)build the binary64 evaluation memo held by the C core.

        _native.Kernel caches nothing but the ROUNDING: the eight float
        coefficient arrays A, A', A'', B, B', B'', N, N', copied once out of
        the exact Fractions so the trace loop never re-rounds them.  It owns no
        state beyond those buffers, so it is process-local by nature -- dropped
        when a Model is pickled and rebuilt on arrival, never shipped.
        """
        try:
            from . import _native
            kernel = _native.Kernel(
                self._fa, self._fap, self._fapp,
                self._fb, self._fbp, self._fbpp,
                self._fn, self._fnp)
        except (ImportError, ValueError):
            kernel = None
        object.__setattr__(self, "_native_kernel", kernel)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_native_kernel", None)
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self._attach_native_kernel()

    # ---------------- floating evaluators (NumPy kernels) ---------------- #

    def A(self, b):        return _ev(self._ca, b)
    def Ap(self, b):       return _ev(self._cap, b)
    def App(self, b):      return _ev(self._capp, b)
    def B(self, b):        return _ev(self._cb, b)
    def Bp(self, b):       return _ev(self._cbp, b)
    def Bpp(self, b):      return _ev(self._cbpp, b)
    def Nval(self, b):     return _ev(self._cn, b)
    def Npval(self, b):    return _ev(self._cnp, b)

    def a_star(self, b):
        if _ordinary_rational_input(b):
            return self.B(b) / self.A(b)
        return _rational_product(b, (self._cb,), (self._ca,))

    def a_star_p(self, b):
        if _ordinary_rational_input(b):
            A, B = self.A(b), self.B(b)
            return self.Bp(b) / A - B * self.Ap(b) / A**2
        return (_rational_product(b, (self._cbp,), (self._ca,))
                - _rational_product(
                    b, (self._cb, self._cap), (self._ca, self._ca)))

    def a_star_pp(self, b):
        if _ordinary_rational_input(b):
            A, B = self.A(b), self.B(b)
            Ap, Bp = self.Ap(b), self.Bp(b)
            return ((self.Bpp(b) * A - B * self.App(b)) / A**2
                    - 2 * Ap * (Bp * A - B * Ap) / A**3)
        return (
            _rational_product(b, (self._cbpp,), (self._ca,))
            - _rational_product(
                b, (self._cb, self._capp), (self._ca, self._ca))
            - 2 * _rational_product(
                b, (self._cap, self._cbp), (self._ca, self._ca))
            + 2 * _rational_product(
                b, (self._cap, self._cap, self._cb),
                (self._ca, self._ca, self._ca)))

    def L(self, a, b):
        return float(self.C) - 2 * a * self.B(b) + a**2 * self.A(b)

    def u(self, b):
        if _ordinary_rational_input(b):
            return float(self.C) - self.B(b)**2 / self.A(b)
        return float(self.C) - _rational_product(
            b, (self._cb, self._cb), (self._ca,))

    def u_p(self, b):
        """u' = B·N/A² (exact identity; see module docstring)."""
        if _ordinary_rational_input(b):
            return self.B(b) * self.Nval(b) / self.A(b)**2
        return _rational_product(
            b, (self._cb, self._cn), (self._ca, self._ca))

    def u_pp(self, b):
        if _ordinary_rational_input(b):
            A, B, Nv = self.A(b), self.B(b), self.Nval(b)
            return ((self.Bp(b) * Nv + B * self.Npval(b)) / A**2
                    - 2 * B * Nv * self.Ap(b) / A**3)
        return (
            _rational_product(
                b, (self._cbp, self._cn), (self._ca, self._ca))
            + _rational_product(
                b, (self._cb, self._cnp), (self._ca, self._ca))
            - 2 * _rational_product(
                b, (self._cb, self._cn, self._cap),
                (self._ca, self._ca, self._ca)))

    def w_of(self, a, b):
        return a - self.a_star(b)

    def P_of(self, b, w):
        """P(b,w) = u' + A'w² - 2Aw·a*'  (so that ḃ = -P under descent)."""
        return (self.u_p(b) + self.Ap(b) * w**2
                - 2 * self.A(b) * w * self.a_star_p(b))

    def gradL(self, a, b):
        return np.array([2 * (a * self.A(b) - self.B(b)),
                         -2 * a * self.Bp(b) + a**2 * self.Ap(b)])

    def hessL(self, a, b):
        h11 = 2 * self.A(b)
        h12 = -2 * self.Bp(b) + 2 * a * self.Ap(b)
        h22 = -2 * a * self.Bpp(b) + a**2 * self.App(b)
        return np.array([[h11, h12], [h12, h22]])

    # --------------------- exact moment response ---------------------- #

    @property
    def moment_count(self) -> int:
        """Number of raw moments on which this model can depend: ``2D+1``."""
        return len(self.moment_A_weights)

    def moment_loss_weights_exact(self, a, b) -> tuple[Fraction, ...]:
        """Exact coefficients of ``mu`` in ``L(a,b; mu)``.

        ``a`` and ``b`` are converted to their exact rational values; a
        binary64 argument therefore denotes its exact dyadic rational.
        """
        aa, bb = P.as_fraction(a), P.as_fraction(b)
        return tuple(
            self.moment_C_weights[m]
            - 2 * aa * P.eval_at(self.moment_B_weights[m], bb)
            + aa * aa * P.eval_at(self.moment_A_weights[m], bb)
            for m in range(self.moment_count))

    def moment_jacobian_exact(self, a, b) -> tuple[tuple[Fraction, ...], ...]:
        """Exact ``2 x (2D+1)`` Jacobian of ``grad L`` with respect to ``mu``."""
        aa, bb = P.as_fraction(a), P.as_fraction(b)
        row_a = []
        row_b = []
        for Am, Bm in zip(self.moment_A_weights, self.moment_B_weights):
            av = P.eval_at(Am, bb)
            bv = P.eval_at(Bm, bb)
            ap = P.eval_at(P.deriv(Am), bb)
            bp = P.eval_at(P.deriv(Bm), bb)
            row_a.append(-2 * bv + 2 * aa * av)
            row_b.append(-2 * aa * bp + aa * aa * ap)
        return tuple(row_a), tuple(row_b)

    def moment_jacobian(self, a: float, b: float) -> np.ndarray:
        """Binary64 ``2 x (2D+1)`` moment Jacobian of ``grad L``.

        The identity

        ``gradL(a,b; mu+delta) - gradL(a,b; mu) = J(a,b) @ delta``

        is algebraically exact.  This method is its fast floating evaluation;
        use :meth:`moment_jacobian_exact` for an EXACT-tier rational object.
        """
        aa, bb = float(a), float(b)
        row_a = np.empty(self.moment_count, dtype=float)
        row_b = np.empty(self.moment_count, dtype=float)
        for m, (Am, Bm) in enumerate(zip(self._moment_A_float,
                                         self._moment_B_float)):
            av = horner(Am, bb)
            bv = horner(Bm, bb)
            ap = horner(tuple(k * Am[k] for k in range(1, len(Am))), bb)
            bp = horner(tuple(k * Bm[k] for k in range(1, len(Bm))), bb)
            row_a[m] = -2.0 * bv + 2.0 * aa * av
            row_b[m] = -2.0 * aa * bp + aa * aa * ap
        return np.vstack((row_a, row_b))

    # ------------- scalar fast path (pure floats, no numpy) ------------- #

    def sA(self, b):    return horner(self._fa, b)
    def sAp(self, b):   return horner(self._fap, b)
    def sApp(self, b):  return horner(self._fapp, b)
    def sB(self, b):    return horner(self._fb, b)
    def sBp(self, b):   return horner(self._fbp, b)
    def sBpp(self, b):  return horner(self._fbpp, b)
    def sN(self, b):    return horner(self._fn, b)
    def sNp(self, b):   return horner(self._fnp, b)

    def s_a_star(self, b):
        return self.sB(b) / self.sA(b)

    def s_a_star_p(self, b):
        A = self.sA(b)
        return self.sBp(b) / A - self.sB(b) * self.sAp(b) / (A * A)

    def s_a_star_pp(self, b):
        A, B = self.sA(b), self.sB(b)
        Ap, Bp = self.sAp(b), self.sBp(b)
        return ((self.sBpp(b) * A - B * self.sApp(b)) / (A * A)
                - 2.0 * Ap * (Bp * A - B * Ap) / (A * A * A))

    def s_u_p(self, b):
        A = self.sA(b)
        return self.sB(b) * self.sN(b) / (A * A)

    def s_u_pp(self, b):
        A, B, Nv = self.sA(b), self.sB(b), self.sN(b)
        return ((self.sBp(b) * Nv + B * self.sNp(b)) / (A * A)
                - 2.0 * B * Nv * self.sAp(b) / (A * A * A))

    def level_curve(self, c, b):
        """Closed-form level curve a_±(b; c); NaN where the level is absent."""
        disc = (c - self.u(b)) / self.A(b)
        s = np.sqrt(np.where(disc >= 0, disc, np.nan))
        return self.a_star(b) - s, self.a_star(b) + s


def build(f, g, mu) -> Model:
    """Build a Model from coefficient iterables (exact over dyadic inputs)."""
    return Model(P.poly(f), P.poly(g), tuple(P.as_fraction(m) for m in mu))


def moments_uniform01(n: int) -> tuple[Fraction, ...]:
    """mu_k = 1/(k+1), k = 0..n-1 (U(0,1))."""
    return tuple(Fraction(1, k + 1) for k in range(n))


def moments_normal01(n: int) -> tuple[Fraction, ...]:
    """mu_k = (k-1)!! for even k, else 0 (standard normal)."""
    out = []
    for k in range(n):
        if k % 2:
            out.append(Fraction(0))
        else:
            v = 1
            for m in range(k - 1, 1, -2):
                v *= m
            out.append(Fraction(v))
    return tuple(out)


def batch_raw_moment_covariance(
        population_mu, moment_count: int, batch_size: int
) -> tuple[tuple[Fraction, ...], ...]:
    """Exact covariance of iid empirical raw moments.

    Returns the covariance of ``(mu_hat_0, ..., mu_hat_{q-1})`` for a batch
    of ``batch_size`` iid draws, where ``q = moment_count``.  Consequently
    ``population_mu`` must reach order ``2q-2``.  This is the with-replacement
    population formula; finite-population sampling and centered sample moments
    have different covariance laws.
    """
    if moment_count < 1:
        raise ValueError("moment_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mu = tuple(P.as_fraction(x) for x in population_mu)
    need = 2 * moment_count - 1
    if len(mu) < need:
        raise ValueError(
            f"need population moments mu_0..mu_{need - 1}, got {len(mu)}")
    if mu[0] != 1:
        raise ValueError("population moment mu_0 must equal 1")
    n = Fraction(batch_size)
    return tuple(tuple((mu[j + k] - mu[j] * mu[k]) / n
                       for k in range(moment_count))
                 for j in range(moment_count))


def gradient_noise_covariance(
        m: Model, a: float, b: float, population_mu, batch_size: int
) -> np.ndarray:
    """Binary64 ``2 x 2`` covariance of an iid batch-gradient estimate.

    The moment covariance is constructed exactly, then evaluated through the
    model's binary64 moment Jacobian.  Thus the formula is exact while this
    returned field value is a floating evaluation of it.
    """
    sigma_mu_exact = batch_raw_moment_covariance(
        population_mu, m.moment_count, batch_size)
    sigma_mu = np.array([[float(x) for x in row]
                         for row in sigma_mu_exact], dtype=float)
    J = m.moment_jacobian(a, b)
    return J @ sigma_mu @ J.T
