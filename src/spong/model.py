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

    def __post_init__(self):
        f, g, mu = self.f, self.g, self.mu
        df, dg = P.degree(f), P.degree(g)
        need = 2 * max(df, dg) + 1
        if len(mu) < need:
            raise ValueError(f"need moments mu_0..mu_{need - 1}, got {len(mu)}")

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

        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "N", n)

        # floating coefficient caches
        object.__setattr__(self, "_ca", _npc(a))
        object.__setattr__(self, "_cb", _npc(b))
        object.__setattr__(self, "_cap", _npc(P.deriv(a)))
        object.__setattr__(self, "_cbp", _npc(P.deriv(b)))
        object.__setattr__(self, "_capp", _npc(P.deriv(P.deriv(a))))
        object.__setattr__(self, "_cbpp", _npc(P.deriv(P.deriv(b))))
        object.__setattr__(self, "_cn", _npc(n))
        object.__setattr__(self, "_cnp", _npc(P.deriv(n)))

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
        try:
            from . import _native
            kernel = _native.Kernel(
                self._fa, self._fap, self._fapp,
                self._fb, self._fbp, self._fbpp,
                self._fn, self._fnp)
        except (ImportError, ValueError):
            kernel = None
        object.__setattr__(self, "_native_kernel", kernel)

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
        return self.B(b) / self.A(b)

    def a_star_p(self, b):
        A, B = self.A(b), self.B(b)
        return self.Bp(b) / A - B * self.Ap(b) / A**2

    def a_star_pp(self, b):
        A, B = self.A(b), self.B(b)
        Ap, Bp = self.Ap(b), self.Bp(b)
        return ((self.Bpp(b) * A - B * self.App(b)) / A**2
                - 2 * Ap * (Bp * A - B * Ap) / A**3)

    def L(self, a, b):
        return float(self.C) - 2 * a * self.B(b) + a**2 * self.A(b)

    def u(self, b):
        return float(self.C) - self.B(b)**2 / self.A(b)

    def u_p(self, b):
        """u' = B·N/A² (exact identity; see module docstring)."""
        return self.B(b) * self.Nval(b) / self.A(b)**2

    def u_pp(self, b):
        A, B, Nv = self.A(b), self.B(b), self.Nval(b)
        return ((self.Bp(b) * Nv + B * self.Npval(b)) / A**2
                - 2 * B * Nv * self.Ap(b) / A**3)

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
