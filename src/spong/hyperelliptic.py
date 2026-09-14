"""The loss pencil as a hyperelliptic surface and its gradient holonomy.

For a fixed loss value ``ell``, put

    y = A(b) a - B(b),
    S_ell(b) = B(b)^2 + (ell-C) A(b).

Then ``L(a,b)=ell`` is exactly the real part of

    X_ell: y^2 = S_ell(b),       a = (B+y)/A.

The merge tree uses only the real branch points of this family.  The Smale
attaching map needs more: it is the holonomy of the gradient trajectories
between regular fibres ``X_ell``.  This module records that holonomy as an
exact algebraic vector field in ``(ell,b,y)`` coordinates.  Numerical or
interval continuation can therefore operate on the hyperelliptic surface
without repeatedly subtracting the valley graph in physical coordinates.

Static periods do not by themselves decide a connection.  An Abel coordinate
on a real component, for example ``integral db/y``, turns branch incidence
into an order problem; its variation with ``ell`` is a Gauss-Manin/holonomy
problem driven by the vector field below.  Certifying that transport is the
global step.  This module now supplies exact rational trapping tubes for that
transport, direct common-fibre rectangle separation, and same-sheet Abel-gap
exclusion.  Rectangle separation is enough for the pairwise Morse--Smale
yes/no question: a noncritical gradient trajectory crosses a regular loss
fibre only once.  Unwrapped positive-genus period transport remains a separate
obligation for global order coordinates.  A validated local invariant-cone
graph launch can now be handed directly to the rational trapping-tube engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Iterable

from . import _poly as P
from . import complex_structure, sturm


@dataclass(frozen=True)
class RationalInterval:
    """Closed rational interval; every arithmetic operation is exact.

    This deliberately small type is used for certificate construction, not
    as a general numerical interval package.  Transcendental evaluation is
    avoided: the genus-zero evaluator below bounds a rational integrand by a
    midpoint rule with an exact rational second-derivative remainder.
    """

    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lo", P.as_fraction(self.lo))
        object.__setattr__(self, "hi", P.as_fraction(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed rational interval")

    @classmethod
    def point(cls, value) -> "RationalInterval":
        q = P.as_fraction(value)
        return cls(q, q)

    @property
    def width(self) -> Fraction:
        return self.hi-self.lo

    @property
    def midpoint(self) -> Fraction:
        return (self.lo+self.hi)/2

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def hull(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(min(self.lo, other.lo),
                                max(self.hi, other.hi))

    def __neg__(self) -> "RationalInterval":
        return RationalInterval(-self.hi, -self.lo)

    def __add__(self, other) -> "RationalInterval":
        other = _interval(other)
        return RationalInterval(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __sub__(self, other) -> "RationalInterval":
        return self+(-_interval(other))

    def __rsub__(self, other) -> "RationalInterval":
        return _interval(other)-self

    def __mul__(self, other) -> "RationalInterval":
        other = _interval(other)
        products = (self.lo*other.lo, self.lo*other.hi,
                    self.hi*other.lo, self.hi*other.hi)
        return RationalInterval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "RationalInterval":
        if self.contains_zero():
            raise ZeroDivisionError("interval divisor contains zero")
        values = (1/self.lo, 1/self.hi)
        return RationalInterval(min(values), max(values))

    def __truediv__(self, other) -> "RationalInterval":
        return self*_interval(other).reciprocal()

    def __rtruediv__(self, other) -> "RationalInterval":
        return _interval(other)/self

    def square(self) -> "RationalInterval":
        if self.contains_zero():
            return RationalInterval(Fraction(0),
                                    max(self.lo*self.lo, self.hi*self.hi))
        values = (self.lo*self.lo, self.hi*self.hi)
        return RationalInterval(min(values), max(values))

    def as_dict(self) -> dict:
        return {
            "lower": float(self.lo), "upper": float(self.hi),
            "lower_exact": (self.lo.numerator, self.lo.denominator),
            "upper_exact": (self.hi.numerator, self.hi.denominator),
        }


def _interval(value) -> RationalInterval:
    return value if isinstance(value, RationalInterval) \
        else RationalInterval.point(value)


def polynomial_interval(polynomial: P.Poly,
                        x: RationalInterval) -> RationalInterval:
    """Natural exact interval extension of a rational polynomial."""
    out = RationalInterval.point(0)
    for coefficient in reversed(polynomial):
        out = out*x+coefficient
    return out


def centered_polynomial_interval(polynomial: P.Poly,
                                 x: RationalInterval) -> RationalInterval:
    """Exact Taylor-form interval extension about the box midpoint.

    This is materially sharper than uncentred Horner evaluation for the loss
    pencil near a saddle, where large polynomial terms cancel to produce a
    tiny value of ``S_level``.
    """
    x = _interval(x)
    centre = x.midpoint
    delta = x-centre
    coefficients = []
    derivative = P.trim(polynomial)
    factorial = 1
    for degree in range(len(derivative)):
        if degree:
            factorial *= degree
        coefficients.append(P.eval_at(derivative, centre)/factorial)
        derivative = P.deriv(derivative)
        if not derivative:
            break
    return polynomial_interval(tuple(coefficients), delta)


def sqrt_interval(value: RationalInterval,
                  bits: int = 128) -> RationalInterval:
    """Dyadic outward enclosure of the nonnegative square root."""
    if value.lo < 0:
        raise ValueError("square root interval reaches the negative axis")
    if bits < 1:
        raise ValueError("sqrt precision must be positive")

    def lower_sqrt(q: Fraction) -> tuple[Fraction, bool]:
        scaled_numerator = q.numerator << (2*bits)
        scaled_floor = scaled_numerator//q.denominator
        root = isqrt(scaled_floor)
        exact = (root*root*q.denominator == scaled_numerator)
        return Fraction(root, 1 << bits), exact

    lo, _ = lower_sqrt(value.lo)
    hi_floor, hi_exact = lower_sqrt(value.hi)
    hi = hi_floor if hi_exact else hi_floor+Fraction(1, 1 << bits)
    return RationalInterval(lo, hi)


def _ceil_dyadic(value: Fraction, bits: int) -> Fraction:
    """Smallest multiple of ``2^-bits`` not below a nonnegative rational."""
    if value < 0:
        raise ValueError("dyadic radius rounding requires a nonnegative value")
    scaled = value.numerator << bits
    return Fraction((scaled+value.denominator-1)//value.denominator,
                    1 << bits)


@dataclass(frozen=True)
class LiftedPoint:
    """An exact point on one fibre of the loss pencil."""

    level: Fraction
    b: Fraction
    y: Fraction
    a: Fraction

    @property
    def sheet(self) -> int:
        return (self.y > 0)-(self.y < 0)


@dataclass(frozen=True)
class HolonomyCentre:
    """A proposed centre in a regular lifted-flow chart.

    Unlike :class:`LiftedPoint`, a centre need not lie exactly on its fibre.
    This distinction matters at the local/global handoff: the validated
    launch is a rational rectangle known to contain the true fibre crossing,
    while its midpoint is normally not itself an algebraic point of the
    fibre.  The trapping proof uses the whole box, not midpoint exactness.
    """

    level: Fraction
    b: Fraction
    y: Fraction


@dataclass(frozen=True)
class LevelTransport:
    """Gradient trajectory differentiated with respect to loss level."""

    db_dlevel: Fraction
    dy_dlevel: Fraction
    loss_a: Fraction
    loss_b: Fraction
    gradient_norm_squared: Fraction


def level_polynomial(model, level) -> P.Poly:
    """Exact ``S_level = B^2 + (level-C)A`` in ascending powers of ``b``."""
    ell = P.as_fraction(level)
    return P.add(P.mul(model.beta, model.beta),
                 P.scale(model.alpha, ell-model.C))


def generic_genus(model) -> int:
    """Genus of a squarefree generic fibre from the degree of its pencil."""
    degree = max(P.degree(P.mul(model.beta, model.beta)),
                 P.degree(model.alpha))
    return max(0, (degree-1)//2)


def lift_exact(model, a, b) -> LiftedPoint:
    """Lift a rational physical point to its exact pencil fibre."""
    aq, bq = P.as_fraction(a), P.as_fraction(b)
    A = P.eval_at(model.alpha, bq)
    B = P.eval_at(model.beta, bq)
    if A == 0:
        raise ZeroDivisionError("transverse divisor vanishes at b")
    y = A*aq-B
    level = model.C-2*aq*B+aq*aq*A
    return LiftedPoint(level, bq, y, aq)


def reconstruct_a_exact(model, b, y) -> Fraction:
    """Return ``a=(B+y)/A`` on a lifted fibre."""
    bq, yq = P.as_fraction(b), P.as_fraction(y)
    A = P.eval_at(model.alpha, bq)
    if A == 0:
        raise ZeroDivisionError("transverse divisor vanishes at b")
    return (P.eval_at(model.beta, bq)+yq)/A


def curve_gradient_exact(model, b, y) -> tuple[Fraction, Fraction]:
    """Return ``(L_a,L_b)`` as rational functions on ``y^2=S_ell(b)``.

    With ``N=A'B-2B'A`` and ``a=(B+y)/A``, the nontrivial identity is

        L_b = (B+y)(N+A'y)/A^2.
    """
    bq, yq = P.as_fraction(b), P.as_fraction(y)
    A = P.eval_at(model.alpha, bq)
    if A == 0:
        raise ZeroDivisionError("transverse divisor vanishes at b")
    B = P.eval_at(model.beta, bq)
    Ap = P.eval_at(P.deriv(model.alpha), bq)
    N = P.eval_at(model.N, bq)
    return 2*yq, (B+yq)*(N+Ap*yq)/(A*A)


def level_transport_exact(model, level, b, y) -> LevelTransport:
    """Exact gradient holonomy with ``ell=L`` as independent variable.

    Away from a critical point,

        db/dell = L_b/(L_a^2+L_b^2).

    Computing ``dy/dell`` from ``y=Aa-B`` rather than by division by ``2y``
    makes this chart regular at an ordinary real branch point ``y=0``.  The
    same unparameterized trajectory is obtained from ascent or descent; only
    the direction in which ``ell`` is traversed changes.
    """
    ell, bq, yq = (P.as_fraction(level), P.as_fraction(b),
                    P.as_fraction(y))
    S = level_polynomial(model, ell)
    if yq*yq != P.eval_at(S, bq):
        raise ValueError("point is not on the requested hyperelliptic fibre")
    La, Lb = curve_gradient_exact(model, bq, yq)
    norm2 = La*La+Lb*Lb
    if norm2 == 0:
        raise ZeroDivisionError("gradient vanishes on the requested fibre")
    db = Lb/norm2
    A = P.eval_at(model.alpha, bq)
    Ap = P.eval_at(P.deriv(model.alpha), bq)
    Bp = P.eval_at(P.deriv(model.beta), bq)
    a = reconstruct_a_exact(model, bq, yq)
    dy = (2*A*yq+(Ap*a-Bp)*Lb)/norm2
    return LevelTransport(db, dy, La, Lb, norm2)


def abel_basis_values_exact(model, b, y) -> tuple[Fraction, ...]:
    """Values of the holomorphic basis ``b^k db/y``, ``0<=k<g``."""
    bq, yq = P.as_fraction(b), P.as_fraction(y)
    if yq == 0:
        raise ZeroDivisionError("Abel differential at a branch point")
    return tuple(bq**k/yq for k in range(generic_genus(model)))


def abel_level_derivative_values_exact(model, b, y
                                       ) -> tuple[Fraction, ...]:
    """Fixed-``b`` level derivatives of ``b^k db/y``.

    Since ``partial S_ell/partial ell=A``, differentiation gives the
    second-kind differentials ``-A b^k db/(2y^3)``.  Reducing these modulo
    exact differentials produces the Gauss-Manin connection matrix; keeping
    the unreduced values here makes that later reduction independently
    testable.
    """
    bq, yq = P.as_fraction(b), P.as_fraction(y)
    if yq == 0:
        raise ZeroDivisionError("Abel differential at a branch point")
    A = P.eval_at(model.alpha, bq)
    return tuple(-A*bq**k/(2*yq**3)
                 for k in range(generic_genus(model)))


# ------------------------------------------------------------------------- #
# Validated fibres and level holonomy                                       #
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FibreCertificate:
    level: Fraction
    roots: complex_structure.RootCertificate
    real_branch_points: int

    @property
    def regular(self) -> bool:
        return self.roots.complete and self.roots.squarefree

    def as_dict(self) -> dict:
        return {
            "level": float(self.level),
            "level_exact": (self.level.numerator, self.level.denominator),
            "regular[VALIDATED]": self.regular,
            "real_branch_points[EXACT]": self.real_branch_points,
            "divisor": self.roots.as_dict(),
        }


def certify_fibre(model, level) -> FibreCertificate:
    """Certify all branch points of one exact rational fibre."""
    ell = P.as_fraction(level)
    polynomial = level_polynomial(model, ell)
    roots = complex_structure.certify_polynomial_roots(polynomial)
    return FibreCertificate(ell, roots, sturm.count_roots(polynomial))


@dataclass(frozen=True)
class IntervalTransport:
    db_dlevel: RationalInterval
    dy_dlevel: RationalInterval
    loss_a: RationalInterval
    loss_b: RationalInterval
    gradient_norm_squared: RationalInterval


def level_transport_interval(model, b: RationalInterval,
                             y: RationalInterval) -> IntervalTransport:
    """Exact rational interval extension of the regular lifted flow.

    The ``(b,y)`` form is regular when ``y=0`` at an ordinary branch point.
    Refusal occurs only when natural interval evaluation cannot exclude
    ``A=0`` or a critical point; subdivision can then sharpen the box.
    """
    b, y = _interval(b), _interval(y)
    A = polynomial_interval(model.alpha, b)
    if A.contains_zero():
        raise ZeroDivisionError("transverse divisor interval contains zero")
    B = polynomial_interval(model.beta, b)
    Ap = polynomial_interval(P.deriv(model.alpha), b)
    Bp = polynomial_interval(P.deriv(model.beta), b)
    N = polynomial_interval(model.N, b)
    a = (B+y)/A
    La = 2*y
    Lb = (B+y)*(N+Ap*y)/A.square()
    norm2 = La.square()+Lb.square()
    if norm2.contains_zero():
        raise ZeroDivisionError("flow box does not exclude a critical point")
    db = Lb/norm2
    dy = (2*A*y+(Ap*a-Bp)*Lb)/norm2
    return IntervalTransport(db, dy, La, Lb, norm2)


def constrained_level_transport_interval(
        model, level: RationalInterval, b: RationalInterval,
        y: RationalInterval) -> IntervalTransport:
    """Lifted transport with ``y^2=S_level(b)`` imposed algebraically."""
    level, b, y = _interval(level), _interval(b), _interval(y)
    if y.contains_zero():
        raise ZeroDivisionError(
            "constrained y-chart transport reaches y=0")
    A = centered_polynomial_interval(model.alpha, b)
    if A.contains_zero():
        raise ZeroDivisionError("transverse divisor interval contains zero")
    B = centered_polynomial_interval(model.beta, b)
    Ap = centered_polynomial_interval(P.deriv(model.alpha), b)
    N = centered_polynomial_interval(model.N, b)
    La = 2*y
    Lb = (B+y)*(N+Ap*y)/A.square()
    norm2 = La.square()+Lb.square()
    if norm2.contains_zero():
        raise ZeroDivisionError("constrained fibre box reaches a critical point")
    db = Lb/norm2
    middle = level.midpoint
    Sb = (centered_polynomial_interval(
        P.deriv(level_polynomial(model, middle)), b)
        +(level-middle)*Ap)
    dy = (A+Sb*db)/(2*y)
    return IntervalTransport(db, dy, La, Lb, norm2)


@dataclass(frozen=True)
class SheetTransportInterval:
    """Exact interval extension after eliminating ``y`` on one sheet.

    The state is only ``b``.  At every evaluation ``y`` is reconstructed as
    ``sheet*sqrt(S_level(b))``; an independently inflated normal coordinate
    can therefore never manufacture a critical point off the loss fibre.
    """

    db_dlevel: RationalInterval
    y: RationalInterval
    loss_b: RationalInterval
    gradient_norm_squared: RationalInterval


@dataclass(frozen=True)
class FibreProjectionCertificate:
    """Validated projection of a lifted rectangle onto one exact sheet."""

    level: Fraction
    input_b: RationalInterval
    input_y: RationalInterval
    sheet: int
    projected_b: RationalInterval | None
    surviving_subboxes: int
    bisection_depth: int
    reason: str | None = None

    @property
    def validated(self) -> bool:
        return self.projected_b is not None

    def as_dict(self) -> dict:
        return {
            "status": "validated" if self.validated else "unresolved",
            "level_exact": (self.level.numerator, self.level.denominator),
            "sheet": self.sheet,
            "input_b_interval": self.input_b.as_dict(),
            "input_y_interval": self.input_y.as_dict(),
            "projected_b_interval": (
                None if self.projected_b is None
                else self.projected_b.as_dict()),
            "surviving_subboxes": self.surviving_subboxes,
            "bisection_depth": self.bisection_depth,
            "fibre_constraint[EXACT]": "y^2 = S_level(b)",
            "reason": self.reason,
        }


def sheet_transport_interval(model, level: RationalInterval,
                             b: RationalInterval, sheet: int, *,
                             sqrt_bits: int = 192
                             ) -> SheetTransportInterval:
    """Enclose ``db/dlevel`` on a strict real sheet of the loss pencil.

    ``level`` and ``b`` may both be intervals.  The identity
    ``y^2=S_level(b)`` is imposed before the gradient field is evaluated.
    Refusal at ``S=0`` is deliberate: that is the boundary of this sheet
    chart, where the regular two-coordinate ``(b,y)`` chart must take over.
    """
    if sheet not in (-1, 1):
        raise ValueError("sheet must be +1 or -1")
    level, b = _interval(level), _interval(b)
    A = centered_polynomial_interval(model.alpha, b)
    if A.contains_zero():
        raise ZeroDivisionError("transverse divisor interval contains zero")
    # Splitting the level dependence off at the midpoint avoids converting
    # an interval coefficient into a polynomial type while retaining exact
    # rational arithmetic.
    middle = level.midpoint
    S = (centered_polynomial_interval(level_polynomial(model, middle), b)
         +(level-middle)*A)
    if S.lo <= 0:
        raise ValueError("fixed-sheet box reaches a real branch point")
    root = sqrt_interval(S, bits=sqrt_bits)
    y = root if sheet > 0 else -root
    B = centered_polynomial_interval(model.beta, b)
    Ap = centered_polynomial_interval(P.deriv(model.alpha), b)
    N = centered_polynomial_interval(model.N, b)
    Lb = (B+y)*(N+Ap*y)/A.square()
    norm2 = 4*y.square()+Lb.square()
    if norm2.contains_zero():
        raise ZeroDivisionError("fixed-sheet box does not exclude a critical point")
    return SheetTransportInterval(Lb/norm2, y, Lb, norm2)


def _affine_compose(polynomial: P.Poly, lower: Fraction,
                    upper: Fraction) -> P.Poly:
    """Return ``polynomial(lower+t*(upper-lower))`` exactly."""
    affine = (lower, upper-lower)
    out = P.ZERO
    for coefficient in reversed(polynomial):
        out = P.add(P.mul(out, affine), (coefficient,))
    return out


def _p2_add(first, second):
    out = dict(first)
    for key, value in second.items():
        out[key] = out.get(key, Fraction(0))+value
    return {key: value for key, value in out.items() if value}


def _p2_scale(poly, value):
    value = P.as_fraction(value)
    return {key: value*coefficient for key, coefficient in poly.items()
            if value*coefficient}


def _p2_mul(first, second):
    out = {}
    for (i, j), x in first.items():
        for (k, l), y in second.items():
            key = i+k, j+l
            out[key] = out.get(key, Fraction(0))+x*y
    return {key: value for key, value in out.items() if value}


def _p2_compose_univariate(polynomial: P.Poly, argument):
    out = {}
    for coefficient in reversed(polynomial):
        out = _p2_add(_p2_mul(out, argument),
                      {(0, 0): coefficient})
    return out


def _p2_interval(poly, first: RationalInterval,
                 second: RationalInterval) -> RationalInterval:
    def power(value, exponent):
        out = RationalInterval.point(1)
        for _ in range(exponent):
            out *= value
        return out

    out = RationalInterval.point(0)
    for (i, j), coefficient in poly.items():
        out += coefficient*power(first, i)*power(second, j)
    return out


def _sheet_tube_domain_interval(model, level0, level1,
                                lower_b0, lower_b1,
                                upper_b0, upper_b1
                                ) -> tuple[RationalInterval,
                                           RationalInterval]:
    """Enclose ``S_level(b)`` on the correlated tube quadrilateral."""
    level0, level1 = P.as_fraction(level0), P.as_fraction(level1)
    lower_b0, lower_b1 = P.as_fraction(lower_b0), P.as_fraction(lower_b1)
    upper_b0, upper_b1 = P.as_fraction(upper_b0), P.as_fraction(upper_b1)
    # Work in centred parameters t=1/2+dt and r=1/2+dr.  For fixed t,
    # r in [0,1] sweeps exactly between the two affine lateral faces.
    one = {(0, 0): Fraction(1)}
    t = {(0, 0): Fraction(1, 2), (1, 0): Fraction(1)}
    r = {(0, 0): Fraction(1, 2), (0, 1): Fraction(1)}

    def affine(first, second):
        return _p2_add(_p2_scale(one, first),
                       _p2_scale(t, second-first))

    level = affine(level0, level1)
    lower = affine(lower_b0, lower_b1)
    upper = affine(upper_b0, upper_b1)
    b = _p2_add(lower, _p2_mul(r, _p2_add(upper, _p2_scale(lower, -1))))
    A = _p2_compose_univariate(model.alpha, b)
    B = _p2_compose_univariate(model.beta, b)
    S = _p2_add(_p2_mul(B, B),
                _p2_mul(_p2_add(level,
                                {(0, 0): -model.C}), A))
    box = RationalInterval(Fraction(-1, 2), Fraction(1, 2))
    return _p2_interval(S, box, box), _p2_interval(A, box, box)


def sheet_transport_affine_interval(
        model, level0, level1, b0, b1, sheet: int, *,
        sqrt_bits: int = 192) -> SheetTransportInterval:
    """Enclose the sheet field along one affine lateral tube face.

    Keeping ``level(t)`` and ``b(t)`` correlated is essential immediately
    below a saddle: their independent rectangle contains the nearby branch
    point even when the sloping face stays strictly inside one sheet.
    """
    if sheet not in (-1, 1):
        raise ValueError("sheet must be +1 or -1")
    ell0, ell1 = P.as_fraction(level0), P.as_fraction(level1)
    b0, b1 = P.as_fraction(b0), P.as_fraction(b1)
    unit = RationalInterval(Fraction(0), Fraction(1))
    A_poly = _affine_compose(model.alpha, b0, b1)
    B_poly = _affine_compose(model.beta, b0, b1)
    Ap_poly = _affine_compose(P.deriv(model.alpha), b0, b1)
    N_poly = _affine_compose(model.N, b0, b1)
    ell_poly = (ell0, ell1-ell0)
    S_poly = P.add(P.mul(B_poly, B_poly),
                   P.mul(P.add(ell_poly, (-model.C,)), A_poly))
    S = centered_polynomial_interval(S_poly, unit)
    if S.lo <= 0:
        raise ValueError("fixed-sheet face reaches a real branch point")
    root = sqrt_interval(S, bits=sqrt_bits)
    y = root if sheet > 0 else -root
    A = centered_polynomial_interval(A_poly, unit)
    if A.contains_zero():
        raise ZeroDivisionError("transverse divisor face contains zero")
    B = centered_polynomial_interval(B_poly, unit)
    Ap = centered_polynomial_interval(Ap_poly, unit)
    N = centered_polynomial_interval(N_poly, unit)
    Lb = (B+y)*(N+Ap*y)/A.square()
    norm2 = 4*y.square()+Lb.square()
    if norm2.contains_zero():
        raise ZeroDivisionError("fixed-sheet face does not exclude a critical point")
    return SheetTransportInterval(Lb/norm2, y, Lb, norm2)


def certify_fibre_projection(
        model, level, b_interval: RationalInterval,
        y_interval: RationalInterval, *,
        max_bisection_depth: int = 24,
        max_subboxes: int = 4096,
        sqrt_bits: int = 192
        ) -> FibreProjectionCertificate:
    """Project ``box intersect {y^2=S_level(b)}`` to a strict sheet.

    Subboxes are discarded only when exact interval evaluation proves that
    their ``S`` range misses ``y_interval^2``.  The hull of every surviving
    subbox therefore contains the projection of the true fibre intersection.
    This contracts correlated Frobenius launch rectangles without assuming a
    smaller launch or trusting a floating centre.
    """
    ell = P.as_fraction(level)
    b_interval, y_interval = _interval(b_interval), _interval(y_interval)
    sheet = 1 if y_interval.lo > 0 else -1 if y_interval.hi < 0 else 0
    if sheet == 0:
        return FibreProjectionCertificate(
            ell, b_interval, y_interval, 0, None, 0, 0,
            "launch y interval does not select one strict sheet")
    y_squared = y_interval.square()
    polynomial = level_polynomial(model, ell)
    pieces = [b_interval]
    survivors = []
    for depth in range(max_bisection_depth+1):
        survivors = []
        for piece in pieces:
            S = centered_polynomial_interval(polynomial, piece)
            if S.hi < y_squared.lo or S.lo > y_squared.hi:
                continue
            survivors.append(piece)
        if not survivors:
            return FibreProjectionCertificate(
                ell, b_interval, y_interval, sheet, None, 0, depth,
                "launch rectangle has no interval-compatible fibre point")
        hull = RationalInterval(min(piece.lo for piece in survivors),
                                max(piece.hi for piece in survivors))
        try:
            sheet_transport_interval(
                model, RationalInterval.point(ell), hull, sheet,
                sqrt_bits=sqrt_bits)
        except (ValueError, ZeroDivisionError):
            pass
        else:
            return FibreProjectionCertificate(
                ell, b_interval, y_interval, sheet, hull, len(survivors),
                depth)
        if depth == max_bisection_depth:
            break
        if 2*len(survivors) > max_subboxes:
            return FibreProjectionCertificate(
                ell, b_interval, y_interval, sheet, None, len(survivors),
                depth, "fibre projection subbox budget reached")
        pieces = []
        for piece in survivors:
            middle = piece.midpoint
            pieces.extend((RationalInterval(piece.lo, middle),
                           RationalInterval(middle, piece.hi)))
    return FibreProjectionCertificate(
        ell, b_interval, y_interval, sheet, None, len(survivors),
        max_bisection_depth,
        "fibre projection did not isolate a strict sheet interval")


@dataclass(frozen=True)
class SheetTubeKnot:
    level: Fraction
    b: Fraction
    b_radius: Fraction
    y_interval: RationalInterval

    @property
    def b_interval(self) -> RationalInterval:
        return RationalInterval(self.b-self.b_radius,
                                self.b+self.b_radius)


@dataclass(frozen=True)
class SheetTubeSlab:
    level_lo: Fraction
    level_hi: Fraction
    lower_face_margin: Fraction
    upper_face_margin: Fraction
    gradient_norm_squared_lower: Fraction


@dataclass(frozen=True)
class SheetFlowTubeCertificate:
    """Validated scalar trapping tube on ``y=sheet*sqrt(S_level(b))``."""

    knots: tuple[SheetTubeKnot, ...]
    slabs: tuple[SheetTubeSlab, ...]
    tube_validated: bool
    launch_validated: bool
    sheet: int
    reason: str | None = None
    level_direction: int = 1
    slab_bisections: int = 0
    handoff: object | None = None
    launch_projection: FibreProjectionCertificate | None = None

    @property
    def status(self) -> str:
        if not self.tube_validated:
            return "unresolved"
        if self.handoff is not None and self.handoff.status != "validated":
            return "unresolved"
        if (self.launch_projection is not None
                and not self.launch_projection.validated):
            return "unresolved"
        return "validated" if self.launch_validated else \
            "conditional_on_launch"

    def as_dict(self) -> dict:
        terminal = self.knots[-1] if self.knots else None
        return {
            "status": self.status,
            "tube[VALIDATED]": self.tube_validated,
            "launch_box[VALIDATED]": self.launch_validated,
            "coordinate": "fixed hyperelliptic sheet",
            "sheet": self.sheet,
            "level_direction": self.level_direction,
            "reason": self.reason,
            "slab_count": len(self.slabs),
            "slab_bisections": self.slab_bisections,
            "two_coordinate_handoff": (
                None if self.handoff is None else self.handoff.as_dict()),
            "fibre_projection": (
                None if self.launch_projection is None
                else self.launch_projection.as_dict()),
            "terminal_level": (None if terminal is None
                               else float(terminal.level)),
            "terminal_b_interval": (None if terminal is None
                                    else terminal.b_interval.as_dict()),
            "terminal_y_interval": (None if terminal is None
                                    else terminal.y_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if not self.slabs else float(min(
                    margin for slab in self.slabs
                    for margin in (slab.lower_face_margin,
                                   slab.upper_face_margin)))),
            "minimum_gradient_norm_squared[EXACT]": (
                None if not self.slabs else float(min(
                    slab.gradient_norm_squared_lower
                    for slab in self.slabs))),
        }


def _sheet_knot(model, point: HolonomyCentre, radius: Fraction,
                sheet: int, sqrt_bits: int) -> SheetTubeKnot:
    box = RationalInterval(point.b-radius, point.b+radius)
    field = sheet_transport_interval(
        model, RationalInterval.point(point.level), box, sheet,
        sqrt_bits=sqrt_bits)
    return SheetTubeKnot(point.level, point.b, radius, field.y)


def certify_sheet_flow_tube(
        model, centres: Iterable[LiftedPoint | HolonomyCentre], *,
        initial_b_interval: RationalInterval,
        sheet: int,
        launch_validated: bool = False,
        max_radius=None,
        max_inflations: int = 32,
        max_endpoint_bits: int = 16384,
        radius_round_bits: int = 192,
        max_slab_bisections: int = 10,
        sqrt_bits: int = 192,
        handoff: object | None = None,
        launch_projection: FibreProjectionCertificate | None = None
        ) -> SheetFlowTubeCertificate:
    """Validate scalar holonomy while preserving the exact fibre relation.

    Each lateral boundary is a piecewise-linear graph ``b_-(level)`` or
    ``b_+(level)``.  Exact interval evaluation proves the scalar vector field
    points inward there.  This is the differential (Nagumo/Gronwall) form of
    a trapping argument, with no a-posteriori quadrature and no independent
    inflation in the normal-to-fibre direction.
    """
    if sheet not in (-1, 1):
        raise ValueError("sheet must be +1 or -1")
    points = tuple(centres)
    initial = _interval(initial_b_interval)
    if len(points) < 2:
        return SheetFlowTubeCertificate(
            (), (), False, launch_validated, sheet,
            "at least two scalar centres are required")
    differences = tuple(points[k+1].level-points[k].level
                        for k in range(len(points)-1))
    direction = (differences[0] > 0)-(differences[0] < 0)
    if direction == 0 or any(direction*difference <= 0
                             for difference in differences):
        return SheetFlowTubeCertificate(
            (), (), False, launch_validated, sheet,
            "centre levels must be strictly monotone")
    if initial.width <= 0:
        raise ValueError("initial scalar interval must have positive width")
    if max_endpoint_bits < 64:
        raise ValueError("tube endpoint bit budget must be at least 64")
    if not 32 <= radius_round_bits < max_endpoint_bits:
        raise ValueError("tube radius precision must fit inside bit budget")
    radius_cap = None if max_radius is None else P.as_fraction(max_radius)
    first = HolonomyCentre(points[0].level, initial.midpoint, points[0].y)
    points = (first, *points[1:])
    radius = initial.width/2
    try:
        knots = [_sheet_knot(model, first, radius, sheet, sqrt_bits)]
    except (ValueError, ZeroDivisionError) as exc:
        return SheetFlowTubeCertificate(
            (), (), False, launch_validated, sheet, str(exc), direction)
    slabs = []
    bisections = 0
    stack = [(points[k], points[k+1], 0)
             for k in range(len(points)-2, -1, -1)]
    while stack:
        left, right, depth = stack.pop()
        h = direction*(right.level-left.level)
        slope = (right.b-left.b)/h
        next_radius = radius
        accepted = None
        failure = None
        for _ in range(max_inflations):
            state = (left.b, right.b, radius, next_radius,
                     left.level, right.level)
            if max(max(abs(q.numerator).bit_length(),
                       q.denominator.bit_length()) for q in state) \
                    > max_endpoint_bits:
                failure = "scalar tube coefficient-swell guard reached"
                break
            try:
                interior_S, interior_A = _sheet_tube_domain_interval(
                    model, left.level, right.level,
                    left.b-radius, right.b-next_radius,
                    left.b+radius, right.b+next_radius)
                if interior_S.lo <= 0:
                    raise ValueError(
                        "fixed-sheet tube reaches a real branch point")
                if interior_A.contains_zero():
                    raise ZeroDivisionError(
                        "transverse divisor tube contains zero")
                lower = sheet_transport_affine_interval(
                    model, left.level, right.level,
                    left.b-radius, right.b-next_radius, sheet,
                    sqrt_bits=sqrt_bits)
                upper = sheet_transport_affine_interval(
                    model, left.level, right.level,
                    left.b+radius, right.b+next_radius, sheet,
                    sqrt_bits=sqrt_bits)
                lower_velocity = direction*lower.db_dlevel
                upper_velocity = direction*upper.db_dlevel
                lower_slope = slope-(next_radius-radius)/h
                upper_slope = slope+(next_radius-radius)/h
                lower_margin = lower_velocity.lo-lower_slope
                upper_margin = upper_slope-upper_velocity.hi
                growth = max(slope-lower_velocity.lo,
                             upper_velocity.hi-slope, Fraction(0))
                required = max(
                    next_radius,
                    _ceil_dyadic(radius+h*growth*Fraction(65, 64),
                                 radius_round_bits))
            except (ValueError, ZeroDivisionError) as exc:
                failure = str(exc)
                break
            if lower_margin >= 0 and upper_margin >= 0:
                accepted = (lower_margin, upper_margin,
                            min(4*interior_S.lo,
                                lower.gradient_norm_squared.lo,
                                upper.gradient_norm_squared.lo))
                break
            if required == next_radius:
                failure = "scalar inward face inequalities did not close"
                break
            next_radius = required
            if radius_cap is not None and next_radius > radius_cap:
                failure = "scalar tube radius cap exceeded"
                break
        if accepted is None:
            if depth < max_slab_bisections:
                middle = HolonomyCentre(
                    (left.level+right.level)/2,
                    (left.b+right.b)/2,
                    (left.y+right.y)/2)
                bisections += 1
                stack.append((middle, right, depth+1))
                stack.append((left, middle, depth+1))
                continue
            return SheetFlowTubeCertificate(
                tuple(knots), tuple(slabs), False, launch_validated, sheet,
                failure or "scalar tube inflation did not close", direction,
                bisections)
        slabs.append(SheetTubeSlab(
            min(left.level, right.level), max(left.level, right.level),
            accepted[0], accepted[1], accepted[2]))
        radius = next_radius
        try:
            knots.append(_sheet_knot(model, right, radius, sheet, sqrt_bits))
        except (ValueError, ZeroDivisionError) as exc:
            return SheetFlowTubeCertificate(
                tuple(knots), tuple(slabs), False, launch_validated, sheet,
                str(exc), direction, bisections)
    return SheetFlowTubeCertificate(
        tuple(knots), tuple(slabs), True, launch_validated, sheet,
        level_direction=direction, slab_bisections=bisections,
        handoff=handoff, launch_projection=launch_projection)


def _level_from_y_b_interval(model, y, b: RationalInterval
                             ) -> RationalInterval:
    y, b = _interval(y), _interval(b)
    A = centered_polynomial_interval(model.alpha, b)
    if A.contains_zero():
        raise ZeroDivisionError("transverse divisor interval contains zero")
    B = centered_polynomial_interval(model.beta, b)
    return model.C+(y.square()-B.square())/A


@dataclass(frozen=True)
class BParameterTransportInterval:
    """Interval field after eliminating loss and using ``b`` as time."""

    dy_db: RationalInterval
    level: RationalInterval
    loss_b: RationalInterval
    gradient_norm_squared: RationalInterval


def b_parameter_transport_affine_interval(
        model, b0, b1, y0, y1) -> BParameterTransportInterval:
    """Enclose ``dy/db`` along an affine face in the ``(b,y)`` graph chart."""
    b0, b1 = P.as_fraction(b0), P.as_fraction(b1)
    y0, y1 = P.as_fraction(y0), P.as_fraction(y1)
    unit = RationalInterval(Fraction(0), Fraction(1))
    y_poly = (y0, y1-y0)
    A_poly = _affine_compose(model.alpha, b0, b1)
    B_poly = _affine_compose(model.beta, b0, b1)
    Ap_poly = _affine_compose(P.deriv(model.alpha), b0, b1)
    Bp_poly = _affine_compose(P.deriv(model.beta), b0, b1)
    N_poly = _affine_compose(model.N, b0, b1)
    By_poly = P.add(B_poly, y_poly)
    Q_poly = P.mul(By_poly, P.add(N_poly, P.mul(Ap_poly, y_poly)))
    A2_poly = P.mul(A_poly, A_poly)
    A4_poly = P.mul(A2_poly, A2_poly)
    R_poly = P.add(P.mul(Ap_poly, By_poly),
                   P.scale(P.mul(A_poly, Bp_poly), Fraction(-1)))
    D_poly = P.add(P.scale(P.mul(A4_poly, y_poly), Fraction(2)),
                   P.mul(R_poly, Q_poly))
    denominator_poly = P.mul(A_poly, Q_poly)
    norm_numerator_poly = P.add(
        P.scale(P.mul(P.mul(y_poly, y_poly), A4_poly), Fraction(4)),
        P.mul(Q_poly, Q_poly))
    level_numerator_poly = P.add(
        P.scale(A_poly, model.C),
        P.add(P.mul(y_poly, y_poly),
              P.scale(P.mul(B_poly, B_poly), Fraction(-1))))

    A = centered_polynomial_interval(A_poly, unit)
    if A.contains_zero():
        raise ZeroDivisionError("b-parameter transverse divisor contains zero")
    denominator = centered_polynomial_interval(denominator_poly, unit)
    if denominator.contains_zero():
        raise ZeroDivisionError(
            "b is not a regular parameter on the affine face")
    norm_numerator = centered_polynomial_interval(
        norm_numerator_poly, unit)
    if norm_numerator.contains_zero():
        raise ZeroDivisionError("b-parameter face reaches a critical point")
    Q = centered_polynomial_interval(Q_poly, unit)
    return BParameterTransportInterval(
        centered_polynomial_interval(D_poly, unit)/denominator,
        centered_polynomial_interval(level_numerator_poly, unit)/A,
        Q/centered_polynomial_interval(A2_poly, unit),
        norm_numerator/centered_polynomial_interval(A4_poly, unit))


def _b_parameter_regularity_interval(
        model, b: RationalInterval,
        y: RationalInterval) -> tuple[RationalInterval, RationalInterval]:
    """Prove that the ``y(b)`` vector field is regular on a full box."""
    b, y = _interval(b), _interval(y)
    A = centered_polynomial_interval(model.alpha, b)
    if A.contains_zero():
        raise ZeroDivisionError("b-parameter transverse divisor contains zero")
    B = centered_polynomial_interval(model.beta, b)
    Ap = centered_polynomial_interval(P.deriv(model.alpha), b)
    N = centered_polynomial_interval(model.N, b)
    Q = (B+y)*(N+Ap*y)
    if Q.contains_zero():
        raise ZeroDivisionError("b is not a regular parameter inside the tube")
    loss_b = Q/A.square()
    norm2 = 4*y.square()+loss_b.square()
    if norm2.contains_zero():
        raise ZeroDivisionError("b-parameter tube reaches a critical point")
    return loss_b, norm2


@dataclass(frozen=True)
class BParameterKnot:
    b: Fraction
    y: Fraction
    y_radius: Fraction
    level_interval: RationalInterval

    @property
    def y_interval(self) -> RationalInterval:
        return RationalInterval(self.y-self.y_radius,
                                self.y+self.y_radius)


@dataclass(frozen=True)
class BParameterSlab:
    b_lo: Fraction
    b_hi: Fraction
    lower_face_margin: Fraction
    upper_face_margin: Fraction
    gradient_norm_squared_lower: Fraction


@dataclass(frozen=True)
class BParameterFlowTubeCertificate:
    """Validated scalar tube in the complementary ``y=y(b)`` chart."""

    knots: tuple[BParameterKnot, ...]
    slabs: tuple[BParameterSlab, ...]
    tube_validated: bool
    launch_validated: bool
    reason: str | None = None
    b_direction: int = 1
    slab_bisections: int = 0
    handoff: object | None = None

    @property
    def status(self) -> str:
        if not self.tube_validated:
            return "unresolved"
        if (self.handoff is not None
                and not getattr(self.handoff, "validated", False)):
            return "unresolved"
        return "validated" if self.launch_validated else \
            "conditional_on_launch"

    def as_dict(self) -> dict:
        terminal = self.knots[-1] if self.knots else None
        return {
            "status": self.status,
            "tube[VALIDATED]": self.tube_validated,
            "launch_box[VALIDATED]": self.launch_validated,
            "coordinate": "fibre-constrained y(b)",
            "reason": self.reason,
            "b_direction": self.b_direction,
            "slab_count": len(self.slabs),
            "slab_bisections": self.slab_bisections,
            "local_handoff": (None if self.handoff is None else
                              self.handoff.as_dict()),
            "terminal_b": None if terminal is None else float(terminal.b),
            "terminal_y_interval": (None if terminal is None else
                                    terminal.y_interval.as_dict()),
            "terminal_level_interval": (None if terminal is None else
                                        terminal.level_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if not self.slabs else float(min(
                    margin for slab in self.slabs for margin in
                    (slab.lower_face_margin, slab.upper_face_margin)))),
            "minimum_gradient_norm_squared[EXACT]": (
                None if not self.slabs else float(min(
                    slab.gradient_norm_squared_lower
                    for slab in self.slabs))),
        }


def certify_b_parameter_flow_tube(
        model, centres: Iterable[LiftedPoint | HolonomyCentre], *,
        initial_y_interval: RationalInterval,
        launch_validated: bool = False,
        max_radius=None,
        max_inflations: int = 32,
        max_endpoint_bits: int = 16384,
        radius_round_bits: int = 192,
        max_slab_bisections: int = 10,
        handoff: object | None = None
        ) -> BParameterFlowTubeCertificate:
    """Validate a scalar trapping tube with ``b`` as independent variable."""
    points = tuple(centres)
    initial = _interval(initial_y_interval)
    if len(points) < 2:
        return BParameterFlowTubeCertificate(
            (), (), False, launch_validated,
            "at least two b-parameter centres are required")
    differences = tuple(points[k+1].b-points[k].b
                        for k in range(len(points)-1))
    direction = (differences[0] > 0)-(differences[0] < 0)
    if direction == 0 or any(direction*difference <= 0
                             for difference in differences):
        return BParameterFlowTubeCertificate(
            (), (), False, launch_validated,
            "centre b values must be strictly monotone")
    if initial.width <= 0:
        raise ValueError("initial b-parameter interval must have positive width")
    if max_endpoint_bits < 64:
        raise ValueError("tube endpoint bit budget must be at least 64")
    if not 32 <= radius_round_bits < max_endpoint_bits:
        raise ValueError("tube radius precision must fit inside bit budget")
    radius_cap = None if max_radius is None else P.as_fraction(max_radius)
    first = HolonomyCentre(points[0].level, points[0].b, initial.midpoint)
    points = (first, *points[1:])
    radius = initial.width/2
    try:
        level_interval = _level_from_y_b_interval(
            model, initial, first.b)
    except (ValueError, ZeroDivisionError) as exc:
        return BParameterFlowTubeCertificate(
            (), (), False, launch_validated, str(exc), direction)
    knots = [BParameterKnot(first.b, first.y, radius, level_interval)]
    slabs = []
    bisections = 0
    stack = [(points[k], points[k+1], 0)
             for k in range(len(points)-2, -1, -1)]
    while stack:
        left, right, depth = stack.pop()
        h = direction*(right.b-left.b)
        slope = (right.y-left.y)/h
        next_radius = radius
        accepted = None
        failure = None
        for _ in range(max_inflations):
            state = (left.y, right.y, radius, next_radius,
                     left.b, right.b)
            if max(max(abs(q.numerator).bit_length(),
                       q.denominator.bit_length()) for q in state) \
                    > max_endpoint_bits:
                failure = "b-parameter tube coefficient-swell guard reached"
                break
            try:
                interior_b = RationalInterval(
                    min(left.b, right.b), max(left.b, right.b))
                interior_y = RationalInterval(
                    min(left.y-radius, right.y-next_radius),
                    max(left.y+radius, right.y+next_radius))
                _interior_loss_b, interior_norm2 = \
                    _b_parameter_regularity_interval(
                        model, interior_b, interior_y)
                lower = b_parameter_transport_affine_interval(
                    model, left.b, right.b,
                    left.y-radius, right.y-next_radius)
                upper = b_parameter_transport_affine_interval(
                    model, left.b, right.b,
                    left.y+radius, right.y+next_radius)
                lower_velocity = direction*lower.dy_db
                upper_velocity = direction*upper.dy_db
                lower_slope = slope-(next_radius-radius)/h
                upper_slope = slope+(next_radius-radius)/h
                lower_margin = lower_velocity.lo-lower_slope
                upper_margin = upper_slope-upper_velocity.hi
                growth = max(slope-lower_velocity.lo,
                             upper_velocity.hi-slope, Fraction(0))
                required = max(
                    next_radius,
                    _ceil_dyadic(radius+h*growth*Fraction(65, 64),
                                 radius_round_bits))
            except (ValueError, ZeroDivisionError) as exc:
                failure = str(exc)
                break
            if lower_margin >= 0 and upper_margin >= 0:
                accepted = (lower_margin, upper_margin,
                            min(interior_norm2.lo,
                                lower.gradient_norm_squared.lo,
                                upper.gradient_norm_squared.lo))
                break
            if required == next_radius:
                failure = "b-parameter inward face inequalities did not close"
                break
            next_radius = required
            if radius_cap is not None and next_radius > radius_cap:
                failure = "b-parameter tube radius cap exceeded"
                break
        if accepted is None:
            if depth < max_slab_bisections:
                middle = HolonomyCentre(
                    (left.level+right.level)/2,
                    (left.b+right.b)/2,
                    (left.y+right.y)/2)
                bisections += 1
                stack.append((middle, right, depth+1))
                stack.append((left, middle, depth+1))
                continue
            return BParameterFlowTubeCertificate(
                tuple(knots), tuple(slabs), False, launch_validated,
                failure or "b-parameter tube did not close", direction,
                bisections)
        radius = next_radius
        endpoint_y = RationalInterval(right.y-radius, right.y+radius)
        try:
            level_interval = _level_from_y_b_interval(
                model, endpoint_y, right.b)
        except (ValueError, ZeroDivisionError) as exc:
            return BParameterFlowTubeCertificate(
                tuple(knots), tuple(slabs), False, launch_validated,
                str(exc), direction, bisections)
        slabs.append(BParameterSlab(
            min(left.b, right.b), max(left.b, right.b),
            accepted[0], accepted[1], accepted[2]))
        knots.append(BParameterKnot(
            right.b, right.y, radius, level_interval))
    return BParameterFlowTubeCertificate(
        tuple(knots), tuple(slabs), True, launch_validated,
        b_direction=direction, slab_bisections=bisections,
        handoff=handoff)


@dataclass(frozen=True)
class BParameterLevelProjectionCertificate:
    """Projection of a validated ``y(b)`` tube onto one exact loss fibre."""

    handoff: BParameterFlowTubeCertificate
    level: Fraction
    b_interval: RationalInterval | None
    y_interval: RationalInterval | None
    candidate_subboxes: int
    bisection_depth: int
    reason: str | None = None

    @property
    def status(self) -> str:
        if (self.handoff.status != "validated" or self.b_interval is None
                or self.y_interval is None):
            return "unresolved"
        return "validated"

    @property
    def knots(self) -> tuple[SheetTubeKnot, ...]:
        if self.b_interval is None or self.y_interval is None:
            return ()
        return (SheetTubeKnot(
            self.level, self.b_interval.midpoint,
            self.b_interval.width/2, self.y_interval),)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "coordinate": "y(b) handoff projected to exact loss fibre",
            "target_level": float(self.level),
            "target_level_exact": (
                self.level.numerator, self.level.denominator),
            "terminal_b_interval": (None if self.b_interval is None else
                                    self.b_interval.as_dict()),
            "terminal_y_interval": (None if self.y_interval is None else
                                    self.y_interval.as_dict()),
            "candidate_subboxes": self.candidate_subboxes,
            "bisection_depth": self.bisection_depth,
            "reason": self.reason,
            "handoff": self.handoff.as_dict(),
        }


def project_b_parameter_tube_to_level(
        model, tube: BParameterFlowTubeCertificate, target_level, *,
        max_bisection_depth: int = 24,
        max_subboxes: int = 4096,
        sqrt_bits: int = 192
        ) -> BParameterLevelProjectionCertificate:
    """Intersect a validated ``y(b)`` tube with an exact regular loss fibre."""
    target = P.as_fraction(target_level)

    def refusal(reason, count=0, depth=0):
        return BParameterLevelProjectionCertificate(
            tube, target, None, None, count, depth, reason)

    if tube.status != "validated" or len(tube.knots) < 2:
        return refusal("validated b-parameter handoff is unavailable")
    selected = None
    for left, right in zip(tube.knots, tube.knots[1:]):
        if left.level_interval.lo > target > right.level_interval.hi:
            selected = left, right
            break
        if left.level_interval.hi < target < right.level_interval.lo:
            selected = left, right
            break
    if selected is None:
        return refusal(
            "no b-parameter slab strictly brackets the target loss")
    left, right = selected

    def slab_box(parameter: RationalInterval
                 ) -> tuple[RationalInterval, RationalInterval]:
        b = (RationalInterval.point(left.b)
             +(right.b-left.b)*parameter)
        lower0, lower1 = (left.y-left.y_radius,
                          right.y-right.y_radius)
        upper0, upper1 = (left.y+left.y_radius,
                          right.y+right.y_radius)
        lower = RationalInterval.point(lower0)+(lower1-lower0)*parameter
        upper = RationalInterval.point(upper0)+(upper1-upper0)*parameter
        return b, lower.hull(upper)

    pieces = [RationalInterval(Fraction(0), Fraction(1))]
    survivors = []
    for depth in range(max_bisection_depth+1):
        survivors = []
        for piece in pieces:
            b, y = slab_box(piece)
            try:
                level = _level_from_y_b_interval(model, y, b)
            except (ValueError, ZeroDivisionError):
                continue
            if level.lo <= target <= level.hi:
                survivors.append(piece)
        if not survivors:
            return refusal(
                "target fibre has no interval-compatible handoff point",
                0, depth)
        if depth == max_bisection_depth:
            break
        if 2*len(survivors) > max_subboxes:
            return refusal(
                "target-fibre projection subbox budget reached",
                len(survivors), depth)
        pieces = []
        for piece in survivors:
            middle = piece.midpoint
            pieces.extend((RationalInterval(piece.lo, middle),
                           RationalInterval(middle, piece.hi)))
    boxes = [slab_box(piece) for piece in survivors]
    b_hull = RationalInterval(
        min(box[0].lo for box in boxes),
        max(box[0].hi for box in boxes))
    y_hull = RationalInterval(
        min(box[1].lo for box in boxes),
        max(box[1].hi for box in boxes))
    sheet = 1 if y_hull.lo > 0 else -1 if y_hull.hi < 0 else 0
    if sheet == 0:
        return refusal(
            "target-fibre handoff does not select one strict sheet",
            len(survivors), max_bisection_depth)
    try:
        exact_y = sheet_transport_interval(
            model, RationalInterval.point(target), b_hull, sheet,
            sqrt_bits=sqrt_bits).y
    except (ValueError, ZeroDivisionError) as exc:
        return refusal(
            "target-fibre handoff is not regular: "+str(exc),
            len(survivors), max_bisection_depth)
    if exact_y.hi < y_hull.lo or exact_y.lo > y_hull.hi:
        return refusal(
            "target-fibre sheet misses the surviving handoff boxes",
            len(survivors), max_bisection_depth)
    return BParameterLevelProjectionCertificate(
        tube, target, b_hull, exact_y, len(survivors),
        max_bisection_depth)


@dataclass(frozen=True)
class NativeBParameterHandoffCertificate:
    """Native GMP proof of a ``y(b)`` tube through one target fibre.

    The centreline is proposal data.  The backend independently validates
    the inward faces, full-slab regularity, and target projection.  This
    object certifies the global handoff conditional on its supplied initial
    interval; whether that interval contains the intended invariant germ is
    the separate local-launch obligation.
    """

    level: Fraction
    b_interval: RationalInterval | None
    y_interval: RationalInterval | None
    backend_status: int
    backend_status_name: str
    reason: str | None
    b_direction: int
    tube_validated: bool
    projection_validated: bool
    minimum_face_margin: Fraction | None
    minimum_gradient_norm_squared: Fraction | None
    work: dict

    @property
    def status(self) -> str:
        return "validated" if (
            self.backend_status == 0 and self.tube_validated
            and self.projection_validated and self.b_interval is not None
            and self.y_interval is not None) else "unresolved"

    @property
    def knots(self) -> tuple[SheetTubeKnot, ...]:
        if self.b_interval is None or self.y_interval is None:
            return ()
        return (SheetTubeKnot(
            self.level, self.b_interval.midpoint,
            self.b_interval.width/2, self.y_interval),)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "engine": "native GMP rational y(b) handoff",
            "target_level_exact": (
                self.level.numerator, self.level.denominator),
            "terminal_b_interval": (
                None if self.b_interval is None
                else self.b_interval.as_dict()),
            "terminal_y_interval": (
                None if self.y_interval is None
                else self.y_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if self.minimum_face_margin is None
                else float(self.minimum_face_margin)),
            "minimum_gradient_norm_squared[EXACT]": (
                None if self.minimum_gradient_norm_squared is None
                else float(self.minimum_gradient_norm_squared)),
            "backend_status": self.backend_status_name,
            "reason": self.reason,
            "b_direction": self.b_direction,
            "work": self.work,
        }


def certify_b_parameter_handoff_native(
        model, centres: Iterable[LiftedPoint | HolonomyCentre], *,
        initial_y_interval: RationalInterval,
        target_level,
        max_radius=None,
        max_inflations: int = 32,
        max_endpoint_bits: int = 16384,
        radius_round_bits: int = 192,
        max_slab_bisections: int = 10,
        max_projection_bisections: int = 24,
        max_projection_subboxes: int = 4096
        ) -> NativeBParameterHandoffCertificate:
    """Ask the native backend to prove a fixed-b handoff and projection."""
    target = P.as_fraction(target_level)
    points = tuple(centres)
    try:
        from . import _native
        result = _native.b_parameter_handoff(
            model.alpha or (Fraction(0),),
            model.beta or (Fraction(0),),
            model.N or (Fraction(0),), model.C,
            tuple((point.level, point.b, point.y) for point in points),
            (initial_y_interval.lo, initial_y_interval.hi), target,
            max_radius=max_radius,
            max_inflations=max_inflations,
            max_slab_bisections=max_slab_bisections,
            max_projection_bisections=max_projection_bisections,
            max_projection_subboxes=max_projection_subboxes,
            max_rational_bits=max_endpoint_bits,
            radius_round_bits=radius_round_bits,
            verify_n_identity=True)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        return NativeBParameterHandoffCertificate(
            target, None, None, -1, "unavailable",
            "native b-parameter handoff unavailable: "+str(exc),
            0, False, False, None, None, {})

    def rational(value):
        return None if value is None else Fraction(*value)

    def interval(value):
        return None if value is None else RationalInterval(
            rational(value[0]), rational(value[1]))

    status = int(result["status"])
    reason = None if status == 0 else (
        f"{result['status_name']}: {result['reason_name']}")
    return NativeBParameterHandoffCertificate(
        target, interval(result["target_b"]), interval(result["target_y"]),
        status, result["status_name"], reason,
        int(result["b_direction"]), bool(result["tube_validated"]),
        bool(result["projection_validated"]),
        rational(result["minimum_face_margin"]),
        rational(result["minimum_gradient_norm_squared"]),
        dict(result["work"]))


@dataclass(frozen=True)
class NativeSheetFlowTubeCertificate:
    """Native GMP proof of an ordinary fixed-sheet ``b(level)`` tube."""

    level: Fraction
    b_interval: RationalInterval | None
    y_interval: RationalInterval | None
    backend_status: int
    backend_status_name: str
    reason: str | None
    level_direction: int
    sheet: int
    tube_validated: bool
    projection_validated: bool
    minimum_face_margin: Fraction | None
    minimum_gradient_norm_squared: Fraction | None
    work: dict
    handoff: object | None = None

    @property
    def status(self) -> str:
        handoff_valid = (self.handoff is None
                         or self.handoff.status == "validated")
        return "validated" if (handoff_valid
            and self.backend_status == 0 and self.tube_validated
            and self.projection_validated and self.b_interval is not None
            and self.y_interval is not None) else "unresolved"

    @property
    def knots(self) -> tuple[SheetTubeKnot, ...]:
        if self.b_interval is None or self.y_interval is None:
            return ()
        return (SheetTubeKnot(
            self.level, self.b_interval.midpoint,
            self.b_interval.width/2, self.y_interval),)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "engine": "native GMP rational b(level) sheet tube",
            "terminal_level": float(self.level),
            "terminal_level_exact": (
                self.level.numerator, self.level.denominator),
            "sheet": self.sheet,
            "level_direction": self.level_direction,
            "terminal_b_interval": (
                None if self.b_interval is None
                else self.b_interval.as_dict()),
            "terminal_y_interval": (
                None if self.y_interval is None
                else self.y_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if self.minimum_face_margin is None
                else float(self.minimum_face_margin)),
            "minimum_gradient_norm_squared[EXACT]": (
                None if self.minimum_gradient_norm_squared is None
                else float(self.minimum_gradient_norm_squared)),
            "backend_status": self.backend_status_name,
            "reason": self.reason,
            "work": self.work,
            "two_coordinate_handoff": (
                None if self.handoff is None else self.handoff.as_dict()),
        }


def certify_sheet_flow_tube_native(
        model, centres: Iterable[LiftedPoint | HolonomyCentre], *,
        initial_b_interval: RationalInterval,
        initial_y_interval: RationalInterval,
        sheet: int,
        max_radius=None,
        max_inflations: int = 32,
        max_endpoint_bits: int = 16384,
        radius_round_bits: int = 192,
        max_slab_bisections: int = 10,
        max_projection_bisections: int = 24,
        max_projection_subboxes: int = 4096,
        sqrt_bits: int = 192,
        handoff: object | None = None
        ) -> NativeSheetFlowTubeCertificate:
    """Ask the native backend to project and prove a fixed-sheet tube."""
    points = tuple(centres)
    terminal_level = points[-1].level if points else Fraction(0)
    try:
        from . import _native
        result = _native.sheet_flow_tube(
            model.alpha or (Fraction(0),),
            model.beta or (Fraction(0),),
            model.N or (Fraction(0),), model.C,
            tuple((point.level, point.b, point.y) for point in points),
            (initial_b_interval.lo, initial_b_interval.hi),
            (initial_y_interval.lo, initial_y_interval.hi), int(sheet),
            max_radius=max_radius,
            max_inflations=max_inflations,
            max_slab_bisections=max_slab_bisections,
            max_projection_bisections=max_projection_bisections,
            max_projection_subboxes=max_projection_subboxes,
            max_rational_bits=max_endpoint_bits,
            radius_round_bits=radius_round_bits,
            sqrt_bits=sqrt_bits,
            verify_n_identity=True)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        return NativeSheetFlowTubeCertificate(
            terminal_level, None, None, -1, "unavailable",
            "native fixed-sheet tube unavailable: "+str(exc),
            0, int(sheet), False, False, None, None, {}, handoff)

    def rational(value):
        return None if value is None else Fraction(*value)

    def interval(value):
        return None if value is None else RationalInterval(
            rational(value[0]), rational(value[1]))

    status = int(result["status"])
    reason = None if status == 0 else (
        f"{result['status_name']}: {result['reason_name']}")
    return NativeSheetFlowTubeCertificate(
        terminal_level, interval(result["terminal_b"]),
        interval(result["terminal_y"]), status, result["status_name"],
        reason, int(result["level_direction"]), int(result["sheet"]),
        bool(result["tube_validated"]),
        bool(result["projection_validated"]),
        rational(result["minimum_face_margin"]),
        rational(result["minimum_gradient_norm_squared"]),
        dict(result["work"]), handoff)


def implicit_fibre_b_interval(model, level: RationalInterval,
                              y: RationalInterval,
                              search: RationalInterval
                              ) -> RationalInterval:
    """Interval-Newton enclosure of ``S_level(b)=y^2`` in one b-chart."""
    level, y, search = _interval(level), _interval(y), _interval(search)
    middle = search.midpoint
    A0 = P.eval_at(model.alpha, middle)
    B0 = P.eval_at(model.beta, middle)
    value = B0*B0+(level-model.C)*A0-y.square()
    level_middle = level.midpoint
    Ap = centered_polynomial_interval(P.deriv(model.alpha), search)
    derivative = (
        centered_polynomial_interval(
            P.deriv(level_polynomial(model, level_middle)), search)
        +(level-level_middle)*Ap)
    if derivative.contains_zero():
        raise ZeroDivisionError("implicit fibre chart does not exclude S_b=0")
    image = middle-value/derivative
    lower, upper = max(search.lo, image.lo), min(search.hi, image.hi)
    if lower > upper or image.lo < search.lo or image.hi > search.hi:
        raise ValueError("interval Newton image does not close in b")
    return RationalInterval(lower, upper)


def implicit_fibre_b_affine_interval(
        model, level0, level1, y0, y1,
        search: RationalInterval) -> RationalInterval:
    """Parametric interval Newton with affine level/y face correlation."""
    ell0, ell1 = P.as_fraction(level0), P.as_fraction(level1)
    y0, y1 = P.as_fraction(y0), P.as_fraction(y1)
    search = _interval(search)
    middle = search.midpoint
    A0 = P.eval_at(model.alpha, middle)
    B0 = P.eval_at(model.beta, middle)
    ell_poly = (ell0, ell1-ell0)
    y_poly = (y0, y1-y0)
    value_poly = P.add(
        (B0*B0,),
        P.add(P.scale(P.add(ell_poly, (-model.C,)), A0),
              P.scale(P.mul(y_poly, y_poly), Fraction(-1))))
    value = centered_polynomial_interval(
        value_poly, RationalInterval(Fraction(0), Fraction(1)))
    levels = RationalInterval(min(ell0, ell1), max(ell0, ell1))
    level_middle = levels.midpoint
    Ap = centered_polynomial_interval(P.deriv(model.alpha), search)
    derivative = (
        centered_polynomial_interval(
            P.deriv(level_polynomial(model, level_middle)), search)
        +(levels-level_middle)*Ap)
    if derivative.contains_zero():
        raise ZeroDivisionError("implicit affine fibre chart does not exclude S_b=0")
    image = middle-value/derivative
    lower, upper = max(search.lo, image.lo), min(search.hi, image.hi)
    if lower > upper or image.lo < search.lo or image.hi > search.hi:
        raise ValueError("affine interval Newton image does not close in b")
    return RationalInterval(lower, upper)


@dataclass(frozen=True)
class YChartKnot:
    level: Fraction
    b: Fraction
    y: Fraction
    lower_y_radius: Fraction
    upper_y_radius: Fraction
    b_interval: RationalInterval

    @property
    def y_interval(self) -> RationalInterval:
        return RationalInterval(self.y-self.lower_y_radius,
                                self.y+self.upper_y_radius)


@dataclass(frozen=True)
class YChartSlab:
    level_lo: Fraction
    level_hi: Fraction
    lower_face_margin: Fraction
    upper_face_margin: Fraction
    gradient_norm_squared_lower: Fraction


@dataclass(frozen=True)
class YChartHandoffCertificate:
    """Experimental face proof for the implicit branch-point y-chart.

    The implementation does not yet prove implicit-chart regularity on each
    complete slab, so it is deliberately incapable of returning a validated
    production status.
    """

    knots: tuple[YChartKnot, ...]
    slabs: tuple[YChartSlab, ...]
    tube_validated: bool
    launch_validated: bool
    reason: str | None = None
    level_direction: int = 1
    slab_bisections: int = 0

    @property
    def status(self) -> str:
        if not self.tube_validated:
            return "unresolved"
        return "conditional_on_interior_regularity"

    def as_dict(self) -> dict:
        terminal = self.knots[-1] if self.knots else None
        return {
            "status": self.status,
            "lateral_faces[VALIDATED]": self.tube_validated,
            "launch_box[VALIDATED]": self.launch_validated,
            "interior_chart_regularity[VALIDATED]": False,
            "coordinate": "implicit fibre y-chart",
            "reason": self.reason,
            "level_direction": self.level_direction,
            "slab_count": len(self.slabs),
            "slab_bisections": self.slab_bisections,
            "terminal_level": (None if terminal is None
                               else float(terminal.level)),
            "terminal_b_interval": (None if terminal is None
                                    else terminal.b_interval.as_dict()),
            "terminal_y_interval": (None if terminal is None
                                    else terminal.y_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if not self.slabs else float(min(
                    margin for slab in self.slabs
                    for margin in (slab.lower_face_margin,
                                   slab.upper_face_margin)))),
            "minimum_gradient_norm_squared[EXACT]": (
                None if not self.slabs else float(min(
                    slab.gradient_norm_squared_lower
                    for slab in self.slabs))),
        }


def certify_y_chart_handoff(
        model, launch, centres: Iterable[LiftedPoint | HolonomyCentre], *,
        max_inflations: int = 32,
        max_radius=None,
        radius_round_bits: int = 192,
        max_slab_bisections: int = 12
        ) -> YChartHandoffCertificate:
    """Advance a validated launch in y while solving b from the exact fibre."""
    if not getattr(launch, "validated", False):
        return YChartHandoffCertificate(
            (), (), False, False, "local graph launch is not validated")
    tail = tuple(centres)
    if not tail:
        return YChartHandoffCertificate(
            (), (), False, True, "at least one y-chart centre is required")
    level0 = launch.section_level
    direction = int(launch.time_direction)
    if direction not in (-1, 1) or any(
            direction*(tail[k].level-(level0 if k == 0
                       else tail[k-1].level)) <= 0
            for k in range(len(tail))):
        return YChartHandoffCertificate(
            (), (), False, True,
            "y-chart centre levels must follow the launch direction")
    lower_radius = launch.y_interval.midpoint-launch.y_interval.lo
    upper_radius = launch.y_interval.hi-launch.y_interval.midpoint
    if min(lower_radius, upper_radius) <= 0:
        raise ValueError("initial y-chart radius must be positive")
    radius_cap = None if max_radius is None else P.as_fraction(max_radius)
    first = HolonomyCentre(level0, launch.b_interval.midpoint,
                           launch.y_interval.midpoint)
    points = (first, *tail)
    b_current = launch.b_interval
    knots = [YChartKnot(first.level, first.b, first.y,
                        lower_radius, upper_radius, b_current)]
    slabs = []
    bisections = 0
    stack = [(points[k], points[k+1], 0)
             for k in range(len(points)-2, -1, -1)]
    while stack:
        left, right, depth = stack.pop()
        h = direction*(right.level-left.level)
        slope = (right.y-left.y)/h
        next_lower = lower_radius
        next_upper = upper_radius
        accepted = None
        failure = None
        for _ in range(max_inflations):
            pad = max(b_current.width,
                      abs(right.b-left.b)/4,
                      Fraction(1, 2**100))
            levels = RationalInterval(min(left.level, right.level),
                                      max(left.level, right.level))
            lower_y0, lower_y1 = (left.y-lower_radius,
                                  right.y-next_lower)
            upper_y0, upper_y1 = (left.y+upper_radius,
                                  right.y+next_upper)
            lower_y = RationalInterval(min(lower_y0, lower_y1),
                                       max(lower_y0, lower_y1))
            upper_y = RationalInterval(min(upper_y0, upper_y1),
                                       max(upper_y0, upper_y1))
            try:
                lower_b = upper_b = search = None
                last_newton_error = None
                for _search_inflation in range(12):
                    search = b_current.hull(RationalInterval(
                        right.b-pad, right.b+pad))
                    try:
                        lower_b = implicit_fibre_b_affine_interval(
                            model, left.level, right.level,
                            lower_y0, lower_y1, search)
                        upper_b = implicit_fibre_b_affine_interval(
                            model, left.level, right.level,
                            upper_y0, upper_y1, search)
                        break
                    except ValueError as exc:
                        last_newton_error = exc
                        pad *= 2
                if lower_b is None or upper_b is None:
                    raise last_newton_error or ValueError(
                        "affine interval Newton search did not close")
                lower = constrained_level_transport_interval(
                    model, levels, lower_b, lower_y)
                upper = constrained_level_transport_interval(
                    model, levels, upper_b, upper_y)
                lower_velocity = direction*lower.dy_dlevel
                upper_velocity = direction*upper.dy_dlevel
                lower_slope = slope-(next_lower-lower_radius)/h
                upper_slope = slope+(next_upper-upper_radius)/h
                lower_margin = lower_velocity.lo-lower_slope
                upper_margin = upper_slope-upper_velocity.hi
                lower_growth = max(
                    slope-lower_velocity.lo, Fraction(0))
                upper_growth = max(
                    upper_velocity.hi-slope, Fraction(0))
                required_lower = max(
                    next_lower,
                    _ceil_dyadic(
                        lower_radius+h*lower_growth*Fraction(65, 64),
                        radius_round_bits))
                required_upper = max(
                    next_upper,
                    _ceil_dyadic(
                        upper_radius+h*upper_growth*Fraction(65, 64),
                        radius_round_bits))
            except (ValueError, ZeroDivisionError) as exc:
                failure = str(exc)
                break
            if lower_margin >= 0 and upper_margin >= 0:
                endpoint_y = RationalInterval(
                    right.y-next_lower, right.y+next_upper)
                endpoint_search = lower_b.hull(upper_b).hull(search)
                try:
                    endpoint_b = implicit_fibre_b_interval(
                        model, RationalInterval.point(right.level),
                        endpoint_y, endpoint_search)
                except (ValueError, ZeroDivisionError) as exc:
                    failure = str(exc)
                    break
                accepted = (lower_margin, upper_margin,
                            min(lower.gradient_norm_squared.lo,
                                upper.gradient_norm_squared.lo), endpoint_b)
                break
            if (required_lower == next_lower
                    and required_upper == next_upper):
                failure = "y-chart inward face inequalities did not close"
                break
            next_lower, next_upper = required_lower, required_upper
            if (radius_cap is not None
                    and max(next_lower, next_upper) > radius_cap):
                failure = "y-chart radius cap exceeded"
                break
        if accepted is None:
            if depth < max_slab_bisections:
                middle = HolonomyCentre(
                    (left.level+right.level)/2,
                    (left.b+right.b)/2,
                    (left.y+right.y)/2)
                bisections += 1
                stack.append((middle, right, depth+1))
                stack.append((left, middle, depth+1))
                continue
            return YChartHandoffCertificate(
                tuple(knots), tuple(slabs), False, True,
                failure or "y-chart tube did not close", direction,
                bisections)
        slabs.append(YChartSlab(
            min(left.level, right.level), max(left.level, right.level),
            accepted[0], accepted[1], accepted[2]))
        lower_radius, upper_radius = next_lower, next_upper
        b_current = accepted[3]
        knots.append(YChartKnot(
            right.level, right.b, right.y,
            lower_radius, upper_radius, b_current))
    return YChartHandoffCertificate(
        tuple(knots), tuple(slabs), True, True,
        level_direction=direction, slab_bisections=bisections)


@dataclass(frozen=True)
class TubeKnot:
    level: Fraction
    b: Fraction
    y: Fraction
    b_radius: Fraction
    y_radius: Fraction

    @property
    def b_interval(self) -> RationalInterval:
        return RationalInterval(self.b-self.b_radius,
                                self.b+self.b_radius)

    @property
    def y_interval(self) -> RationalInterval:
        return RationalInterval(self.y-self.y_radius,
                                self.y+self.y_radius)


@dataclass(frozen=True)
class TubeSlab:
    level_lo: Fraction
    level_hi: Fraction
    lower_face_margins: tuple[Fraction, Fraction]
    upper_face_margins: tuple[Fraction, Fraction]
    gradient_norm_squared_lower: Fraction


@dataclass(frozen=True)
class FlowTubeCertificate:
    knots: tuple[TubeKnot, ...]
    slabs: tuple[TubeSlab, ...]
    tube_validated: bool
    launch_validated: bool
    reason: str | None = None
    level_direction: int = 1
    slab_bisections: int = 0

    @property
    def status(self) -> str:
        if not self.tube_validated:
            return "unresolved"
        return "validated" if self.launch_validated else \
            "conditional_on_launch"

    def as_dict(self) -> dict:
        terminal = self.knots[-1] if self.knots else None
        return {
            "status": self.status,
            "tube[VALIDATED]": self.tube_validated,
            "launch_box[VALIDATED]": self.launch_validated,
            "level_direction": self.level_direction,
            "reason": self.reason,
            "slab_count": len(self.slabs),
            "slab_bisections": self.slab_bisections,
            "terminal_level": (None if terminal is None
                               else float(terminal.level)),
            "terminal_b_interval": (None if terminal is None
                                    else terminal.b_interval.as_dict()),
            "terminal_y_interval": (None if terminal is None
                                    else terminal.y_interval.as_dict()),
            "minimum_face_margin[EXACT]": (
                None if not self.slabs else float(min(
                    margin for slab in self.slabs
                    for margin in (*slab.lower_face_margins,
                                   *slab.upper_face_margins)))),
            "minimum_gradient_norm_squared[EXACT]": (
                None if not self.slabs else float(min(
                    slab.gradient_norm_squared_lower for slab in self.slabs))),
        }


def _face_box(z0, z1, r0, r1, dimension: int, upper: bool
              ) -> tuple[RationalInterval, RationalInterval]:
    full = [RationalInterval(min(z0[k]-r0[k], z1[k]-r1[k]),
                             max(z0[k]+r0[k], z1[k]+r1[k]))
            for k in range(2)]
    sign = 1 if upper else -1
    full[dimension] = RationalInterval(
        min(z0[dimension]+sign*r0[dimension],
            z1[dimension]+sign*r1[dimension]),
        max(z0[dimension]+sign*r0[dimension],
            z1[dimension]+sign*r1[dimension]))
    return full[0], full[1]


def certify_flow_tube(model, centres: Iterable[LiftedPoint | HolonomyCentre], *,
                      initial_b_radius=Fraction(1, 2**40),
                      initial_y_radius=Fraction(1, 2**40),
                      launch_validated: bool = False,
                      max_radius=None,
                      max_inflations: int = 32,
                      max_endpoint_bits: int = 4096,
                      radius_round_bits: int = 192,
                      max_slab_bisections: int = 10
                      ) -> FlowTubeCertificate:
    """Validate a piecewise-linear trapping tube for lifted holonomy.

    For every lateral face of every monotone ``ell`` slab, exact interval
    arithmetic proves that ``(direction,direction*F_b,direction*F_y)`` points
    inward, where ``direction`` is ``+1`` for increasing loss and ``-1`` for
    decreasing loss.  The resulting statement is a genuine flow enclosure
    for *every* lifted trajectory entering the first box.

    A slab whose faces do not close is bisected at the linear midpoint of
    its two centres, to depth ``max_slab_bisections``.  The inserted centre
    is a proposal exactly like the supplied ones; nothing about it is
    trusted.  Bisection is what lets a tiny validated launch box hand off to
    coarsely spaced downstream centres: a long first slab hulls the box back
    toward the saddle, and ``|grad L|^2 >= 4y^2`` then cannot exclude the
    critical point.

    ``launch_validated`` must only be true when an independent local
    invariant-manifold theorem proves that the desired separatrix enters that
    box; sampled Poincare residuals alone do not meet that contract.
    """
    points = tuple(centres)
    if len(points) < 2:
        return FlowTubeCertificate((), (), False, launch_validated,
                                   "at least two lifted centres are required")
    differences = tuple(points[k+1].level-points[k].level
                        for k in range(len(points)-1))
    level_direction = ((differences[0] > 0)-(differences[0] < 0))
    if level_direction == 0 or any(
            level_direction*difference <= 0 for difference in differences):
        return FlowTubeCertificate((), (), False, launch_validated,
                                   "centre levels must be strictly monotone")
    r = (P.as_fraction(initial_b_radius), P.as_fraction(initial_y_radius))
    if min(r) <= 0:
        raise ValueError("initial tube radii must be positive")
    if max_endpoint_bits < 64:
        raise ValueError("tube endpoint bit budget must be at least 64")
    if not 32 <= radius_round_bits < max_endpoint_bits:
        raise ValueError("tube radius precision must fit inside bit budget")
    radius_cap = None if max_radius is None else P.as_fraction(max_radius)
    knots = [TubeKnot(points[0].level, points[0].b, points[0].y,
                      r[0], r[1])]
    slabs = []
    bisections = 0
    # Slabs are processed in level order; a failing slab is split and both
    # halves pushed back, so the radius state ``r`` always belongs to the
    # slab about to be examined.
    stack = [(points[k], points[k+1], 0)
             for k in range(len(points)-2, -1, -1)]
    while stack:
        first, second, depth = stack.pop()
        h = level_direction*(second.level-first.level)
        z0, z1 = (first.b, first.y), (second.b, second.y)
        slope = ((z1[0]-z0[0])/h, (z1[1]-z0[1])/h)
        r1 = list(r)
        accepted = None
        failure = None
        for _ in range(max_inflations):
            state = (*z0, *z1, *r, *r1)
            if max(max(abs(q.numerator).bit_length(), q.denominator.bit_length())
                   for q in state) > max_endpoint_bits:
                failure = "tube coefficient-swell guard reached"
                break
            lower_margins, upper_margins = [], []
            norm_lowers = []
            required = list(r1)
            try:
                interior_b = RationalInterval(
                    min(z0[0]-r[0], z1[0]-r1[0]),
                    max(z0[0]+r[0], z1[0]+r1[0]))
                interior_y = RationalInterval(
                    min(z0[1]-r[1], z1[1]-r1[1]),
                    max(z0[1]+r[1], z1[1]+r1[1]))
                interior = level_transport_interval(
                    model, interior_b, interior_y)
                norm_lowers.append(interior.gradient_norm_squared.lo)
                for dimension in range(2):
                    blo, ylo = _face_box(z0, z1, r, r1, dimension, False)
                    bhi, yhi = _face_box(z0, z1, r, r1, dimension, True)
                    flo = level_transport_interval(model, blo, ylo)
                    fhi = level_transport_interval(model, bhi, yhi)
                    vlo = level_direction*(
                        flo.db_dlevel, flo.dy_dlevel)[dimension]
                    vhi = level_direction*(
                        fhi.db_dlevel, fhi.dy_dlevel)[dimension]
                    lower_slope = slope[dimension]-(r1[dimension]-r[dimension])/h
                    upper_slope = slope[dimension]+(r1[dimension]-r[dimension])/h
                    lower_margins.append(vlo.lo-lower_slope)
                    upper_margins.append(upper_slope-vhi.hi)
                    norm_lowers.extend((flo.gradient_norm_squared.lo,
                                        fhi.gradient_norm_squared.lo))
                    growth = max(slope[dimension]-vlo.lo,
                                 vhi.hi-slope[dimension], Fraction(0))
                    required[dimension] = max(
                        required[dimension],
                        _ceil_dyadic(
                            r[dimension]+h*growth*Fraction(65, 64),
                            radius_round_bits))
            except (ValueError, ZeroDivisionError) as exc:
                failure = str(exc)
                break
            # Nonnegative is the exact Nagumo inward condition for a closed
            # trapping tube.  Equality is important for invariant faces such
            # as b=constant; requiring artificial strictness would widen an
            # already exact coordinate and can manufacture a critical point.
            if all(x >= 0 for x in (*lower_margins, *upper_margins)):
                accepted = (tuple(lower_margins), tuple(upper_margins),
                            min(norm_lowers))
                break
            if required == r1:
                failure = "inward face inequalities did not close"
                break
            r1 = required
            if radius_cap is not None and max(r1) > radius_cap:
                failure = "tube radius cap exceeded"
                break
        if accepted is None:
            if depth < max_slab_bisections:
                middle = HolonomyCentre((first.level+second.level)/2,
                                        (first.b+second.b)/2,
                                        (first.y+second.y)/2)
                bisections += 1
                stack.append((middle, second, depth+1))
                stack.append((first, middle, depth+1))
                continue
            return FlowTubeCertificate(tuple(knots), tuple(slabs), False,
                                       launch_validated,
                                       failure or "tube inflation did not close",
                                       level_direction=level_direction,
                                       slab_bisections=bisections)
        slabs.append(TubeSlab(min(first.level, second.level),
                              max(first.level, second.level),
                              accepted[0], accepted[1], accepted[2]))
        r = tuple(r1)
        knots.append(TubeKnot(second.level, second.b, second.y, r[0], r[1]))
    return FlowTubeCertificate(tuple(knots), tuple(slabs), True,
                               launch_validated,
                               level_direction=level_direction,
                               slab_bisections=bisections)


def certify_flow_tube_from_launch(
        model, launch, centres: Iterable[LiftedPoint | HolonomyCentre], **kwargs
        ) -> FlowTubeCertificate:
    """Compose a validated local graph launch with global holonomy boxes.

    ``launch`` is intentionally accepted by structural interface rather than
    imported from :mod:`spong.local_certificate`, avoiding a module cycle.
    Its exact section rectangle becomes the first tube knot and first radii.
    Later centres are numerical proposals only; all load-bearing statements
    are replayed by :func:`certify_flow_tube` with rational intervals.
    """
    if not getattr(launch, "validated", False):
        return FlowTubeCertificate(
            (), (), False, False, "local graph launch is not validated")
    level = getattr(launch, "section_level", None)
    b_interval = getattr(launch, "b_interval", None)
    y_interval = getattr(launch, "y_interval", None)
    direction = getattr(launch, "time_direction", 0)
    if (level is None or b_interval is None or y_interval is None
            or direction not in (-1, 1)):
        return FlowTubeCertificate(
            (), (), False, False, "local graph launch is incomplete")
    if b_interval.width <= 0 or y_interval.width <= 0:
        return FlowTubeCertificate(
            (), (), False, False,
            "local graph launch must have positive box widths")
    tail = tuple(centres)
    first = HolonomyCentre(level, b_interval.midpoint, y_interval.midpoint)
    if tail and direction*(tail[0].level-level) <= 0:
        return FlowTubeCertificate(
            (), (), False, True,
            "first holonomy centre does not follow launch level direction",
            level_direction=direction)
    options = dict(kwargs)
    options["initial_b_radius"] = b_interval.width/2
    options["initial_y_radius"] = y_interval.width/2
    options["launch_validated"] = True
    return certify_flow_tube(model, (first, *tail), **options)


# ------------------------------------------------------------------------- #
# Abel gap inside one sheet chart                                           #
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AbelGapCertificate:
    level: Fraction
    b_gap: RationalInterval
    abel_gap: RationalInterval | None
    zero_excluded: bool
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "level": float(self.level),
            "level_exact": (self.level.numerator, self.level.denominator),
            "b_gap": self.b_gap.as_dict(),
            "abel_gap": (None if self.abel_gap is None
                         else self.abel_gap.as_dict()),
            "zero_excluded[VALIDATED]": self.zero_excluded,
            "reason": self.reason,
        }


def certify_abel_gap(model, level, first_b: RationalInterval,
                     first_y: RationalInterval,
                     second_b: RationalInterval,
                     second_y: RationalInterval,
                     *, sqrt_bits: int = 128) -> AbelGapCertificate:
    """Exclude equality of two crossings in a common real sheet chart.

    On a fixed sheet ``db/y`` has constant sign, so a disjoint ``b`` order
    is equivalent to a nonzero Abel gap.  The returned integral interval is
    bounded directly from ``S_ell`` on the entire joining ``b`` interval.
    Opposite-sheet and branch-point-spanning comparisons honestly refuse;
    those require an unwrapped period coordinate.
    """
    ell = P.as_fraction(level)
    first_b, first_y = _interval(first_b), _interval(first_y)
    second_b, second_y = _interval(second_b), _interval(second_y)
    gap = second_b-first_b
    first_sheet = 1 if first_y.lo > 0 else -1 if first_y.hi < 0 else 0
    second_sheet = 1 if second_y.lo > 0 else -1 if second_y.hi < 0 else 0
    if first_sheet == 0 or first_sheet != second_sheet:
        return AbelGapCertificate(
            ell, gap, None, False,
            "crossings are not certified in one common sheet chart")
    if gap.contains_zero():
        return AbelGapCertificate(ell, gap, None, False,
                                  "crossing boxes overlap in b")
    if gap.hi < 0:
        # Reuse the positive-orientation calculation and negate at the end.
        swapped = certify_abel_gap(
            model, ell, second_b, second_y, first_b, first_y,
            sqrt_bits=sqrt_bits)
        return AbelGapCertificate(
            ell, gap, None if swapped.abel_gap is None else -swapped.abel_gap,
            swapped.zero_excluded, swapped.reason)
    path = first_b.hull(second_b)
    S_range = polynomial_interval(level_polynomial(model, ell), path)
    if S_range.lo <= 0:
        return AbelGapCertificate(
            ell, gap, None, False,
            "joining chart does not exclude a real branch point")
    root = sqrt_interval(S_range, bits=sqrt_bits)
    length = RationalInterval(second_b.lo-first_b.hi,
                              second_b.hi-first_b.lo)
    magnitude = RationalInterval(length.lo/root.hi,
                                 length.hi/root.lo)
    abel = magnitude if first_sheet > 0 else -magnitude
    return AbelGapCertificate(ell, gap, abel, not abel.contains_zero())


@dataclass(frozen=True)
class FibreSeparationCertificate:
    """Exact coordinate separation of two boxes on one regular fibre.

    A saddle connection would make the stable and unstable crossings the same
    point.  Disjoint projection in either lifted coordinate therefore excludes
    a connection, without requiring a common sheet or an Abel-coordinate
    branch.
    """

    b_gap: RationalInterval
    y_gap: RationalInterval

    @property
    def rectangles_disjoint(self) -> bool:
        return not self.b_gap.contains_zero() or not self.y_gap.contains_zero()

    @property
    def separated_coordinates(self) -> tuple[str, ...]:
        return tuple(name for name, gap in (("b", self.b_gap),
                                             ("y", self.y_gap))
                     if not gap.contains_zero())

    def as_dict(self) -> dict:
        return {
            "b_gap": self.b_gap.as_dict(),
            "y_gap": self.y_gap.as_dict(),
            "separated_coordinates[EXACT]": self.separated_coordinates,
            "rectangles_disjoint[VALIDATED]": self.rectangles_disjoint,
        }


def certify_fibre_separation(
        first_b: RationalInterval, first_y: RationalInterval,
        second_b: RationalInterval, second_y: RationalInterval
        ) -> FibreSeparationCertificate:
    """Certify two lifted crossing rectangles disjoint in ``(b,y)``."""
    first_b, first_y = _interval(first_b), _interval(first_y)
    second_b, second_y = _interval(second_b), _interval(second_y)
    return FibreSeparationCertificate(second_b-first_b, second_y-first_y)


@dataclass(frozen=True)
class SmaleConnectionCertificate:
    """One stable/unstable incidence decision on a common regular fibre."""

    unstable: FlowTubeCertificate
    stable: FlowTubeCertificate
    gap: AbelGapCertificate | None
    reason: str | None = None
    fibre_separation: FibreSeparationCertificate | None = None

    @property
    def status(self) -> str:
        separated = (self.fibre_separation is not None
                     and self.fibre_separation.rectangles_disjoint)
        abel_excluded = self.gap is not None and self.gap.zero_excluded
        if not (separated or abel_excluded):
            return "unresolved"
        if (self.unstable.status == "validated"
                and self.stable.status == "validated"):
            return "connection_excluded"
        return "conditional_exclusion"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "connection_excluded[VALIDATED]": (
                self.status == "connection_excluded"),
            "reason": self.reason,
            "unstable_holonomy": self.unstable.as_dict(),
            "stable_holonomy": self.stable.as_dict(),
            "fibre_separation": (
                None if self.fibre_separation is None
                else self.fibre_separation.as_dict()),
            "abel_gap": None if self.gap is None else self.gap.as_dict(),
        }


def certify_connection_exclusion(
        model, unstable: FlowTubeCertificate,
        stable: FlowTubeCertificate) -> SmaleConnectionCertificate:
    """Exclude equality of two validated crossings on one exact fibre."""
    if not unstable.tube_validated or not stable.tube_validated:
        return SmaleConnectionCertificate(
            unstable, stable, None, "one or both holonomy tubes are unresolved")
    if not unstable.knots or not stable.knots:
        return SmaleConnectionCertificate(
            unstable, stable, None, "one or both holonomy tubes are empty")
    u, s = unstable.knots[-1], stable.knots[-1]
    if u.level != s.level:
        return SmaleConnectionCertificate(
            unstable, stable, None,
            "terminal boxes are not on the same exact rational fibre")
    separation = certify_fibre_separation(
        u.b_interval, u.y_interval, s.b_interval, s.y_interval)
    gap = certify_abel_gap(model, u.level, u.b_interval, u.y_interval,
                           s.b_interval, s.y_interval)
    if separation.rectangles_disjoint:
        return SmaleConnectionCertificate(
            unstable, stable, gap, None,
            fibre_separation=separation)
    return SmaleConnectionCertificate(
        unstable, stable, gap, gap.reason,
        fibre_separation=separation)


@dataclass(frozen=True)
class SheetConnectionCertificate:
    """Pairwise exclusion from two fibre-constrained scalar tubes."""

    unstable: object
    stable: object
    fibre_separation: FibreSeparationCertificate | None
    reason: str | None = None

    @property
    def status(self) -> str:
        if (self.fibre_separation is None
                or not self.fibre_separation.rectangles_disjoint):
            return "unresolved"
        if (self.unstable.status == "validated"
                and self.stable.status == "validated"):
            return "connection_excluded"
        return "conditional_exclusion"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "connection_excluded[VALIDATED]": (
                self.status == "connection_excluded"),
            "reason": self.reason,
            "unstable_holonomy": self.unstable.as_dict(),
            "stable_holonomy": self.stable.as_dict(),
            "fibre_separation": (
                None if self.fibre_separation is None
                else self.fibre_separation.as_dict()),
        }


def certify_sheet_connection_exclusion(
        unstable: object,
        stable: object) -> SheetConnectionCertificate:
    """Exclude equality of two validated fixed-sheet crossings."""
    if unstable.status != "validated" or stable.status != "validated":
        return SheetConnectionCertificate(
            unstable, stable, None,
            "one or both fibre-constrained tubes are unresolved")
    if not unstable.knots or not stable.knots:
        return SheetConnectionCertificate(
            unstable, stable, None,
            "one or both fibre-constrained tubes are empty")
    u, s = unstable.knots[-1], stable.knots[-1]
    if u.level != s.level:
        return SheetConnectionCertificate(
            unstable, stable, None,
            "terminal intervals are not on the same exact rational fibre")
    separation = certify_fibre_separation(
        u.b_interval, u.y_interval, s.b_interval, s.y_interval)
    reason = None if separation.rectangles_disjoint else \
        "terminal fibre intervals overlap"
    return SheetConnectionCertificate(unstable, stable, separation, reason)


# ------------------------------------------------------------------------- #
# Genus-zero residue--logarithm fast path                                   #
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RationalLogCertificate:
    numerator: P.Poly
    denominator: P.Poly
    polynomial_primitive: P.Poly
    proper_numerator: P.Poly
    pole_divisor: complex_structure.RootCertificate
    logarithmic_root_sum: bool
    interval: RationalInterval | None
    subintervals: int
    reason: str | None = None

    @property
    def validated(self) -> bool:
        return (self.interval is not None and self.pole_divisor.complete
                and self.logarithmic_root_sum)

    def as_dict(self) -> dict:
        return {
            "status": "validated" if self.validated else "unresolved",
            "genus[EXACT]": 0,
            "representation": (
                "Q(x1)-Q(x0) + sum_{D(alpha)=0} "
                "R(alpha)/D'(alpha) log((x1-alpha)/(x0-alpha))"),
            "logarithmic_root_sum[EXACT]": self.logarithmic_root_sum,
            "polynomial_primitive[EXACT]": [
                (x.numerator, x.denominator)
                for x in self.polynomial_primitive],
            "proper_numerator[EXACT]": [
                (x.numerator, x.denominator) for x in self.proper_numerator],
            "pole_divisor": self.pole_divisor.as_dict(),
            "definite_integral[VALIDATED]": (
                None if self.interval is None else self.interval.as_dict()),
            "rational_midpoint_subintervals": self.subintervals,
            "reason": self.reason,
        }


def _integrate_polynomial(polynomial: P.Poly) -> P.Poly:
    return (Fraction(0), *(
        polynomial[k]/Fraction(k+1) for k in range(len(polynomial))))


def certify_genus_zero_integral(numerator: P.Poly,
                                denominator: P.Poly,
                                lower, upper, *,
                                tolerance=Fraction(1, 10**8),
                                max_subintervals: int = 65536
                                ) -> RationalLogCertificate:
    """Certify a definite rational integral and its residue--log form.

    Pulling a meromorphic differential back to a rational parametrisation of
    a conic gives ``N(x)/D(x) dx``.  After exact cancellation and division,
    a squarefree ``D`` proves the root-sum identity

        integral = algebraic polynomial term
                 + sum R(alpha)/D'(alpha) log(x-alpha).

    The numeric enclosure does not trust complex floating logarithms.  It
    uses exact rational midpoint bounds with an interval enclosure of the
    second derivative.  Thus the complex disks certify *which algebraic
    poles occur*, while the final real interval is independently replayable.
    """
    N, D = P.trim(numerator), P.trim(denominator)
    lo, hi = P.as_fraction(lower), P.as_fraction(upper)
    if not D:
        raise ZeroDivisionError("zero rational denominator")
    if hi < lo:
        cert = certify_genus_zero_integral(
            N, D, hi, lo, tolerance=tolerance,
            max_subintervals=max_subintervals)
        return RationalLogCertificate(
            cert.numerator, cert.denominator, cert.polynomial_primitive,
            cert.proper_numerator, cert.pole_divisor,
            cert.logarithmic_root_sum,
            None if cert.interval is None else -cert.interval,
            cert.subintervals, cert.reason)
    common = P.gcd_poly(N, D)
    N, nr = P.divmod_exact(N, common)
    D, dr = P.divmod_exact(D, common)
    if nr or dr:
        raise ArithmeticError("failed to reduce rational differential")
    quotient, remainder = P.divmod_exact(N, D)
    primitive = _integrate_polynomial(quotient)
    poles = complex_structure.certify_polynomial_roots(D)
    squarefree = P.degree(P.gcd_poly(D, P.deriv(D))) <= 0
    logarithmic = bool(squarefree and poles.complete)
    if P.eval_at(D, lo) == 0 or P.eval_at(D, hi) == 0 \
            or sturm.count_roots(D, lo, hi):
        return RationalLogCertificate(
            N, D, primitive, remainder, poles, logarithmic, None, 0,
            "integration interval contains a pole")
    exact_polynomial = P.eval_at(primitive, hi)-P.eval_at(primitive, lo)
    if not remainder or lo == hi:
        value = RationalInterval.point(exact_polynomial)
        return RationalLogCertificate(
            N, D, primitive, remainder, poles, logarithmic, value, 0,
            None if logarithmic else "pole divisor was not fully certified")

    # f'' = ((R'D-RD')'D - 2(R'D-RD')D') / D^3.
    Dp = P.deriv(D)
    first_numerator = P.sub(P.mul(P.deriv(remainder), D),
                            P.mul(remainder, Dp))
    second_numerator = P.sub(
        P.mul(P.deriv(first_numerator), D),
        P.scale(P.mul(first_numerator, Dp), Fraction(2)))
    tol = P.as_fraction(tolerance)
    if tol <= 0:
        raise ValueError("integral tolerance must be positive")
    n = 1
    enclosure = None
    refusal = None
    while n <= max_subintervals:
        step = (hi-lo)/n
        total_lo = total_hi = exact_polynomial
        valid = True
        for k in range(n):
            left, right = lo+k*step, lo+(k+1)*step
            cell = RationalInterval(left, right)
            denominator_range = polynomial_interval(D, cell)
            if denominator_range.contains_zero():
                valid = False
                break
            midpoint = (left+right)/2
            midpoint_value = P.eval_at(remainder, midpoint) / \
                P.eval_at(D, midpoint)
            second_range = (polynomial_interval(second_numerator, cell)
                            / (denominator_range*denominator_range
                               * denominator_range))
            second_bound = max(abs(second_range.lo), abs(second_range.hi))
            error = step**3*second_bound/Fraction(24)
            centre = step*midpoint_value
            total_lo += centre-error
            total_hi += centre+error
        if valid:
            enclosure = RationalInterval(total_lo, total_hi)
            if enclosure.width <= tol:
                break
        n *= 2
    if enclosure is None or enclosure.width > tol:
        refusal = ("midpoint enclosure budget exhausted" if enclosure
                   else "natural denominator interval did not exclude zero")
        enclosure = None
    if not logarithmic:
        refusal = refusal or "squarefree complete pole divisor required"
    return RationalLogCertificate(
        N, D, primitive, remainder, poles, logarithmic,
        enclosure if logarithmic else None, min(n, max_subintervals), refusal)
