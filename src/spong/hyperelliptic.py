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
transport and same-sheet Abel-gap exclusion.  Unwrapped positive-genus period
transport remains a separate obligation.  A validated local invariant-cone
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
class SmaleConnectionCertificate:
    """One stable/unstable incidence decision on a common regular fibre."""

    unstable: FlowTubeCertificate
    stable: FlowTubeCertificate
    gap: AbelGapCertificate | None
    reason: str | None = None

    @property
    def status(self) -> str:
        if self.gap is None or not self.gap.zero_excluded:
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
            "abel_gap": None if self.gap is None else self.gap.as_dict(),
        }


def certify_connection_exclusion(
        model, unstable: FlowTubeCertificate,
        stable: FlowTubeCertificate) -> SmaleConnectionCertificate:
    """Compose two holonomy tubes with a same-fibre Abel zero exclusion."""
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
    gap = certify_abel_gap(model, u.level, u.b_interval, u.y_interval,
                           s.b_interval, s.y_interval)
    return SmaleConnectionCertificate(unstable, stable, gap, gap.reason)


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
