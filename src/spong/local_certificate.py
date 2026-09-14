"""Validated local invariant-cone launch for a Morse saddle.

The floating Hadamard/Poincare graph is an excellent proposal, but its grid
residual is not an enclosure.  This module supplies the load-bearing local
statement needed by ``spong.hyperelliptic``.  In a rational approximation to
the Hessian eigenframe it encloses the exact algebraic critical centre,
constructs the centered polynomial gradient with interval coefficients, and
proves that a one-sided cone

    u = orientation*t,       |s| <= cone_slope*t,       0 <= t <= reach

is invariant for the time orientation which moves away from the saddle.
Every lateral-face test is a polynomial inequality after the common factor
``t`` is removed.  Hyperbolicity plus the cone condition places the selected
local invariant manifold inside the cone.  Two transverse cone faces then
bracket an exact rational loss level; their slab supplies a validated
``(b,y)`` launch box on that fibre.

For a near-backbone branch the linear cone can be too wide in the physical
``y=Aa-B`` coordinate.  The hard path replays the selected quadratic
Poincare map exactly and validates a piecewise-linear graph tube inside the
cone.  Its complete slabs have inward faces, positive chart Jacobian, and
outward longitudinal flow.  That tube can be cut at fixed loss or fixed
physical ``b`` without moving the launch artificially closer to the saddle.

The implementation is intentionally C-shaped: two components, rectangular
coefficient arrays, bounded dyadic searches, integer status codes, and a
small work record.  It is the independent Python/Fraction oracle for the
eventual GMP C kernel, not the performance implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from math import factorial

from . import _poly as P
from .hyperelliptic import (RationalInterval, level_transport_interval,
                            polynomial_interval)


class LocalCertificateStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    FRAME_SINGULAR = 2
    CONE_UNRESOLVED = 3
    SECTION_UNRESOLVED = 4
    WORK_BUDGET = 5


@dataclass(frozen=True)
class LocalCertificateWork:
    coefficient_intervals: int
    cone_tests: int
    reach_halvings: int
    slope_doublings: int
    peak_endpoint_bits: int
    section_bisections: int = 0
    section_retries: int = 0


@dataclass(frozen=True)
class LocalGraphKnot:
    t: Fraction
    s: Fraction
    radius: Fraction


@dataclass(frozen=True)
class LocalGraphTubeCertificate:
    knots: tuple[LocalGraphKnot, ...]
    lower_margins: tuple[Fraction, ...]
    upper_margins: tuple[Fraction, ...]
    validated: bool
    reason: str | None = None
    minimum_flow_margin: Fraction | None = None

    def as_dict(self) -> dict:
        terminal = self.knots[-1] if self.knots else None
        return {
            "status": "validated" if self.validated else "unresolved",
            "slab_count": len(self.lower_margins),
            "terminal_t": None if terminal is None else float(terminal.t),
            "terminal_s_interval": (None if terminal is None else
                RationalInterval(terminal.s-terminal.radius,
                                 terminal.s+terminal.radius).as_dict()),
            "minimum_face_margin[EXACT]": (
                None if not self.lower_margins else float(min(
                    *self.lower_margins, *self.upper_margins))),
            "minimum_flow_margin[EXACT]": (
                None if self.minimum_flow_margin is None
                else float(self.minimum_flow_margin)),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PoincareBSectionCertificate:
    """Fixed-``b`` section cut from a validated local cone or graph tube."""

    b: Fraction | None
    y_interval: RationalInterval | None
    level_interval: RationalInterval | None
    t_interval: RationalInterval | None
    b_direction: int
    validated: bool
    reason: str | None = None
    graph_tube: LocalGraphTubeCertificate | None = None

    def as_dict(self) -> dict:
        def exact(value):
            return None if value is None else (
                value.numerator, value.denominator)

        return {
            "status": "validated" if self.validated else "unresolved",
            "coordinate": "fixed-b section of exact local invariant tube",
            "b": None if self.b is None else float(self.b),
            "b_exact": exact(self.b),
            "b_direction": self.b_direction,
            "y_interval": (None if self.y_interval is None else
                           self.y_interval.as_dict()),
            "level_interval": (None if self.level_interval is None else
                               self.level_interval.as_dict()),
            "t_interval": (None if self.t_interval is None else
                           self.t_interval.as_dict()),
            "graph_tube": (None if self.graph_tube is None else
                           self.graph_tube.as_dict()),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PoincareLaunchCertificate:
    status_code: int
    manifold: str
    orientation: int
    time_direction: int
    reach: Fraction | None
    cone_slope: Fraction | None
    flow_margin: Fraction | None
    lower_face_margin: Fraction | None
    upper_face_margin: Fraction | None
    section_level: Fraction | None
    b_interval: RationalInterval | None
    y_interval: RationalInterval | None
    work: LocalCertificateWork
    reason: str | None = None
    cone_power: int = 1
    tangent_slope_interval: RationalInterval | None = None
    graph_tube: LocalGraphTubeCertificate | None = None
    engine: str = "python_fraction_oracle"
    b_section: PoincareBSectionCertificate | None = None

    @property
    def validated(self) -> bool:
        return self.status_code == int(LocalCertificateStatus.OK)

    def as_dict(self) -> dict:
        def q(value):
            return None if value is None else (
                value.numerator, value.denominator)

        return {
            "status": "validated" if self.validated else "unresolved",
            "status_code": self.status_code,
            "method": "exact-rational invariant-cone graph transform",
            "engine": self.engine,
            "manifold": self.manifold,
            "orientation": self.orientation,
            "time_direction": self.time_direction,
            "reach": None if self.reach is None else float(self.reach),
            "reach_exact": q(self.reach),
            "cone_slope": (None if self.cone_slope is None
                           else float(self.cone_slope)),
            "cone_slope_exact": q(self.cone_slope),
            "cone_power": self.cone_power,
            "tangent_slope_interval": (
                None if self.tangent_slope_interval is None
                else self.tangent_slope_interval.as_dict()),
            "validated_graph_tube": (
                None if self.graph_tube is None
                else self.graph_tube.as_dict()),
            "fixed_b_section": (
                None if self.b_section is None
                else self.b_section.as_dict()),
            "flow_margin[EXACT]": (None if self.flow_margin is None
                                  else float(self.flow_margin)),
            "lower_face_margin[EXACT]": (
                None if self.lower_face_margin is None
                else float(self.lower_face_margin)),
            "upper_face_margin[EXACT]": (
                None if self.upper_face_margin is None
                else float(self.upper_face_margin)),
            "section_level": (None if self.section_level is None
                              else float(self.section_level)),
            "section_level_exact": q(self.section_level),
            "section_b_interval": (None if self.b_interval is None
                                   else self.b_interval.as_dict()),
            "section_y_interval": (None if self.y_interval is None
                                   else self.y_interval.as_dict()),
            "work": {
                "coefficient_intervals": self.work.coefficient_intervals,
                "cone_tests": self.work.cone_tests,
                "reach_halvings": self.work.reach_halvings,
                "slope_doublings": self.work.slope_doublings,
                "peak_endpoint_bits": self.work.peak_endpoint_bits,
                "section_bisections": self.work.section_bisections,
                "section_retries": self.work.section_retries,
            },
            "reason": self.reason,
        }


def certify_poincare_launch_native(model, point, chart, orientation: int, *,
                                   require_frobenius: bool = True,
                                   max_reach_halvings: int = 24,
                                   max_slope_doublings: int = 96,
                                   tangent_bisections: int = 80,
                                   section_bisections: int = 40,
                                   frobenius_subdivisions: int = 1,
                                   max_endpoint_bits: int = 16384
                                   ) -> PoincareLaunchCertificate:
    """Native GMP launch proof from the original exact loss pencil.

    The backend independently checks the critical root interval and rebuilds
    the centered, framed, quadratically transformed interval field.  Python
    supplies no load-bearing transformed coefficients.
    """
    from . import _native

    empty = LocalCertificateWork(0, 0, 0, 0, 0)
    if orientation not in (-1, 1) or point.source not in ("B", "N"):
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.INVALID_INPUT), chart.manifold,
            orientation, 0, None, None, None, None, None, None, None, None,
            empty, "invalid native launch source or orientation",
            engine="native_gmp")
    root = point.local.center_interval
    result = _native.local_launch(
        model.alpha, model.beta or (Fraction(0),),
        model.N or (Fraction(0),), model.C, point.source,
        (root.lo, root.hi),
        tuple(Fraction(float(value))
              for row in chart.frame for value in row),
        tuple(Fraction(float(value))
              for row in chart.selected_map for value in row),
        Fraction(float(chart.eigenvalues[0])),
        Fraction(float(chart.desired_reach)), int(orientation),
        max_reach_halvings=max_reach_halvings,
        max_slope_doublings=max_slope_doublings,
        tangent_bisections=tangent_bisections,
        section_bisections=section_bisections,
        frobenius_subdivisions=frobenius_subdivisions,
        max_rational_bits=max_endpoint_bits,
        require_frobenius=bool(require_frobenius))

    def q(value):
        return None if value is None else Fraction(*value)

    def interval(value):
        return None if value is None else RationalInterval(
            q(value[0]), q(value[1]))

    work_data = result["work"]
    work = LocalCertificateWork(
        int(work_data["coefficient_intervals"]),
        int(work_data["cone_tests"]),
        int(work_data["reach_halvings"]),
        int(work_data["slope_doublings"]),
        int(work_data["peak_rational_bits"]),
        int(work_data["section_bisections"]),
        int(work_data["section_retries"]))
    validated = bool(result["validated"])
    b_section = (PoincareBSectionCertificate(
        q(result["b_section_b"]), interval(result["b_section_y"]),
        interval(result["b_section_level"]), None,
        int(result["b_direction"]), True,
        graph_tube=None)
        if bool(result["b_section_validated"]) else None)
    return PoincareLaunchCertificate(
        (int(LocalCertificateStatus.OK) if validated else
         int(LocalCertificateStatus.WORK_BUDGET)
         if result["status"] == _native.SPONG_SMALE_WORK_LIMIT else
         int(LocalCertificateStatus.CONE_UNRESOLVED)),
        chart.manifold, int(result["orientation"]),
        int(result["time_direction"]), q(result["reach"]),
        q(result["cone_slope"]), q(result["flow_margin"]),
        q(result["lower_face_margin"]), q(result["upper_face_margin"]),
        q(result["section_level"]), interval(result["section_b"]),
        interval(result["section_y"]), work,
        None if validated else result["reason_name"],
        cone_power=int(result["cone_power"] or 1),
        tangent_slope_interval=interval(result["tangent_slope"]),
        engine="native_gmp", b_section=b_section)


IntervalPoly2 = dict[tuple[int, int], RationalInterval]


def _iv(value) -> RationalInterval:
    return value if isinstance(value, RationalInterval) \
        else RationalInterval.point(value)


def _add(first: IntervalPoly2, second: IntervalPoly2
         ) -> IntervalPoly2:
    out = dict(first)
    for key, value in second.items():
        out[key] = out.get(key, _iv(0))+value
    return out


def _scale(poly: IntervalPoly2, value) -> IntervalPoly2:
    scale = _iv(value)
    return {key: coefficient*scale for key, coefficient in poly.items()}


def _mul(first: IntervalPoly2, second: IntervalPoly2
         ) -> IntervalPoly2:
    out: IntervalPoly2 = {}
    for (i, j), x in first.items():
        for (k, l), y in second.items():
            key = i+k, j+l
            out[key] = out.get(key, _iv(0))+x*y
    return out


def _power(poly: IntervalPoly2, exponent: int) -> IntervalPoly2:
    out = {(0, 0): _iv(1)}
    for _ in range(exponent):
        out = _mul(out, poly)
    return out


def _substitute_linear(poly: IntervalPoly2,
                       first: tuple[Fraction, Fraction],
                       second: tuple[Fraction, Fraction]) -> IntervalPoly2:
    u = {(1, 0): _iv(first[0]), (0, 1): _iv(first[1])}
    s = {(1, 0): _iv(second[0]), (0, 1): _iv(second[1])}
    out: IntervalPoly2 = {}
    for (i, j), coefficient in poly.items():
        term = _scale(_mul(_power(u, i), _power(s, j)), coefficient)
        out = _add(out, term)
    return out


def _substitute_polynomial(poly: IntervalPoly2,
                           first: IntervalPoly2,
                           second: IntervalPoly2) -> IntervalPoly2:
    out: IntervalPoly2 = {}
    for (i, j), coefficient in poly.items():
        term = _scale(_mul(_power(first, i), _power(second, j)),
                      coefficient)
        out = _add(out, term)
    return out


def _partial(poly: IntervalPoly2, dimension: int) -> IntervalPoly2:
    out = {}
    for (i, j), coefficient in poly.items():
        exponent = i if dimension == 0 else j
        if exponent == 0:
            continue
        key = (i-1, j) if dimension == 0 else (i, j-1)
        out[key] = coefficient*exponent
    return out


def _shift_intervals(polynomial: P.Poly, centre: RationalInterval
                     ) -> tuple[RationalInterval, ...]:
    out = []
    derivative = polynomial
    for order in range(len(polynomial)):
        out.append(polynomial_interval(derivative, centre)/factorial(order))
        derivative = P.deriv(derivative)
    return tuple(out)


def _exact_centered_field(model, point, chart
                          ) -> tuple[IntervalPoly2, IntervalPoly2,
                                     IntervalPoly2, IntervalPoly2,
                                     RationalInterval,
                                     RationalInterval, RationalInterval,
                                     IntervalPoly2, IntervalPoly2,
                                     IntervalPoly2]:
    """Interval polynomial in a fixed rational frame at the algebraic root."""
    root_interval = (point.local.center_interval
                     if point.local is not None else point.interval)
    bc = RationalInterval(root_interval.lo, root_interval.hi)
    A = _shift_intervals(model.alpha, bc)
    B = _shift_intervals(model.beta, bc)
    Ap = _shift_intervals(P.deriv(model.alpha), bc)
    Bp = _shift_intervals(P.deriv(model.beta), bc)
    ac = polynomial_interval(model.beta, bc)/polynomial_interval(model.alpha,
                                                                  bc)

    ga: IntervalPoly2 = {}
    gb: IntervalPoly2 = {}
    for j in range(max(len(A), len(B))):
        Aj = A[j] if j < len(A) else _iv(0)
        Bj = B[j] if j < len(B) else _iv(0)
        ga[(0, j)] = _iv(0) if j == 0 else 2*(ac*Aj-Bj)
        ga[(1, j)] = 2*Aj
    for j in range(max(len(Ap), len(Bp))):
        Aj = Ap[j] if j < len(Ap) else _iv(0)
        Bj = Bp[j] if j < len(Bp) else _iv(0)
        gb[(0, j)] = _iv(0) if j == 0 else ac.square()*Aj-2*ac*Bj
        gb[(1, j)] = 2*ac*Aj-2*Bj
        gb[(2, j)] = Aj

    # Integrate the exact centered gradient before changing coordinates.
    # The critical-value constant is kept separate, so cancellation at the
    # algebraic saddle is structural rather than delegated to interval
    # evaluation of C-2aB+a^2A.
    loss: IntervalPoly2 = {}
    for (i, j), coefficient in ga.items():
        loss[(i+1, j)] = coefficient/Fraction(i+1)
    for (i, j), coefficient in gb.items():
        if i == 0:
            loss[(0, j+1)] = loss.get((0, j+1), _iv(0)) \
                + coefficient/Fraction(j+1)

    frame = chart.frame
    v00, v01 = (Fraction(float(frame[0][0])),
                 Fraction(float(frame[0][1])))
    v10, v11 = (Fraction(float(frame[1][0])),
                 Fraction(float(frame[1][1])))
    determinant = v00*v11-v01*v10
    if determinant == 0:
        raise ZeroDivisionError("rational eigenframe is singular")
    p, q = (tuple(Fraction(float(value)) for value in row)
            for row in chart.selected_map)
    U: IntervalPoly2 = {
        (1, 0): _iv(1), (2, 0): _iv(p[0]),
        (1, 1): _iv(p[1]), (0, 2): _iv(p[2]),
    }
    S: IntervalPoly2 = {
        (0, 1): _iv(1), (2, 0): _iv(q[0]),
        (1, 1): _iv(q[1]), (0, 2): _iv(q[2]),
    }
    da = _add(_scale(U, v00), _scale(S, v01))
    db = _add(_scale(U, v10), _scale(S, v11))
    physical_ga = _substitute_polynomial(ga, da, db)
    physical_gb = _substitute_polynomial(gb, da, db)
    chart_y = _scale(physical_ga, Fraction(1, 2))
    chart_loss = _substitute_polynomial(loss, da, db)

    da_du, da_ds = _partial(da, 0), _partial(da, 1)
    db_du, db_ds = _partial(db, 0), _partial(db, 1)
    jacobian = _add(_mul(da_du, db_ds),
                    _scale(_mul(da_ds, db_du), -1))
    # Cramer's-rule numerators.  Multiplying both by the constant sign of
    # the frame determinant makes the separately certified Jacobian positive;
    # division by it is unnecessary for every cone/face sign test.
    sign = 1 if determinant > 0 else -1
    fu = _scale(_add(_mul(db_ds, physical_ga),
                     _scale(_mul(da_ds, physical_gb), -1)), sign)
    fs = _scale(_add(_scale(_mul(db_du, physical_ga), -1),
                     _mul(da_du, physical_gb)), sign)
    jacobian = _scale(jacobian, sign)
    fu.pop((0, 0), None)
    fs.pop((0, 0), None)
    critical_loss = model.C-polynomial_interval(model.beta, bc).square() \
        / polynomial_interval(model.alpha, bc)
    return (fu, fs, chart_loss, chart_y, ac, bc, critical_loss,
            da, db, jacobian)


def _eval_scaled(poly: IntervalPoly2, t: RationalInterval,
                 v: RationalInterval, orientation: int,
                 time_direction: int) -> RationalInterval:
    """Evaluate ``time_direction*F(orientation*t,t*v)/t``."""
    out = _iv(0)
    for (i, j), coefficient in poly.items():
        total = i+j
        if total == 0:
            continue
        term = coefficient*(orientation**i)*time_direction
        if total > 1:
            term = term*_pow_interval(t, total-1)
        if j:
            term = term*_pow_interval(v, j)
        out = out+term
    return out


def _eval_parabolic_scaled(poly: IntervalPoly2, t: RationalInterval,
                           v: RationalInterval, orientation: int,
                           time_direction: int) -> RationalInterval:
    """Evaluate ``time_direction*F(orientation*t,t^2*v)/t``."""
    out = _iv(0)
    for (i, j), coefficient in poly.items():
        exponent = i+2*j-1
        if exponent < 0:
            continue
        term = coefficient*(orientation**i)*time_direction
        if exponent:
            term = term*_pow_interval(t, exponent)
        if j:
            term = term*_pow_interval(v, j)
        out = out+term
    return out


def _pow_interval(value: RationalInterval, exponent: int
                  ) -> RationalInterval:
    out = _iv(1)
    for _ in range(exponent):
        out = out*value
    return out


def _eval_cone_poly(poly: IntervalPoly2, t: RationalInterval,
                    v: RationalInterval, orientation: int
                    ) -> RationalInterval:
    """Evaluate ``P(orientation*t,t*v)`` on a cone box."""
    out = _iv(0)
    for (i, j), coefficient in poly.items():
        term = coefficient*(orientation**i)
        if i+j:
            term = term*_pow_interval(t, i+j)
        if j:
            term = term*_pow_interval(v, j)
        out = out+term
    return out


def _eval_parabolic_poly(poly: IntervalPoly2, t: RationalInterval,
                         v: RationalInterval, orientation: int
                         ) -> RationalInterval:
    """Evaluate ``P(orientation*t,t^2*v)`` on a parabolic graph cone."""
    out = _iv(0)
    for (i, j), coefficient in poly.items():
        term = coefficient*(orientation**i)
        exponent = i+2*j
        if exponent:
            term = term*_pow_interval(t, exponent)
        if j:
            term = term*_pow_interval(v, j)
        out = out+term
    return out


def _physical_box(ac: RationalInterval, bc: RationalInterval, frame,
                  orientation: int, t: RationalInterval,
                  cone_slope: Fraction
                  ) -> tuple[RationalInterval, RationalInterval]:
    v00, v01 = Fraction(float(frame[0][0])), Fraction(float(frame[0][1]))
    v10, v11 = Fraction(float(frame[1][0])), Fraction(float(frame[1][1]))
    u = orientation*t
    s = t*RationalInterval(-cone_slope, cone_slope)
    return ac+v00*u+v01*s, bc+v10*u+v11*s


def _physical_parabolic_box(
        ac: RationalInterval, bc: RationalInterval, frame,
        orientation: int, t: RationalInterval, curvature: Fraction
        ) -> tuple[RationalInterval, RationalInterval]:
    v00, v01 = Fraction(float(frame[0][0])), Fraction(float(frame[0][1]))
    v10, v11 = Fraction(float(frame[1][0])), Fraction(float(frame[1][1]))
    u = orientation*t
    s = t.square()*RationalInterval(-curvature, curvature)
    return ac+v00*u+v01*s, bc+v10*u+v11*s


def _eval_frobenius_poly(poly: IntervalPoly2, t: RationalInterval,
                         tangent: RationalInterval, curvature: Fraction,
                         orientation: int) -> RationalInterval:
    # s=t*(q+t*r), q in the certified tangent-slope interval and |r|<=K.
    v = tangent+t*RationalInterval(-curvature, curvature)
    return _eval_cone_poly(poly, t, v, orientation)


def _physical_frobenius_box(
        ac: RationalInterval, bc: RationalInterval, frame,
        orientation: int, t: RationalInterval,
        tangent: RationalInterval, curvature: Fraction
        ) -> tuple[RationalInterval, RationalInterval]:
    v00, v01 = Fraction(float(frame[0][0])), Fraction(float(frame[0][1]))
    v10, v11 = Fraction(float(frame[1][0])), Fraction(float(frame[1][1]))
    u = orientation*t
    s = t*(tangent+t*RationalInterval(-curvature, curvature))
    return ac+v00*u+v01*s, bc+v10*u+v11*s


def _linear_tangent_bracket(fu: IntervalPoly2, fs: IntervalPoly2,
                            linear_slope: Fraction, orientation: int,
                            time_direction: int,
                            bisections: int = 80
                            ) -> RationalInterval:
    """Isolate the invariant eigenslope in the rational chart."""
    zero = RationalInterval.point(0)

    def defect(q: Fraction) -> RationalInterval:
        qbox = RationalInterval.point(q)
        U = orientation*_eval_scaled(
            fu, zero, qbox, orientation, time_direction)
        V = _eval_scaled(fs, zero, qbox, orientation, time_direction)
        return V-q*U

    lower, upper = -linear_slope, linear_slope
    if defect(lower).lo <= 0 or defect(upper).hi >= 0:
        return RationalInterval(lower, upper)
    for _ in range(bisections):
        middle = (lower+upper)/2
        value = defect(middle)
        if value.lo > 0:
            lower = middle
        elif value.hi < 0:
            upper = middle
        elif value.lo == 0 == value.hi:
            lower = upper = middle
            break
        else:
            break
    return RationalInterval(lower, upper)


def _interval_poly_add(first, second):
    size = max(len(first), len(second))
    return tuple((first[k] if k < len(first) else _iv(0))
                 +(second[k] if k < len(second) else _iv(0))
                 for k in range(size))


def _interval_poly_scale(poly, scalar):
    scalar = _iv(scalar)
    return tuple(coefficient*scalar for coefficient in poly)


def _interval_poly_mul(first, second):
    if not first or not second:
        return ()
    out = [_iv(0)]*(len(first)+len(second)-1)
    for i, x in enumerate(first):
        for j, y in enumerate(second):
            out[i+j] = out[i+j]+x*y
    return tuple(out)


def _interval_poly_power(poly, exponent):
    out = (_iv(1),)
    for _ in range(exponent):
        out = _interval_poly_mul(out, poly)
    return out


def _interval_poly_eval(poly, value):
    out = _iv(0)
    for coefficient in reversed(poly):
        out = out*value+coefficient
    return out


def _interval_poly_eval_punctured(poly, value):
    """Evaluate after removing an exact common power of positive ``t``."""
    first = 0
    while (first+1 < len(poly)
           and poly[first].lo == 0 == poly[first].hi):
        first += 1
    return _interval_poly_eval(poly[first:], value)


def _scaled_affine_face_poly(poly: IntervalPoly2,
                             tangent: RationalInterval,
                             curvature: Fraction, orientation: int,
                             time_direction: int):
    """Symbolically substitute ``s=t*(q+curvature*t)`` into ``F/t``."""
    v = (tangent, RationalInterval.point(curvature))
    out = ()
    for (i, j), coefficient in poly.items():
        exponent = i+j-1
        if exponent < 0:
            continue
        term = _interval_poly_power(v, j)
        if exponent:
            term = (_iv(0),)*exponent+term
        term = _interval_poly_scale(
            term, coefficient*(orientation**i)*time_direction)
        out = _interval_poly_add(out, term)
    return out


def _affine_curve_poly(poly: IntervalPoly2,
                       u0: Fraction, u1: Fraction,
                       s0: Fraction, s1: Fraction):
    u = (u0, u1-u0)
    s = (s0, s1-s0)
    max_u = max((i for i, _ in poly), default=0)
    max_s = max((j for _, j in poly), default=0)
    upows = [P.poly((1,))]
    spows = [P.poly((1,))]
    for _ in range(max_u):
        upows.append(P.mul(upows[-1], u))
    for _ in range(max_s):
        spows.append(P.mul(spows[-1], s))
    degree = max((i+j for i, j in poly), default=-1)
    out = [_iv(0)]*(degree+1)
    for (i, j), coefficient in poly.items():
        monomial = P.mul(upows[i], spows[j])
        for k, value in enumerate(monomial):
            out[k] = out[k]+coefficient*value
    return tuple(out)


def _graph_face_margin(fu: IntervalPoly2, fs: IntervalPoly2,
                       t0: Fraction, t1: Fraction,
                       s0: Fraction, s1: Fraction,
                       orientation: int, time_direction: int,
                       upper: bool) -> Fraction:
    slope = (s1-s0)/(t1-t0)
    if t0 == 0:
        t = RationalInterval(Fraction(0), t1)
        v = RationalInterval.point(slope)
        U = orientation*_eval_scaled(
            fu, t, v, orientation, time_direction)
        V = _eval_scaled(fs, t, v, orientation, time_direction)
    else:
        t = RationalInterval(t0, t1)
        u = orientation*t
        s = RationalInterval(min(s0, s1), max(s0, s1))
        U = orientation*time_direction*_eval_interval_poly2(fu, u, s)
        V = time_direction*_eval_interval_poly2(fs, u, s)
        cheap = (slope*U-V).lo if upper else (V-slope*U).lo
        if cheap < 0:
            u0, u1 = orientation*t0, orientation*t1
            U_poly = _interval_poly_scale(
                _affine_curve_poly(fu, u0, u1, s0, s1),
                orientation*time_direction)
            V_poly = _interval_poly_scale(
                _affine_curve_poly(fs, u0, u1, s0, s1), time_direction)
            defect = (_interval_poly_add(
                _interval_poly_scale(U_poly, slope),
                _interval_poly_scale(V_poly, -1)) if upper else
                _interval_poly_add(V_poly,
                                   _interval_poly_scale(U_poly, -slope)))
            symbolic = _interval_poly_eval(
                defect, RationalInterval(Fraction(0), Fraction(1))).lo
            return max(cheap, symbolic)
        return cheap
    return ((slope*U-V).lo if upper else (V-slope*U).lo)


def _graph_face_interval(poly: IntervalPoly2,
                         t: RationalInterval,
                         centre0: Fraction, centre1: Fraction,
                         radius0: Fraction, radius1: Fraction,
                         t0: Fraction, t1: Fraction,
                         orientation: int) -> RationalInterval:
    # Natural evaluation is used only after the section t-slab has been
    # narrowed by rational bisection.
    fraction = (t-t0)/(t1-t0)
    centre = centre0+fraction*(centre1-centre0)
    radius = radius0+fraction*(radius1-radius0)
    u = orientation*t
    s = centre+RationalInterval(-radius.hi, radius.hi)
    return _eval_interval_poly2(poly, u, s)


def _eval_interval_poly2(poly: IntervalPoly2,
                         u: RationalInterval,
                         s: RationalInterval) -> RationalInterval:
    out = _iv(0)
    for (i, j), coefficient in poly.items():
        out = out+coefficient*_pow_interval(u, i)*_pow_interval(s, j)
    return out


def _frobenius_face_defect_poly(
        fu: IntervalPoly2, fs: IntervalPoly2,
        tangent: Fraction, curvature: Fraction,
        orientation: int, time_direction: int, upper: bool):
    q = RationalInterval.point(tangent)
    U = _interval_poly_scale(_scaled_affine_face_poly(
        fu, q, curvature, orientation, time_direction), orientation)
    V = _scaled_affine_face_poly(
        fs, q, curvature, orientation, time_direction)
    boundary_slope = (q, RationalInterval.point(2*curvature))
    tangent_velocity = _interval_poly_mul(boundary_slope, U)
    defect = (_interval_poly_add(tangent_velocity,
                                 _interval_poly_scale(V, -1))
              if upper else
              _interval_poly_add(V,
                                 _interval_poly_scale(tangent_velocity, -1)))
    return defect


def _frobenius_face_margin(fu: IntervalPoly2, fs: IntervalPoly2,
                           t: RationalInterval,
                           tangent: Fraction, curvature: Fraction,
                           orientation: int, time_direction: int,
                           upper: bool) -> Fraction:
    defect = _frobenius_face_defect_poly(
        fu, fs, tangent, curvature, orientation, time_direction, upper)
    return _interval_poly_eval(defect, t).lo


def _certify_frobenius_cone(fu: IntervalPoly2, fs: IntervalPoly2,
                            reach: Fraction, linear_slope: Fraction,
                            orientation: int, time_direction: int,
                            subdivisions: int = 1):
    """Prove ``q_- t-Kt^2 <= s <= q_+ t+Kt^2`` invariant."""
    t = RationalInterval(Fraction(0), reach)
    tangent = _linear_tangent_bracket(
        fu, fs, linear_slope, orientation, time_direction)
    available_lower = tangent.lo+linear_slope
    available_upper = linear_slope-tangent.hi
    if min(available_lower, available_upper) <= 0:
        return None
    maximum = min(available_lower, available_upper)/reach
    curvature = maximum/Fraction(2**16)
    for _ in range(17):
        lower_defect = _frobenius_face_defect_poly(
            fu, fs, tangent.lo, -curvature,
            orientation, time_direction, False)
        upper_defect = _frobenius_face_defect_poly(
            fu, fs, tangent.hi, curvature,
            orientation, time_direction, True)
        flow_margins = []
        lower_margins = []
        upper_margins = []
        for index in range(subdivisions):
            slab = RationalInterval(
                reach*Fraction(index, subdivisions),
                reach*Fraction(index+1, subdivisions))
            interior = tangent+slab*RationalInterval(
                -curvature, curvature)
            flow_margins.append((orientation*_eval_scaled(
                fu, slab, interior, orientation, time_direction)).lo)
            lower_margins.append(
                _interval_poly_eval_punctured(lower_defect, slab).lo)
            upper_margins.append(
                _interval_poly_eval_punctured(upper_defect, slab).lo)
        flow_margin = min(flow_margins)
        lower_margin = min(lower_margins)
        upper_margin = min(upper_margins)
        if flow_margin > 0 and lower_margin > 0 and upper_margin > 0:
            return ((reach, curvature, flow_margin,
                     lower_margin, upper_margin), tangent)
        curvature *= 2
    return None


def _linear_cone_at_slope(fu: IntervalPoly2, fs: IntervalPoly2,
                          jacobian_poly: IntervalPoly2,
                          reach: Fraction, slope: Fraction,
                          orientation: int, time_direction: int):
    t = RationalInterval(Fraction(0), reach)
    interior = RationalInterval(-slope, slope)
    jacobian = _eval_cone_poly(
        jacobian_poly, t, interior, orientation)
    ft = orientation*_eval_scaled(
        fu, t, interior, orientation, time_direction)
    lower_fs = _eval_scaled(
        fs, t, RationalInterval.point(-slope),
        orientation, time_direction)
    lower_ft = orientation*_eval_scaled(
        fu, t, RationalInterval.point(-slope),
        orientation, time_direction)
    upper_fs = _eval_scaled(
        fs, t, RationalInterval.point(slope),
        orientation, time_direction)
    upper_ft = orientation*_eval_scaled(
        fu, t, RationalInterval.point(slope),
        orientation, time_direction)
    lower_margin = lower_fs.lo+slope*lower_ft.lo
    upper_margin = slope*upper_ft.lo-upper_fs.hi
    if (jacobian.lo > 0 and ft.lo > 0
            and lower_margin > 0 and upper_margin > 0):
        return (reach, slope, ft.lo, lower_margin, upper_margin)
    return None


def _widen_linear_cone(fu: IntervalPoly2, fs: IntervalPoly2,
                       jacobian_poly: IntervalPoly2, cone,
                       orientation: int, time_direction: int,
                       doublings: int = 4):
    best = cone
    slope = cone[1]
    for _ in range(doublings):
        slope *= 2
        candidate = _linear_cone_at_slope(
            fu, fs, jacobian_poly, cone[0], slope,
            orientation, time_direction)
        if candidate is None:
            break
        best = candidate
    return best


def _certify_parabolic_cone(fu: IntervalPoly2, fs: IntervalPoly2,
                            reach: Fraction, linear_slope: Fraction,
                            orientation: int, time_direction: int):
    """Prove ``|s| <= K t^2`` invariant inside a validated linear cone."""
    t = RationalInterval(Fraction(0), reach)
    maximum = linear_slope/reach
    curvature = maximum/Fraction(2**16)
    for _ in range(17):
        interior = RationalInterval(-curvature, curvature)
        ft = orientation*_eval_parabolic_scaled(
            fu, t, interior, orientation, time_direction)
        lower_fu = orientation*_eval_parabolic_scaled(
            fu, t, RationalInterval.point(-curvature),
            orientation, time_direction)
        lower_fs = _eval_parabolic_scaled(
            fs, t, RationalInterval.point(-curvature),
            orientation, time_direction)
        upper_fu = orientation*_eval_parabolic_scaled(
            fu, t, RationalInterval.point(curvature),
            orientation, time_direction)
        upper_fs = _eval_parabolic_scaled(
            fs, t, RationalInterval.point(curvature),
            orientation, time_direction)
        # On s=+/-K t^2, differentiate s-/+K t^2 along the
        # time-oriented field and divide the result by t.
        lower_margin = (lower_fs+2*curvature*t*lower_fu).lo
        upper_margin = (2*curvature*t*upper_fu-upper_fs).lo
        if ft.lo > 0 and lower_margin > 0 and upper_margin > 0:
            return (reach, curvature, ft.lo,
                    lower_margin, upper_margin)
        curvature *= 2
    return None


def _peak_bits(*items) -> int:
    values = []
    for item in items:
        if isinstance(item, RationalInterval):
            values.extend((item.lo, item.hi))
        elif isinstance(item, Fraction):
            values.append(item)
    return max((max(abs(q.numerator).bit_length(), q.denominator.bit_length())
                for q in values), default=0)


def _certify_numerical_graph_tube(point, chart, orientation: int,
                                  time_direction: int,
                                  reach: Fraction,
                                  linear_slope: Fraction,
                                  fu: IntervalPoly2, fs: IntervalPoly2,
                                  jacobian: IntervalPoly2,
                                  *, nodes: int = 65,
                                  max_inflations: int = 128
                                  ) -> LocalGraphTubeCertificate:
    """Validate a thin tube around the non-load-bearing graph proposal."""
    try:
        _, diag = chart.graph(
            point.local, orientation, n=max(129, 2*nodes-1),
            reach=float(reach))
        raw_t = tuple(float(value) for value in diag["normal_x"])
        raw_s = tuple(float(value) for value in diag["normal_h"])
    except (ArithmeticError, FloatingPointError, OverflowError,
            RuntimeError, ValueError) as exc:
        return LocalGraphTubeCertificate(
            (), (), (), False, "graph proposal failed: "+str(exc))
    if not raw_t or len(raw_t) != len(raw_s):
        return LocalGraphTubeCertificate(
            (), (), (), False, "graph proposal is empty")
    indices = tuple(sorted({
        (k*(len(raw_t)-1))//(nodes-1) for k in range(nodes)}))
    points = [(Fraction(0), Fraction(0))]
    for index in indices:
        t = Fraction(raw_t[index])
        s = Fraction(raw_s[index])
        if t > points[-1][0]:
            points.append((t, s))
    if points[-1][0] < reach:
        points.append((reach, Fraction(raw_s[-1])))

    radius = Fraction(0)
    knots = [LocalGraphKnot(Fraction(0), Fraction(0), radius)]
    lower_margins = []
    upper_margins = []
    flow_margins = []
    for (t0, s0), (t1, s1) in zip(points, points[1:]):
        next_radius = max(radius, t1/Fraction(2**120))
        maximum = min(linear_slope*t1-abs(s1),
                      linear_slope*t0-abs(s0) if t0 else
                      linear_slope*t1-abs(s1))
        if maximum <= 0:
            return LocalGraphTubeCertificate(
                tuple(knots), tuple(lower_margins), tuple(upper_margins),
                False, "graph proposal leaves the validated linear cone")
        accepted = None
        for _ in range(max_inflations):
            if next_radius > maximum:
                break
            lower = _graph_face_margin(
                fu, fs, t0, t1, s0-radius, s1-next_radius,
                orientation, time_direction, False)
            upper = _graph_face_margin(
                fu, fs, t0, t1, s0+radius, s1+next_radius,
                orientation, time_direction, True)
            ubox = RationalInterval(
                min(orientation*t0, orientation*t1),
                max(orientation*t0, orientation*t1))
            sbox = RationalInterval(
                min(s0-radius, s1-next_radius),
                max(s0+radius, s1+next_radius))
            # The quadratic Poincare map must remain a coordinate chart on
            # the whole trapping slab.  Positivity only on its two lateral
            # faces would not rule out a fold in the interior.
            interior_jacobian = _eval_interval_poly2(
                jacobian, ubox, sbox)
            if t0 == 0:
                tbox = RationalInterval(Fraction(0), t1)
                slopes = RationalInterval(
                    (s1-next_radius)/t1, (s1+next_radius)/t1)
                flow = (orientation*_eval_scaled(
                    fu, tbox, slopes, orientation, time_direction)).lo
            else:
                tbox = RationalInterval(t0, t1)
                flow = (orientation*time_direction*_eval_interval_poly2(
                    fu, orientation*tbox, sbox)).lo
            if (lower >= 0 and upper >= 0
                    and interior_jacobian.lo > 0
                    and flow > 0):
                accepted = (lower, upper, flow)
                break
            next_radius *= 2
        if accepted is None:
            return LocalGraphTubeCertificate(
                tuple(knots), tuple(lower_margins), tuple(upper_margins),
                False, "piecewise graph face inequalities did not close")
        radius = next_radius
        lower_margins.append(accepted[0])
        upper_margins.append(accepted[1])
        flow_margins.append(accepted[2])
        knots.append(LocalGraphKnot(t1, s1, radius))
    return LocalGraphTubeCertificate(
        tuple(knots), tuple(lower_margins), tuple(upper_margins), True,
        minimum_flow_margin=min(flow_margins))


def _section_from_graph_tube_terminal(
        graph: LocalGraphTubeCertificate,
        loss: IntervalPoly2, chart_y: IntervalPoly2,
        coordinate_a: IntervalPoly2, coordinate_b: IntervalPoly2,
        ac: RationalInterval, bc: RationalInterval,
        critical_loss: RationalInterval,
        orientation: int, time_direction: int, model) -> _Section:
    if not graph.validated or len(graph.knots) < 2:
        return _Section(None, None, None, 0, 0,
                        graph.reason or "validated graph tube unavailable")

    def at_face(poly, knot):
        return _eval_interval_poly2(
            poly, RationalInterval.point(orientation*knot.t),
            RationalInterval(knot.s-knot.radius,
                             knot.s+knot.radius))

    selected = None
    for first, second in reversed(tuple(zip(graph.knots, graph.knots[1:]))):
        first_loss = critical_loss+at_face(loss, first)
        second_loss = critical_loss+at_face(loss, second)
        gap = (second_loss.lo-first_loss.hi if time_direction > 0
               else first_loss.lo-second_loss.hi)
        if gap > 0:
            selected = first, second, first_loss, second_loss
            break
    if selected is None:
        return _Section(None, None, None, 0, 0,
                        "graph tube faces do not bracket a regular loss level")
    first, second, first_loss, second_loss = selected
    level = ((first_loss.hi+second_loss.lo)/2 if time_direction > 0
             else (second_loss.hi+first_loss.lo)/2)
    signed_level = level if time_direction > 0 else -level

    def signed_face(t_value: Fraction) -> RationalInterval:
        tbox = RationalInterval.point(t_value)
        value = critical_loss+_graph_face_interval(
            loss, tbox, first.s, second.s,
            first.radius, second.radius,
            first.t, second.t, orientation)
        return value if time_direction > 0 else -value

    lower_safe, upper_probe = first.t, second.t
    for _ in range(48):
        middle = (lower_safe+upper_probe)/2
        if signed_face(middle).hi < signed_level:
            lower_safe = middle
        else:
            upper_probe = middle
    lower_probe, upper_safe = first.t, second.t
    for _ in range(48):
        middle = (lower_probe+upper_safe)/2
        if signed_face(middle).lo > signed_level:
            upper_safe = middle
        else:
            lower_probe = middle
    if not lower_safe < upper_safe:
        return _Section(None, None, None, 0, 96,
                        "graph-section bisection did not leave a bracket")
    tbox = RationalInterval(lower_safe, upper_safe)
    a_box = ac+_graph_face_interval(
        coordinate_a, tbox, first.s, second.s,
        first.radius, second.radius, first.t, second.t, orientation)
    b_box = bc+_graph_face_interval(
        coordinate_b, tbox, first.s, second.s,
        first.radius, second.radius, first.t, second.t, orientation)
    y_box = _graph_face_interval(
        chart_y, tbox, first.s, second.s,
        first.radius, second.radius, first.t, second.t, orientation)
    peak = _peak_bits(a_box, b_box, y_box, level)
    try:
        level_transport_interval(model, b_box, y_box)
    except (ValueError, ZeroDivisionError) as exc:
        return _Section(None, None, None, peak, 96,
                        "graph section is not a flow box: "+str(exc))
    if y_box.contains_zero():
        return _Section(None, None, None, peak, 96,
                        "graph section does not select one sheet")
    return _Section(level, b_box, y_box, peak, 96, None)


def _section_from_graph_tube(
        graph: LocalGraphTubeCertificate,
        loss: IntervalPoly2, chart_y: IntervalPoly2,
        coordinate_a: IntervalPoly2, coordinate_b: IntervalPoly2,
        ac: RationalInterval, bc: RationalInterval,
        critical_loss: RationalInterval,
        orientation: int, time_direction: int, model) -> _Section:
    """Return the farthest graph-tube prefix with a regular sheet section."""
    last = _Section(None, None, None, 0, 0,
                    graph.reason or "validated graph tube unavailable")
    for end in range(len(graph.knots), 1, -1):
        prefix = LocalGraphTubeCertificate(
            graph.knots[:end], graph.lower_margins[:end-1],
            graph.upper_margins[:end-1], graph.validated, graph.reason,
            graph.minimum_flow_margin)
        last = _section_from_graph_tube_terminal(
            prefix, loss, chart_y, coordinate_a, coordinate_b,
            ac, bc, critical_loss, orientation, time_direction, model)
        if last.reason is None:
            return last
    return last


def certify_poincare_b_section(model, point, chart,
                               launch: PoincareLaunchCertificate
                               ) -> PoincareBSectionCertificate:
    """Cut a fixed-``b`` section from a validated Poincare graph tube.

    This is the complementary handoff used when a fixed loss sheet is too
    curved near ``y=0``.  The graph tube already traps the selected local
    invariant manifold.  Here we additionally prove that physical ``b`` is
    strictly monotone on one slab, bracket an exact rational ``b`` value
    between its transverse faces, and enclose the corresponding ``y`` and
    loss values of the invariant graph.
    """
    graph = launch.graph_tube
    if graph is None or not graph.validated or len(graph.knots) < 2:
        return PoincareBSectionCertificate(
            None, None, None, None, 0, False,
            "validated local graph tube is unavailable")
    try:
        (fu, fs, loss, chart_y, ac, bc, critical_loss,
         _coordinate_a, coordinate_b, jacobian) = \
            _exact_centered_field(model, point, chart)
    except ZeroDivisionError as exc:
        return PoincareBSectionCertificate(
            None, None, None, None, 0, False, str(exc))

    b_flow = _add(_mul(_partial(coordinate_b, 0), fu),
                  _mul(_partial(coordinate_b, 1), fs))

    def at_face(poly, knot):
        return _eval_interval_poly2(
            poly, RationalInterval.point(launch.orientation*knot.t),
            RationalInterval(knot.s-knot.radius,
                             knot.s+knot.radius))

    refusal = "no graph slab has separated, monotone physical-b faces"
    for first, second in reversed(tuple(zip(graph.knots, graph.knots[1:]))):
        first_b = bc+at_face(coordinate_b, first)
        second_b = bc+at_face(coordinate_b, second)
        if second_b.lo > first_b.hi:
            direction = 1
            section_b = (first_b.hi+second_b.lo)/2
        elif second_b.hi < first_b.lo:
            direction = -1
            section_b = (second_b.hi+first_b.lo)/2
        else:
            continue
        tbox = RationalInterval(first.t, second.t)
        ubox = launch.orientation*tbox
        sbox = RationalInterval(
            min(first.s-first.radius, second.s-second.radius),
            max(first.s+first.radius, second.s+second.radius))
        chart_det = _eval_interval_poly2(jacobian, ubox, sbox)
        flow = (direction*launch.time_direction
                *_eval_interval_poly2(b_flow, ubox, sbox))
        if chart_det.lo <= 0 or flow.lo <= 0:
            refusal = "physical b is not certified monotone on the graph slab"
            continue
        signed_b = direction*section_b

        def signed_face(t_value: Fraction) -> RationalInterval:
            value = bc+_graph_face_interval(
                coordinate_b, RationalInterval.point(t_value),
                first.s, second.s, first.radius, second.radius,
                first.t, second.t, launch.orientation)
            return direction*value

        lower_safe, upper_probe = first.t, second.t
        for _ in range(56):
            middle = (lower_safe+upper_probe)/2
            if signed_face(middle).hi < signed_b:
                lower_safe = middle
            else:
                upper_probe = middle
        lower_probe, upper_safe = first.t, second.t
        for _ in range(56):
            middle = (lower_probe+upper_safe)/2
            if signed_face(middle).lo > signed_b:
                upper_safe = middle
            else:
                lower_probe = middle
        if lower_safe >= upper_safe:
            refusal = "fixed-b graph-section bisection did not close"
            continue
        crossing_t = RationalInterval(lower_safe, upper_safe)
        y_box = _graph_face_interval(
            chart_y, crossing_t, first.s, second.s,
            first.radius, second.radius, first.t, second.t,
            launch.orientation)
        level_box = critical_loss+_graph_face_interval(
            loss, crossing_t, first.s, second.s,
            first.radius, second.radius, first.t, second.t,
            launch.orientation)
        try:
            transport = level_transport_interval(
                model, RationalInterval.point(section_b), y_box)
        except (ValueError, ZeroDivisionError) as exc:
            refusal = "fixed-b graph section is not a flow box: "+str(exc)
            continue
        if transport.loss_b.contains_zero():
            refusal = "fixed-b graph section does not exclude L_b=0"
            continue
        return PoincareBSectionCertificate(
            section_b, y_box, level_box, crossing_t,
            direction, True, graph_tube=graph)
    return PoincareBSectionCertificate(
        None, None, None, None, 0, False, refusal)


def certify_poincare_launch(model, point, chart, orientation: int, *,
                            max_reach_halvings: int = 24,
                            max_slope_doublings: int = 96,
                            initial_slope=Fraction(1, 2**96),
                            maximum_slope=Fraction(1, 2),
                            max_endpoint_bits: int = 16384,
                            frobenius_graph: bool = False,
                            validated_graph: bool = False
                            ) -> PoincareLaunchCertificate:
    """Certify one local invariant branch and an exact loss-section box."""
    empty_work = LocalCertificateWork(0, 0, 0, 0, 0)
    if orientation not in (-1, 1) or chart.manifold not in (
            "stable", "unstable"):
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.INVALID_INPUT), chart.manifold,
            orientation, 0, None, None, None, None, None, None, None, None,
            empty_work, "invalid manifold or orientation")
    ld = Fraction(float(chart.eigenvalues[0]))
    if ld == 0:
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.INVALID_INPUT), chart.manifold,
            orientation, 0, None, None, None, None, None, None, None, None,
            empty_work, "departing eigenvalue rounded to zero")
    time_direction = 1 if ld > 0 else -1
    try:
        (fu, fs, loss, chart_y, ac, bc, critical_loss,
         coordinate_a, coordinate_b, coordinate_jacobian) = \
            _exact_centered_field(model, point, chart)
    except ZeroDivisionError as exc:
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.FRAME_SINGULAR), chart.manifold,
            orientation, time_direction, None, None, None, None, None,
            None, None, None, empty_work, str(exc))

    coefficient_count = len(fu)+len(fs)+len(loss)+len(chart_y)
    cone_tests = slope_doublings = section_retries = 0
    peak = _peak_bits(ac, bc)
    reach = Fraction(float(chart.desired_reach))

    def refusal(status, reason, *, halvings, cone=None, bisections=0):
        work = LocalCertificateWork(
            coefficient_count, cone_tests, halvings, slope_doublings, peak,
            bisections, section_retries)
        if cone is None:
            return PoincareLaunchCertificate(
                int(status), chart.manifold, orientation, time_direction,
                None, None, None, None, None, None, None, None, work, reason)
        reach_, slope_, flow_, lower_, upper_ = cone
        return PoincareLaunchCertificate(
            int(status), chart.manifold, orientation, time_direction,
            reach_, slope_, flow_, lower_, upper_, None, None, None, work,
            reason)

    # The reach loop owns both stages.  The cone can fail to close at a
    # reach, and the section rectangle it yields can fail to be a flow box:
    # at a backbone-tangent departure y=A(a-a*) is second order along the
    # branch, while the cone's transverse slack, thin in (a,b), is stretched
    # by A into a y-width of the same order.  The slack scales as R^2 and
    # the signal as R, so halving the reach resolves it; pushing the
    # section outward would not.
    section_reason = "no invariant cone closed within budget"
    slope_start = P.as_fraction(initial_slope)
    previous_graph_candidate = None
    for reach_halvings in range(max_reach_halvings+1):
        t = RationalInterval(Fraction(0), reach)
        # Warm start after a halving.  The cone must hold the branch's
        # curvature s ~ c u^2, so the closing slope scales like c*R and the
        # slope at R/2 is expected near K/2; starting the doubling search
        # at K/4 leaves one octave of slack below that and costs two or
        # three tests instead of ninety.  Starting from a larger slope
        # never weakens the certificate: any slope whose three face
        # inequalities close is a valid cone; a smaller one is only a
        # tighter box.  A GMP C kernel should do the same.
        slope = slope_start
        accepted = None
        for doubling in range(max_slope_doublings+1):
            if slope > maximum_slope:
                break
            cone_tests += 1
            slope_doublings = max(slope_doublings, doubling)
            interior = RationalInterval(-slope, slope)
            jacobian = _eval_cone_poly(
                coordinate_jacobian, t, interior, orientation)
            ft = orientation*_eval_scaled(
                fu, t, interior, orientation, time_direction)
            lower_fs = _eval_scaled(
                fs, t, RationalInterval.point(-slope),
                orientation, time_direction)
            lower_ft = orientation*_eval_scaled(
                fu, t, RationalInterval.point(-slope),
                orientation, time_direction)
            upper_fs = _eval_scaled(
                fs, t, RationalInterval.point(slope),
                orientation, time_direction)
            upper_ft = orientation*_eval_scaled(
                fu, t, RationalInterval.point(slope),
                orientation, time_direction)
            lower_margin = lower_fs.lo+slope*lower_ft.lo
            upper_margin = slope*upper_ft.lo-upper_fs.hi
            peak = max(peak, _peak_bits(
                jacobian, ft, lower_fs, lower_ft, upper_fs, upper_ft,
                reach, slope))
            if peak > max_endpoint_bits:
                return refusal(
                    LocalCertificateStatus.WORK_BUDGET,
                    "local certificate endpoint-bit budget reached",
                    halvings=reach_halvings)
            if (jacobian.lo > 0 and ft.lo > 0
                    and lower_margin > 0 and upper_margin > 0):
                accepted = (reach, slope, ft.lo, lower_margin, upper_margin)
                break
            slope *= 2
        if accepted is None:
            section_reason = "no invariant cone closed within budget"
            slope_start = P.as_fraction(initial_slope)
            reach /= 2
            continue
        slope_start = max(P.as_fraction(initial_slope), accepted[1]/4)

        # The linear cone establishes the branch and a safe neighbourhood.
        # Inside it, certify the sharper Frobenius graph shape s=O(u^2).
        # This often permits a useful departure section at the original
        # reach, where the linear cone's transverse slack would swamp y.
        frobenius = (_certify_frobenius_cone(
            fu, fs, reach, accepted[1], orientation, time_direction)
            if frobenius_graph else None)
        if frobenius is not None:
            parabolic, tangent = frobenius
            parabolic_section = _certify_section(
                parabolic, loss, chart_y, ac, bc, critical_loss,
                chart.frame, orientation, time_direction, model,
                cone_power=2, tangent_slope=tangent,
                coordinate_maps=(coordinate_a, coordinate_b))
            peak = max(peak, parabolic_section.peak)
            if peak > max_endpoint_bits:
                return refusal(
                    LocalCertificateStatus.WORK_BUDGET,
                    "local certificate endpoint-bit budget reached",
                    halvings=reach_halvings, cone=parabolic,
                    bisections=parabolic_section.bisections)
            if parabolic_section.reason is None:
                work = LocalCertificateWork(
                    coefficient_count, cone_tests, reach_halvings,
                    slope_doublings, peak, parabolic_section.bisections,
                    section_retries)
                reach_, curvature, flow_margin, lower_margin, upper_margin = \
                    parabolic
                return PoincareLaunchCertificate(
                    int(LocalCertificateStatus.OK), chart.manifold,
                    orientation, time_direction, reach_, curvature,
                    flow_margin, lower_margin, upper_margin,
                    parabolic_section.level, parabolic_section.b_slab,
                    parabolic_section.y_slab, work, cone_power=2,
                    tangent_slope_interval=tangent)

        section = _certify_section(
            accepted, loss, chart_y, ac, bc, critical_loss, chart.frame,
            orientation, time_direction, model,
            coordinate_maps=(coordinate_a, coordinate_b))
        peak = max(peak, section.peak)
        if peak > max_endpoint_bits:
            return refusal(
                LocalCertificateStatus.WORK_BUDGET,
                "local certificate endpoint-bit budget reached",
                halvings=reach_halvings, cone=accepted,
                bisections=section.bisections)
        if section.reason is None:
            if validated_graph:
                candidates = []
                if previous_graph_candidate is not None:
                    candidates.append(previous_graph_candidate)
                candidates.append((reach, _widen_linear_cone(
                    fu, fs, coordinate_jacobian, accepted,
                    orientation, time_direction)))
                for graph_reach, graph_cone in candidates:
                    graph_tube = _certify_numerical_graph_tube(
                        point, chart, orientation, time_direction,
                        graph_reach, 2*graph_cone[1], fu, fs,
                        coordinate_jacobian)
                    if not graph_tube.validated:
                        continue
                    graph_section = _section_from_graph_tube(
                        graph_tube, loss, chart_y,
                        coordinate_a, coordinate_b, ac, bc,
                        critical_loss, orientation, time_direction, model)
                    peak = max(peak, graph_section.peak)
                    if graph_section.reason is not None:
                        continue
                    work = LocalCertificateWork(
                        coefficient_count, cone_tests, reach_halvings,
                        slope_doublings, peak,
                        graph_section.bisections, section_retries)
                    reach_, slope_, flow_margin, lower_margin, \
                        upper_margin = graph_cone
                    return PoincareLaunchCertificate(
                        int(LocalCertificateStatus.OK), chart.manifold,
                        orientation, time_direction, reach_, slope_,
                        flow_margin, lower_margin, upper_margin,
                        graph_section.level, graph_section.b_slab,
                        graph_section.y_slab, work,
                        graph_tube=graph_tube)
            work = LocalCertificateWork(
                coefficient_count, cone_tests, reach_halvings,
                slope_doublings, peak, section.bisections, section_retries)
            reach_, slope_, flow_margin, lower_margin, upper_margin = accepted
            return PoincareLaunchCertificate(
                int(LocalCertificateStatus.OK), chart.manifold, orientation,
                time_direction, reach_, slope_, flow_margin, lower_margin,
                upper_margin, section.level, section.b_slab, section.y_slab,
                work)
        section_reason = section.reason
        section_retries += 1
        if validated_graph:
            previous_graph_candidate = (
                reach, _widen_linear_cone(
                    fu, fs, coordinate_jacobian, accepted,
                    orientation, time_direction))
        reach /= 2

    return refusal(
        LocalCertificateStatus.SECTION_UNRESOLVED
        if section_retries else LocalCertificateStatus.CONE_UNRESOLVED,
        section_reason+" within the reach budget",
        halvings=max_reach_halvings)


@dataclass(frozen=True)
class _Section:
    level: Fraction | None
    b_slab: RationalInterval | None
    y_slab: RationalInterval | None
    peak: int
    bisections: int
    reason: str | None


def _certify_section(accepted, loss, chart_y, ac, bc, critical_loss, frame,
                     orientation: int, time_direction: int, model, *,
                     cone_power: int = 1,
                     tangent_slope: RationalInterval | None = None,
                     coordinate_maps: tuple[IntervalPoly2,
                                            IntervalPoly2] | None = None
                     ) -> _Section:
    """Bracket an exact loss level between two transverse cone faces and
    return the slab's ``(b,y)`` rectangle, or the reason there is none.

    The rectangle is accepted only if it is a flow box for the lifted
    transport, i.e. ``|grad L|^2`` excludes zero on it, and lies on one
    sheet, i.e. ``y`` is one-signed.  The first is exactly what
    :func:`hyperelliptic.certify_flow_tube_from_launch` will demand of its
    first knot; the second is what a same-sheet comparison downstream means,
    and is what keeps the first slab's ``y``-radius from inflating.
    """
    reach, slope, _flow, _lower, _upper = accepted
    if tangent_slope is None:
        physical_box = (_physical_box if cone_power == 1
                        else _physical_parabolic_box)
        evaluate = (_eval_cone_poly if cone_power == 1
                    else _eval_parabolic_poly)

        def evaluate_at(poly, t):
            return evaluate(poly, t, RationalInterval(-slope, slope),
                            orientation)
    else:
        def evaluate_at(poly, t):
            return _eval_frobenius_poly(
                poly, t, tangent_slope, slope, orientation)

    if coordinate_maps is None:
        if tangent_slope is None:
            def box_at(t):
                return physical_box(ac, bc, frame, orientation, t, slope)
        else:
            def box_at(t):
                return _physical_frobenius_box(
                    ac, bc, frame, orientation, t, tangent_slope, slope)
    else:
        coordinate_a, coordinate_b = coordinate_maps

        def box_at(t):
            return (ac+evaluate_at(coordinate_a, t),
                    bc+evaluate_at(coordinate_b, t))

    a_outer, b_outer = box_at(RationalInterval.point(reach))
    loss_outer = critical_loss+evaluate_at(
        loss, RationalInterval.point(reach))
    section_gap = Fraction(-1)
    section_level = None
    inner = reach/2
    a_inner = b_inner = loss_inner = None
    for exponent in range(1, 9):
        inner = reach/Fraction(2**exponent)
        a_inner, b_inner = box_at(RationalInterval.point(inner))
        loss_inner = critical_loss+evaluate_at(
            loss, RationalInterval.point(inner))
        if time_direction > 0:
            section_gap = loss_outer.lo-loss_inner.hi
            section_level = (loss_inner.hi+loss_outer.lo)/2
        else:
            section_gap = loss_inner.lo-loss_outer.hi
            section_level = (loss_outer.hi+loss_inner.lo)/2
        if section_gap > 0:
            break
    peak = _peak_bits(a_inner, b_inner, a_outer, b_outer,
                      loss_inner, loss_outer)
    if section_gap <= 0:
        return _Section(None, None, None, peak, 0,
                        "cone faces do not bracket an exact regular loss "
                        "section")

    signed_level = (section_level if time_direction > 0
                    else -section_level)

    def signed_face(t_value: Fraction) -> RationalInterval:
        value = critical_loss+evaluate_at(
            loss, RationalInterval.point(t_value))
        return value if time_direction > 0 else -value

    # Find a face certainly before the section and one certainly after it.
    # The interval between them contains the invariant graph's unique loss
    # crossing.  Separate searches retain rigor when the cone makes the
    # midpoint face itself overlap the chosen level.
    lower_safe, upper_probe = inner, reach
    for _ in range(40):
        middle = (lower_safe+upper_probe)/2
        if signed_face(middle).hi < signed_level:
            lower_safe = middle
        else:
            upper_probe = middle
    lower_probe, upper_safe = inner, reach
    for _ in range(40):
        middle = (lower_probe+upper_safe)/2
        if signed_face(middle).lo > signed_level:
            upper_safe = middle
        else:
            lower_probe = middle
    if not lower_safe < upper_safe:
        return _Section(None, None, None, peak, 80,
                        "loss-section bisection did not leave an ordered "
                        "bracket")

    section_t = RationalInterval(lower_safe, upper_safe)
    a_slab, b_slab = box_at(section_t)
    y_slab = evaluate_at(chart_y, section_t)
    peak = max(peak, _peak_bits(a_slab, b_slab, y_slab, section_level))
    try:
        level_transport_interval(model, b_slab, y_slab)
    except (ValueError, ZeroDivisionError) as exc:
        return _Section(None, None, None, peak, 80,
                        "section rectangle is not a flow box: "+str(exc))
    # One sheet: the launch must determine the sign of y.  A rectangle that
    # straddles y=0 may still be a flow box (L_b can exclude zero on it),
    # but the 2Ay/|grad L|^2 term of dy/dlevel then straddles with a spread
    # of order A*width(y), and the tube inflates faster than the branch's
    # own y grows.  Signal ~R, slack ~R^2: halving the reach resolves it.
    if y_slab.contains_zero():
        return _Section(None, None, None, peak, 80,
                        "section rectangle straddles y=0; sheet undetermined")
    return _Section(section_level, b_slab, y_slab, peak, 80, None)
