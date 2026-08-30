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
from .hyperelliptic import RationalInterval, polynomial_interval


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
            "manifold": self.manifold,
            "orientation": self.orientation,
            "time_direction": self.time_direction,
            "reach": None if self.reach is None else float(self.reach),
            "reach_exact": q(self.reach),
            "cone_slope": (None if self.cone_slope is None
                           else float(self.cone_slope)),
            "cone_slope_exact": q(self.cone_slope),
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
            },
            "reason": self.reason,
        }


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


def _shift_intervals(polynomial: P.Poly, centre: RationalInterval
                     ) -> tuple[RationalInterval, ...]:
    out = []
    derivative = polynomial
    for order in range(len(polynomial)):
        out.append(polynomial_interval(derivative, centre)/factorial(order))
        derivative = P.deriv(derivative)
    return tuple(out)


def _exact_centered_field(model, point, frame
                          ) -> tuple[IntervalPoly2, IntervalPoly2,
                                     IntervalPoly2, IntervalPoly2,
                                     RationalInterval,
                                     RationalInterval, RationalInterval]:
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

    v00, v01 = (Fraction(float(frame[0][0])),
                 Fraction(float(frame[0][1])))
    v10, v11 = (Fraction(float(frame[1][0])),
                 Fraction(float(frame[1][1])))
    determinant = v00*v11-v01*v10
    if determinant == 0:
        raise ZeroDivisionError("rational eigenframe is singular")
    physical_ga = _substitute_linear(ga, (v00, v01), (v10, v11))
    physical_gb = _substitute_linear(gb, (v00, v01), (v10, v11))
    chart_y = _scale(physical_ga, Fraction(1, 2))
    chart_loss = _substitute_linear(loss, (v00, v01), (v10, v11))
    fu = _add(_scale(physical_ga, v11/determinant),
              _scale(physical_gb, -v01/determinant))
    fs = _add(_scale(physical_ga, -v10/determinant),
              _scale(physical_gb, v00/determinant))
    fu.pop((0, 0), None)
    fs.pop((0, 0), None)
    critical_loss = model.C-polynomial_interval(model.beta, bc).square() \
        / polynomial_interval(model.alpha, bc)
    return fu, fs, chart_loss, chart_y, ac, bc, critical_loss


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


def _physical_box(ac: RationalInterval, bc: RationalInterval, frame,
                  orientation: int, t: RationalInterval,
                  cone_slope: Fraction
                  ) -> tuple[RationalInterval, RationalInterval]:
    v00, v01 = Fraction(float(frame[0][0])), Fraction(float(frame[0][1]))
    v10, v11 = Fraction(float(frame[1][0])), Fraction(float(frame[1][1]))
    u = orientation*t
    s = t*RationalInterval(-cone_slope, cone_slope)
    return ac+v00*u+v01*s, bc+v10*u+v11*s


def _peak_bits(*items) -> int:
    values = []
    for item in items:
        if isinstance(item, RationalInterval):
            values.extend((item.lo, item.hi))
        elif isinstance(item, Fraction):
            values.append(item)
    return max((max(abs(q.numerator).bit_length(), q.denominator.bit_length())
                for q in values), default=0)


def certify_poincare_launch(model, point, chart, orientation: int, *,
                            max_reach_halvings: int = 24,
                            max_slope_doublings: int = 96,
                            initial_slope=Fraction(1, 2**96),
                            maximum_slope=Fraction(1, 2),
                            max_endpoint_bits: int = 16384
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
        fu, fs, loss, chart_y, ac, bc, critical_loss = _exact_centered_field(
            model, point, chart.frame)
    except ZeroDivisionError as exc:
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.FRAME_SINGULAR), chart.manifold,
            orientation, time_direction, None, None, None, None, None,
            None, None, None, empty_work, str(exc))

    coefficient_count = len(fu)+len(fs)+len(loss)+len(chart_y)
    cone_tests = slope_doublings = 0
    peak = _peak_bits(ac, bc)
    reach = Fraction(float(chart.desired_reach))
    accepted = None
    for reach_halvings in range(max_reach_halvings+1):
        t = RationalInterval(Fraction(0), reach)
        slope = P.as_fraction(initial_slope)
        for doubling in range(max_slope_doublings+1):
            if slope > maximum_slope:
                break
            cone_tests += 1
            slope_doublings = max(slope_doublings, doubling)
            interior = RationalInterval(-slope, slope)
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
                ft, lower_fs, lower_ft, upper_fs, upper_ft, reach, slope))
            if peak > max_endpoint_bits:
                work = LocalCertificateWork(
                    coefficient_count, cone_tests, reach_halvings,
                    slope_doublings, peak)
                return PoincareLaunchCertificate(
                    int(LocalCertificateStatus.WORK_BUDGET), chart.manifold,
                    orientation, time_direction, None, None, None, None,
                    None, None, None, None, work,
                    "local certificate endpoint-bit budget reached")
            if ft.lo > 0 and lower_margin > 0 and upper_margin > 0:
                accepted = (reach, slope, ft.lo,
                            lower_margin, upper_margin, reach_halvings)
                break
            slope *= 2
        if accepted is not None:
            break
        reach /= 2

    if accepted is None:
        work = LocalCertificateWork(
            coefficient_count, cone_tests, max_reach_halvings,
            slope_doublings, peak)
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.CONE_UNRESOLVED), chart.manifold,
            orientation, time_direction, None, None, None, None, None,
            None, None, None, work, "no invariant cone closed within budget")

    reach, slope, flow_margin, lower_margin, upper_margin, halvings = accepted
    a_outer, b_outer = _physical_box(
        ac, bc, chart.frame, orientation, RationalInterval.point(reach), slope)
    loss_outer = critical_loss+_eval_cone_poly(
        loss, RationalInterval.point(reach),
        RationalInterval(-slope, slope), orientation)
    section_gap = Fraction(-1)
    section_level = None
    inner = reach/2
    a_inner = b_inner = loss_inner = None
    for exponent in range(1, 9):
        inner = reach/Fraction(2**exponent)
        a_inner, b_inner = _physical_box(
            ac, bc, chart.frame, orientation,
            RationalInterval.point(inner), slope)
        loss_inner = critical_loss+_eval_cone_poly(
            loss, RationalInterval.point(inner),
            RationalInterval(-slope, slope), orientation)
        if time_direction > 0:
            section_gap = loss_outer.lo-loss_inner.hi
            section_level = (loss_inner.hi+loss_outer.lo)/2
        else:
            section_gap = loss_inner.lo-loss_outer.hi
            section_level = (loss_outer.hi+loss_inner.lo)/2
        if section_gap > 0:
            break
    if section_gap <= 0:
        work = LocalCertificateWork(
            coefficient_count, cone_tests, halvings, slope_doublings, peak)
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.SECTION_UNRESOLVED), chart.manifold,
            orientation, time_direction, reach, slope, flow_margin,
            lower_margin, upper_margin, None, None, None, work,
            "cone faces do not bracket an exact regular loss section")

    signed_level = (section_level if time_direction > 0
                    else -section_level)

    def signed_face(t_value: Fraction) -> RationalInterval:
        value = critical_loss+_eval_cone_poly(
            loss, RationalInterval.point(t_value),
            RationalInterval(-slope, slope), orientation)
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
        work = LocalCertificateWork(
            coefficient_count, cone_tests, halvings, slope_doublings,
            peak, 80)
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.SECTION_UNRESOLVED), chart.manifold,
            orientation, time_direction, reach, slope, flow_margin,
            lower_margin, upper_margin, None, None, None, work,
            "loss-section bisection did not leave an ordered bracket")

    a_slab, b_slab = _physical_box(
        ac, bc, chart.frame, orientation,
        RationalInterval(lower_safe, upper_safe), slope)
    y_slab = _eval_cone_poly(
        chart_y, RationalInterval(lower_safe, upper_safe),
        RationalInterval(-slope, slope), orientation)
    peak = max(peak, _peak_bits(
        a_inner, b_inner, a_outer, b_outer, loss_inner, loss_outer,
        a_slab, b_slab, y_slab, section_level))
    work = LocalCertificateWork(
        coefficient_count, cone_tests, halvings, slope_doublings, peak, 80)
    if peak > max_endpoint_bits:
        return PoincareLaunchCertificate(
            int(LocalCertificateStatus.WORK_BUDGET), chart.manifold,
            orientation, time_direction, reach, slope, flow_margin,
            lower_margin, upper_margin, None, None, None, work,
            "local certificate endpoint-bit budget reached")
    return PoincareLaunchCertificate(
        int(LocalCertificateStatus.OK), chart.manifold, orientation,
        time_direction, reach, slope, flow_margin, lower_margin,
        upper_margin, section_level, b_slab, y_slab, work)
