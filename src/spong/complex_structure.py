"""Validated complex structure of the reduced rational backbone.

Floating root finders are useful proposal engines, but their output is not a
certificate.  This module rounds each proposed root to an exact Gaussian
rational centre and proves its disk count with either a cheap exact linear-term
Rouche witness or the exact Schur-Cohn recursion used by the Lehmer-Schur
method.  If disjoint disk counts sum to the polynomial degree, they account for
every complex root without trusting the proposals.

The certificate distinguishes three divisors that an unreduced ``A`` blurs:

    A                         transverse zero divisor,
    a* = B/V                  valley-chart pole divisor,
    u = C - P/D, u' = H/D^2  backbone pole and critical divisors.

It does not claim that a pole-free disk is a convergence disk for an invariant
manifold.  That additional statement needs a majorant for the nonlinear orbit
equation; keeping the distinction explicit is part of the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, gcd, lcm

import numpy as np

from . import _poly as P
from ._poly import Poly


Gaussian = tuple[Fraction, Fraction]
_MAX_SCHUR_BITS = 32_768


class _SchurCoefficientSwell(ArithmeticError):
    pass


def _gadd(a: Gaussian, b: Gaussian) -> Gaussian:
    return a[0] + b[0], a[1] + b[1]


def _gmul(a: Gaussian, b: Gaussian) -> Gaussian:
    return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def _gscale(a: Gaussian, q: Fraction) -> Gaussian:
    return a[0]*q, a[1]*q


def _gconj(a: Gaussian) -> Gaussian:
    return a[0], -a[1]


def _gpow(a: Gaussian, n: int) -> Gaussian:
    out = (Fraction(1), Fraction(0))
    base = a
    while n:
        if n & 1:
            out = _gmul(out, base)
        base = _gmul(base, base)
        n //= 2
    return out


def _abs_upper(z: Gaussian) -> Fraction:
    """Rational upper bound on |z|."""
    return abs(z[0]) + abs(z[1])


def _abs_lower(z: Gaussian) -> Fraction:
    """Rational lower bound on |z|."""
    return max(abs(z[0]), abs(z[1]))


def _q(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _taylor_about(p: Poly, centre: Gaussian) -> tuple[Gaussian, ...]:
    """Exact coefficients of p(centre+z), in ascending powers of z."""
    out = []
    for k in range(len(p)):
        coefficient = (Fraction(0), Fraction(0))
        for j in range(k, len(p)):
            coefficient = _gadd(
                coefficient,
                _gscale(_gpow(centre, j-k), p[j]*comb(j, k)))
        out.append(coefficient)
    return tuple(out)


def _rouche_margin(coeffs: tuple[Gaussian, ...], radius: Fraction
                    ) -> Fraction:
    """Positive iff the linear Taylor term dominates on |z|=radius."""
    if len(coeffs) < 2:
        return Fraction(-1)
    lhs = _abs_lower(coeffs[1])*radius
    rhs = _abs_upper(coeffs[0])
    power = radius*radius
    for coefficient in coeffs[2:]:
        rhs += _abs_upper(coefficient)*power
        power *= radius
    return lhs-rhs


def _gtrim(p: tuple[Gaussian, ...]) -> tuple[Gaussian, ...]:
    n = len(p)
    while n and p[n-1] == (0, 0):
        n -= 1
    return p[:n]


def _gprimitive(p: tuple[Gaussian, ...]) -> tuple[Gaussian, ...]:
    """Clear denominators and integer content to control Schur swell.

    Multiplication by a nonzero scalar changes neither the roots nor any
    Schur-Cohn sign.  Primitive Gaussian-integer representatives prevent the
    repeated quadratic transforms from accumulating irrelevant common
    factors.
    """
    p = _gtrim(p)
    if not p:
        return p
    denominator = 1
    for re, im in p:
        denominator = lcm(denominator, re.denominator, im.denominator)
    integers = tuple((re.numerator*(denominator//re.denominator),
                      im.numerator*(denominator//im.denominator))
                     for re, im in p)
    content = 0
    for re, im in integers:
        content = gcd(content, abs(re))
        content = gcd(content, abs(im))
    if content == 0:
        return ()
    integers = tuple((re//content, im//content) for re, im in integers)
    if max((max(abs(re).bit_length(), abs(im).bit_length())
            for re, im in integers), default=0) > _MAX_SCHUR_BITS:
        raise _SchurCoefficientSwell
    return tuple((Fraction(re), Fraction(im)) for re, im in integers)


def _schur_transform(p: tuple[Gaussian, ...]) -> tuple[Gaussian, ...]:
    """Schur transform, ascending coefficients; the leading term cancels."""
    n = len(p)-1
    if all(re.denominator == 1 and im.denominator == 1 for re, im in p):
        if 2*max((max(abs(re.numerator).bit_length(),
                         abs(im.numerator).bit_length())
                  for re, im in p), default=0) > _MAX_SCHUR_BITS:
            raise _SchurCoefficientSwell
        a, b = p[0][0].numerator, p[0][1].numerator
        e, f = p[-1][0].numerator, p[-1][1].numerator
        out = []
        for k in range(n+1):
            c, d = p[k][0].numerator, p[k][1].numerator
            g, h = p[n-k][0].numerator, p[n-k][1].numerator
            # conj(p0)*p[k] - p[n]*conj(p[n-k])
            real = a*c+b*d-(e*g+f*h)
            imag = a*d-b*c-(-e*h+f*g)
            out.append((Fraction(real), Fraction(imag)))
        return _gtrim(tuple(out))
    p0_conjugate = _gconj(p[0])
    leading = p[-1]
    return _gtrim(tuple(
        _gadd(_gmul(p0_conjugate, p[k]),
              _gscale(_gmul(leading, _gconj(p[n-k])), Fraction(-1)))
        for k in range(n+1)))


def _schur_cohn_count_unit(p: tuple[Gaussian, ...]
                           ) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Exact number of roots in |z|<1, or None for a singular/boundary case.

    The trace records ``(degree, sign(delta))`` at every Schur transform and
    is sufficient to replay the root-count recurrence.  A zero delta is an
    honest refusal: the selected circle must be perturbed or subdivided.
    """
    p = _gprimitive(p)
    centred = 0
    while len(p) > 1 and p[0] == (0, 0):
        centred += 1
        p = p[1:]
    p = _gprimitive(p)
    if len(p) <= 1:
        return centred, ()
    degree = len(p)-1
    delta = (p[0][0]*p[0][0] + p[0][1]*p[0][1]
             - p[-1][0]*p[-1][0] - p[-1][1]*p[-1][1])
    if delta == 0:
        return None
    transformed = _schur_transform(p)
    if not transformed:
        return None
    child = _schur_cohn_count_unit(transformed)
    if child is None:
        return None
    child_count, trace = child
    count = child_count if delta > 0 else degree-child_count
    return centred+count, ((degree, (delta > 0)-(delta < 0)), *trace)


def _count_disk(polynomial: Poly, centre: Gaussian, radius: Fraction
                ) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    if radius <= 0:
        raise ValueError("disk radius must be positive")
    shifted = _taylor_about(P.trim(polynomial), centre)
    power = Fraction(1)
    scaled = []
    for coefficient in shifted:
        scaled.append(_gscale(coefficient, power))
        power *= radius
    return _schur_cohn_count_unit(tuple(scaled))


def schur_cohn_count_disk(polynomial: Poly, centre: Gaussian,
                          radius: Fraction
                          ) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Count roots in an exact complex disk by the Lehmer-Schur test.

    ``q(z)=p(centre+radius*z)`` has exact Gaussian-rational coefficients, so
    every Schur-Cohn sign is exact.  ``None`` means the chosen boundary, a
    singular recursion, or the explicit coefficient-swell guard prevented a
    decision; no certificate is emitted in any of those cases.
    """
    try:
        return _count_disk(polynomial, centre, radius)
    except _SchurCoefficientSwell:
        return None


@dataclass(frozen=True)
class RootDisk:
    """An exact disk containing one polynomial root, counting multiplicity."""

    centre_re: Fraction
    centre_im: Fraction
    radius: Fraction
    root_count: int
    schur_trace: tuple[tuple[int, int], ...]
    rouche_margin: Fraction | None = None

    @property
    def centre(self) -> Gaussian:
        return self.centre_re, self.centre_im

    def disjoint(self, other: "RootDisk") -> bool:
        dr = self.centre_re-other.centre_re
        di = self.centre_im-other.centre_im
        return dr*dr+di*di > (self.radius+other.radius)**2

    def real_axis_clearance(self) -> Fraction:
        return max(Fraction(0), abs(self.centre_im)-self.radius)

    def distance_from_real_interval(self, lo: Fraction,
                                    hi: Fraction) -> Fraction:
        if self.centre_re < lo:
            horizontal = lo-self.centre_re
        elif self.centre_re > hi:
            horizontal = self.centre_re-hi
        else:
            horizontal = Fraction(0)
        # max(|dx|, |dy|) <= Euclidean distance, so this remains a lower
        # bound without introducing an inexact square root.
        return max(Fraction(0), max(horizontal, abs(self.centre_im))-self.radius)

    def as_dict(self) -> dict:
        witness = ("exact-Lehmer-Schur-disk-count" if self.schur_trace
                   else "exact-linear-Rouche-disk-count")
        return {
            "centre": [float(self.centre_re), float(self.centre_im)],
            "centre_exact": [_q(self.centre_re), _q(self.centre_im)],
            "radius": float(self.radius),
            "radius_exact": _q(self.radius),
            "root_count[VALIDATED]": self.root_count,
            "witness": witness,
            "schur_cohn_trace[EXACT]": [list(row) for row in self.schur_trace],
            "rouche_margin": (float(self.rouche_margin)
                               if self.rouche_margin is not None else None),
            "rouche_margin_exact": (_q(self.rouche_margin)
                                     if self.rouche_margin is not None else None),
        }


@dataclass(frozen=True)
class RootCertificate:
    polynomial: Poly
    disks: tuple[RootDisk, ...]
    complete: bool
    squarefree: bool
    reason: str | None = None

    @property
    def degree(self) -> int:
        return P.degree(self.polynomial)

    def real_axis_clearance(self) -> Fraction | None:
        if not self.complete or not self.disks:
            return None
        return min(disk.real_axis_clearance() for disk in self.disks)

    def as_dict(self) -> dict:
        clearance = self.real_axis_clearance()
        return {
            "polynomial_primitive[EXACT]": list(P.int_primitive(self.polynomial)),
            "degree[EXACT]": self.degree,
            "squarefree[EXACT]": self.squarefree,
            "all_roots_isolated[VALIDATED]": self.complete,
            "method": "exact-Lehmer-Schur/linear-Rouche-disk-count",
            "reason": self.reason,
            "root_disks": [disk.as_dict() for disk in self.disks],
            "real_axis_clearance_lower[VALIDATED]": (
                float(clearance) if clearance is not None else None),
            "real_axis_clearance_lower_exact": (
                _q(clearance) if clearance is not None else None),
        }


def _newton_proposals(p: Poly) -> list[complex]:
    degree = P.degree(p)
    if degree < 1:
        return []
    coefficients = np.asarray([float(c) for c in p], dtype=float)
    scale = float(np.max(np.abs(coefficients)))
    if not np.isfinite(scale) or scale == 0.0:
        return []
    roots = np.roots((coefficients/scale)[::-1])
    polynomial = (coefficients/scale)[::-1]
    derivative = np.polyder(polynomial)
    out = []
    for root in roots:
        z = complex(root)
        for _ in range(12):
            value = complex(np.polyval(polynomial, z))
            slope = complex(np.polyval(derivative, z))
            if slope == 0.0 or not np.isfinite(slope.real+slope.imag):
                break
            step = value/slope
            candidate = z-step
            if not np.isfinite(candidate.real+candidate.imag):
                break
            z = candidate
            if abs(step) <= 4*np.finfo(float).eps*max(1.0, abs(z)):
                break
        if np.isfinite(z.real+z.imag):
            out.append(z)
    return out


def _certify_proposal(p: Poly, proposal: complex,
                      neighbour_distance: float) -> RootDisk | None:
    centre = (Fraction(float(proposal.real)), Fraction(float(proposal.imag)))
    coeffs = _taylor_about(p, centre)
    derivative_lower = _abs_lower(coeffs[1]) if len(coeffs) > 1 else Fraction(0)
    if derivative_lower == 0:
        return None
    correction = _abs_upper(coeffs[0])/derivative_lower
    numerical_floor = Fraction(float(
        32*np.finfo(float).eps*max(1.0, abs(proposal))))
    radius = max(8*correction, numerical_floor)
    if np.isfinite(neighbour_distance):
        cap = Fraction(float(0.45*neighbour_distance))
    else:
        cap = Fraction(float(max(1.0, abs(proposal)+1.0)))
    if cap <= 0:
        return None
    for _ in range(24):
        trial = min(radius, cap)
        margin = _rouche_margin(coeffs, trial)
        # For high-degree, well-separated simple roots the linear Rouche
        # witness is both stronger and dramatically cheaper than driving a
        # Schur recurrence through irrelevant coefficient swell.
        if margin > 0 and P.degree(p) > 10:
            return RootDisk(centre[0], centre[1], trial, 1, (), margin)
        try:
            counted = _count_disk(p, centre, trial)
        except _SchurCoefficientSwell:
            if margin > 0:
                return RootDisk(centre[0], centre[1], trial, 1, (), margin)
            # Changing the radius does not cure pathological recurrence
            # growth.  Refuse this proposal rather than repeating the same
            # expensive exact transform 24 times.
            return None
        if counted is not None and counted[0] == 1:
            return RootDisk(centre[0], centre[1], trial, 1, counted[1],
                            margin if margin > 0 else None)
        if trial == cap:
            break
        radius *= 2
    return None


@lru_cache(maxsize=256)
def certify_polynomial_roots(polynomial: Poly) -> RootCertificate:
    """Return exact one-root disk witnesses and an honest completeness flag."""
    p = P.trim(polynomial)
    degree = P.degree(p)
    squarefree = degree < 1 or P.degree(P.gcd_poly(p, P.deriv(p))) <= 0
    if degree < 1:
        return RootCertificate(p, (), True, squarefree)
    proposals = _newton_proposals(p)
    if len(proposals) != degree:
        return RootCertificate(
            p, (), False, squarefree,
            f"proposal count {len(proposals)} does not equal degree {degree}")
    disks = []
    for i, root in enumerate(proposals):
        separation = min((abs(root-other) for j, other in enumerate(proposals)
                          if i != j), default=float("inf"))
        disk = _certify_proposal(p, root, separation)
        if disk is not None:
            disks.append(disk)
    pairwise = all(a.disjoint(b) for i, a in enumerate(disks)
                   for b in disks[i+1:])
    complete = sum(disk.root_count for disk in disks) == degree and pairwise
    reason = None
    if not complete:
        reason = (f"validated {len(disks)}/{degree} roots"
                  if pairwise else "validated disks overlap")
    return RootCertificate(p, tuple(disks), complete, squarefree, reason)


@dataclass(frozen=True)
class BackboneCertificate:
    cancelled_factor: Poly
    transverse: RootCertificate
    valley_denominator: RootCertificate
    denominator: RootCertificate
    critical: RootCertificate

    @property
    def complete(self) -> bool:
        return (self.transverse.complete
                and self.valley_denominator.complete
                and self.denominator.complete
                and self.critical.complete)

    def pole_clearance(self, interval) -> Fraction | None:
        """Lower distance from a real interval to every reduced ``u`` pole."""
        if not self.denominator.complete or not self.denominator.disks:
            return None
        return min(disk.distance_from_real_interval(interval.lo, interval.hi)
                   for disk in self.denominator.disks)

    def valley_clearance(self, interval) -> Fraction | None:
        """Lower distance from a real interval to every reduced ``a*`` pole."""
        if (not self.valley_denominator.complete
                or not self.valley_denominator.disks):
            return None
        return min(disk.distance_from_real_interval(interval.lo, interval.hi)
                   for disk in self.valley_denominator.disks)

    def as_dict(self) -> dict:
        return {
            "status": "validated" if self.complete else "partial",
            "scope": ("complete complex transverse, reduced valley-chart, "
                      "reduced backbone, and critical divisors; not an "
                      "invariant-manifold convergence proof"),
            "cancelled_factor_degree[EXACT]": P.degree(self.cancelled_factor),
            "cancelled_factor_primitive[EXACT]": list(
                P.int_primitive(self.cancelled_factor)),
            "transverse_zero_divisor": self.transverse.as_dict(),
            "valley_chart_poles": self.valley_denominator.as_dict(),
            "backbone_poles": self.denominator.as_dict(),
            "critical_divisor": self.critical.as_dict(),
        }


def certify_backbone(model) -> BackboneCertificate:
    """Certified complex divisor of the model's reduced rational backbone."""
    valley_common = P.gcd_poly(model.alpha, model.beta)
    valley_denominator, remainder = P.divmod_exact(
        model.alpha, valley_common)
    if remainder:
        raise ArithmeticError("failed to reduce exact valley chart")
    return BackboneCertificate(
        model.backbone_common,
        certify_polynomial_roots(model.alpha),
        certify_polynomial_roots(valley_denominator),
        certify_polynomial_roots(model.backbone_den),
        certify_polynomial_roots(model.critical_reduced),
    )
