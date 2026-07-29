"""Arithmetic qualification measurements for a computed SPONG model.

The qualification boundary is intentionally separate from the mathematical
genericity and a-posteriori topology certificates.  This module only measures
the finite exact data that binary64 will consume.  Policy thresholds are not
embedded here: they are to be fixed from a calibration corpus and then recorded
by the caller.

All coefficient construction has already taken place over ``Fraction`` in
``Model``.  Consequently the useful questions are:

* how large and how complicated are the required moments;
* did cancellation or compensation leave well-scaled exact A, B, and N;
* can every nonzero coefficient be represented by binary64; and
* after exact critical-point enumeration, are distinct roots and local
  eigendata resolved at the binary64 scale?
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Iterable

import numpy as np

from . import _poly as P
from .model import Model


def _log2_int(n: int) -> float:
    """Accurate log2 for an arbitrarily large positive integer."""
    if n <= 0:
        raise ValueError("log2 integer must be positive")
    bits = n.bit_length()
    keep = min(bits, 53)
    top = n >> (bits - keep)
    return math.log2(top) + bits - keep


def _log2_abs(x: Fraction) -> float:
    if not x:
        return -math.inf
    return _log2_int(abs(x.numerator)) - _log2_int(x.denominator)


def _fraction_bits(x: Fraction) -> int:
    return max(abs(x.numerator).bit_length(), x.denominator.bit_length())


def _binary64_relative_error(x: Fraction) -> float:
    if not x:
        return 0.0
    try:
        y = float(x)
    except OverflowError:
        return math.inf
    if not math.isfinite(y) or y == 0.0:
        return math.inf
    return float(abs(Fraction.from_float(y) - x) / abs(x))


def _sequence_profile(xs: Iterable[Fraction]) -> dict:
    values = tuple(xs)
    nonzero = tuple(x for x in values if x)
    logs = tuple(_log2_abs(x) for x in nonzero)
    errors = tuple(_binary64_relative_error(x) for x in nonzero)
    representable = all(math.isfinite(e) for e in errors)
    return {
        "count": len(values),
        "nonzero": len(nonzero),
        "max_fraction_bits": max((_fraction_bits(x) for x in values),
                                 default=0),
        "min_abs_log2": min(logs, default=None),
        "max_abs_log2": max(logs, default=None),
        "dynamic_range_bits": (
            max(logs) - min(logs) if len(logs) > 1 else 0.0),
        "binary64_all_nonzero_finite": representable,
        "max_binary64_relative_error": (
            max(errors, default=0.0) if representable else math.inf),
    }


def _central_moment(mu: tuple[Fraction, ...], k: int) -> Fraction:
    mass = mu[0]
    mean = mu[1] / mass if len(mu) > 1 else Fraction(0)
    return sum(
        Fraction(math.comb(k, j)) * (-mean) ** (k - j) * mu[j]
        for j in range(k + 1)
    ) / mass


def moment_profile(m: Model) -> dict:
    """Profile exactly the prefix of the moment sequence consumed by m."""
    degree = max(P.degree(m.f), P.degree(m.g))
    used = tuple(m.mu[:2 * degree + 1])
    out = _sequence_profile(used)
    out["highest_order"] = len(used) - 1
    if not used or used[0] <= 0:
        out.update({
            "mass_positive": False,
            "mean_abs_log2": None,
            "variance_log2": None,
            "max_standardized_central_moment_log2": None,
        })
        return out

    mass = used[0]
    mean = used[1] / mass if len(used) > 1 else Fraction(0)
    variance = _central_moment(used, 2) if len(used) > 2 else Fraction(0)
    standardized = []
    if variance > 0:
        log_variance = _log2_abs(variance)
        for k in range(3, len(used)):
            ck = _central_moment(used, k)
            if ck:
                standardized.append(_log2_abs(ck) - 0.5*k*log_variance)
    else:
        log_variance = None
    out.update({
        "mass_positive": True,
        "mean_abs_log2": _log2_abs(mean) if mean else None,
        "variance_log2": log_variance,
        "max_standardized_central_moment_log2": (
            max(standardized, default=None)),
    })
    return out


def polynomial_profile(p) -> dict:
    out = _sequence_profile(tuple(p))
    out["degree"] = P.degree(p)
    return out


def _exact_quotient(p, q):
    quotient, remainder = P.divmod_exact(p, q)
    if remainder:
        raise ArithmeticError("internal non-exact polynomial quotient")
    return quotient


def reduced_backbone_polynomials(m: Model) -> dict:
    """Exact reduced representation of u = numerator/denominator.

    The additive constant C is retained in ``loss_numerator``.  The
    ``variable_numerator`` represents B² after cancelling its common factor
    with A.  The derivative numerator is for u itself:

        u' = derivative_numerator / denominator².
    """
    common = m.backbone_common
    variable_numerator = m.backbone_num
    denominator = m.backbone_den
    loss_numerator = P.sub(P.scale(denominator, m.C),
                           variable_numerator)
    derivative_numerator = m.critical_reduced
    if derivative_numerator:
        derivative_common = P.gcd_poly(
            derivative_numerator, P.deriv(derivative_numerator))
        derivative_squarefree = _exact_quotient(
            derivative_numerator, derivative_common)
    else:
        derivative_squarefree = P.ZERO
    return {
        "cancelled_factor": common,
        "variable_numerator": variable_numerator,
        "loss_numerator": loss_numerator,
        "denominator": denominator,
        "derivative_numerator": derivative_numerator,
        "derivative_squarefree": derivative_squarefree,
    }


def backbone_profile(m: Model) -> dict:
    """Realized algebraic complexity of the loss restricted to the backbone."""
    parts = reduced_backbone_polynomials(m)
    return {
        "absolute_loss_numerator": polynomial_profile(
            parts["loss_numerator"]),
        "variable_numerator": polynomial_profile(
            parts["variable_numerator"]),
        "denominator": polynomial_profile(parts["denominator"]),
        "derivative_numerator": polynomial_profile(
            parts["derivative_numerator"]),
        "critical_squarefree": polynomial_profile(
            parts["derivative_squarefree"]),
        "cancelled_factor_degree": P.degree(parts["cancelled_factor"]),
    }


def _integer_polynomial_preflight(p) -> dict:
    """Cheap root-isolation predictors after primitive integer scaling."""
    integers = P.int_primitive(p)
    degree = len(integers)-1
    height_bits = max((abs(x).bit_length() for x in integers), default=0)
    nonzero = sum(x != 0 for x in integers)
    if degree >= 1:
        lc = abs(integers[-1])
        other = max((abs(x) for x in integers[:-1]), default=0)
        root_bound = Fraction(lc+other, lc)
        root_bound_log2 = _log2_abs(root_bound)
    else:
        root_bound_log2 = None

    # Mignotte's separation bound for a squarefree integer polynomial:
    # sep > sqrt(3)/(n^(n/2+1) ||p||_2^(n-1)).  It is a valid but usually
    # very pessimistic ceiling on the bisection depth.
    if degree >= 2:
        norm_squared = sum(x*x for x in integers)
        log2_norm = 0.5*_log2_int(norm_squared)
        separation_depth_bound = max(
            0.0,
            (degree/2.0+1.0)*math.log2(degree)
            +(degree-1.0)*log2_norm-0.5*math.log2(3.0))
    else:
        separation_depth_bound = 0.0
    return {
        "degree": degree,
        "nonzero": nonzero,
        "primitive_height_bits": height_bits,
        "cauchy_root_bound_log2": root_bound_log2,
        "worst_case_separation_depth_bits": separation_depth_bound,
    }


def native_sturm_profile(p, *, max_coefficient_bits: int = 0,
                         max_chain_coefficients: int = 0,
                         max_prs_steps: int = 0) -> dict | None:
    """Certified C exact-work profile, or None without the native backend."""
    integers = P.int_primitive(p)
    if not integers:
        return {
            "status": 0, "distinct_real_roots": 0,
            "repeated_real_roots": 0, "input_degree": -1,
            "squarefree_degree": -1, "prs_steps": 0,
            "chain_polynomials": 0, "chain_coefficients": 0,
            "peak_coefficient_bits": 0,
        }
    try:
        from . import _native
    except ImportError:
        return None
    return dict(_native.sturm_analyze(
        integers, max_coefficient_bits,
        max_chain_coefficients, max_prs_steps))


def morse_preflight(m: Model, include_sturm_chain: bool = False) -> dict:
    """Predict exact Morse work before root subdivision.

    The cheap fields are available once A, B and N have been constructed.
    ``include_sturm_chain`` performs the first substantial exact-algebra step
    and reports coefficient swell.  The chain is memoized by ``sturm``, so
    this online preflight work is reused by subsequent enumeration.
    """
    backbone = reduced_backbone_polynomials(m)
    targets = {
        "A_positivity": m.alpha,
        "B_roots": m.beta,
        "N_roots": m.N,
        "reduced_u_prime": backbone["derivative_squarefree"],
    }
    profiles = {
        name: _integer_polynomial_preflight(poly)
        for name, poly in targets.items()
    }
    out = {
        "polynomials": profiles,
        "max_degree": max((x["degree"] for x in profiles.values()),
                          default=-1),
        "max_primitive_height_bits": max(
            (x["primitive_height_bits"] for x in profiles.values()),
            default=0),
        "max_worst_case_separation_depth_bits": max(
            (x["worst_case_separation_depth_bits"]
             for x in profiles.values()), default=0.0),
    }
    native_profiles = {
        name: native_sturm_profile(poly)
        for name, poly in targets.items()
    }
    if all(profile is not None for profile in native_profiles.values()):
        out["native_exact"] = native_profiles
        out["native_peak_coefficient_bits"] = max(
            (profile["peak_coefficient_bits"]
             for profile in native_profiles.values()), default=0)
        out["native_total_prs_steps"] = sum(
            profile["prs_steps"] for profile in native_profiles.values())
        out["native_total_chain_coefficients"] = sum(
            profile["chain_coefficients"]
            for profile in native_profiles.values())
    if include_sturm_chain:
        from . import sturm
        chains = {}
        for name, poly in targets.items():
            squarefree = sturm.squarefree_part(poly)
            chain = sturm.sturm_chain(squarefree)
            coefficient_bits = [
                abs(c).bit_length() for q in chain for c in q]
            chains[name] = {
                "length": len(chain),
                "coefficient_count": sum(len(q) for q in chain),
                "peak_coefficient_bits": max(coefficient_bits, default=0),
                "total_coefficient_bits": sum(coefficient_bits),
            }
        out["sturm_chains"] = chains
        out["peak_sturm_coefficient_bits"] = max(
            (x["peak_coefficient_bits"] for x in chains.values()),
            default=0)
        out["total_sturm_coefficient_bits"] = sum(
            x["total_coefficient_bits"] for x in chains.values())
    return out


def arithmetic_profile(m: Model) -> dict:
    """Observation-only finite arithmetic profile; no policy decisions."""
    return {
        "schema": 1,
        "degrees": {
            "f": P.degree(m.f),
            "g": P.degree(m.g),
        },
        "moments": moment_profile(m),
        "polynomials": {
            "f": polynomial_profile(m.f),
            "g": polynomial_profile(m.g),
            "A": polynomial_profile(m.alpha),
            "B": polynomial_profile(m.beta),
            "N": polynomial_profile(m.N),
        },
        "backbone": backbone_profile(m),
        "morse_preflight": morse_preflight(m),
    }


def _separation_ulps(left: float, right: float) -> float:
    gap = right - left
    if not math.isfinite(gap) or gap <= 0.0:
        return 0.0
    scale = max(abs(left), abs(right))
    ulp = float(np.spacing(scale if scale else 1.0))
    return gap / abs(ulp)


def _root_sensitivity(h, b: Fraction) -> Fraction | None:
    """Normwise absolute root sensitivity at a refined root representative.

    For coefficient perturbations bounded at the scale of H, first-order root
    displacement is bounded by eps * sensitivity.  The (1+|b|)^k envelope is
    deliberately conservative at b=0 and avoids declaring an exact zero
    coefficient immune to arithmetic perturbation.
    """
    hp = abs(P.eval_at(P.deriv(h), b))
    if not hp:
        return None
    scale = 1 + abs(b)
    envelope = sum(
        (abs(c)*scale**k for k, c in enumerate(h)), Fraction(0))
    return envelope/hp


def _morse_collision_margin_log2(m: Model, points) -> float | None:
    """Smallest adjacent-root collision margin, measured in binary64 eps.

    If each coefficient of the squarefree numerator of u' is perturbed at a
    normwise binary64 scale, this is the first-order number of such epsilons
    required to close an adjacent exact isolating-interval gap.
    """
    if len(points) < 2:
        return None
    h = reduced_backbone_polynomials(m)["derivative_squarefree"]
    sensitivities = [
        _root_sensitivity(h, point.local.center_interval.mid)
        if point.local is not None else None
        for point in points
    ]
    margins = []
    for i in range(len(points)-1):
        gap = points[i+1].interval.lo-points[i].interval.hi
        si, sj = sensitivities[i], sensitivities[i+1]
        if gap <= 0 or si is None or sj is None:
            margins.append(-math.inf)
        else:
            # eps64 = 2^-52.
            margins.append(_log2_abs(gap/(si+sj))+52.0)
    return min(margins, default=None)


def _evaluation_envelope(p, b: Fraction) -> Fraction:
    scale = 1 + abs(b)
    return sum((abs(c)*scale**k for k, c in enumerate(p)), Fraction(0))


def local_gamma_l1_profile(local) -> dict:
    """Conservative exact-jet gamma bound in coefficient l1 norm.

    For F=grad L centered at a critical point, the stored Taylor coefficient
    of a degree-k monomial is D^k F/k!.  We precondition every homogeneous
    term by DF(0)^-1 exactly and use the coefficient l1 sum as an operator-norm
    upper bound.  This is a conservative normed analogue of Smale's gamma,
    invariant under multiplying the whole loss by a nonzero scalar.
    """
    H = local.exact_hessian
    det = H[0][0]*H[1][1]-H[0][1]*H[1][0]
    if not det:
        return {
            "gamma_l1_upper_log2": math.inf,
            "nonlinear_radius_lower_log2": -math.inf,
            "highest_nonlinear_degree": None,
        }
    terms = {}
    for component, polynomial in enumerate(local.exact_grad):
        for i, row in enumerate(polynomial):
            for j, coefficient in enumerate(row):
                if coefficient and i+j >= 2:
                    terms.setdefault((i, j), [Fraction(0), Fraction(0)])[
                        component] = coefficient
    by_degree = {}
    for (i, j), (c0, c1) in terms.items():
        x0 = (H[1][1]*c0-H[0][1]*c1)/det
        x1 = (-H[1][0]*c0+H[0][0]*c1)/det
        by_degree[i+j] = by_degree.get(i+j, Fraction(0)) \
            + abs(x0)+abs(x1)
    candidates = [
        (_log2_abs(bound)/(degree-1), degree, bound)
        for degree, bound in by_degree.items() if bound
    ]
    if not candidates:
        return {
            "gamma_l1_upper_log2": -math.inf,
            "nonlinear_radius_lower_log2": math.inf,
            "highest_nonlinear_degree": None,
        }
    gamma_log2 = max(x[0] for x in candidates)
    return {
        "gamma_l1_upper_log2": gamma_log2,
        "nonlinear_radius_lower_log2": -gamma_log2,
        "highest_nonlinear_degree": max(by_degree),
        "homogeneous_preconditioned_l1_log2": {
            str(degree): _log2_abs(bound)
            for degree, bound in sorted(by_degree.items()) if bound
        },
    }


def skeleton_profile(m: Model, enumeration) -> dict:
    """Binary64 resolution and near-Morse measurements after enumeration."""
    points = enumeration.points
    separations = [
        _separation_ulps(points[i].b, points[i + 1].b)
        for i in range(len(points) - 1)
    ]
    spectral = []
    hessian_relative = []
    backbone_curvature_logs = []
    transverse_curvature_logs = []
    transverse_resolution_logs = []
    backbone_shear = []
    gamma_logs = []
    gamma_target_logs = []
    critical_profiles = []
    global_margin = []
    for point_index, point in enumerate(points):
        if point.local is None:
            continue
        local = point.local
        point_profile = {
            "a": float(point.a), "b": float(point.b),
            "kind": point.kind, "source": point.source,
        }
        gamma = local_gamma_l1_profile(local)
        gamma_log = gamma["gamma_l1_upper_log2"]
        point_profile.update(gamma)
        if math.isfinite(gamma_log):
            gamma_logs.append(gamma_log)
            neighbor_distance = min((
                math.hypot(point.a-other.a, point.b-other.b)
                for j, other in enumerate(points) if j != point_index),
                default=0.2)
            target_scale = min(0.1, 0.5*neighbor_distance)
            point_profile["neighbor_target_scale"] = target_scale
            if target_scale > 0:
                product_log = gamma_log+math.log2(target_scale)
                gamma_target_logs.append(product_log)
                point_profile["gamma_target_product_log2"] = product_log
        eigenvalues = tuple(abs(x) for x in local.spectral.decimal_eigenvalues)
        large = max(eigenvalues)
        small = min(eigenvalues)
        if large:
            relative = small/large
            hessian_relative.append(float(relative))
            spectral.append(float(relative/Decimal(2)**-52))
            point_profile["hessian_relative_nonsingularity"] = \
                float(relative)
            point_profile["spectral_resolution_margin"] = float(
                relative/Decimal(2)**-52)
        h00 = local.exact_hessian[0][0]
        if h00:
            A0 = h00/2
            transverse_curvature_logs.append(_log2_abs(2*A0))
            point_profile["transverse_curvature_abs_log2"] = \
                _log2_abs(2*A0)
            envelope = _evaluation_envelope(
                m.alpha, local.center_interval.mid)
            if envelope:
                # A / (eps64 * evaluation envelope).
                transverse_resolution_logs.append(
                    _log2_abs(A0/envelope)+52.0)
                point_profile["A_evaluation_margin_log2_eps"] = \
                    _log2_abs(A0/envelope)+52.0
            u2 = local.spectral.determinant/h00
            if u2:
                backbone_curvature_logs.append(_log2_abs(u2))
                point_profile["backbone_curvature_abs_log2"] = \
                    _log2_abs(u2)
            backbone_shear.append(float(abs(
                local.exact_hessian[0][1]/h00)))
            point_profile["backbone_shear_abs"] = float(abs(
                local.exact_hessian[0][1]/h00))
        for stub in point.stubs:
            cert = dict(stub.certificates)
            if "global_resolution_margin" in cert:
                global_margin.append(float(cert["global_resolution_margin"]))
        critical_profiles.append(point_profile)
    return {
        "critical_count": len(points),
        "morse_exact": bool(enumeration.morse),
        "alpha_positive_exact": bool(enumeration.psi_positive),
        "alternates_exact": bool(enumeration.alternates),
        "critical_coordinates_binary64_distinct": (
            len({point.b for point in points}) == len(points)),
        "min_adjacent_root_separation_ulps": min(separations, default=None),
        "morse_root_collision_margin_log2_eps":
            _morse_collision_margin_log2(m, points),
        "max_abs_a": max((abs(point.a) for point in points), default=0.0),
        "max_abs_b": max((abs(point.b) for point in points), default=0.0),
        "min_backbone_curvature_abs_log2": min(
            backbone_curvature_logs, default=None),
        "max_backbone_curvature_abs_log2": max(
            backbone_curvature_logs, default=None),
        "min_transverse_curvature_abs_log2": min(
            transverse_curvature_logs, default=None),
        "max_transverse_curvature_abs_log2": max(
            transverse_curvature_logs, default=None),
        "min_A_evaluation_margin_log2_eps": min(
            transverse_resolution_logs, default=None),
        "max_backbone_shear_abs": max(backbone_shear, default=None),
        "max_gamma_l1_upper_log2": max(gamma_logs, default=None),
        "max_gamma_target_product_log2": max(
            gamma_target_logs, default=None),
        "min_hessian_relative_nonsingularity": min(
            hessian_relative, default=None),
        "min_spectral_resolution_margin": min(spectral, default=None),
        "min_global_resolution_margin": min(global_margin, default=None),
        "critical_profiles": critical_profiles,
    }
