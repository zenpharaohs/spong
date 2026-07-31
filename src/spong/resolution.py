"""Total public resolution contract for a SPONG model.

Every valid call terminates in one of three states:

* the loss is certified non-Morse;
* the loss is Morse, but the requested arithmetic/policy did not certify a
  portrait; or
* a certified portrait is returned.

The exact Morse decision is upstream of every binary64 policy.  Numerical
admission thresholds are deliberately supplied by ``ResolutionPolicy`` rather
than hidden in the measurements: calibration may tighten them without changing
the meaning of the terminal states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from . import portrait, qualification, sturm
from .model import Model


class ResolutionStatus(str, Enum):
    CERTIFIED_NON_MORSE = "certified_non_morse"
    MORSE_NUMERICALLY_UNRESOLVED = "morse_numerically_unresolved"
    CERTIFIED_PORTRAIT = "certified_portrait"


class ResolutionReason(str, Enum):
    NONE = "none"
    EXACT_NON_MORSE = "exact_non_morse"
    MODEL_HYPOTHESIS = "model_hypothesis"
    ROOT_COLLISION_MARGIN = "root_collision_margin"
    HESSIAN_RESOLUTION = "hessian_resolution"
    LOCAL_NONLINEARITY = "local_nonlinearity"
    BINARY64_COORDINATE_COLLISION = "binary64_coordinate_collision"
    ARITHMETIC_FAILURE = "arithmetic_failure"
    BRANCH_ABORT = "branch_abort"
    TOPOLOGY_UNRESOLVED = "topology_unresolved"


@dataclass(frozen=True)
class ResolutionPolicy:
    """Requested arithmetic envelope and geometry budget.

    ``None`` disables a prospective threshold.  The default policy observes
    all margins and lets the a-posteriori geometry certificate decide.  A
    released frontend can freeze calibrated values without changing
    ``resolve`` or its three terminal outcomes.
    """

    min_root_collision_margin_log2_eps: float | None = None
    max_hessian_condition_loss_bits: float | None = None
    max_gamma_target_product_log2: float | None = None
    require_distinct_binary64_coordinates: bool = True
    max_geometry_level: int = 2


@dataclass(frozen=True)
class Resolution:
    status: ResolutionStatus
    reason: ResolutionReason
    message: str
    exact_morse: bool | None
    enumeration: sturm.Enumeration | None = None
    arithmetic_profile: dict[str, Any] | None = None
    skeleton_profile: dict[str, Any] | None = None
    portrait: portrait.Portrait | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return self.status in (
            ResolutionStatus.CERTIFIED_NON_MORSE,
            ResolutionStatus.CERTIFIED_PORTRAIT,
        )


def _policy_refusals_python(profile: dict, policy: ResolutionPolicy) -> list[
        tuple[ResolutionReason, str]]:
    refusals = []
    if (policy.require_distinct_binary64_coordinates
            and not profile["critical_coordinates_binary64_distinct"]):
        refusals.append((
            ResolutionReason.BINARY64_COORDINATE_COLLISION,
            "distinct exact critical coordinates collide in binary64"))

    margin = profile["morse_root_collision_margin_log2_eps"]
    floor = policy.min_root_collision_margin_log2_eps
    if floor is not None and margin is not None and margin < floor:
        refusals.append((
            ResolutionReason.ROOT_COLLISION_MARGIN,
            f"root-collision margin {margin:.3g} bits is below {floor:.3g}"))

    relative = profile["min_hessian_relative_nonsingularity"]
    ceiling = policy.max_hessian_condition_loss_bits
    if ceiling is not None and relative is not None and relative > 0.0:
        loss = -math.log2(relative)
        if loss > ceiling:
            refusals.append((
                ResolutionReason.HESSIAN_RESOLUTION,
                f"Hessian condition loss {loss:.3g} bits exceeds "
                f"{ceiling:.3g}"))

    gamma = profile["max_gamma_target_product_log2"]
    gamma_ceiling = policy.max_gamma_target_product_log2
    if (gamma_ceiling is not None and gamma is not None
            and gamma > gamma_ceiling):
        refusals.append((
            ResolutionReason.LOCAL_NONLINEARITY,
            f"scaled local gamma {gamma:.3g} bits exceeds "
            f"{gamma_ceiling:.3g}"))
    return refusals


def _policy_refusals(profile: dict, policy: ResolutionPolicy) -> list[
        tuple[ResolutionReason, str]]:
    """Production C policy decision with the Python implementation as oracle."""
    oracle = _policy_refusals_python(profile, policy)
    try:
        from . import _native
    except ImportError:
        return oracle

    enabled = 0
    if policy.min_root_collision_margin_log2_eps is not None:
        enabled |= _native.SPONG_POLICY_ROOT_COLLISION
    if policy.max_hessian_condition_loss_bits is not None:
        enabled |= _native.SPONG_POLICY_HESSIAN_CONDITION
    if policy.max_gamma_target_product_log2 is not None:
        enabled |= _native.SPONG_POLICY_LOCAL_GAMMA
    if policy.require_distinct_binary64_coordinates:
        enabled |= _native.SPONG_POLICY_DISTINCT_BINARY64

    margin = profile["morse_root_collision_margin_log2_eps"]
    relative = profile["min_hessian_relative_nonsingularity"]
    gamma = profile["max_gamma_target_product_log2"]
    status, primary, mask = _native.resolution_preflight(
        True, bool(profile["alpha_positive_exact"]),
        bool(profile["critical_coordinates_binary64_distinct"]),
        margin is not None, margin or 0.0,
        relative is not None, relative or 0.0,
        gamma is not None, gamma or 0.0,
        enabled,
        policy.min_root_collision_margin_log2_eps or 0.0,
        policy.max_hessian_condition_loss_bits or 0.0,
        policy.max_gamma_target_product_log2 or 0.0)
    if status == _native.SPONG_RESOLUTION_PROCEED:
        return []
    native_reasons = {
        ResolutionReason.ROOT_COLLISION_MARGIN:
            _native.SPONG_REASON_ROOT_COLLISION_MARGIN,
        ResolutionReason.HESSIAN_RESOLUTION:
            _native.SPONG_REASON_HESSIAN_RESOLUTION,
        ResolutionReason.LOCAL_NONLINEARITY:
            _native.SPONG_REASON_LOCAL_NONLINEARITY,
        ResolutionReason.BINARY64_COORDINATE_COLLISION:
            _native.SPONG_REASON_BINARY64_COORDINATE_COLLISION,
    }
    selected = [
        item for item in oracle
        if mask & (1 << (native_reasons[item[0]]-1))
    ]
    if not selected or native_reasons[selected[0][0]] != primary:
        raise RuntimeError("native/Python resolution policy disagreement")
    return selected


def _geometry_terminal(topology: dict) -> tuple[
        ResolutionStatus, ResolutionReason]:
    certified = topology["status"] == "certified"
    branch_abort = topology.get("resolution_reason") == "branch_abort"
    try:
        from . import _native
    except ImportError:
        if certified:
            return ResolutionStatus.CERTIFIED_PORTRAIT, ResolutionReason.NONE
        return (
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
            ResolutionReason.BRANCH_ABORT if branch_abort
            else ResolutionReason.TOPOLOGY_UNRESOLVED)
    status, reason, _mask = _native.resolution_finalize(
        certified, branch_abort)
    status_map = {
        _native.SPONG_CERTIFIED_PORTRAIT:
            ResolutionStatus.CERTIFIED_PORTRAIT,
        _native.SPONG_MORSE_NUMERICALLY_UNRESOLVED:
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
    }
    reason_map = {
        _native.SPONG_REASON_NONE: ResolutionReason.NONE,
        _native.SPONG_REASON_BRANCH_ABORT: ResolutionReason.BRANCH_ABORT,
        _native.SPONG_REASON_TOPOLOGY_UNRESOLVED:
            ResolutionReason.TOPOLOGY_UNRESOLVED,
    }
    return status_map[status], reason_map[reason]


def _exact_terminal(enumeration: sturm.Enumeration) -> tuple[
        ResolutionStatus | None, ResolutionReason]:
    """Shared C state-machine decision immediately after exact enumeration."""
    try:
        from . import _native
    except ImportError:
        if not enumeration.psi_positive:
            return (ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
                    ResolutionReason.MODEL_HYPOTHESIS)
        if not enumeration.morse:
            return (ResolutionStatus.CERTIFIED_NON_MORSE,
                    ResolutionReason.EXACT_NON_MORSE)
        return None, ResolutionReason.NONE
    status, reason, _mask = _native.resolution_preflight(
        bool(enumeration.morse), bool(enumeration.psi_positive), True,
        False, 0.0, False, 0.0, False, 0.0,
        0, 0.0, 0.0, 0.0)
    status_map = {
        _native.SPONG_RESOLUTION_PROCEED: None,
        _native.SPONG_CERTIFIED_NON_MORSE:
            ResolutionStatus.CERTIFIED_NON_MORSE,
        _native.SPONG_MORSE_NUMERICALLY_UNRESOLVED:
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
    }
    reason_map = {
        _native.SPONG_REASON_NONE: ResolutionReason.NONE,
        _native.SPONG_REASON_EXACT_NON_MORSE:
            ResolutionReason.EXACT_NON_MORSE,
        _native.SPONG_REASON_MODEL_HYPOTHESIS:
            ResolutionReason.MODEL_HYPOTHESIS,
    }
    return status_map[status], reason_map[reason]


def resolve(m: Model, *, view=None,
            policy: ResolutionPolicy | None = None) -> Resolution:
    """Return exactly one terminal resolution for ``m``.

    Internal programming errors are intentionally not relabeled as numerical
    refusals.  Expected finite-arithmetic failures are caught and returned as
    ``MORSE_NUMERICALLY_UNRESOLVED``; bugs remain visible to tests.
    """
    policy = policy or ResolutionPolicy()
    arithmetic = qualification.arithmetic_profile(m)
    try:
        enumeration = sturm.enumerate_critical_points(m)
    except (ArithmeticError, FloatingPointError, OverflowError) as exc:
        return Resolution(
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
            ResolutionReason.ARITHMETIC_FAILURE,
            f"critical-point analysis exceeded the requested arithmetic: "
            f"{exc}",
            None,
            arithmetic_profile=arithmetic,
            diagnostics=(type(exc).__name__,),
        )

    exact_status, exact_reason = _exact_terminal(enumeration)
    if exact_reason is ResolutionReason.MODEL_HYPOTHESIS:
        return Resolution(
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
            ResolutionReason.MODEL_HYPOTHESIS,
            "the exact positivity hypothesis A(b)>0 is not certified",
            enumeration.morse,
            enumeration=enumeration,
            arithmetic_profile=arithmetic,
        )

    if exact_status is ResolutionStatus.CERTIFIED_NON_MORSE:
        return Resolution(
            ResolutionStatus.CERTIFIED_NON_MORSE,
            ResolutionReason.EXACT_NON_MORSE,
            "the exact critical set is not Morse",
            False,
            enumeration=enumeration,
            arithmetic_profile=arithmetic,
        )

    skeleton = qualification.skeleton_profile(m, enumeration)
    refusals = _policy_refusals(skeleton, policy)
    if refusals:
        return Resolution(
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
            refusals[0][0],
            "Morse exactly, but outside the requested numerical envelope",
            True,
            enumeration=enumeration,
            arithmetic_profile=arithmetic,
            skeleton_profile=skeleton,
            diagnostics=tuple(message for _, message in refusals),
        )

    try:
        computed = portrait.certified_compute(
            m, view=view,
            max_geometry_level=policy.max_geometry_level,
            _enumeration=enumeration)
    except (ArithmeticError, FloatingPointError, OverflowError) as exc:
        return Resolution(
            ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
            ResolutionReason.ARITHMETIC_FAILURE,
            f"Morse exactly, but geometry exceeded the requested arithmetic: "
            f"{exc}",
            True,
            enumeration=enumeration,
            arithmetic_profile=arithmetic,
            skeleton_profile=skeleton,
            diagnostics=(type(exc).__name__,),
        )

    topology = computed.ledger["topology"]
    terminal_status, terminal_reason = _geometry_terminal(topology)
    if terminal_status is ResolutionStatus.CERTIFIED_PORTRAIT:
        return Resolution(
            terminal_status,
            terminal_reason,
            "certified Morse phase portrait",
            True,
            enumeration=computed.enumeration,
            arithmetic_profile=arithmetic,
            skeleton_profile=qualification.skeleton_profile(
                m, computed.enumeration),
            portrait=computed,
        )

    diagnostics = [
        f"topology_status={topology['status']}",
        f"ambiguous_contacts={topology.get('ambiguous_count', 0)}",
        f"forbidden_intersections={topology.get('forbidden_count', 0)}",
    ]
    if topology.get("resolution_reason"):
        diagnostics.append(
            f"resolution_reason={topology['resolution_reason']}")
    return Resolution(
        ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED,
        terminal_reason,
        "Morse exactly, but the geometry certificate did not close",
        True,
        enumeration=computed.enumeration,
        arithmetic_profile=arithmetic,
        skeleton_profile=qualification.skeleton_profile(
            m, computed.enumeration),
        portrait=computed,
        diagnostics=tuple(diagnostics),
    )
