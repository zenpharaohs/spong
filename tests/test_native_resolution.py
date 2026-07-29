"""Parity gates for the frontend-independent C resolution ABI."""

import random

from spong import _native
from spong import resolution


def _profile(rng):
    hessian_loss = rng.uniform(0.0, 90.0)
    return {
        "alpha_positive_exact": True,
        "critical_coordinates_binary64_distinct": rng.choice([True, False]),
        "morse_root_collision_margin_log2_eps": rng.uniform(-30.0, 60.0),
        "min_hessian_relative_nonsingularity": 2.0**(-hessian_loss),
        "max_gamma_target_product_log2": rng.uniform(-5.0, 100.0),
    }


def test_native_policy_matches_python_oracle_across_margin_combinations():
    rng = random.Random(20260729)
    for _ in range(400):
        profile = _profile(rng)
        policy = resolution.ResolutionPolicy(
            min_root_collision_margin_log2_eps=(
                rng.uniform(-10.0, 40.0) if rng.random() < 0.8 else None),
            max_hessian_condition_loss_bits=(
                rng.uniform(10.0, 70.0) if rng.random() < 0.8 else None),
            max_gamma_target_product_log2=(
                rng.uniform(10.0, 80.0) if rng.random() < 0.8 else None),
            require_distinct_binary64_coordinates=rng.choice([True, False]),
        )
        assert resolution._policy_refusals(
            profile, policy) == resolution._policy_refusals_python(
                profile, policy)


def test_native_exact_and_geometry_terminal_states_are_stable():
    assert _native.SPONG_ABI_VERSION == 1
    none = 0.0
    exact_non_morse = _native.resolution_preflight(
        False, True, True, False, none, False, none, False, none,
        0, none, none, none)
    assert exact_non_morse == (
        _native.SPONG_CERTIFIED_NON_MORSE,
        _native.SPONG_REASON_EXACT_NON_MORSE,
        1 << (_native.SPONG_REASON_EXACT_NON_MORSE-1))

    assert _native.resolution_finalize(True, False) == (
        _native.SPONG_CERTIFIED_PORTRAIT,
        _native.SPONG_REASON_NONE, 0)
    assert _native.resolution_finalize(False, True) == (
        _native.SPONG_MORSE_NUMERICALLY_UNRESOLVED,
        _native.SPONG_REASON_BRANCH_ABORT,
        1 << (_native.SPONG_REASON_BRANCH_ABORT-1))
    assert _native.resolution_finalize(False, False) == (
        _native.SPONG_MORSE_NUMERICALLY_UNRESOLVED,
        _native.SPONG_REASON_TOPOLOGY_UNRESOLVED,
        1 << (_native.SPONG_REASON_TOPOLOGY_UNRESOLVED-1))
