"""Parity of the fused demo Adam kernel with the Python optimizer stream."""

import numpy as np
import pytest

from demos import native_adam
from demos import optimizers as opt


F = [3, 3]
G = [4, -3, 0, 3, -1, 2, -1, -3, 3, -1, 1, -3, -1, 1]


@pytest.mark.parametrize("distribution,samples", [
    ("uniform01", None), ("normal01", None),
    ("absolute_x", None), ("empirical", [-1.0, 0.25, 1.5]),
])
def test_many_batches_preserve_the_incremental_rng_stream(
        distribution, samples):
    options = dict(distribution=distribution, samples=samples)
    incremental = opt.BatchGradient(
        F, G, 7, np.random.default_rng(17), **options)
    grouped = opt.BatchGradient(
        F, G, 7, np.random.default_rng(17), **options)
    expected = np.stack([incremental._draw() for _ in range(37)])
    np.testing.assert_array_equal(grouped._draw_many(37), expected)
    np.testing.assert_array_equal(grouped._draw(), incremental._draw())


@pytest.mark.parametrize("distribution,samples", [
    ("uniform01", None), ("normal01", None),
    ("absolute_x", None), ("empirical", [-1.0, 0.25, 1.5]),
])
@pytest.mark.parametrize("schedule", ["constant", "inverse-sqrt", "cosine"])
@pytest.mark.parametrize("b2,eps", [(0.999, 1e-8), (0.95, 1e-10)])
def test_native_adam_chunk_matches_python_iterates_and_state(
        distribution, samples, schedule, b2, eps, monkeypatch):
    if not native_adam.available():
        pytest.skip("native Adam C compiler unavailable")
    monkeypatch.setattr(native_adam, "SAMPLES_PER_BLOCK", 35)
    lr = (0.001 if schedule == "constant" else
          opt.inverse_sqrt_schedule(0.001, 10) if schedule == "inverse-sqrt"
          else opt.cosine_schedule(0.001, 80))
    options = dict(distribution=distribution, samples=samples)
    python_gradient = opt.BatchGradient(
        F, G, 7, np.random.default_rng(17), **options)
    native_gradient = opt.BatchGradient(
        F, G, 7, np.random.default_rng(17), **options)
    python_state = opt.AdamState(np.array([0.3, 0.4]), lr, b2=b2, eps=eps)
    native_state = opt.AdamState(np.array([0.3, 0.4]), lr, b2=b2, eps=eps)
    expected = []
    for _ in range(37):
        point, failure = opt.checked_step(python_state, python_gradient)
        assert failure is None
        expected.append(point.copy())
    actual, reason = native_adam.run_chunk(native_state, native_gradient, 37)
    assert reason is None
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(native_state.z, python_state.z,
                               rtol=0, atol=2e-15)
    np.testing.assert_allclose(native_state.first, python_state.first,
                               rtol=0, atol=2e-15)
    np.testing.assert_allclose(native_state.second, python_state.second,
                               rtol=0, atol=2e-15)
    assert native_state.t == python_state.t
    np.testing.assert_array_equal(native_gradient._draw(),
                                  python_gradient._draw())


def test_native_adam_stops_at_last_finite_iterate():
    if not native_adam.available():
        pytest.skip("native Adam C compiler unavailable")
    options = dict(distribution="empirical", samples=[2.0])
    py_grad = opt.BatchGradient([1.0], [0.0, 0.0, 0.0, 1.0], 4,
                                np.random.default_rng(41), **options)
    c_grad = opt.BatchGradient([1.0], [0.0, 0.0, 0.0, 1.0], 4,
                               np.random.default_rng(41), **options)
    py_state = opt.AdamState(np.array([1.0, 1e200]), 0.001)
    c_state = opt.AdamState(np.array([1.0, 1e200]), 0.001)
    point, expected_reason = opt.checked_step(py_state, py_grad)
    points, reason = native_adam.run_chunk(c_state, c_grad, 10)
    assert reason == expected_reason == "nonfinite_gradient"
    assert len(points) == 0
    np.testing.assert_array_equal(c_state.z, point)
    assert c_state.t == py_state.t == 0


def test_native_adam_can_fall_back_without_advancing_rng(monkeypatch):
    monkeypatch.setenv("SPONG_DEMO_ADAM_NATIVE", "0")
    gradient = opt.BatchGradient(
        F, G, 7, np.random.default_rng(19))
    reference = opt.BatchGradient(
        F, G, 7, np.random.default_rng(19))
    state = opt.AdamState(np.array([0.3, 0.4]), 0.001)
    assert native_adam.run_chunk(state, gradient, 10) is None
    assert state.t == 0
    np.testing.assert_array_equal(gradient._draw(), reference._draw())
