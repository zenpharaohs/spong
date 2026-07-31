"""Phase-6 sanity: the optimizer demos are functional consumers.

The showcase script itself is demo-ware (runs the 40 s tricky portrait)
and is exercised manually; here the optimizers run on the cheap d=2
oracle model.
"""

from fractions import Fraction

import numpy as np
import pytest

from demos import initializers
from demos import adam_flow
from demos import hybrid_saddle_connection
from demos import optimizer_moustaches
from demos import optimizers as opt
from demos import saddle_connections
from demos import thompson
from spong import model, sturm


def test_batch_gradient_matches_mean_field(d2):
    """E[batch gradient] -> model gradient (the moments story)."""
    m, _ = d2
    g = opt.BatchGradient([1, 1, 1], [1, 1, 1], batch_size=20000,
                          rng=np.random.default_rng(0))
    a, b = 1.3, 0.4
    est = np.mean([g(a, b) for _ in range(20)], axis=0)
    exact = m.gradL(a, b)
    assert np.allclose(est, exact, rtol=0.05, atol=0.02)


def test_lbfgs_reaches_a_minimum(d2):
    m, e = d2
    tr = opt.run_lbfgs(m, (1.4, 0.2), n_steps=100)
    cp, dist = opt.nearest_critical(e, tr[-1])
    assert cp.kind == "min" and dist < 1e-3
    assert m.L(*tr[-1]) <= m.L(1.4, 0.2)


def test_adam_descends(d2):
    m, _ = d2
    g = opt.BatchGradient([1, 1, 1], [1, 1, 1], batch_size=64,
                          rng=np.random.default_rng(1))
    tr = opt.run_adam(g, (1.4, 0.2), lr=0.02, n_steps=2000)
    assert np.all(np.isfinite(tr))
    assert m.L(*tr[-1]) < m.L(1.4, 0.2)


def test_sgd_finite_on_mild_terrain(d2):
    m, _ = d2
    g = opt.BatchGradient([1, 1, 1], [1, 1, 1], batch_size=64,
                          rng=np.random.default_rng(2))
    tr = opt.run_sgd(g, (1.4, 0.2), lr=5e-3, n_steps=2000)
    assert np.all(np.isfinite(tr))
    assert m.L(*tr[-1]) < m.L(1.4, 0.2)


def test_initialization_designs_are_reproducible_and_in_box():
    box = (-2.0, 3.0, -4.0, 5.0)
    halton = initializers.low_discrepancy(100, box)
    blue1 = initializers.blue_noise(100, box, seed=17)
    blue2 = initializers.blue_noise(100, box, seed=17)
    assert halton.shape == blue1.shape == (100, 2)
    assert np.array_equal(blue1, blue2)
    for points in (halton, blue1):
        assert np.all((box[0] <= points[:, 0]) & (points[:, 0] <= box[1]))
        assert np.all((box[2] <= points[:, 1]) & (points[:, 1] <= box[3]))
        assert len(np.unique(points, axis=0)) == 100


def test_resumable_sgd_matches_one_shot_run():
    gradient = lambda a, b: np.array([2*a, 3*b])
    schedule = opt.cosine_schedule(0.01, 20)
    whole = opt.run_sgd(
        gradient, (1.0, -2.0), schedule, 20,
        momentum=0.9, nesterov=True)
    state = opt.SGDState(
        np.array([1.0, -2.0]), schedule, momentum=0.9, nesterov=True)
    first = opt.run_state(state, gradient, 7)
    second = opt.run_state(state, gradient, 13)
    joined = np.vstack((first, second[1:]))
    assert np.array_equal(joined, whole)


def test_vector_muon_surrogate_normalizes_the_2x1_update():
    state = opt.VectorMuonState(
        np.array([1.0, 1.0]), lr=0.2, momentum=0.0, nesterov=False)
    before = state.z.copy()
    state.step(np.array([3.0, 4.0]))
    assert np.hypot(*(state.z-before)) == pytest.approx(0.2)


def test_muon_auxiliary_adamw_fallback_is_available():
    gradient = lambda a, b: np.array([a, b])
    trajectory = opt.run_adamw(
        gradient, (1.0, -2.0), lr=3e-4, n_steps=5,
        b1=0.9, b2=0.95, eps=1e-10, weight_decay=0.0)
    assert trajectory.shape == (6, 2)
    assert np.all(np.isfinite(trajectory))


def test_muon_fallback_is_not_a_default_duplicate_panel():
    assert optimizer_moustaches.DEFAULT_METHODS == (
        "sgd", "sgd-momentum", "adam")
    assert "muon-adamw" in optimizer_moustaches.METHODS


class _DeterministicPosterior:
    def __init__(self, draws):
        self.draws = iter(draws)
        self.updates = []

    def draw_all(self):
        return np.asarray(next(self.draws), dtype=float)

    def update(self, arm, observation):
        self.updates.append((arm, observation))


def test_thompson_allocates_immediately_without_capture_information():
    states = [
        opt.SGDState(np.array([1.0, 0.0]), lr=0.1),
        opt.SGDState(np.array([2.0, 0.0]), lr=0.1),
    ]
    gradients = [
        lambda a, b: np.array([a, b]),
        lambda a, b: np.array([a, b]),
    ]
    posterior = _DeterministicPosterior(([0.8, 0.2], [0.1, 0.9]))
    result = thompson.allocate(
        states, gradients, lambda a, b: a*a+b*b,
        posterior, rounds=4, chunk_steps=1)
    # One forced pull per arm, then posterior choices 1 and 0.
    assert np.array_equal(result.choices, [0, 1, 1, 0])
    assert np.array_equal(result.allocations, [2, 2])
    assert [arm for arm, _ in posterior.updates] == [0, 1, 1, 0]
    assert all(0 <= value <= 1 for _, value in posterior.updates)


def test_saddle_connection_coefficients_are_bounded_dyadics():
    q = saddle_connections.quantized_unit_coefficients([3.0, 4.0], bits=12)
    assert all(x.denominator <= 2**12 for x in q)
    assert np.hypot(*map(float, q)) == pytest.approx(1.0, abs=5e-4)


def test_saddle_connection_affine_segment_is_exact_and_continuous():
    left = (Fraction(1, 4), Fraction(-1, 2))
    right = (Fraction(3, 4), Fraction(1, 2))
    midpoint = saddle_connections.affine_coefficients(
        left, right, Fraction(1, 2))
    assert midpoint == (Fraction(1, 2), Fraction(0))


def test_hybrid_minimum_norm_rows_avoids_normal_equations():
    rows = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    rhs = np.array([2.0, -1.0])
    solution = hybrid_saddle_connection.minimum_norm_rows(rows, rhs)
    assert np.allclose(rows @ solution, rhs, rtol=0.0, atol=1e-14)
    # The null vector (1,1,-1) is orthogonal to the minimum-norm solution.
    assert np.dot(solution, [1.0, 1.0, -1.0]) == pytest.approx(0.0)


def test_saddle_connection_strip_contracts_with_degree():
    assert saddle_connections.central_strip_width(3) \
        > saddle_connections.central_strip_width(11) > 1.0


def test_linear_activation_cannot_supply_saddle_connection():
    f = saddle_connections.quantized_unit_coefficients([1.0, -0.3])
    g = saddle_connections.quantized_unit_coefficients([0.4, 1.0])
    result = saddle_connections.evaluate(f, g)
    assert not result.valid
    assert result.reason in {
        "fewer_than_two_saddles", "no_unequal_saddle_levels"}


def test_stable_unstable_point_cloud_nearness():
    first = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    second = np.array([[0.0, 2.0], [1.0, 0.25], [2.0, 2.0]])
    distance, i, j = saddle_connections.point_cloud_nearness(first, second)
    assert distance == pytest.approx(0.25)
    assert (i, j) == (1, 1)


def test_level_section_shooting_residual_has_sign_and_zero(d2):
    m, _ = d2
    level = m.L(1.0, 0.0)
    first = np.array([[0.0, 0.0], [2.0, 0.0]])
    crossing = saddle_connections.level_crossing(m, first, level)
    assert crossing is not None
    assert m.L(*crossing) == pytest.approx(level, abs=1e-12)
    zero = saddle_connections.shooting_residual(
        m, first, first, level + 1.0, level - 1.0, 1.0)
    assert zero is not None
    assert zero[1] == pytest.approx(0.0, abs=1e-14)


def test_common_level_distance_rejects_close_unequal_loss_vertices():
    class LinearLoss:
        @staticmethod
        def L(a, b):
            return a

    m = LinearLoss()
    first = np.array([[2.0, 0.0], [0.0, 0.0]])
    second = np.array([[0.0, 0.1], [2.0, 10.0]])
    result = saddle_connections.common_level_separation(
        m, first, second, 2.0, 0.0)
    assert result is not None
    assert result[3] > saddle_connections.point_cloud_nearness(
        first, second)[0]


def test_level_section_rejects_distant_tangent_projection_false_positive(d2):
    m, _ = d2
    level = m.L(1.0, 0.0)
    first = np.array([[0.0, 0.0], [2.0, 0.0]])
    # This chord is arranged to hit the same level far away in the gradient
    # direction.  Its tangent projection can be tiny, but it is not in the
    # local level-set chart of the first crossing.
    second = np.array([[0.0, -4.0], [2.0, -4.0]])
    assert saddle_connections.shooting_residual(
        m, first, second, level+1.0, level-1.0, scale=1.0) is None


def test_spsa_coefficient_step_is_clipped():
    f = saddle_connections.quantized_unit_coefficients([1, 2, 3])
    g = saddle_connections.quantized_unit_coefficients([3, 2, 1])
    delta = np.ones(3)
    f1, g1 = saddle_connections.spsa_update(
        f, g, 10.0, 0.0, delta, delta,
        epsilon=1e-6, learning_rate=1.0, max_step=1e-3)
    displacement = np.r_[
        np.asarray(list(map(float, f1))) - np.asarray(list(map(float, f))),
        np.asarray(list(map(float, g1))) - np.asarray(list(map(float, g)))]
    # Renormalization and dyadic rounding can add a small radial component.
    assert np.hypot.reduce(displacement) < 2e-3


@pytest.mark.parametrize("value, expected", [
    (0.0, 0.0), (1.0, 0.5), (3.0, 0.75),
    (float("inf"), 1.0), (-1e-15, 0.0)])
def test_transformed_loss(value, expected):
    assert thompson.transformed_loss(value) == pytest.approx(expected)


def test_empirical_adam_innovation_matches_empirical_loss_gradient():
    inputs, rationals = adam_flow.empirical_uniform_grid(16)
    moments = adam_flow.empirical_moments(rationals, 5)
    m = model.build([1, -2, .5], [.3, 1, -.2], moments)
    a, b = 1.2, -.7
    innovation = adam_flow.negative_sample_gradient(
        [1, -2, .5], [.3, 1, -.2],
        np.array([a]), np.array([b]), inputs)[0]
    assert np.allclose(np.mean(innovation, axis=0), -m.gradL(a, b),
                       rtol=2e-14, atol=2e-14)


def test_zero_residual_is_an_exact_stochastic_adam_equilibrium():
    inputs, _ = adam_flow.empirical_uniform_grid(32)
    innovation = adam_flow.negative_sample_gradient(
        [1, -2, .5], [1, -2, .5],
        np.array([1.0]), np.array([1.0]), inputs)[0]
    assert np.array_equal(innovation, np.zeros_like(innovation))


def test_adam_field_grid_is_reproducible_and_reports_chain_error():
    inputs, _ = adam_flow.empirical_uniform_grid(16)
    kwargs = dict(
        f=[1, 0, 1], g=[1, 1], inputs=inputs,
        view=(-1, 1, -1, 1), grid=5, batch_size=4,
        burn_in=20, samples=30, chains=2, seed=9,
        alpha=.5, beta=.8)
    first = adam_flow.estimate_grid(**kwargs)
    second = adam_flow.estimate_grid(**kwargs)
    assert np.array_equal(first.field, second.field)
    assert first.field.shape == (5, 5, 2)
    assert np.all(np.isfinite(first.field))
    assert first.diagnostics["chain_rms_difference"] >= 0
