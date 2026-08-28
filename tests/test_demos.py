"""Phase-6 sanity: the optimizer demos are functional consumers.

The showcase script itself is demo-ware (runs the 40 s tricky portrait)
and is exercised manually; here the optimizers run on the cheap d=2
oracle model.
"""

from fractions import Fraction
import warnings

import numpy as np
import pytest

from demos import initializers
from demos import adam_flow
from demos import batch_moment_portraits
from demos import cb_sampler
from demos import hybrid_saddle_connection
from demos import l2_l4_phase_portraits
from demos import optimizer_moustaches
from demos import optimizers as opt
from demos import saddle_connections
from demos import saddle_connection_comparison
from demos import saddle_connection_triptych
from demos import thompson
from demos import thompson_moustaches
from spong import charts, model, portrait, sturm, zoo


def test_batch_gradient_matches_mean_field(d2):
    """E[batch gradient] -> model gradient (the moments story)."""
    m, _ = d2
    g = opt.BatchGradient([1, 1, 1], [1, 1, 1], batch_size=20000,
                          rng=np.random.default_rng(0))
    a, b = 1.3, 0.4
    est = np.mean([g(a, b) for _ in range(20)], axis=0)
    exact = m.gradL(a, b)
    assert np.allclose(est, exact, rtol=0.05, atol=0.02)


def test_batch_gradient_supports_normal_portrait_law():
    m = model.build(
        [1, 1, 1], [1, 1, 1], model.moments_normal01(5))
    gradient = opt.BatchGradient(
        [1, 1, 1], [1, 1, 1], batch_size=40000,
        rng=np.random.default_rng(31), distribution="normal01")
    estimate = np.mean([gradient(0.7, 0.2) for _ in range(8)], axis=0)
    assert np.allclose(estimate, m.gradL(0.7, 0.2), rtol=0.03, atol=0.03)


def test_batch_gradient_resamples_empirical_portrait_support():
    samples = np.array([-1.25, 0.0, 0.75])
    gradient = opt.BatchGradient(
        [1, -0.5], [0.25, 1], batch_size=200,
        rng=np.random.default_rng(37), distribution="empirical",
        samples=samples)
    assert set(np.unique(gradient._draw())) <= set(samples)


def test_cb_library_resolver_honors_explicit_override(tmp_path):
    library = tmp_path / "libcb_core.test"
    library.touch()
    assert cb_sampler.resolve_library(library) == library.resolve()


def test_cb_library_resolver_can_build_discovered_sibling_source(
        tmp_path, monkeypatch):
    source = tmp_path / "cb_core.c"
    source.write_text("/* exact sampler test source */\n")
    built = tmp_path / "libcb_core.test"
    monkeypatch.delenv("CB_CORE_LIBRARY", raising=False)
    monkeypatch.delenv("CB_CORE_SOURCE", raising=False)
    monkeypatch.setattr(cb_sampler, "_source_candidates", lambda: iter([source]))
    monkeypatch.setattr(cb_sampler, "_built_candidates", lambda _source: iter(()))
    monkeypatch.setattr(cb_sampler, "_build_shared", lambda found: built)
    assert cb_sampler.resolve_library(auto_build=True) == built


@pytest.mark.parametrize("observation", [-1e-12, 1.0 + 1e-12, np.nan, np.inf])
def test_cb_bank_rejects_observations_outside_closed_unit_interval(observation):
    bank = object.__new__(cb_sampler.ContinuousBernoulliBank)
    with pytest.raises(ValueError, match="closed interval"):
        bank.update(0, observation)


def test_cb_bank_closed_endpoints_remain_direct_point_masses():
    library = cb_sampler.resolve_library(auto_build=True)
    with cb_sampler.ContinuousBernoulliBank(2, library=library) as bank:
        for _ in range(1000):
            bank.update(0, 0.0)
            bank.update(1, 1.0)
            assert np.array_equal(bank.draw_all(), [0.0, 1.0])


def test_cb_bank_can_replace_descent_sample_and_hold_stats():
    library = cb_sampler.resolve_library(auto_build=True)
    with cb_sampler.ContinuousBernoulliBank(1, library=library) as bank:
        bank.set_observation(0, 0.25, 7)
        assert np.isfinite(bank.draw_all()[0])
        bank.set_observation(0, 0.75, 8)
        assert np.isfinite(bank.draw_all()[0])


def test_l4_exact_fit_has_zero_hessian_and_quartic_leading_loss():
    q = l2_l4_phase_portraits.QuarticModel(
        [1, 1, 1], [1, 1, 1], model.moments_uniform01(9))
    loss, gradient, hessian = q.values(1.0, 1.0)
    assert loss == pytest.approx(0.0, abs=1e-30)
    assert np.array_equal(gradient, [0.0, 0.0])
    assert np.array_equal(hessian, np.zeros((2, 2)))
    direction = np.array([0.7, -0.4])
    values = [q.L(*(np.ones(2)+scale*direction))
              for scale in (1e-2, 5e-3)]
    assert values[0]/values[1] == pytest.approx(16.0, rel=0.025)


def test_l4_conditional_minimizer_annuls_a_gradient():
    q = l2_l4_phase_portraits.QuarticModel(
        [1, 1, 1], [1, 1, 1], model.moments_normal01(9))
    for b in (-1.0, 0.0, 0.6, 1.0):
        a = q.a_star(b)
        assert q.gradL(a, b)[0] == pytest.approx(0.0, abs=2e-10)


def test_l4_closed_form_symmetric_eigenpairs_have_small_residuals():
    hessian = np.array([[8537.22993, -68.1664021],
                        [-68.1664021, 0.542783399]])
    for smaller in (True, False):
        eigenvalue, vector = l2_l4_phase_portraits._sym2_eigenpair(
            hessian, smaller=smaller)
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert np.linalg.norm(hessian @ vector-eigenvalue*vector) \
            <= 2e-12*np.linalg.norm(hessian)


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


def test_optimizer_run_stops_at_last_finite_iterate_without_warnings():
    gradient = opt.BatchGradient(
        [1.0], [0.0, 0.0, 0.0, 1.0], batch_size=4,
        rng=np.random.default_rng(39), distribution="empirical",
        samples=[2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        trajectory = opt.run_adam(
            gradient, (1.0, 1e200), lr=1e-3, n_steps=4)
    assert trajectory.shape == (1, 2)
    assert np.array_equal(trajectory[-1], [1.0, 1e200])


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


class _FakeContinuousBernoulliBank:
    def __init__(self, n_arms, seed=0, library=None):
        self.n_arms = n_arms
        self.library_path = "/fake/libcb_core"

    def draw_all(self):
        return np.arange(self.n_arms, 0, -1, dtype=float)

    def update(self, arm, observation):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        pass


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
    assert np.array_equal(result.allocation_steps, [1, 1, 1, 1])
    assert np.allclose(result.allocation_losses, [.81, 3.24, 2.6244, .6561])
    assert [arm for arm, _ in posterior.updates] == [0, 1, 1, 0]
    assert all(0 <= value <= 1 for _, value in posterior.updates)


def test_thompson_replaces_sample_and_hold_stats_when_supported():
    class HoldingPosterior(_DeterministicPosterior):
        def __init__(self):
            super().__init__(([0.8, 0.2], [0.1, 0.9]))
            self.held = []

        def set_observation(self, arm, observation, selections):
            self.held.append((arm, observation, selections))

        def update(self, arm, observation):
            raise AssertionError("descent allocation must replace, not add")

    states = [
        opt.SGDState(np.array([1.0, 0.0]), lr=0.1),
        opt.SGDState(np.array([2.0, 0.0]), lr=0.1),
    ]
    gradients = [lambda a, b: np.array([a, b])] * 2
    posterior = HoldingPosterior()
    result = thompson.allocate(
        states, gradients, lambda a, b: a*a+b*b,
        posterior, rounds=4, chunk_steps=1)
    assert np.array_equal(result.choices, [0, 1, 1, 0])
    assert [(arm, selections) for arm, _, selections in posterior.held] == [
        (0, 1), (1, 1), (1, 2), (0, 2)]


def test_inverse_sqrt_allocator_schedule_is_budget_prefix_consistent():
    short = thompson_moustaches._schedule(
        "inverse-sqrt", 0.01, horizon=1000, warmup_steps=80)
    long = thompson_moustaches._schedule(
        "inverse-sqrt", 0.01, horizon=100000, warmup_steps=80)
    assert [short(t) for t in (1, 40, 80, 2000)] == [
        long(t) for t in (1, 40, 80, 2000)]


def test_thompson_allocation_honors_interactive_stop_hook():
    state = opt.SGDState(np.array([1.0, 0.0]), lr=0.1)
    posterior = _DeterministicPosterior(())
    with pytest.raises(TimeoutError, match="interactive time limit"):
        thompson.allocate(
            [state], [lambda a, b: np.array([a, b])],
            lambda a, b: a*a+b*b, posterior,
            rounds=1, chunk_steps=10, should_stop=lambda: True)
    assert state.t == 0
    assert posterior.updates == []


def test_thompson_marks_divergent_arm_terminal_without_retrying_it():
    state = opt.AdamState(np.array([1.0, 1e200]), lr=1e-3)
    gradient = opt.BatchGradient(
        [1.0], [0.0, 0.0, 0.0, 1.0], batch_size=4,
        rng=np.random.default_rng(41), distribution="empirical",
        samples=[2.0])
    posterior = _DeterministicPosterior(([0.0], [0.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = thompson.allocate(
            [state], [gradient], lambda a, b: a*a+b*b,
            posterior, rounds=3, chunk_steps=2)
    assert np.array_equal(result.allocations, [1])
    assert np.array_equal(result.executed_steps, [0])
    assert np.array_equal(result.terminated, [True])
    assert result.termination_reasons == ("nonfinite_gradient",)
    assert result.observations == [[1.0]]
    assert np.array_equal(result.allocation_steps, [0])
    assert np.all(np.isinf(result.allocation_losses))
    assert result.trajectories[0].shape == (1, 2)


def test_thompson_escalates_nonfinite_loss_without_sampling_from_one():
    state = opt.SGDState(np.array([1e200, 0.0]), lr=0.1)
    posterior = _DeterministicPosterior(())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = thompson.allocate(
            [state], [lambda a, b: np.zeros(2)],
            lambda a, b: a*a+b*b, posterior,
            rounds=1, chunk_steps=1)
    assert result.observations == [[1.0]]
    assert result.terminated[0]
    assert result.termination_reasons == ("nonfinite_loss",)
    assert result.executed_steps[0] == 1
    assert np.array_equal(result.allocation_steps, [1])
    assert np.all(np.isinf(result.allocation_losses))


def test_work_loss_histogram_accounts_for_every_executed_step():
    from types import SimpleNamespace

    equal = SimpleNamespace(
        allocation_losses=np.array([1e-3, 1e-1, np.inf]),
        allocation_steps=np.array([10, 20, 0]))
    adaptive = SimpleNamespace(
        allocation_losses=np.array([1e-2, 0.0, np.inf]),
        allocation_steps=np.array([4, 5, 6]))
    histogram = thompson_moustaches.work_loss_histogram(
        equal, adaptive, bins=8)
    assert len(histogram["edges"]) == 9
    assert histogram["weight"] == "executed_optimizer_steps"
    assert sum(histogram["equal"]["steps"]) == 30
    assert histogram["equal"]["total_steps"] == 30
    assert sum(histogram["thompson"]["steps"]) == 4
    assert histogram["thompson"]["zero_steps"] == 5
    assert histogram["thompson"]["nonfinite_steps"] == 6
    assert histogram["thompson"]["total_steps"] == 15


def test_equal_thompson_comparison_accepts_arbitrary_model_and_view(
        d2, monkeypatch):
    m, _ = d2
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "ContinuousBernoulliBank",
        _FakeContinuousBernoulliBank)
    result = thompson_moustaches.compare_allocators(
        m, [1, 1, 1], [1, 1, 1], (-0.5, 0.5, -0.75, 0.75),
        starts=3, rounds=6, chunk_steps=1, batch_size=4,
        method="adam", schedule="constant", seed=41)
    starts = result["starts"]
    assert np.all((-0.5 <= starts[:, 0]) & (starts[:, 0] <= 0.5))
    assert np.all((-0.75 <= starts[:, 1]) & (starts[:, 1] <= 0.75))
    assert np.array_equal(result["equal"].allocations, [2, 2, 2])
    assert np.array_equal(result["thompson"].allocations, [1, 1, 4])
    assert np.array_equal(result["equal"].executed_steps, [2, 2, 2])
    assert not np.any(result["equal"].terminated)
    assert result["library_path"] == "/fake/libcb_core"


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
    (0.0, 0.0),
    (1.0, 0.5), (3.0, 0.75),
    (float("inf"), 1.0),
    (-1e-15, 0.0)])
def test_transformed_loss(value, expected):
    assert thompson.transformed_loss(value) == pytest.approx(expected)


def test_transformed_loss_uses_closed_family_boundaries():
    values = [0.0, -1.0, 1e-300, 1.0, 1e300, np.inf, np.nan]
    observations = [thompson.transformed_loss(value) for value in values]
    assert observations[:2] == [0.0, 0.0]
    assert observations[-2:] == [1.0, 1.0]
    assert all(0.0 <= observation <= 1.0 for observation in observations)


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


def test_batch_portrait_moments_are_exact_for_binary64_samples():
    moments = batch_moment_portraits.exact_empirical_moments(
        [0.0, 0.5, 1.0], 5)
    assert moments == (
        Fraction(1), Fraction(1, 2), Fraction(5, 12),
        Fraction(3, 8), Fraction(17, 48))


def test_midpoint_256_control_uses_the_adam_oracles_exact_law():
    _, support = adam_flow.empirical_uniform_grid(256)
    expected = adam_flow.empirical_moments(support, 11)
    actual = batch_moment_portraits.exact_midpoint_moments(256, 11)
    assert actual == expected
    control = batch_moment_portraits.matched_midpoint_control()
    lo, hi = control["wall_bracket"]
    assert lo < control["wall_parameter"] < hi
    assert control["shift_from_population_wall"] < 0
    assert abs(control["shift_from_population_wall"]) < 2e-5
    assert control["adam_minus_sd_wall"] > 0.2
    assert not control["affine_span_probe"]["proof"]
    assert "Numerical-oracle" in control["bracket_protocol"]


def test_batch_portrait_streams_are_nested_and_independent():
    prefix = batch_moment_portraits.nested_uniform_batch(19, 2, 8)
    longer = batch_moment_portraits.nested_uniform_batch(19, 2, 32)
    other = batch_moment_portraits.nested_uniform_batch(19, 3, 8)
    assert np.array_equal(prefix, longer[:8])
    assert not np.array_equal(prefix, other)


def test_batch_portrait_report_disclaims_cross_moment_branch_identity():
    report = {
        "view": (-1, 1, -1, 1),
        "matched_midpoint_256": (
            batch_moment_portraits.matched_midpoint_control()),
        "results": [{
            "batch_size": 8, "batch_index": 0, "certified": False,
            "branches": [], "critical_points": [],
            "moment_error_max": 0.1,
        }],
    }
    html = batch_moment_portraits._html(report)
    normalized = " ".join(html.split())
    assert "No branch identity is asserted across moment space" in normalized
    assert "Only portraits whose global topology audit certified" in normalized


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


def test_saddle_connection_triptych_is_a_distinct_zoo_wall_family():
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    assert family.name in zoo.wall_family_names()
    assert family.name not in zoo.names()
    assert family.below_parameter < family.wall_parameter \
        < family.above_parameter
    base = zoo.get(family.base_case)
    for member, lam in (
            ("below", family.below_parameter),
            ("wall", family.wall_parameter),
            ("above", family.above_parameter)):
        case = zoo.rheostat_member(family, member)
        root = np.sqrt(lam)
        assert np.allclose(case.f, np.asarray(base.f)/root,
                           rtol=0, atol=0)
        assert np.allclose(case.g, root*np.asarray(base.g),
                           rtol=0, atol=0)


def test_wall_family_bracket_is_the_citable_object():
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    assert family.wall_bracket is not None
    lo, hi = family.wall_bracket
    # ordering: the representative center lies strictly inside the bracket,
    # which lies strictly inside the chamber parameters
    assert family.below_parameter < lo < family.wall_parameter \
        < hi < family.above_parameter
    # the bracket is tight relative to the chamber gap but has nonzero width
    assert 0 < hi-lo < 1e-9*family.wall_parameter
    # a bracket must carry its verification protocol
    assert family.bracket_protocol
    assert "Radau" in family.bracket_protocol
    assert "DOP853" in family.bracket_protocol


def test_wall_family_bracket_validation_rejects_malformed_records():
    import dataclasses

    family = zoo.get_wall_family("nonnearest-saddle-connection")
    with pytest.raises(ValueError):
        dataclasses.replace(
            family, wall_bracket=(family.above_parameter,
                                  family.above_parameter+1))
    with pytest.raises(ValueError):
        dataclasses.replace(family, bracket_protocol="")


def test_wall_limit_surgery_removes_only_the_two_involved_continuations():
    from types import SimpleNamespace

    family = zoo.get_wall_family("nonnearest-saddle-connection")
    source = SimpleNamespace(
        kind="saddle", a=0.0, b=family.source_b)
    target = SimpleNamespace(
        kind="saddle", a=1.0, b=family.target_b)
    enumeration = SimpleNamespace(points=[source, target])

    def branch(kind, Y, **diag):
        return charts.Branch(kind, np.asarray(Y, dtype=float), "box_exit",
                             {}, diag)

    branches = [
        branch("unstable", [(0, family.source_b), (1, 1)],
               saddle_b=family.source_b, unstable_direction=1),
        branch("stable", [(1, family.target_b), (.01, family.source_b+.01)],
               saddle_b=family.target_b, stable_sign=1),
        branch("stable", [(1, family.target_b), (4, 4)],
               saddle_b=family.target_b, stable_sign=-1),
        branch("unstable", [(0, family.source_b), (-1, -1)],
               saddle_b=family.source_b, unstable_direction=-1),
    ]
    original = portrait.Portrait(
        None, enumeration, branches, (-5, 5, -5, 5), None, {})
    wall, diagnostics = saddle_connection_triptych.wall_limit_portrait(
        original, family, np.array(((0, family.source_b),
                                    (1, family.target_b))))
    assert diagnostics["removed_source_unstable"] == 0
    assert diagnostics["removed_target_stable"] == 1
    assert len(wall.branches) == 2
    assert wall.ledger["comparison"]["geometry_method"] \
        == "geometric wall limit"


def test_triptych_svg_contains_three_nested_panels():
    panels = [
        '<svg xmlns="http://www.w3.org/2000/svg"><text>A</text></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><text>B</text></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><text>C</text></svg>',
    ]
    combined = saddle_connection_triptych.triptych_svg(panels)
    assert combined.count("<text>") == 3
    assert 'width="1920"' in combined
    assert combined.count("<line ") == 2
    compact = saddle_connection_triptych.triptych_svg(
        panels, panel_height=330)
    assert 'height="330"' in compact
    assert 'viewBox="0 0 1920 330"' in compact


def test_saddle_connection_comparison_geometry_helpers():
    point = np.array([1.0, 0.25])
    curve = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert saddle_connection_comparison.point_polyline_distance(
        point, curve) == pytest.approx(0.25)

    family = zoo.get_wall_family("nonnearest-saddle-connection")
    branches = [charts.Branch(
        "unstable", np.array([[0.0, family.source_b], [1.0, 1.0]]),
        "capture", {}, {
            "saddle_b": family.source_b,
            "unstable_direction": family.unstable_direction,
        })]
    candidate = portrait.Portrait(None, None, branches, None, None, {})
    assert saddle_connection_comparison.tracked_branch(
        candidate, family) is branches[0]

    svg = saddle_connection_comparison.transverse_gallery_svg([
        ("method A", [(-0.06, -2e-3), (0.0, -1e-3), (0.06, 1e-3)]),
        ("method B", [(-0.06, 2e-6), (0.0, 1e-6), (0.06, -1e-6)]),
    ])
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert svg.count('<polyline class="wu"') == 2
    assert svg.count('<polyline class="ws"') == 2
    assert "max |d|=2.000e-03" in svg
    assert "max |d|=2.000e-06" in svg


@pytest.mark.slow
def test_saddle_connection_side_chambers_certify_in_requested_view():
    """Regression for coincident terminal tails in the handle-slide zoo."""
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    for member in ("below", "above"):
        p = portrait.compute(
            saddle_connection_triptych.build_member(member),
            view=family.default_view)
        assert p.box[0] < family.default_view[0]
        assert p.box[1] > family.default_view[1]
        assert p.box[2] < family.default_view[2]
        assert p.box[3] > family.default_view[3]
        assert p.ledger["topology"]["status"] == "certified"
        assert p.ledger["topology"]["raw_event_count"] == 0
        assert all(tail["method"] == "exact_superlevel_product"
                   for tail in p.ledger["topology"]["stable_tails"])
