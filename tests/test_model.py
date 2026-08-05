"""spong.model: exact identities of SPONG_FOUNDING Part II sections 1-3."""

from fractions import Fraction as F

import numpy as np
import pytest

from spong import model
from spong import _poly as P


@pytest.fixture(scope="module")
def m():
    # d = 2 example from the mse-bundle sessions: f = g = [1,1,1], U(0,1).
    return model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))


def test_exact_coefficients(m):
    # alpha_m = mu_m * sum_{i+j=m} g_i g_j ; beta_j = g_j sum_i f_i mu_{i+j}
    assert m.alpha == (F(1), F(1), F(1), F(1, 2), F(1, 5))
    assert m.beta[0] == F(1) + F(1, 2) + F(1, 3)          # 11/6
    assert m.C == F(1) + 2 * F(1, 2) + F(1, 3) * 3 + 2 * F(1, 4) + F(1, 5)


def test_completing_the_square(m):
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.uniform(-3, 3, 2)
        w = m.w_of(a, b)
        assert m.L(a, b) == pytest.approx(m.u(b) + m.A(b) * w**2, rel=1e-12)


def test_gradient_matches_finite_difference(m):
    rng = np.random.default_rng(1)
    h = 1e-6
    for _ in range(20):
        a, b = rng.uniform(-2, 2, 2)
        g = m.gradL(a, b)
        ga = (m.L(a + h, b) - m.L(a - h, b)) / (2 * h)
        gb = (m.L(a, b + h) - m.L(a, b - h)) / (2 * h)
        assert g[0] == pytest.approx(ga, rel=1e-6, abs=1e-5)
        assert g[1] == pytest.approx(gb, rel=1e-6, abs=1e-5)


def test_u_prime_identity(m):
    # u' = B*N/A^2 exactly
    b = np.linspace(-4, 4, 101)
    h = 1e-6
    lhs = (m.u(b + h) - m.u(b - h)) / (2 * h)
    rhs = m.u_p(b)
    assert np.allclose(lhs, rhs, rtol=1e-5, atol=1e-7)


def test_hessian_identities_on_backbone(m):
    # H12 = -2A a*' and det H = 2A u'' at backbone points
    for b in [-2.0, -0.7, 0.3, 1.5]:
        a = m.a_star(b)
        H = m.hessL(a, b)
        assert H[0, 1] == pytest.approx(-2 * m.A(b) * m.a_star_p(b), rel=1e-10)
        det = H[0, 0] * H[1, 1] - H[0, 1] ** 2
        assert det == pytest.approx(2 * m.A(b) * m.u_pp(b), rel=1e-9)


def test_level_curves_closed_form(m):
    b = np.linspace(-2, 2, 41)
    c = m.u(0.5) + 0.7           # a level above the valley floor at b=0.5
    lo, hi = m.level_curve(c, b)
    mask = ~np.isnan(lo)
    assert mask.any()
    assert np.allclose(m.L(lo[mask], b[mask]), c, rtol=1e-10)
    assert np.allclose(m.L(hi[mask], b[mask]), c, rtol=1e-10)


def test_P_is_minus_bdot(m):
    # gradL_b at (a*(b)+w, b) equals P(b, w)
    rng = np.random.default_rng(2)
    for _ in range(20):
        b = rng.uniform(-2, 2)
        w = rng.uniform(-0.5, 0.5)
        a = m.a_star(b) + w
        assert m.gradL(a, b)[1] == pytest.approx(m.P_of(b, w),
                                                 rel=1e-9, abs=1e-11)


def test_normal_moments():
    mu = model.moments_normal01(8)
    assert [float(x) for x in mu] == [1, 0, 1, 0, 3, 0, 15, 0]


def test_degree_one_model_native_coefficients_are_not_empty():
    m = model.build([1, 2], [1, 2], model.moments_uniform01(3))
    assert m.App(0.0) == pytest.approx(8 / 3)
    assert m.Bpp(0.0) == pytest.approx(0.0)
    assert m._fbpp == (0.0,)
    assert m._fnp == (2 / 3,)


def test_exact_moment_jacobian_is_the_full_gradient_difference():
    f = [F(3, 2), F(-2, 3), F(5, 7)]
    g = [F(-1, 4), F(4, 5), F(2, 9)]
    mu = model.moments_uniform01(5)
    delta = (F(0), F(1, 101), F(-2, 103), F(3, 107), F(-4, 109))
    perturbed = model.build(f, g, tuple(x + dx for x, dx in zip(mu, delta)))
    base = model.build(f, g, mu)
    a, b = F(7, 11), F(-5, 13)

    def exact_grad(m):
        return (
            2 * (a * P.eval_at(m.alpha, b) - P.eval_at(m.beta, b)),
            -2 * a * P.eval_at(P.deriv(m.beta), b)
            + a * a * P.eval_at(P.deriv(m.alpha), b),
        )

    lhs = tuple(y - x for x, y in zip(exact_grad(base), exact_grad(perturbed)))
    J = base.moment_jacobian_exact(a, b)
    rhs = tuple(sum((row[k] * delta[k] for k in range(base.moment_count)), F(0))
                for row in J)
    assert lhs == rhs

    J_float = base.moment_jacobian(float(a), float(b))
    assert np.allclose(J_float, np.array(J, dtype=float), rtol=2e-15, atol=2e-15)


def test_exact_batch_raw_moment_covariance():
    mu = model.moments_uniform01(9)
    cov = model.batch_raw_moment_covariance(mu, moment_count=5, batch_size=8)
    assert cov[0] == (F(0),) * 5
    assert all(cov[j][k] == cov[k][j] for j in range(5) for k in range(5))
    assert cov[1][1] == (F(1, 3) - F(1, 2) ** 2) / 8
    assert cov[2][4] == (F(1, 7) - F(1, 3) * F(1, 5)) / 8

    with pytest.raises(ValueError, match="mu_0..mu_8"):
        model.batch_raw_moment_covariance(mu[:5], moment_count=5, batch_size=8)


def test_gradient_noise_covariance_is_symmetric_positive_semidefinite(m):
    # The degree-two model needs mu_0..mu_4 for its loss, but mu_0..mu_8
    # for the covariance of that five-component empirical moment vector.
    cov = model.gradient_noise_covariance(
        m, a=0.7, b=-1.2, population_mu=model.moments_uniform01(9),
        batch_size=32)
    assert np.allclose(cov, cov.T, rtol=1e-14, atol=1e-16)
    assert cov[0, 0] >= 0.0
    assert cov[1, 1] >= 0.0
    assert cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2 >= -1e-15
