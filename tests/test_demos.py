"""Phase-6 sanity: the optimizer demos are functional consumers.

The showcase script itself is demo-ware (runs the 40 s tricky portrait)
and is exercised manually; here the optimizers run on the cheap d=2
oracle model.
"""

import numpy as np
import pytest

from demos import optimizers as opt
from spong import model, sturm


@pytest.fixture(scope="module")
def d2():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    return m, e


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
