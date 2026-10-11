"""Exact rational-response construction and its Chebyshev critical points."""

from fractions import Fraction

import pytest

from demos import optimizers
from demos.explorer import serve
from spong import _poly as poly, filter_family, model, portrait, sturm


@pytest.mark.parametrize("degree", (3, 5, 7, 9, 11))
def test_chebyshev_family_is_exactly_equiripple(degree):
    p = filter_family.chebyshev_coefficients(degree)
    m = filter_family.chebyshev_response_model(degree)
    assert m.beta == p
    assert m.alpha == poly.add((Fraction(1),), poly.mul(p, p))
    assert m.C == 1
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse and e.alternates
    assert (len(e.saddles), len(e.minima)) == (degree, degree - 1)
    assert all(q.a == pytest.approx(0) and m.u(q.b) == pytest.approx(1)
               for q in e.saddles)
    assert all(abs(q.a) == pytest.approx(0.5)
               and m.u(q.b) == pytest.approx(0.5) for q in e.minima)


def test_rational_response_with_real_poles_has_positive_model_a():
    # Q has roots at +/-1, where the backbone reaches a perfect fit.
    p, q = poly.poly((0, 1)), poly.poly((1, 0, -1))
    m = filter_family.rational_response_model(p, q, gain=2)
    assert m.alpha == poly.add(poly.mul(p, p),
                               poly.scale(poly.mul(q, q), Fraction(4)))
    assert m.beta == p and m.C == 1
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse
    assert m.u(-1) == pytest.approx(0)
    assert m.u(1) == pytest.approx(0)


@pytest.mark.parametrize("p,q,gain", [
    ((1,), (1,), 1), ((0, 1), (0, 1), 1),
    ((0, 1), (1, 1), 1), ((0, 1), (1,), 0),
    ((0, 1, 0, 1), (1, 0, 1), 1),
])
def test_rational_response_rejects_invalid_inputs(p, q, gain):
    with pytest.raises(ValueError):
        filter_family.rational_response_model(p, q, gain)


def test_rademacher_inspector_and_batch_sampler_agree():
    assert model.moments_rademacher(7) == (1, 0, 1, 0, 1, 0, 1)
    assert serve.moment_vector({"kind": "rademacher"}, 7) == model.moments_rademacher(7)
    import numpy as np
    sampler = optimizers.BatchGradient((0, 1), (1, -3, 0, 4),
                                        100, np.random.default_rng(11),
                                        distribution="rademacher")
    assert set(sampler._draw()) == {-1, 1}
    assert set(sampler._draw_many(3).ravel()) == {-1, 1}


def test_degree_five_portrait_certifies():
    p = portrait.compute(filter_family.chebyshev_response_model(5))
    assert p.ledger["summary"]["topology_status"] == "certified"
    assert p.ledger["summary"]["representation_attested"]
