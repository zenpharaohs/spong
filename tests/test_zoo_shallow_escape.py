"""Symmetric normal-moment zoo example and prescribed-root extension."""

from fractions import Fraction

import pytest

from spong import merge_tree, model, sturm, zoo
from spong import _poly as poly


def build_case(case):
    moments = model.moments_normal01(2 * max(len(case.f), len(case.g)) - 1)
    return model.build(case.f, case.g, moments)


def test_original_symmetric_escape_case():
    case = zoo.get("normal-symmetric-shallow-escape")
    m = build_case(case)
    e = sturm.enumerate_critical_points(m)
    assert e.morse and e.psi_positive
    assert (len(e.minima), len(e.saddles)) == (2, 3)
    assert merge_tree.backbone_level_at_infinity(m) == Fraction(18, 5)
    assert all(m.u(q.b) == pytest.approx(197.87324039814374)
               for q in e.minima)
    assert all(m.u(q.b) == pytest.approx(198) for q in e.saddles)
    assert all(not coefficient for coefficient in m.alpha[1::2])
    assert all(not coefficient for coefficient in m.beta[::2])


@pytest.mark.parametrize("degree", (3, 5, 7))
def test_prescribed_root_family_has_many_shallow_minima(degree):
    case = zoo.symmetric_shallow_escape_case(degree)
    if degree in (5, 7):
        assert zoo.get(case.name) == case
    m = build_case(case)
    e = sturm.enumerate_critical_points(m)
    assert e.morse and e.psi_positive
    assert len(e.saddles) == degree
    assert len(e.minima) >= degree - 1
    assert merge_tree.backbone_level_at_infinity(m) == 0
    for root in range(-(degree - 1) // 2, (degree - 1) // 2 + 1):
        assert poly.eval_at(m.beta, Fraction(root)) == 0
    assert all(0 < float(m.C) - m.u(q.b) < float(m.C) / 1000
               for q in e.minima)
    assert all(not coefficient for coefficient in m.alpha[1::2])
    assert all(not coefficient for coefficient in m.beta[::2])


def test_prescribed_root_family_validates_parameters():
    for degree in (0, 2, 4, True, 3.0):
        with pytest.raises(ValueError, match="degree"):
            zoo.symmetric_shallow_escape_case(degree)
    for shield in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="shield"):
            zoo.symmetric_shallow_escape_case(3, shield)
    assert zoo.symmetric_shallow_escape_case(3, 200).name.endswith("-s200")
