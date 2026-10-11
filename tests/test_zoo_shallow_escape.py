"""Symmetric normal-moment zoo example and prescribed-root extension."""

from fractions import Fraction

import pytest

from spong import merge_tree, model, portrait, sturm, zoo
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


@pytest.mark.parametrize("degree", (3, 5, 7, 9, 11))
def test_small_integer_sequence_has_off_axis_alternating_minima(degree):
    case = zoo.small_integer_symmetric_escape_case(degree)
    if degree in (3, 5, 7, 9):
        assert zoo.get(case.name) == case
    else:
        assert case.name not in zoo.names()
    assert max(abs(x) for x in (*case.f, *case.g)) <= 24
    m = build_case(case)
    e = sturm.enumerate_critical_points(m)
    assert e.morse and e.psi_positive
    assert (len(e.points), len(e.minima), len(e.saddles)) == (
        2 * degree - 1, degree - 1, degree)
    assert all(abs(q.a) < 1e-30 for q in e.saddles)
    signs = [1 if q.a > 0 else -1 for q in e.minima]
    assert all(left != right for left, right in zip(signs, signs[1:]))
    assert all(abs(q.a) > 0.05 for q in e.minima if degree <= 7)
    infinity_loss = float(merge_tree.backbone_level_at_infinity(m))
    assert all(infinity_loss < m.u(q.b) < float(m.C) for q in e.minima)


def test_small_integer_sequence_rejects_unverified_degrees():
    for degree in (1, 2, 4, 13, True, 5.0):
        with pytest.raises(ValueError, match="degree"):
            zoo.small_integer_symmetric_escape_case(degree)


@pytest.mark.parametrize("degree", (5, 7))
def test_named_small_integer_portraits_certify(degree):
    case = zoo.get(f"normal-symmetric-small-integer-d{degree}")
    p = portrait.compute(build_case(case))
    summary = p.ledger["summary"]
    assert summary["topology_status"] == "certified"
    assert summary["representation_attested"]


def test_normal_symmetric_flat_degree5_certifies():
    case = zoo.get("normal-symmetric-flat-d5")
    assert case.f == (0, 1, 0, 1, 0, 1)
    assert case.g == (-3, -1, -3, 4, -3, -2)
    m = build_case(case)
    e = sturm.enumerate_critical_points(m)
    assert e.morse and e.psi_positive
    assert (len(e.minima), len(e.saddles)) == (4, 5)
    assert m.C == 1207
    assert tuple(m.beta) == (0, -19, 0, 492, 0, -2130)
    assert merge_tree.backbone_level_at_infinity(m) == Fraction(142, 21)
    assert all(abs(q.a) < 1e-30 and m.u(q.b) == pytest.approx(1207)
               for q in e.saddles)
    assert all(0 < (m.C - m.u(q.b)) / float(m.C) < 0.00067
               for q in e.minima)
    assert all(m.u(q.b) > Fraction(142, 21) for q in e.minima)
    assert [1 if q.a > 0 else -1 for q in e.minima] == [-1, 1, -1, 1]
    summary = portrait.compute(m).ledger["summary"]
    assert summary["topology_status"] == "certified"
    assert summary["representation_attested"]


def test_normal_symmetric_flat_degree7_certifies():
    case = zoo.get("normal-symmetric-flat-d7")
    assert case.f == (0, 1, 0, 1, 0, 1, 0, 1)
    assert case.g == (-3, -1, -3, 4, -3, -2, -3, Fraction(1, 5))
    m = build_case(case)
    e = sturm.enumerate_critical_points(m)
    assert e.morse and e.psi_positive
    assert (len(e.minima), len(e.saddles)) == (6, 7)
    assert m.C == 159232
    assert tuple(m.beta) == (0, -124, 0, 4272, 0, -22920, 0, 29316)
    tail = Fraction(101968, 429)
    assert merge_tree.backbone_level_at_infinity(m) == tail
    assert all(abs(q.a) < 1e-30 and m.u(q.b) == pytest.approx(m.C)
               for q in e.saddles)
    assert all(tail < m.u(q.b) < m.C for q in e.minima)
    assert max((m.C - m.u(q.b)) / float(m.C) for q in e.minima) < 0.000618
    assert [1 if q.a > 0 else -1 for q in e.minima] == [1, -1, 1, -1, 1, -1]
    summary = portrait.compute(m).ledger["summary"]
    assert summary["topology_status"] == "certified"
    assert summary["representation_attested"]


@pytest.mark.heavy
def test_degree9_stable_retrace_uses_its_actual_box():
    case = zoo.get("normal-symmetric-small-integer-d9")
    p = portrait.compute(build_case(case))
    assert p.ledger["topology"]["status"] == "certified"
    retraced = [b for b in p.branches
               if b.kind == "stable" and b.diag.get("stable_box_retrace")]
    assert retraced
    assert all(b.diag["stable_box_retrace"]["box_a"][1] > p.box[1]
               for b in retraced)
    assert all(x["certified"] for x in p.ledger["topology"]["stable_tails"])
