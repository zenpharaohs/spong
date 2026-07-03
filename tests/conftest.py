"""Shared session fixtures: each model's exact enumeration (the 9 s bigint
certificate pass) and the tricky portrait are computed ONCE per test run
instead of once per test file."""

import pytest

from spong import model, portrait, sturm
from tests.test_enumeration import TRICKY_F


@pytest.fixture(scope="session")
def tricky():
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    e = sturm.enumerate_critical_points(m)
    return m, e


@pytest.fixture(scope="session")
def d2():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    return m, e


@pytest.fixture(scope="session")
def tricky_portrait(tricky):
    m, _ = tricky
    return portrait.compute(m, view=(-1.5, 2.5, -4.0, 3.0))
