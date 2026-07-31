import math

import numpy as np
import pytest

from spong import (
    atlas, comparison, comparison_cli, model, portrait, render, sturm, zoo)


class _LinearModel:
    def __init__(self, rate):
        self.rate = rate

    def gradL(self, a, b):
        return np.array([self.rate*a, self.rate*b])

    def hessL(self, a, b):
        return np.eye(2)*self.rate

    def A(self, b):
        return 1.0

    def B(self, b):
        return 0.0

    def Ap(self, b):
        return 0.0

    def Bp(self, b):
        return 0.0


@pytest.mark.parametrize("method, factor", [
    ("forward-euler", lambda x: 1-x),
    ("backward-euler", lambda x: 1/(1+x)),
    ("explicit-midpoint", lambda x: 1-x+x*x/2),
    ("implicit-midpoint", lambda x: (1-x/2)/(1+x/2)),
])
def test_textbook_fixed_steps_match_linear_closed_forms(method, factor):
    m = _LinearModel(7.0)
    y = np.array([1.0, -2.0])
    h = 0.03
    actual, ok = comparison._fixed_step(m, method, y, h, +1)
    assert ok
    assert np.allclose(actual, factor(7*h)*y, rtol=2e-13, atol=2e-13)


def test_rkf45_trial_has_expected_accuracy_on_linear_flow():
    m = _LinearModel(3.0)
    y = np.array([1.0, -2.0])
    h = 0.1
    actual, error = comparison._rkf45_trial(m, y, h, +1)
    expected = math.exp(-0.3)*y
    assert np.max(np.abs(actual-expected)) < 2e-6
    assert np.max(np.abs(error)) > 0


def test_ros2_trial_is_second_order_and_stable_on_linear_flow():
    m = _LinearModel(30.0)
    y = np.array([1.0, -2.0])
    h = 0.01
    actual, error, ok = comparison._ros2_trial(m, y, h, +1)
    assert ok
    expected = math.exp(-0.3)*y
    assert np.max(np.abs(actual-expected)) < 3e-2
    assert np.max(np.abs(error)) > 0


def test_common_resolution_angle_diagnostic_is_zero_on_straight_flow():
    m = _LinearModel(1.0)
    Y = np.column_stack((np.linspace(1.0, 0.1, 100), np.zeros(100)))
    detail = comparison.integral_curve_diagnostics(m, Y, spacing=0.01)
    assert detail["angle_energy_common"] < 1e-28
    assert detail["angle_rms_deg"] < 1e-12


def _quadratic_stiff():
    case = zoo.get("quadratic-stiff")
    degree = max(len(case.f)-1, len(case.g)-1)
    return model.build(
        case.f, case.g, model.moments_uniform01(2*degree+1))


def test_canonical_tricky_case_is_available_to_comparison_cli():
    case = zoo.get("tricky-d11")
    assert len(case.f)-1 == 11
    assert case.f == case.g
    assert case.expected_connections[0] == pytest.approx(
        (-2.738230515199397, -0.7895860210707522))


def test_grid_newton_is_explicitly_resolution_dependent():
    m = _quadratic_stiff()
    certified = sturm.enumerate_critical_points(m)
    box = atlas.compute_box(m, certified)
    coarse, _ = comparison.grid_newton_critical_points(m, box, grid=9)
    finer, _ = comparison.grid_newton_critical_points(m, box, grid=13)
    assert len(coarse.points) < len(certified.points)
    assert len(finer.points) == len(certified.points)
    assert not coarse.morse and not finer.morse


@pytest.mark.parametrize("method", comparison.GEOMETRY_METHODS)
def test_casual_geometry_is_separate_and_marked_uncertified(method):
    m = _quadratic_stiff()
    enumeration = sturm.enumerate_critical_points(m)
    p = comparison.casual_portrait(
        m, method, reference_enumeration=enumeration,
        step_size=0.05, max_steps=20)
    assert len(p.branches) == 4*len(enumeration.saddles)
    assert p.ledger["comparison"]["uncertified"]
    assert p.ledger["comparison"]["geometry_method"] == method
    assert all(br.diag["elapsed_time"] <= 1.0+1e-14 for br in p.branches)
    assert all("angle_rms_deg" in br.certs for br in p.branches)
    svg = render.plane_view(p, width=320, height=240, n_levels=4, n_grid=101)
    assert "UNCERTIFIED COMPARISON" in svg


def test_installed_comparison_command_writes_gallery(monkeypatch, tmp_path):
    def lightweight_reference(m, view=None):
        enumeration = sturm.enumerate_critical_points(m)
        box = atlas.compute_box(m, enumeration)
        return portrait.Portrait(
            m, enumeration, [], box, box, {
                "summary": {
                    "worst_angle_energy": 0.0,
                    "balanced": True,
                    "worst_max_turn_deg": 0.0,
                },
            })

    monkeypatch.setattr(
        comparison_cli.portrait, "certified_compute", lightweight_reference)
    status = comparison_cli.main([
        "--zoo", "quadratic-stiff",
        "--methods", "forward-euler",
        "--critical-method", "grid-newton",
        "--critical-grid", "9",
        "--max-steps", "20",
        "--width", "320", "--height", "240",
        "--output-dir", str(tmp_path),
    ])
    assert status == 0
    assert (tmp_path/"quadratic-stiff_comparison.html").exists()
    assert (tmp_path/"quadratic-stiff_comparison.json").exists()
    assert (tmp_path/"quadratic-stiff_certified.svg").exists()
    assert (tmp_path/"quadratic-stiff_grid-newton_forward-euler.svg").exists()
