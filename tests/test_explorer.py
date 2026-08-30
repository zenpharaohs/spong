"""Focused backend contracts for the interactive explorer demo."""

import subprocess
import warnings

import numpy as np
import pytest

from demos import thompson_moustaches
from demos.explorer import serve
from spong import zoo


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


def test_png_export_redraws_live_portrait_and_backbone_at_two_x():
    page = serve.PAGE.read_text()
    assert 'id="exportPng"' in page
    assert 'id="exportStatus" role="status" aria-live="polite"' in page
    assert "let renderScale = 1;" in page
    assert "cP.width = saved.pw * scale" in page
    assert "cB.width = saved.bw * scale" in page
    assert "drawPortrait();\n    drawBone();" in page
    assert "ctx.drawImage(cP" in page
    assert "ctx.drawImage(cB" in page
    assert '"image/png"' in page
    assert "link.download = filename" in page
    assert "S.lastBody && S.lastBody.zoo" in page
    assert "WALL.atWall" in page


def test_portrait_has_no_selection_ring_or_gesture_zoom():
    page = serve.PAGE.read_text()
    assert "xP.arc(sx, sy, 5" not in page
    assert 'const p = toModel(e), k = e.deltaY > 0' not in page
    assert '#cPort{touch-action:none}' in page
    assert 'cP.addEventListener("wheel", e =>' in page
    assert "if (e.ctrlKey && e.cancelable) e.preventDefault();" in page
    assert '["gesturestart", "gesturechange"]' in page


def test_wall_slider_starts_at_exact_stored_wall_coordinate():
    page = serve.PAGE.read_text()
    assert 'id="lam" type="range" min="0" max="1" step="any"' in page
    assert page.count('els("lam").value = WALL.wall;') == 2
    assert "Math.round(1000 * t)" not in page
    assert "wall_limit: !!WALL.atWall" in page
    assert 'kind": "stable_unstable"' in serve.Path(serve.__file__).read_text()
    assert 'css("--connection")' in page
    assert 'endRadius:equal ? 0 : 1.8' in page
    assert 'markerEndpoint:!equal' in page
    assert 'xP.fillStyle = marker.color' in page
    assert 'markerRing' not in page

    wall = zoo.get_wall_family("nonnearest-saddle-connection")
    f, g, view, spec, _ = serve._resolve({"wall": wall.name})
    base = zoo.get(wall.base_case)
    root = np.sqrt(wall.wall_parameter)
    assert f == pytest.approx(np.asarray(base.f) / root)
    assert g == pytest.approx(np.asarray(base.g) * root)
    assert view == wall.default_view
    assert spec == {"kind": base.moment_dist}


def test_wall_limit_snaps_model_and_serializes_common_orbit(monkeypatch):
    from types import SimpleNamespace

    wall = zoo.get_wall_family("nonnearest-saddle-connection")
    literal = np.nextafter(wall.wall_parameter, np.inf)
    exact = serve._resolve({
        "wall": wall.name, "lam": literal, "wall_limit": True})
    ordinary = serve._resolve({"wall": wall.name, "lam": literal})
    assert exact[0] != ordinary[0]

    source = SimpleNamespace(a=1.0, b=wall.source_b)
    target = SimpleNamespace(a=2.0, b=wall.target_b)
    original = SimpleNamespace(model=object())
    display = SimpleNamespace()
    monkeypatch.setattr(
        serve.saddle_wall, "critical_near",
        lambda p, b: source if b == wall.source_b else target)
    unstable = SimpleNamespace(
        kind="unstable",
        Y=np.array([[source.a, source.b], [1.5, 0.0], [target.a, target.b]]),
        diag={"saddle_b": wall.source_b,
              "unstable_direction": wall.unstable_direction})
    stable = SimpleNamespace(
        kind="stable",
        Y=np.array([[target.a, target.b], [source.a, source.b]]),
        diag={"saddle_b": wall.target_b, "stable_sign": -1})
    original.branches = [unstable, stable]
    monkeypatch.setattr(
        serve.saddle_wall, "wall_limit_portrait",
        lambda p, family, connection: (display, {"removed": [3, 7]}))

    result, connections, diagnostics = serve._geometric_wall_limit(
        {"wall": wall.name, "wall_limit": True}, original)
    assert result is display
    assert connections == [{
        "kind": "stable_unstable",
        "label": "saddle connection (Wu = Ws)",
        "source_b": wall.source_b,
        "target_b": wall.target_b,
        "n_traced": 3,
        "points": [[1.0, wall.source_b], [1.5, 0.0],
                   [2.0, wall.target_b]],
        "numerical_miss": 0.0,
    }]
    assert diagnostics["geometry_method"] == "geometric wall limit"
    assert diagnostics["parameter"] == wall.wall_parameter

    # At ANY Lambda the pair is identified on the ordinary portrait and the
    # closest approaches are reported, so the viewer can print how near the
    # two continuations come rather than leave it to the eye.
    pair = serve._wall_pair({"wall": wall.name}, original)
    assert pair["source_unstable"] == 0
    assert pair["target_stable"] == 1
    assert pair["unstable_to_target"] == 0.0
    assert pair["stable_to_source"] == 0.0
    assert pair["closest_index"] == 2
    assert serve._wall_pair({}, original) is None


def test_allocator_endpoint_uses_active_empirical_portrait_and_visible_view(
        monkeypatch):
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "ContinuousBernoulliBank",
        _FakeContinuousBernoulliBank)
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "resolve_library",
        lambda library=None, auto_build=False: "/fake/libcb_core")
    serve._MODELS.clear()
    result = serve.allocator({
        "f": [1.0, -0.5],
        "g": [0.25, 1.0],
        "moments": {"kind": "empirical", "samples": [-1.0, 0.25, 1.5]},
        "view": [-4.0, 4.0, -4.0, 4.0],
        "allocation_view": [-0.5, 0.75, -1.25, 1.0],
        "starts": 3,
        "rounds": 6,
        "chunk_steps": 1,
        "batch_size": 4,
        "method": "adam",
        "schedule": "constant",
        "learning_rate": 0.001,
        "time_limit_sec": 7,
        "seed": 43,
    })
    assert result["format"] == "spong-equal-thompson-traces-v1"
    assert result["allocation_view"] == [-0.5, 0.75, -1.25, 1.0]
    assert result["configuration"]["distribution"] == "empirical"
    assert result["configuration"]["cb_library"] == "/fake/libcb_core"
    assert result["configuration"]["time_limit_sec"] == 7
    assert result["equal"]["allocations"] == [2, 2, 2]
    assert result["thompson"]["allocations"] == [1, 1, 4]
    assert result["equal"]["executed_optimizer_steps"] == 6
    assert result["thompson"]["executed_optimizer_steps"] == 6
    histogram = result["work_loss_histogram"]
    assert histogram["equal"]["total_steps"] == 6
    assert histogram["thompson"]["total_steps"] == 6
    for policy in ("equal", "thompson"):
        h = histogram[policy]
        assert sum(h["steps"])+h["zero_steps"]+h["nonfinite_steps"] == 6
    assert result["equal"]["terminated_arms"] == 0
    assert result["thompson"]["terminated_arms"] == 0
    assert len(result["equal"]["traces"]) == 3
    assert all(trace["points"] for trace in result["thompson"]["traces"])
    assert all(trace["termination"] is None
               for trace in result["thompson"]["traces"])
    assert all(not trace["mark_start"]
               for trace in result["equal"]["traces"])
    assert all(trace["mark_start"]
               for trace in result["thompson"]["traces"])
    assert all(not trace["mark_end"] for trace in result["equal"]["traces"])
    assert all(trace["mark_end"] for trace in result["thompson"]["traces"])
    assert max(trace["width"] for trace in result["equal"]["traces"]) \
        > max(trace["width"] for trace in result["thompson"]["traces"])
    assert max(trace["opacity"] for trace in result["equal"]["traces"]) \
        < max(trace["opacity"] for trace in result["thompson"]["traces"])


def test_allocator_endpoint_reports_numerical_terminations_without_warnings(
        monkeypatch):
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "ContinuousBernoulliBank",
        _FakeContinuousBernoulliBank)
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "resolve_library",
        lambda library=None, auto_build=False: "/fake/libcb_core")
    serve._MODELS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = serve.allocator({
            "f": [1.0],
            "g": [0.0, 0.0, 0.0, 1.0],
            "moments": {"kind": "empirical", "samples": [2.0]},
            "view": [-1.0, 1.0, -1.0, 1.0],
            "allocation_view": [0.5, 1.5, 1e200, 1.1e200],
            "starts": 2,
            "rounds": 4,
            "chunk_steps": 2,
            "batch_size": 4,
            "method": "adam",
            "schedule": "constant",
            "learning_rate": 0.001,
            "seed": 47,
        })
    for policy in (result["equal"], result["thompson"]):
        assert policy["terminated_arms"] == 2
        assert policy["executed_optimizer_steps"] == 0
        assert policy["final_observations"] == [1.0, 1.0]
        assert all(trace["termination"] == "nonfinite_gradient"
                   for trace in policy["traces"])
        assert all(len(trace["points"]) == 1 for trace in policy["traces"])
    for policy in ("equal", "thompson"):
        h = result["work_loss_histogram"][policy]
        assert h["total_steps"] == 0
        assert sum(h["steps"])+h["zero_steps"]+h["nonfinite_steps"] == 0


def test_allocator_endpoint_enforces_interactive_time_limit(monkeypatch):
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "ContinuousBernoulliBank",
        _FakeContinuousBernoulliBank)
    monkeypatch.setattr(
        thompson_moustaches.cb_sampler, "resolve_library",
        lambda library=None, auto_build=False: "/fake/libcb_core")
    ticks = iter((100.0, 102.0))
    monkeypatch.setattr(serve.time, "perf_counter", lambda: next(ticks))
    serve._MODELS.clear()
    with pytest.raises(TimeoutError, match="interactive time limit"):
        serve.allocator({
            "f": [1.0, 1.0],
            "g": [1.0, 1.0],
            "moments": {"kind": "uniform01"},
            "view": [-1.0, 1.0, -1.0, 1.0],
            "allocation_view": [-1.0, 1.0, -1.0, 1.0],
            "starts": 1,
            "rounds": 1,
            "chunk_steps": 1,
            "batch_size": 1,
            "method": "adam",
            "schedule": "constant",
            "learning_rate": 0.001,
            "time_limit_sec": 1,
            "seed": 53,
        })


def test_isolated_allocator_kills_worker_stuck_past_deadline(monkeypatch):
    class StuckWorker:
        def __init__(self):
            self.killed = False

        def communicate(self, payload=None, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("allocator", timeout)
            return "", ""

        def kill(self):
            self.killed = True

    worker = StuckWorker()
    monkeypatch.setattr(serve.subprocess, "Popen", lambda *a, **k: worker)
    with pytest.raises(TimeoutError, match="hard interactive time limit"):
        serve.isolated_allocator({"time_limit_sec": 1})
    assert worker.killed
