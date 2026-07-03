import json
import builtins

from spong import cli, model, portrait


def test_random_phase_portrait_cli_smoke(tmp_path):
    rc = cli.main([
        "--seed", "123",
        "--count", "1",
        "--same",
        "--f-degree", "2",
        "--view", "-4", "4", "-10", "4",
        "--output-dir", str(tmp_path),
        "--prefix", "smoke",
        "--levels", "6",
        "--grid", "101",
        "--no-stable",
        "--zoom-close", "1",
        "--zoom-grid", "51",
    ])
    assert rc == 0
    svg = tmp_path / "smoke_0000_seed_66316748_plane.svg"
    zoom = tmp_path / "smoke_0000_seed_66316748_zoom_00.svg"
    summary = tmp_path / "smoke_0000_seed_66316748_summary.json"
    index = tmp_path / "smoke_index.json"
    assert svg.exists()
    assert zoom.exists()
    assert summary.exists()
    assert index.exists()
    data = json.loads(summary.read_text())
    assert data["seed"] == 66316748
    assert data["enumeration"]["n_critical"] >= 1
    assert data["ledger_summary"]["balanced"] is True
    assert len(data["zooms"]) == 1
    assert data["zooms"][0]["svg"].endswith("_zoom_00.svg")


def test_zoo_quadratic_stiff_cli(tmp_path):
    rc = cli.main([
        "--zoo", "quadratic-stiff",
        "--output-dir", str(tmp_path),
        "--prefix", "zoo",
        "--levels", "6",
        "--grid", "101",
        "--no-stable",
    ])
    assert rc == 0
    svg = tmp_path / "zoo_quadratic-stiff_plane.svg"
    summary = tmp_path / "zoo_quadratic-stiff_summary.json"
    index = tmp_path / "zoo_zoo_index.json"
    assert svg.exists()
    assert summary.exists()
    assert index.exists()
    data = json.loads(summary.read_text())
    assert data["zoo"]["name"] == "quadratic-stiff"
    assert data["zoo"]["discovery_seed"] == 2735729614
    assert data["enumeration"]["n_critical"] == 6
    assert data["enumeration"]["n_min"] == 3
    assert data["enumeration"]["n_saddle"] == 3


def test_degree_one_cli_smoke(tmp_path):
    rc = cli.main([
        "--seed", "5",
        "--count", "1",
        "--same",
        "--f-degree", "1",
        "--output-dir", str(tmp_path),
        "--prefix", "d1",
        "--levels", "4",
        "--grid", "51",
    ])
    assert rc == 0
    summary = tmp_path / "d1_0000_seed_2881021351_summary.json"
    assert summary.exists()
    data = json.loads(summary.read_text())
    assert data["enumeration"]["n_critical"] == 2
    assert data["enumeration"]["n_min"] == 1
    assert data["enumeration"]["n_saddle"] == 1
    assert data["ledger_summary"]["balanced"] is True


def test_random_phase_portrait_cli_open_smoke(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "_open_one",
                        lambda path, viewer="auto":
                        opened.append((path, viewer)) or True)
    rc = cli.main([
        "--seed", "7",
        "--count", "1",
        "--same",
        "--f-degree", "2",
        "--output-dir", str(tmp_path),
        "--prefix", "open",
        "--levels", "4",
        "--grid", "51",
        "--no-stable",
        "--open",
        "--viewer", "inkscape",
    ])
    assert rc == 0
    assert len(opened) == 1
    assert opened[0][0].endswith("_plane.svg")
    assert opened[0][1] == "inkscape"


def test_viewer_auto_uses_platform_default_even_if_inkscape_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_inkscape_command",
                        lambda: ["open", "-a", "Inkscape"])
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda cmd, **kw: calls.append(cmd))
    assert cli._open_one("/tmp/a.svg", viewer="auto")
    assert calls == [["open", "/tmp/a.svg"]]


def test_viewer_inkscape_is_explicit(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_inkscape_command",
                        lambda: ["open", "-a", "Inkscape"])
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda cmd, **kw: calls.append(cmd))
    assert cli._open_one("/tmp/a.svg", viewer="inkscape")
    assert calls == [["open", "-a", "Inkscape", "/tmp/a.svg"]]


def test_viewer_auto_falls_back_to_open(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_inkscape_command", lambda: None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda cmd, **kw: calls.append(cmd))
    assert cli._open_one("/tmp/a.svg", viewer="auto")
    assert calls == [["open", "/tmp/a.svg"]]


def test_parse_view_text():
    assert cli._parse_view_text("-1, 2, -3, 4") == (-1.0, 2.0, -3.0, 4.0)


def test_fit_view_to_compute_box():
    assert cli._fit_view_to_box((-10, 10, -10, 10),
                                (-2, 5, -6, 3)) == (-2, 5, -6, 3)
    assert cli._fit_view_to_box((-1, 1, -2, 2),
                                (-2, 5, -6, 3)) == (-1, 1, -2, 2)


def test_pause_view_command_writes_view(tmp_path, monkeypatch):
    opened = []
    replies = iter(["v", "-1 1 -2 2", ""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(replies))
    monkeypatch.setattr(cli, "_open_one",
                        lambda path, viewer="auto":
                        opened.append((path, viewer)) or True)
    rc = cli.main([
        "--seed", "7",
        "--count", "1",
        "--same",
        "--f-degree", "2",
        "--output-dir", str(tmp_path),
        "--prefix", "paused",
        "--levels", "4",
        "--grid", "51",
        "--no-stable",
        "--pause",
    ])
    assert rc == 0
    view = tmp_path / "paused_0000_seed_4058335882_view_00.svg"
    assert view.exists()
    assert any(path.endswith("_view_00.svg") for path, _viewer in opened)


def test_pause_view_recomputes_when_view_exceeds_compute_box(tmp_path,
                                                            monkeypatch):
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m, view=(0.95, 1.05, 0.95, 1.05),
                         trace_stable_branches=False)
    p = portrait.Portrait(
        p.model, p.enumeration, p.branches,
        (0.9, 1.1, 0.9, 1.1), p.view, p.ledger)
    rendered_boxes = []
    replies = iter(["v", "-4 4 -10 4", ""])
    args = cli.build_parser().parse_args([
        "--levels", "4",
        "--grid", "51",
        "--no-stable",
        "--pause",
    ])
    args.view = cli._view(args.view)
    args.auto_open = False
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(replies))

    def fake_render(p_rendered, path, view, args, title):
        rendered_boxes.append(p_rendered.box)
        return path

    monkeypatch.setattr(cli, "_render_plane_path", fake_render)
    assert cli._pause_loop(p, tmp_path, "recompute", "title", args) == "next"
    assert rendered_boxes
    assert rendered_boxes[0][0] <= -4
    assert rendered_boxes[0][1] >= 4
    assert rendered_boxes[0][2] <= -10
    assert rendered_boxes[0][3] >= 4


def test_pause_view_clamps_when_recompute_cannot_cover_view(tmp_path,
                                                            monkeypatch):
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m, trace_stable_branches=False)
    p = portrait.Portrait(
        p.model, p.enumeration, p.branches,
        (-2.0, 2.0, -3.0, 3.0), p.view, p.ledger)
    rendered_views = []
    replies = iter(["v", "-10 10 -10 10", ""])
    args = cli.build_parser().parse_args([
        "--levels", "4",
        "--grid", "51",
        "--no-stable",
        "--pause",
    ])
    args.view = cli._view(args.view)
    args.auto_open = False
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(replies))
    monkeypatch.setattr(cli.portrait, "compute", lambda *a, **kw: p)

    def fake_render(_p, _path, view, _args, _title):
        rendered_views.append(view)
        return _path

    monkeypatch.setattr(cli, "_render_plane_path", fake_render)
    assert cli._pause_loop(p, tmp_path, "clamp", "title", args) == "next"
    assert rendered_views == [(-2.0, 2.0, -3.0, 3.0)]
