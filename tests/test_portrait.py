"""Phase-5 gates: full ledger emitted; parity on the tricky portrait;
rendering guarantees; the d=2 oracle end-to-end."""

import numpy as np
import pytest

from spong import model, portrait, render
from tests.test_enumeration import TRICKY_F


def test_ledger_complete_and_clean(tricky_portrait):
    p = tricky_portrait
    led = p.ledger
    e = led["enumeration"]
    assert e["n_critical"] == 8 and e["n_min"] == 4 and e["n_saddle"] == 4
    assert e["psi_positive[EXACT]"] and e["morse[EXACT]"]
    assert e["u2_alternation[EXACT]"]
    assert led["index_balance[EXACT]"]["balanced"]
    assert led["genericity[EXACT]"]["generic"]
    assert len(led["branches"]) == 16          # 4 saddles x 4 branches
    assert led["summary"]["all_branches_clean"]
    assert led["summary"]["worst_angle_energy"] < 1e-6
    assert led["summary"]["worst_max_turn_deg"] < 3.0


def test_tricky_branch_parity(tricky_portrait):
    """The founding MATLAB-parity gate, now through the full assembly."""
    p = tricky_portrait
    tricky = [b for b in p.ledger["branches"]
              if b["kind"] == "unstable"
              and abs(b["saddle_b"] - (-2.738230515199397)) < 1e-6]
    assert len(tricky) == 2
    for b in tricky:
        assert b["term"] == "capture"
        assert b["adjacency[RESIDUAL,thm-backed]"]
        assert b["angle_energy[RESIDUAL]"] < 1e-12
        # ...and it must have measured something: angle_energy skips vertices
        # whose grad L direction is below the digit budget, so E alone cannot
        # distinguish a clean pass from a vacuous one.  Whatever it skipped is
        # certified algebraically instead (out there the branch IS a* = B/A).
        assert b["angle_resolved"] > 1500
        if b["angle_unresolved"]:
            assert b["backbone_residual[RESIDUAL]"] < 1e-8


def test_adjacency_everywhere(tricky_portrait):
    for b in tricky_portrait.ledger["branches"]:
        if b["kind"] == "unstable":
            assert b["adjacency[RESIDUAL,thm-backed]"]


def test_asymptote_certificates_present(tricky_portrait):
    """Certificates exist on all separatrices; residuals are meaningful
    (small) only where the exit radius is genuinely asymptotic — the
    certificate must be honest about pre-asymptotic exits, not silent."""
    p = tricky_portrait
    stables = [(br, b) for br, b in zip(p.branches, p.ledger["branches"])
               if b["kind"] == "stable"]
    assert len(stables) == 8
    n_asymptotic = 0
    for br, b in stables:
        assert "asymptote_residual[RESIDUAL]" in b
        r_exit = float(np.hypot(*br.Y[-1]))
        if r_exit > 8.0:                       # genuinely toward the rim
            assert b["asymptote_residual[RESIDUAL]"] < 0.2
            n_asymptotic += 1
    assert n_asymptotic >= 2                   # the tricky saddle's pair


def test_render_svg(tricky_portrait, tmp_path):
    svg = render.plane_view(tricky_portrait, title="test")
    assert svg.startswith("<svg") and svg.count("<path") > 100
    assert 'vector-effect="non-scaling-stroke"' in svg
    assert 'fill="white" stroke="#111111"' in svg
    disk = render.disk_view(tricky_portrait)
    assert disk.startswith("<svg") and disk.count("<path") > 50
    assert 'vector-effect="non-scaling-stroke"' in disk
    assert 'fill="white" stroke="#111111"' in disk
    f = render.save(svg, str(tmp_path / "p.svg"))
    assert (tmp_path / "p.svg").stat().st_size > 10000


def test_close_unstable_zooms(tricky_portrait):
    zooms = render.close_unstable_zooms(tricky_portrait, n=1, samples=400)
    assert len(zooms) == 1
    z = zooms[0]
    assert z["separation"] > 0
    assert len(z["view"]) == 4
    assert z["view"][0] < z["center"][0] < z["view"][1]
    assert z["view"][2] < z["center"][1] < z["view"][3]


def test_branch_segment_crossing_view_is_rendered():
    X = np.array([0.0, 0.0])
    Y = np.array([-1.4, -2.9])
    runs = render._clip_polylines(X, Y, (-0.5, 0.5, -2.0, -1.0))
    assert len(runs) == 1
    assert runs[0][0] == (0.0, -1.4)
    assert runs[0][-1] == (0.0, -2.0)


def test_default_display_view_is_smaller_than_trace_box():
    f = [-0.27126925828072923, -0.7363598557165663, 0.7989868625933855]
    m = model.build(f, f, model.moments_uniform01(5))
    p = portrait.compute(m)
    assert p.view is not None
    assert p.box[0] < p.view[0] and p.box[1] > p.view[1]
    assert p.box[2] < p.view[2] and p.box[3] > p.view[3]
    assert any(
        br.kind == "stable"
        and (np.min(br.Y[:, 0]) < p.view[0] or np.max(br.Y[:, 0]) > p.view[1]
             or np.min(br.Y[:, 1]) < p.view[2] or np.max(br.Y[:, 1]) > p.view[3])
        for br in p.branches
    )


def test_backbone_sampling_is_adaptive_in_screen_space():
    f = [-0.3259270001197266, -0.3989071127950847, -0.08631058733881063]
    m = model.build(f, f, model.moments_uniform01(5))
    p = portrait.compute(m)
    view = p.box
    to_px = render._mapper(view, 1200, 900, 40)
    A, B = render._adaptive_backbone(m, view, to_px)
    runs = render._clip_polylines(A, B, view)
    assert runs
    for run in runs:
        pts = np.array([to_px(a, b) for a, b in run])
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        assert np.max(seg) <= 8.1


def test_d2_far_lower_minimum_connection_expands_compute_box():
    f = [1.0511595983436535, 2.207740477509359, -0.3128201040655276]
    m = model.build(f, f, model.moments_uniform01(5))
    p = portrait.compute(m)
    lower_min = min(p.enumeration.minima, key=lambda q: q.b)
    lower_saddle = min(p.enumeration.saddles, key=lambda q: q.b)
    branches = [br for br in p.branches
                if br.kind == "unstable"
                and abs(br.diag.get("saddle_b") - lower_saddle.b) < 1e-9
                and br.diag.get("target") is not None
                and abs(br.diag["target"][1] - lower_min.b) < 1e-9]
    assert len(branches) == 1
    br = branches[0]
    assert br.term == "capture"
    assert br.Y[-1, 0] == pytest.approx(lower_min.a, abs=1e-12)
    assert br.Y[-1, 1] == pytest.approx(lower_min.b, abs=1e-12)
    assert p.box[0] < np.min(br.Y[:, 0])


def test_linear_vs_d17_horizontal_branch_reaches_global_minimum():
    """Seed 1158725111: a high-degree fit to linear data made a finite
    branch almost horizontal, so |Δb|-only spacing exhausted max_steps."""
    f = [-0.9514652373581963, -1.7945943562259494]
    g = [
        -0.2302392536989173, 2.4650232077321212,
        -1.3228162550152864, 3.120814010750335,
        -1.4209087818998132, -2.0452998402729996,
        -0.10981879692405587, -1.5932352722307273,
        -0.6365486378841433, 0.19535772631955794,
        0.26391606930746814, -0.7697423065439618,
        -0.7428291736196287, -0.21651008677854963,
        1.7960186839315102, 0.4286765825415307,
        0.8227526313695055, 0.15929379398579482,
    ]
    m = model.build(f, g, model.moments_uniform01(35))
    p = portrait.compute(m, trace_stable_branches=False)
    saddle_b = 0.16855072623386724
    target_b = -0.1527183018008041
    branches = [
        br for br in p.branches
        if br.kind == "unstable"
        and abs(br.diag.get("saddle_b") - saddle_b) < 1e-9
        and br.diag.get("target") is not None
        and abs(br.diag["target"][1] - target_b) < 1e-9
    ]
    assert len(branches) == 1
    br = branches[0]
    assert br.term == "capture"
    assert br.Y[-1, 0] == pytest.approx(br.diag["target"][0], abs=1e-12)
    assert br.Y[-1, 1] == pytest.approx(br.diag["target"][1], abs=1e-12)
    assert len(br.Y) < 100000


def test_d2_end_to_end():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m)
    led = p.ledger
    assert led["enumeration"]["n_critical"] == 4
    assert len(led["branches"]) == 8
    assert led["summary"]["all_branches_clean"]
    assert led["summary"]["balanced"]
    assert led["summary"]["worst_angle_energy"] < 1e-6
    svg = render.plane_view(p)
    assert svg.startswith("<svg")
