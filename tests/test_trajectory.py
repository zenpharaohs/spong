"""Evidence ownership, adversarial proposals, and production assembly paths."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from spong import engine, model, portrait, zoo
from spong.trajectory import PieceRecorder, TrajectoryPiece as Piece
from spong.trajectory_check import inventory, inspect_trajectory, measure_jet


def test_partition_refuses_missing_duplicate_unknown_and_invalid_ranges():
    points = np.zeros((4, 2))
    good = [Piece("potential_ascent", 0, 2), Piece("handoff", 2, 3)]
    assert inventory(points, good)["status"] == "complete"
    for bad in (good[:1], good+good, [replace(good[0], start=1)],
                [replace(good[0], stop=4)], [Piece("potential_ascent", 0, 0)],
                [replace(good[0], stop=2.0)]):
        assert inventory(points, bad)["status"] == "invalid"
    assert inventory(points, [Piece("new_unchecked_chart", 0, 3)])["reason"] == "unknown_piece"
    points[1, 0] = np.nan
    assert inventory(points, good)["reason"] == "nonfinite_points"


def test_recorder_preserves_geometry_and_separates_seam_and_connector():
    pts = [(0., 0.)]
    segment = [(0., 1e-20), (1., 1.), (2., 2.), (3., 3.)]
    rec = PieceRecorder()
    rec.append(pts, segment, "potential_prefix", term="capture")
    assert pts == [(0., 0.)]+segment[1:]
    assert [p.kind for p in rec.pieces] == [
        "handoff", "potential_prefix", "capture_connector"]
    assert inventory(pts, rec.pieces)["status"] == "complete"
    rec.truncate(2)
    assert inventory(pts[:3], rec.pieces)["status"] == "complete"
    rec.truncate(0)
    assert rec.pieces == []


def polynomial_proposal():
    # L=a²: a=t, b=t². At t=1/2, the true tangent is (1,1).
    m = SimpleNamespace(alpha=(Q(1),), beta=(Q(0),))
    evidence = {"representation": "centered_polynomial", "center": (0., 0.),
                "coefficients": ((0., 1.), (0., 0., 1.)),
                "parameters": (0., 0., 0.5, 1.),
                "tangents": ((999., 999.),)*4}
    points = np.array([[0., 0.], [0., 0.], [.5, .25], [1., 1.]])
    return m, points, evidence


def test_jet_checker_rederives_tangent_and_checks_flow_and_geometry():
    m, points, evidence = polynomial_proposal()
    result = measure_jet(m, points, evidence, 1)
    assert result["status"] == "measured"
    assert result["max_sin_squared"] == pytest.approx(4/5)
    assert result["opposed_flow"] == 0
    assert result["samples"] == 5  # vertices and midparameters, duplicate origin once
    evidence["tangents"] = ()  # producer tangent evidence cannot change the answer
    assert measure_jet(m, points, evidence, 1) == result
    assert measure_jet(m, points, evidence, -1)["opposed_flow"] == 4
    points[2, 0] += .01
    assert measure_jet(m, points, evidence, 1)["reason"] == "geometry_mismatch"


def test_centered_offsets_survive_global_rounding_in_independent_evaluation():
    # L=(a-2**60)² (constant omitted), so exact centered evaluation is
    # nonstationary even though every displayed a coordinate is identical.
    m = SimpleNamespace(alpha=(Q(1),), beta=(Q(2**60),))
    evidence = {"representation": "centered_polynomial", "center": (2.**60, 0.),
                "coefficients": ((0., 1.), (0.,)), "parameters": (0., .5, 1.)}
    points = np.array([[2.**60, 0.]]*3)
    result = measure_jet(m, points, evidence, 1)
    assert result["status"] == "measured"
    assert result["stationary_samples"] == result["opposed_flow"] == 0
    assert result["max_sin_squared"] == 0


@pytest.mark.parametrize("change", [
    {"coefficients": ((np.nan,), (0.,))}, {"parameters": (0., 1., .5, 2.)},
    {"center": (0.,)}, {"parameters": ()}, {"representation": "invented"},
])
def test_malformed_jet_evidence_is_refused(change):
    m, points, evidence = polynomial_proposal()
    evidence.update(change)
    assert measure_jet(m, points, evidence, 1)["status"] == "invalid"


def test_known_polyline_is_unresolved_and_unknown_piece_never_passes():
    m, points, _ = polynomial_proposal()
    branch = SimpleNamespace(Y=points, kind="stable",
                             pieces=(Piece("potential_ascent", 0, 3),))
    result = inspect_trajectory(m, branch, measure_jets=True)
    assert result["coverage"]["status"] == "complete"
    assert result["status"] == "unresolved" and not result["certified"]
    assert result["pieces"][0]["reason"] == "smooth_evidence_unavailable"
    branch.pieces = (Piece("invented", 0, 3),)
    assert inspect_trajectory(m, branch)["coverage"]["reason"] == "unknown_piece"
    branch.pieces = (Piece("manifold_jet", 0, 3, None),)
    assert inspect_trajectory(m, branch)["coverage"]["reason"] == "invalid_evidence"


@pytest.mark.parametrize("stub_mode", ["jet", "grid"])
@pytest.mark.parametrize("name", ["minimal-quartet", "dead-neuron-far-saddle-d3"])
def test_production_phase_inventory_and_retained_evidence(monkeypatch, stub_mode, name):
    monkeypatch.setenv("SPONG_STUB_MODE", stub_mode)
    case = zoo.get(name)
    m = model.build(list(case.f), list(case.g),
                    model.moments_uniform01(2*max(len(case.f), len(case.g))-1))
    with engine.using("native"):
        p = portrait.certified_compute(m)
    assert len(p.ledger["trajectory_evidence"]) == len(p.branches)
    for br in p.branches:
        assert inventory(br.Y, br.pieces)["status"] == "complete"
        assert not any(piece.kind == "unknown" for piece in br.pieces)
        for piece in br.pieces:
            if piece.kind == "manifold_jet":
                e = piece.evidence
                assert len(e["parameters"]) == piece.stop-piece.start+1
                assert len(e["tangents"]) == len(e["parameters"])
                # Check one actual nontrivial launch in exact arithmetic;
                # the full every-sample probe remains a deliberate tool.
                selection = [0, 1, len(e["parameters"])//2, len(e["parameters"])-1]
                sparse = {**e, "parameters": tuple(e["parameters"][i] for i in selection)}
                result = measure_jet(m, br.Y[np.array(selection)+piece.start],
                                     sparse, 1 if br.kind == "stable" else -1)
                assert result["status"] == "measured"
    if stub_mode == "jet":
        br = next(b for b in p.branches if b.pieces and b.pieces[0].kind == "manifold_jet")
        tampered = deepcopy(br)
        tampered.Y[10, 0] += 1
        assert inspect_trajectory(m, tampered, measure_jets=True)["status"] == "invalid"


@pytest.mark.parametrize("seed,expected", [
    (206935347, {"poincare_grid", "continuation_after_shallow_rejection"}),
    (483412341, {"centered_grid_extension", "hadamard_graph"}),
    (1283395251, {"centered_arrival", "capture_connector"}),
])
def test_rare_construction_regressions(monkeypatch, seed, expected):
    import random
    from pathlib import Path
    from spong import sturm

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from qualify import directed_model

    monkeypatch.setenv("SPONG_STUB_MODE", "jet")
    m = directed_model(random.Random(seed), 5)[0]
    with engine.using("native"):
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
        p = portrait.compute(m, _enumeration=e, _skip_audit=True)
    kinds = {piece.kind for b in p.branches for piece in b.pieces}
    assert expected <= kinds
    for br in p.branches:
        assert inventory(br.Y, br.pieces)["status"] == "complete"
