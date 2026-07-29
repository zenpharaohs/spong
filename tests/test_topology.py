import numpy as np

from spong import model, portrait, sturm, topology
from spong.charts import Branch


def test_bvh_audit_finds_a_forbidden_transverse_crossing(d2):
    m, e = d2
    branches = [
        Branch("stable", np.array([[-1., -1.], [1., 1.]]), "box_exit"),
        Branch("unstable", np.array([[-1., 1.], [1., -1.]]), "box_exit"),
    ]
    result = topology.audit(m, e, branches, (-2., 2., -2., 2.))
    assert result["status"] == "fp64_unresolved"
    assert len(result["forbidden_intersections"]) == 1
    np.testing.assert_allclose(
        result["forbidden_intersections"][0]["point"], (0., 0.))


def test_same_minimum_unstable_contacts_are_trimmed_but_separatrix_is_not(d2):
    """Basin-interior arcs may be rerouted after their common capture; a
    crossing with a stable basin boundary remains topology-changing."""
    m, e = d2
    target = np.array([2.0, 0.0])
    unstable = [
        Branch("unstable",
               np.array([[-1., -1.], [0., 0.], target]), "capture"),
        Branch("unstable",
               np.array([[-1., 1.], [0., 0.], target]), "capture"),
    ]
    same_basin = topology.audit(m, e, unstable, (-2., 3., -2., 2.))
    assert not same_basin["forbidden_intersections"]
    assert not same_basin["ambiguous_contacts"]

    stable = Branch(
        "stable", np.array([[-1., -.5], [1., -.5]]), "box_exit")
    separated = topology.audit(
        m, e, [unstable[0], stable], (-2., 3., -2., 2.))
    assert separated["forbidden_intersections"]


def test_portraitist_runs_stable_first_and_emits_topology_certificate():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m)
    nstable = 2*len(p.enumeration.saddles)
    assert all(br.kind == "stable" for br in p.branches[:nstable])
    assert all(br.kind == "unstable" for br in p.branches[nstable:])
    cert = p.ledger["topology"]
    assert cert["status"] in ("certified", "fp64_unresolved")
    assert not cert["forbidden_intersections"]
    assert "backbone_crossings" in cert
    assert len(cert["unstable_candidates"]) == nstable


def test_audit_refuses_incomplete_branch_without_intersection_scan():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    branch = Branch(
        "unstable", np.array([[0.0, 0.0], [1.0, 1.0]]),
        "abort_step_failure")
    result = topology.audit(m, e, [branch], (-2., 2., -2., 2.))
    assert result["status"] == "fp64_unresolved"
    assert result["resolution_reason"] == "branch_abort"
    assert result["aborted_branches"] == [0]
    assert result["raw_event_count"] == 0


def test_common_saddle_stub_germs_use_local_transversality_certificate(d2):
    """Centered samples can coalesce within global FP64 predicate tolerance.

    The local fixed-point/injectivity certificate, rather than a second
    poorly conditioned global orientation test, owns this common germ.
    """
    m, e = d2
    stable = Branch(
        "stable",
        np.array([[0., 0.], [1e-14, 1e-6], [1., 1.]]),
        "box_exit", diag={"saddle_b": 0.0, "critical_steps": 2})
    unstable = Branch(
        "unstable",
        np.array([[0., 0.], [-1e-14, 1e-6], [-1., 1.]]),
        "box_exit", diag={"saddle_b": 0.0, "critical_steps": 2})
    result = topology.audit(
        m, e, [stable, unstable], (-2., 2., -2., 2.))
    assert result["forbidden_count"] == 0
    assert result["ambiguous_count"] == 0


def test_loss_floor_excludes_higher_minimum_candidates(d2):
    m, e = d2
    p = portrait.compute(m, trace_stable_branches=False)
    candidates = p.ledger["topology"]["unstable_candidates"]
    levels = {q.b: m.L(q.a, q.b) for q in e.minima}
    for item in candidates:
        for b in item["eligible_minimum_b"]:
            assert levels[b] <= item["sampled_loss_floor"] + 1e-12


def test_certified_compute_records_stepup_or_success():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.certified_compute(
        m, max_geometry_level=0, trace_stable_branches=False)
    attempts = p.ledger["topology"]["attempts"]
    assert len(attempts) == 1
    attempt = dict(attempts[0])
    elapsed = attempt.pop("elapsed_sec")
    assert elapsed >= 0.0
    assert attempt == {
        "geometry_level": 0, "status": "certified", "reason": None,
        "forbidden": 0, "ambiguous": 0, "uncertified_tails": 0}


def test_minimum_basin_uses_conditioned_critical_spectrum(d2, monkeypatch):
    m, e = d2

    def forbidden_global_hessian(*_args, **_kwargs):
        raise AssertionError("global FP64 Hessian must not be diagonalized")

    monkeypatch.setattr(type(m), "hessL", forbidden_global_hessian)
    radii = topology._minimum_basin_radii(m, e)
    assert set(radii) == {(float(q.a), float(q.b)) for q in e.minima}
