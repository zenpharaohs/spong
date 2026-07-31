from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from spong import model, portrait, sturm, topology
from spong.charts import Branch


def _canonical_events(events):
    return sorted(events, key=lambda event: (event[0], event[1], event[2]))


def test_native_contact_scan_matches_python_bvh_oracle():
    native = pytest.importorskip("spong._native")
    if not hasattr(native, "ContactScan"):
        pytest.skip("native contact scanner not built")
    rng = np.random.default_rng(41017)
    first = np.cumsum(rng.normal(size=(83, 2)), axis=0)
    second = np.cumsum(rng.normal(size=(71, 2)), axis=0)
    tolerance = 128*np.finfo(float).eps*max(
        1.0, np.max(np.abs(first)), np.max(np.abs(second)))

    expected_pair = _canonical_events(topology._pair_events(
        first, topology._tree(first), second, topology._tree(second),
        tolerance))
    actual_pair = _canonical_events(native.ContactScan(
        np.ascontiguousarray(first), np.ascontiguousarray(second),
        tolerance, False))
    expected_self = _canonical_events(topology._self_events(
        first, topology._tree(first), tolerance))
    actual_self = _canonical_events(native.ContactScan(
        np.ascontiguousarray(first), None, tolerance, True))

    for expected, actual in ((expected_pair, actual_pair),
                             (expected_self, actual_self)):
        assert [(x[0], x[1], x[2]) for x in actual] == [
            (x[0], x[1], x[2]) for x in expected]
        np.testing.assert_allclose(
            [x[3] for x in actual], [x[3] for x in expected],
            rtol=0.0, atol=4*np.finfo(float).eps)


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
    minimum = max(e.minima, key=lambda q: q.b)
    target = np.array([minimum.a, minimum.b])
    unstable = [
        Branch("unstable",
               np.array([[.9, .9], target]), "capture",
               diag={"target": target}),
        Branch("unstable",
               np.array([[1.1, .9], target]), "capture",
               diag={"target": target}),
    ]
    same_basin = topology.audit(m, e, unstable, (-2., 3., -2., 2.))
    assert not same_basin["forbidden_intersections"]
    assert not same_basin["ambiguous_contacts"]

    stable = Branch(
        "stable", np.array([[-2., .95], [3., .95]]), "box_exit")
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
    assert len(cert["unstable_ends"]) == nstable
    assert cert["branch_inventory"]["certified"]
    assert all(end["certified"] for end in cert["unstable_ends"])
    assert all(tail["certified"] for tail in cert["stable_tails"])
    assert all(tail["level_lower"] is not None
               for tail in cert["stable_tails"])


def test_capture_certificate_uses_a_preconnector_sublevel_tube():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m)
    captures = [
        end for end in p.ledger["topology"]["unstable_ends"]
        if end["kind"] == "finite_capture"]
    assert captures
    for end in captures:
        assert end["certified"]
        branch = p.branches[end["branch"]]
        assert end["entry_index"] < len(branch.Y)-1
        assert end["b_interval"][0] is not None
        assert end["b_interval"][1] is not None


def test_exact_level_inventory_ignores_additive_loss_constant(d2):
    m, e = d2
    q = e.minima[0]
    point = np.array([q.a+1e-5, q.b+1e-5])
    base = topology._sublevel_component_inventory(m, e, point)
    shifted_model = replace(m)
    object.__setattr__(shifted_model, "C", m.C+Fraction(10**100))
    shifted = topology._sublevel_component_inventory(
        shifted_model, e, point)
    assert base["certified"] and shifted["certified"]
    assert base["bounded"] == shifted["bounded"]
    assert base["unbounded_sides"] == shifted["unbounded_sides"]
    assert [q.b for q in base["minima"]] == [q.b for q in shifted["minima"]]
    assert [q.b for q in base["saddles"]] == [q.b for q in shifted["saddles"]]


def test_asymptote_residual_alone_does_not_certify_stable_escape(d2):
    m, e = d2
    q = e.saddles[0]
    branch = Branch(
        "stable",
        np.array([[q.a, q.b], [q.a+1e-4, q.b+1e-4]]),
        "box_exit",
        certs={"asymptote": {
            "residual": 0.0, "radii": [1.0, 2.0, 4.0]}},
        diag={"saddle_b": q.b})
    result = topology.audit(m, e, [branch], (-2., 2., -2., 2.))
    assert result["status"] == "fp64_unresolved"
    assert not result["stable_tails"][0]["certified"]
    assert result["stable_tails"][0]["reason"] == "no_box_boundary_crossing"


def test_asymptote_residual_does_not_suppress_tail_contact(d2):
    m, e = d2
    tiny = 16*np.finfo(float).eps
    asymptote = {"residual": 0.0, "radii": [1.0, 2.0, 4.0]}
    branches = [
        Branch("stable", np.array([[-1., 0.], [1., 0.]]), "box_exit",
               certs={"asymptote": asymptote}),
        Branch("stable", np.array([[-1., tiny], [1., tiny]]), "box_exit",
               certs={"asymptote": asymptote}),
    ]
    result = topology.audit(m, e, branches, (-2., 2., -2., 2.))
    assert result["status"] == "fp64_unresolved"
    assert result["ambiguous_contacts"]


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


def test_strictly_monotone_subarc_is_not_a_global_self_contact(d2):
    m, e = d2
    tiny = 16*np.finfo(float).eps
    branch = Branch(
        "unstable",
        np.array([[0., 0.], [1., 0.], [1.+tiny, 0.], [2., 0.]]),
        "abort_step_failure")
    # Exercise the self-event predicate directly because an incomplete
    # branch intentionally short-circuits the full topology audit.
    root = topology._tree(branch.Y)
    events = list(topology._self_events(branch.Y, root, 1e-12))
    assert events
    kept = [event for event in events
            if not (event[2] == "ambiguous"
                    and topology._strictly_monotone_subarc(
                        branch.Y, event[0], event[1]))]
    assert not kept


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
    p = portrait.compute(m)
    candidates = p.ledger["topology"]["unstable_candidates"]
    levels = {q.b: m.L(q.a, q.b) for q in e.minima}
    for item in candidates:
        for b in item["eligible_minimum_b"]:
            assert levels[b] <= item["sampled_loss_floor"] + 1e-12


def test_certified_compute_records_stepup_or_success():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.certified_compute(m, max_geometry_level=0)
    attempts = p.ledger["topology"]["attempts"]
    assert len(attempts) == 1
    attempt = dict(attempts[0])
    elapsed = attempt.pop("elapsed_sec")
    uncertified_ends = attempt.pop("uncertified_ends")
    assert elapsed >= 0.0
    assert uncertified_ends == 0
    assert attempt == {
        "geometry_level": 0, "status": "certified",
        "reason": None,
        "forbidden": 0, "ambiguous": 0, "uncertified_tails": 0}


def test_minimum_basin_uses_conditioned_critical_spectrum(d2, monkeypatch):
    m, e = d2

    def forbidden_global_hessian(*_args, **_kwargs):
        raise AssertionError("global FP64 Hessian must not be diagonalized")

    monkeypatch.setattr(type(m), "hessL", forbidden_global_hessian)
    radii = topology._minimum_basin_radii(m, e)
    assert set(radii) == {(float(q.a), float(q.b)) for q in e.minima}
