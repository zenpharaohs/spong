"""Public three-outcome resolution contract."""

from types import SimpleNamespace

from spong import (ResolutionPolicy, ResolutionReason, ResolutionStatus,
                   model, resolve)
from spong import portrait, qualification, resolution, sturm


def test_exact_non_morse_returns_terminal_certificate_without_geometry(
        monkeypatch):
    # g is independent of b, so L has an entire critical line rather than
    # isolated nondegenerate critical points.
    m = model.build([1], [1], model.moments_uniform01(1))

    def geometry_must_not_run(*args, **kwargs):
        raise AssertionError("non-Morse input reached the geometry engine")

    monkeypatch.setattr(portrait, "certified_compute", geometry_must_not_run)
    result = resolve(m)
    assert result.status is ResolutionStatus.CERTIFIED_NON_MORSE
    assert result.reason is ResolutionReason.EXACT_NON_MORSE
    assert result.certified and result.exact_morse is False
    assert result.portrait is None


def test_morse_policy_refusal_is_terminal_and_precedes_geometry(monkeypatch):
    m = model.build([1, 1, 1], [1, 1, 1],
                    model.moments_uniform01(5))
    enumeration = sturm.enumerate_critical_points(m)
    margin = qualification.skeleton_profile(
        m, enumeration)["morse_root_collision_margin_log2_eps"]

    def geometry_must_not_run(*args, **kwargs):
        raise AssertionError("refused input reached the geometry engine")

    monkeypatch.setattr(portrait, "certified_compute", geometry_must_not_run)
    result = resolve(
        m, policy=ResolutionPolicy(
            min_root_collision_margin_log2_eps=margin+1.0))
    assert result.status is ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED
    assert result.reason is ResolutionReason.ROOT_COLLISION_MARGIN
    assert not result.certified and result.exact_morse is True
    assert result.portrait is None
    assert "root-collision margin" in result.diagnostics[0]


def test_a_posteriori_geometry_refusal_is_a_terminal_answer(monkeypatch):
    m = model.build([1, 1, 1], [1, 1, 1],
                    model.moments_uniform01(5))

    def unresolved_geometry(_model, **kwargs):
        enumeration = sturm.materialize_stubs(
            _model, kwargs["_enumeration"])
        return SimpleNamespace(
            enumeration=enumeration,
            ledger={"topology": {
                "status": "fp64_unresolved",
                "resolution_reason": None,
                "ambiguous_count": 2,
                "forbidden_count": 0,
            }})

    monkeypatch.setattr(
        resolution.portrait, "certified_compute", unresolved_geometry)
    result = resolve(m)
    assert result.status is ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED
    assert result.reason is ResolutionReason.TOPOLOGY_UNRESOLVED
    assert not result.certified and result.exact_morse is True
    assert result.portrait is not None
    assert "ambiguous_contacts=2" in result.diagnostics


def test_successful_end_to_end_call_returns_only_certified_portrait():
    m = model.build([1, 1, 1], [1, 1, 1],
                    model.moments_uniform01(5))
    result = resolve(
        m, policy=ResolutionPolicy(max_geometry_level=0))
    assert result.status is ResolutionStatus.CERTIFIED_PORTRAIT
    assert result.reason is ResolutionReason.NONE
    assert result.certified and result.exact_morse is True
    assert result.portrait is not None
    assert result.portrait.ledger["topology"]["status"] == "certified"
