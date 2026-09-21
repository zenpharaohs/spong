"""Proof boundaries for the connectivity stage, independent of full portraits."""
from dataclasses import replace
from fractions import Fraction

import pytest

from spong import model
from spong import connectivity as C


def build(f,g):
    return model.build(f,g,model.moments_uniform01(8))


def test_forced_two_minimum_attachment_needs_no_transport(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('forced branches must not request transport')
    monkeypatch.setattr(C,'_extend_proposal',forbidden)
    result=C.determine_connectivity(build([1,1],[1,0,1]))
    assert result.complete and result.finite_plane_smale
    assert [b.minimum for b in result.branches]==[0,2]
    assert all(b.witness=='validated_launch_component' for b in result.branches)
    assert not result.candidate_connections


def test_section_transport_resolves_capture_without_full_portrait():
    m=build([1],[1,1])
    first=C.determine_connectivity(m,policy=C.ConnectivityPolicy(transport=False))
    assert not first.complete and first.finite_plane_smale
    assert first.branches[0].end=='b_minus_infinity'
    result=C.determine_connectivity(m,enumeration=first.enumeration)
    assert result.complete and result.finite_plane_smale
    capture=next(b for b in result.branches if b.status=='capture')
    assert capture.minimum==1 and capture.witness=='validated_section_component'
    assert capture.sections==1 and capture.proof.status=='validated'
    assert result.as_dict()['rheostat_distance_lower_bound'] is None
    assert not result.as_dict()['compactified_smale_certified']


def test_transport_budget_exhaustion_is_not_near_wall_evidence():
    result=C.determine_connectivity(build([1],[1,1]),
        policy=C.ConnectivityPolicy(max_proposal_steps=0))
    assert not result.complete and result.finite_plane_smale
    assert any('budget exhausted' in (b.reason or '') for b in result.branches)
    assert not result.rheostat


def test_unresolved_critical_sign_cannot_omit_a_possible_destination(monkeypatch):
    monkeypatch.setattr(C.merge_tree,'value_sign',lambda *args: None)
    with pytest.raises(ValueError,match='unresolved critical-value signs'):
        C._checked_components(None,type('Enumeration',(),{'points':[object()]})(),Fraction(1))


def test_model_hypothesis_and_non_morse_are_distinct():
    assert C.determine_connectivity(build([1],[0,1])).status=='model_hypothesis_failed'
    assert C.determine_connectivity(build([1],[1,0,1])).status=='certified_non_morse'


def test_missing_unstable_branch_cannot_certify(monkeypatch):
    m=build([1,1],[1,0,1])
    e=C.sturm.materialize_stubs(m,C.sturm.enumerate_critical_points(m))
    e=replace(e,points=tuple(replace(p,stubs=tuple(s for s in p.stubs
        if s.manifold!='unstable' or s.orientation==-1)) if p.kind=='saddle' else p
        for p in e.points))
    result=C.determine_connectivity(m,enumeration=e)
    assert not result.complete and not result.finite_plane_smale
    assert result.diagnostics


def test_numerical_wall_never_promotes_connectivity(monkeypatch):
    monkeypatch.setattr(C,'_rheostat_diagnostic',lambda *args:{
        'grade':'NUMERICAL_ORACLE','wall_parameter':'1.0000000000000001'})
    result=C.determine_connectivity(build([1,1,1],[1,1,1]),
        policy=C.ConnectivityPolicy(transport=False,rheostat_diagnostics=True))
    assert len(result.branches)==4
    assert sum(b.certified for b in result.branches)==3
    assert len(result.candidate_connections)==2 and len(result.rheostat)==2
    assert not result.complete and not result.finite_plane_smale
    assert result.as_dict()['rheostat_distance_lower_bound'] is None


@pytest.mark.parametrize('policy',[
    C.ConnectivityPolicy(proposal_step=float('nan')),
    C.ConnectivityPolicy(max_proposal_steps=-1),
    C.ConnectivityPolicy(rheostat_radius=1),
])
def test_invalid_policy_is_rejected(policy):
    with pytest.raises(ValueError,match='invalid connectivity policy'):
        C.determine_connectivity(build([1],[1,1]),policy=policy)
