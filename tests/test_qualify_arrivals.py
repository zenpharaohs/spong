"""The comparison harness must retain refusals and use fixed criteria."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('qualify_arrivals',Path(__file__).resolve().parents[1]/'scripts/qualify_arrivals.py')
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)


def test_cohort_is_reproducible_and_retains_all_recorded_calls():
    a=qa.cohort(20260925,2,2)
    assert a==qa.cohort(20260925,2,2)
    assert a[:4]!=qa.cohort(20260926,2,2)[:4]
    assert [r['cohort'] for r in a[:4]]==['random','random','directed','directed']
    assert sum(r['cohort']=='zoo' for r in a)==7
    assert sum(r['cohort']=='corpus' for r in a)==22


def test_no_small_residual_can_hide_wrong_flow_missing_geometry_or_budget():
    good={'status':'measured','max_sin_squared':1e-9,'binding_eps':.5,'join_eps':.5}
    assert qa.geometry_ok(good)
    for change in [{'status':'unresolved'},{'status':'invalid'}, {'opposed_flow':1},
                   {'stationary_samples':1},{'unvalidated_steps':1},
                   {'binding_eps':65},{'join_eps':65},{'max_sin_squared':1e-5}]:
        assert not qa.geometry_ok({**good,**change})
    assert not qa.geometry_ok({})
    assert qa.connector_ok({'status':'measured','sin_squared':1e-9,'flow_alignment':1,'loss_direction':1})
    assert not qa.connector_ok({'status':'measured','sin_squared':1e-30,'flow_alignment':-1,'loss_direction':1})


def test_proposal_exception_is_recorded_not_dropped():
    def fails():raise ValueError('arrival_resonance_or_small_divisor')
    result=qa.attempt(fails)
    assert result['status']=='error'
    assert 'arrival_resonance_or_small_divisor' in result['reason']
    assert result['seconds']>=0
