"""Independent attachment/boundary checks and policy refusal controls."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from spong import engine, model, portrait, zoo, merge_tree, sturm
from spong.portrait_check import CredibilityPolicy, assess, check_launch, check_end


@pytest.fixture(scope='module')
def proposal():
    z = zoo.get('minimal-quartet')
    m = model.build(list(z.f),list(z.g),model.moments_uniform01(2*max(len(z.f),len(z.g))-1))
    with engine.using('native'):
        p = portrait.certified_compute(m)
    e = sturm.enumerate_critical_points(m)
    return p,e,merge_tree.build(m,e)


def test_launch_recomputes_eigenspace_and_side(proposal):
    p,e,_ = proposal
    for b in p.branches:
        assert check_launch(p.model,e,b)['status'] == 'measured'
    b = deepcopy(p.branches[0])
    b.diag['stable_sign'] *= -1
    assert check_launch(p.model,e,b)['status'] == 'invalid'
    b = deepcopy(p.branches[0])
    ev = b.pieces[0].evidence
    ev['coefficients'] = tuple(tuple(-x for x in c) for c in ev['coefficients'])
    assert check_launch(p.model,e,b)['status'] == 'invalid'
    b.Y[0,0] += 1
    assert check_launch(p.model,e,b)['reason'] == 'launch_center'


def test_endpoint_fate_and_boundary_orientation_are_separate(proposal):
    p,e,tree = proposal
    b = next(b for b in p.branches if b.term=='box_exit')
    result = check_end(p.model,e,tree,b,p.box)
    assert result['fate_exact'] and result['outward'] and result['flow_aligned']
    # Same exact fate, but no longer outside the claimed boundary.
    result = check_end(p.model,e,tree,b,(-1e100,1e100,-1e100,1e100))
    assert result['fate_exact'] and not result['outward']
    bad = deepcopy(b)
    last = bad.pieces[-1]
    data = np.frombuffer(last.evidence['trace'],dtype=float).copy().reshape(-1,24)
    dense_ids = np.where(data[:,0]==1)[0]
    data[dense_ids[-1],6:14] *= -1
    ev = {**last.evidence,'trace':data.tobytes()}
    bad.pieces = (*bad.pieces[:-1],replace(last,evidence=ev))
    result = check_end(p.model,e,tree,bad,p.box)
    assert not result['flow_aligned']


def test_capture_checks_connector_against_original_field(proposal):
    p,e,tree = proposal
    b = next(b for b in p.branches if b.term=='capture')
    r = check_end(p.model,e,tree,b,p.box)
    assert r['fate_exact']
    assert r['connector']['sin_squared'] > 1e-6
    bad = deepcopy(b); bad.Y[-1,0] += 1
    assert check_end(p.model,e,tree,bad,p.box)['status'] != 'measured'


def test_missing_measurements_and_stale_ledger_never_promote(proposal):
    p,_,_ = proposal
    policy = CredibilityPolicy(1e-6,1e-12,64)
    bad = replace(p, ledger=deepcopy(p.ledger))
    bad.ledger['topology']['status']='certified'
    r = assess(bad,policy,measure=False)
    assert r['status']=='unresolved' and r['certified'] is False
    assert all('trajectory_geometry' in b['issues'] for b in r['branches'])
    # A tiny budget explicitly refuses the long native capture tails.
    r = assess(bad,replace(policy,sample_budget=1))
    assert r['status']=='unresolved'
    assert any(x.get('reason')=='measurement_budget' for b in r['branches'] for x in b['trajectory']['pieces'])


@pytest.mark.parametrize('kwargs', [
    {'max_sin_squared':0}, {'max_sin_squared':float('nan')},
    {'max_eigen_sin_squared':1}, {'max_binding_eps':float('inf')},
    {'sample_budget':0}, {'sample_budget':1.5},
])
def test_policy_requires_explicit_valid_targets(kwargs):
    args=dict(max_sin_squared=1e-6,max_eigen_sin_squared=1e-12,max_binding_eps=64)
    args.update(kwargs)
    with pytest.raises(ValueError):
        CredibilityPolicy(**args)


def test_checker_import_does_not_load_constructor():
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]/"src")}
    subprocess.run([sys.executable, "-c", "import sys; import spong.portrait_check; "
                    "assert 'spong.charts' not in sys.modules; "
                    "assert 'spong.engine' not in sys.modules; "
                    "assert 'spong.local' not in sys.modules; "
                    "import spong; assert 'Resolution' in dir(spong); "
                    "from spong import resolve, ResolutionPolicy; "
                    "assert callable(resolve); assert ResolutionPolicy is not None"],
                   check=True, env=env)
