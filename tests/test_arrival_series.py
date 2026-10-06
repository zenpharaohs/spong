"""Minimum entry charts: incoming modes, exact controls and refusals."""
from copy import deepcopy
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from spong import engine, model, portrait, zoo
from spong.arrival_series import parameterization, _propose, refine_capture
from spong.arrival_check import measure_arrival, check_target
from spong.trajectory_check import inventory, inspect_trajectory
from spong.local import _exact_symmetric_spectral


def linear_local(ratio):
    H = ((Q(1),Q(0)),(Q(0),Q(ratio)))
    return SimpleNamespace(exact_hessian=H,
                           exact_grad=(((Q(0),),(Q(1),)),((Q(0),Q(ratio)),)),
                           spectral=_exact_symmetric_spectral(H))


def test_two_mode_matching_retains_fast_transient():
    local = linear_local(Q(5,2))
    chart = parameterization(local,order=6)
    chart['center'] = (Q(0),Q(0))
    points,e = _propose(local,(.25,-.125),(0,0),chart=chart)
    assert points[0].tolist()==[.25,-.125] and points[-1].tolist()==[0,0]
    with localcontext() as ctx:
        ctx.prec=80
        for p,s in zip(points[:-1],e['parameters'][:-1]):
            t=D(s)
            assert p[0]==float(D('.25')*(-t).exp())
            assert p[1]==float(D('-.125')*(-D('2.5')*t).exp())
    assert all(x!='0' for x in e['amplitudes'])
    assert e['terminal']=='analytic_limit'


@pytest.mark.parametrize('ratio',[2,3,4,Q('2.0000000000001')])
def test_resonance_and_small_divisor_are_explicit_refusals(ratio):
    with pytest.raises(ValueError,match='resonance_or_small_divisor'):
        parameterization(linear_local(ratio),order=6)


MODEL = SimpleNamespace(alpha=(Q(1),Q(0),Q(5,2)),beta=(Q(1),))


def analytic_proposal():
    e={'representation':'poincare_frobenius_arrival','schema':1,
       'center':('1','0'),'terms':(('1','-.01','0'),),
       'parameters':('0','1','2',None),'terminal':'analytic_limit'}
    with localcontext() as ctx:
        ctx.prec=90
        points=np.array([[float(1-D('.01')*(-D(t)).exp()),0] for t in (0,1,2)]+[[1,0]])
    return points,e


def test_independent_analytic_control_and_binding():
    points,e=analytic_proposal()
    r=measure_arrival(MODEL,points,e)
    assert r['status']=='measured' and r['max_sin_squared']==0
    assert r['opposed_flow']==r['zero_tangents']==r['stationary_samples']==0
    assert r['limit']['positive_hessian'] and r['limit']['inward']
    assert r['binding_eps']<1
    points[1,0]+=.01
    assert measure_arrival(MODEL,points,e)['binding_eps']>64
    assert measure_arrival(MODEL,points,e,sample_budget=1)['reason']=='measurement_budget'


def test_checker_samples_fast_mode_even_when_producer_does_not():
    points,e=analytic_proposal()
    e['terms'] += (('1000000','0','.001'),('2000000','0','-.001'))
    r=measure_arrival(MODEL,points,e)
    # The extra terms cancel at the first point and vanish at the other
    # producer parameters. They bend the curve in a thin omitted layer.
    assert r['binding_eps']<1
    assert r['max_sin_squared']>.9


@pytest.mark.parametrize('change',[
    {'schema':2},{'parameters':('0','1','1',None)},
    {'parameters':('0',None,'2',None)}, {'terminal':'capture_radius'},
    {'terms':(('0','1','0'),)},{'terms':(('nan','1','0'),)},
    {'terms':(('1','0','0'),)}, {'center':('nan','0')},
])
def test_malformed_and_constant_arrivals_refuse(change):
    points,e=analytic_proposal();e.update(change)
    assert measure_arrival(MODEL,points,e)['status']=='invalid'


@pytest.fixture(scope='module',params=['minimal-quartet','dead-neuron-far-saddle-d3'])
def proposal(request):
    z=zoo.get(request.param)
    m=model.build(list(z.f),list(z.g),model.moments_uniform01(2*max(len(z.f),len(z.g))-1))
    with engine.using('native'):
        return portrait.certified_compute(m)


def test_entry_chart_preserves_prefix_and_removes_only_connector(proposal):
    p=proposal
    for b in p.branches:
        if b.term!='capture': continue
        saved=b.Y.copy()
        candidate=refine_capture(p.model,p.enumeration,b)
        piece=candidate.pieces[-1]
        assert np.array_equal(candidate.Y[:len(b.Y)-1],b.Y[:-1])
        assert np.array_equal(b.Y,saved)
        assert candidate.certs=={}
        assert candidate.Y[-1].tolist()==b.Y[-1].tolist()
        assert piece.kind=='minimum_arrival'
        assert inventory(candidate.Y,candidate.pieces)['status']=='complete'
        r=measure_arrival(p.model,candidate.Y[piece.start:],piece.evidence)
        assert r['status']=='measured' and r['max_sin_squared']<1e-6
        assert not r['opposed_flow'] and r['binding_eps']<1
        target=next(q for q in p.enumeration.minima if np.array_equal(b.Y[-1],(q.a,q.b)))
        attachment=check_target(p.model,target,piece.evidence)
        assert attachment['center_bound'] and attachment['inward']
        bad=deepcopy(piece.evidence);bad['center']=('0','0')
        assert check_target(p.model,target,bad)['status']=='invalid'
        coarse=refine_capture(p.model,p.enumeration,b,order=6,precision=60)
        c=coarse.pieces[-1]
        rc=measure_arrival(p.model,coarse.Y[c.start:],c.evidence,precision=70)
        # More terms improve the largest truncation error; no tolerance is
        # inferred from whether a particular zoo case happened to pass.
        assert r['max_sin_squared']<=max(1e-14,rc['max_sin_squared']*1.01)
        report=inspect_trajectory(p.model,candidate)
        assert not report['certified']
        assert report['pieces'][-1]['reason']=='measurement_not_requested'
