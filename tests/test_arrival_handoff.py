"""A later analytic entry must retain and independently check its prefix."""
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from spong.trajectory import TrajectoryPiece as Piece
from spong.arrival_check import measure_handoff
from spong.native_arrival import checked_path


def line_prefix():
    vertex0 = np.zeros(24); vertex0[0] = 3; vertex0[4:6] = (1,0); vertex0[23] = -1
    dense = np.zeros(24); dense[0] = 2; dense[3] = 1
    dense[4:12] = (1,0,.5,0,-.5,0,-.5,0)
    vertex1 = np.zeros(24); vertex1[0] = 3; vertex1[4:6] = (.5,0)
    vertex1[20], vertex1[22], vertex1[23] = 1,1,1
    points = np.array([[1.,0],[.5,0],[0,0]])
    prefix = Piece('centered_arrival',0,1,{'representation':'native_dense_trace','schema':1,
        'trace':np.array([vertex0,dense,vertex1]).tobytes(),'source_points':2,'branch_offset':0})
    arrival = Piece('minimum_arrival',1,2,{'representation':'poincare_frobenius_arrival',
        'center':('0','0'),'terms':[('1','.5','0')]})
    return points,prefix,arrival


def test_join_checks_representations_not_shared_display_point():
    m = SimpleNamespace(alpha=(Q(1),),beta=(Q(0),))
    points, prefix, tail = line_prefix()
    r = measure_handoff(m,points,prefix,tail)
    assert r['status']=='measured' and r['same_orientation']
    assert r['join_eps']==r['sin_squared']==0
    displaced = replace(tail,evidence={**tail.evidence,'terms':[('1','.6','0')]})
    assert measure_handoff(m,points,prefix,displaced)['join_eps']>64
    reversed_flow = replace(tail,evidence={**tail.evidence,'terms':[('-1','.5','0')]})
    assert not measure_handoff(m,points,prefix,reversed_flow)['same_orientation']
    gap = replace(tail,start=2)
    assert measure_handoff(m,points,prefix,gap)['status']=='invalid'


def test_prefix_geometry_refusal_cannot_be_hidden_by_good_tail(monkeypatch):
    from spong import native_arrival as na
    import spong.native_trace as nt
    points,prefix,tail = line_prefix()
    calls=[]
    def proposal(m,q,start,target,**kwargs):
        calls.append(tuple(start))
        if tuple(start)==(1.,0.): raise ValueError('arrival_matching_not_converged')
        return points[1:],tail.evidence,{'geometry':{'status':'measured'},'certified':False}
    monkeypatch.setattr(na,'checked_proposal',proposal)
    monkeypatch.setattr(na,'_entry_trace',lambda *args:(points[:2],prefix.evidence,{}))
    monkeypatch.setattr(nt,'measure_native',lambda *args,**kwargs:{
        'status':'measured','max_sin_squared':.1,'binding_eps':0,'join_eps':0})
    with pytest.raises(ValueError,match='arrival_handoff_prefix_rejected') as exc:
        checked_path(None,None,(1.,0.),(0.,0.))
    assert len(calls)==2 and exc.value.handoff['attempts'][0]['prefix']['max_sin_squared']==.1


@pytest.fixture(scope='module')
def recovered():
    import importlib.util,random
    from pathlib import Path
    from spong import sturm
    spec=importlib.util.spec_from_file_location('handoff_qualify',Path(__file__).resolve().parents[1]/'scripts/qualify.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m,_=module.directed_model(random.Random(462615314),5)
    en=sturm.materialize_stubs(m,sturm.enumerate_critical_points(m))
    start=(0.16759403906990844,2.5960993416905467)
    target=(0.21191280559088493,2.411776756996781)
    q=next(q for q in en.minima if (q.a,q.b)==target)
    return m,en,q,start,target


def test_real_handoff_retains_every_chord_and_passes_fresh_checks(recovered):
    from spong.trajectory_check import inventory
    from spong.native_trace import measure_native
    from spong.arrival_check import measure_arrival
    m,en,q,start,target=recovered
    points,pieces,checks=checked_path(m,q,start,target,sample_budget=50000)
    assert [p.kind for p in pieces]==['centered_arrival','minimum_arrival']
    assert np.array_equal(points[0],start) and np.array_equal(points[-1],target)
    assert inventory(points,pieces)['status']=='complete' and not checks['certified']
    assert measure_native(m,points,pieces[0],-1)['max_sin_squared']<1e-6
    tail=pieces[-1]
    g=measure_arrival(m,points[tail.start:],tail.evidence,precision=120,sample_budget=50000)
    assert g['status']=='measured' and g['max_sin_squared']<1e-6
    assert measure_handoff(m,points,*pieces)['same_orientation']


def test_refinement_shifts_trace_indices_and_preserves_original_prefix(recovered):
    from spong.charts import Branch
    from spong.arrival_series import refine_capture
    from spong.native_trace import measure_native
    from spong.trajectory_check import inventory
    m,en,q,start,target=recovered
    before=(start[0]+.001,start[1]+.001)
    branch=Branch(kind='unstable',Y=np.array([before,start,target]),term='capture',
        diag={},certs={'old':True},pieces=(Piece('chart_continuation',0,1),Piece('capture_connector',1,2)))
    result=refine_capture(m,en,branch)
    assert np.array_equal(result.Y[:2],branch.Y[:2]) and result.certs=={}
    assert inventory(result.Y,result.pieces)['status']=='complete'
    prefix=result.pieces[-2]
    assert prefix.evidence['branch_offset']==1 and prefix.start==1
    assert measure_native(m,result.Y,prefix,-1)['status']=='measured'
    assert result.diag['arrival_refinement']['checks']['handoff']['selected_index']>0


def test_native_trace_must_preserve_incoming_anchor(monkeypatch):
    from spong import _native
    from spong.native_arrival import _entry_trace
    q=SimpleNamespace(local=SimpleNamespace(native=object(),a=0.,b=0.,
        spectral=SimpleNamespace(eigenvalues=(1.,2.))))
    raw=[0]*17
    raw[-2]=np.array([[.75,0.],[.5,0.]]).tobytes()
    raw[-1]=b''
    monkeypatch.setattr(_native,'centered_arrival',lambda *args:tuple(raw)+({},))
    with pytest.raises(ValueError,match='incoming_anchor_lost'):
        _entry_trace(q,(1.,0.),(0.,0.))


def test_bad_representation_join_refuses_even_when_both_piece_checks_pass(monkeypatch):
    from spong import native_arrival as na
    import spong.native_trace as nt
    points,prefix,tail=line_prefix()
    wrong={**tail.evidence,'terms':[('1','.6','0')]}
    def proposal(m,q,start,target,**kwargs):
        if tuple(start)==(1.,0.):raise ValueError('arrival_matching_not_converged')
        return points[1:],wrong,{'geometry':{'status':'measured'},'certified':False}
    monkeypatch.setattr(na,'checked_proposal',proposal)
    monkeypatch.setattr(na,'_entry_trace',lambda *args:(points[:2],prefix.evidence,{}))
    monkeypatch.setattr(nt,'measure_native',lambda *args,**kwargs:{
        'status':'measured','max_sin_squared':0,'binding_eps':0,'join_eps':0})
    m=SimpleNamespace(alpha=(Q(1),),beta=(Q(0),))
    with pytest.raises(ValueError,match='arrival_handoff_no_checked_entry') as exc:
        checked_path(m,None,(1.,0.),(0.,0.))
    assert exc.value.handoff['attempts'][0]['reason']=='arrival_handoff_join_rejected'
