"""Analytic events, conditioning refusals, and independent event-algorithm parity."""
import numpy as np
import pytest
from spong import model,zoo
from spong.wall_shoot import _level_crossing
from spong.zoo import _connection_contexts
from decimal import Decimal as D,localcontext
from section_oracle import _level_crossing as reference

@pytest.mark.parametrize('shift',[0.,100.])
@pytest.mark.parametrize('direction,start', [(-1,2.),(1,1.25)])
def test_analytic_invariant_axis_and_loss_translation(shift,direction,start):
    m=model.build([1],[1,1],model.moments_normal01(3))
    r=m._native_kernel.section_trace(start,0.,direction,.25+shift,float(m.C)+shift,.17,100)
    assert r['status']==0, r['reason']
    assert r['points'][-1]==pytest.approx((1.5,0),abs=3e-12)
    assert abs(r['loss_residual'])<=r['loss_roundoff_scale']
    assert r['refinements']>0


def test_section_explicit_refusals_and_recovery():
    m=model.build([1],[1,1],model.moments_normal01(3));k=m._native_kernel
    assert k.section_trace(2,0,1,.25,1,.1,20)['status']==1
    assert k.section_trace(1,0,1,.25,1,.1,20)['status']==3
    assert k.section_trace(2,0,-1,.25,1,.01,1)['status']==2
    assert k.section_trace(2,0,-1,1000.25,1001,.1,20)['status']==3
    assert k.section_trace(float('inf'),0,-1,.25,1,.1,20)['status']!=0
    assert k.section_trace(2,0,-1,.25,1,.1,20)['status']==0


@pytest.mark.parametrize('name',['two-asymmetric-connections','two-strongly-asymmetric-connections'])
def test_native_section_against_python_event_oracle(name):
    c=zoo.get_numerical_connection(name);m=c.build();lam=D(c.parameter)
    for _,_,_,_,context in _connection_contexts(c):
        germs=context.launch(lam)
        with localcontext() as ctx:
            ctx.prec=80;level=float(context.level/lam)
            starts=[(float(v.point[0]/lam),float(v.point[1])) for v in (germs.unstable,germs.stable)]
        endpoints=[]
        for i,start in enumerate(starts):
            actual=_level_crossing(m,m._native_kernel,*start,bool(i),level,.001,10000)
            expected=reference(m,m._native_kernel,*start,bool(i),level,.001,10000)
            assert np.linalg.norm(actual[-1]-expected[-1])<1e-11
            losses=np.array([float(m.L(*z)) for z in actual])
            assert np.all((1 if i else -1)*np.diff(losses)>-1e-13)
            assert abs(losses[-1]-level)<1e-12
            endpoints.append(actual[-1])
        assert np.linalg.norm(endpoints[0]-endpoints[1])<1e-11
