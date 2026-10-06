from fractions import Fraction as Q
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from spong import _native
from spong.trajectory import TrajectoryPiece
from spong.precision_bridge_check import endpoint,measure_precision_bridge


def scalar_path(steps=4):
    r=_native.precision_bridge(('2','2','0','0'),2,1,'2','2',80,steps)
    e={'representation':'precision_bridge_cubics','schema':1,'origin':('1','0'),
       'working_bits':r['working_bits'],'polynomials':r['polynomials']}
    points=np.array([(1.,0)]+[tuple(map(float,endpoint(e,i)[0])) for i in range(1,len(r['polynomials'])+1)])
    return SimpleNamespace(alpha=(Q(1),),beta=(Q(0),)),points,TrajectoryPiece('precision_bridge',0,steps,e)


def test_radau_matches_exact_stability_function_and_is_checked():
    m,points,p=scalar_path()
    z=Q(-1,4);R=(1+2*z/5+z*z/20)/(1-3*z/5+3*z*z/20-z**3/60)
    assert abs(endpoint(p.evidence,1)[0][0]-R)<Q(1,10**90)
    check=measure_precision_bridge(m,points,p)
    assert check['status']=='measured' and check['max_sin_squared']==0 and check['opposed_flow']==0
    assert check['binding_eps']<1 and check['join_eps']<1
    assert endpoint(p.evidence,1)[0][0]!=Q(float(points[1,0]))


def test_independent_checker_rejects_missing_budget_bad_binding_and_opposed_flow():
    m,points,p=scalar_path(1)
    assert measure_precision_bridge(m,points,p,sample_budget=4)['status']=='unresolved'
    bad=points.copy();bad[0,0]+=.01
    assert measure_precision_bridge(m,bad,p)['binding_eps']>64
    polys=[[list(c) for c in s] for s in p.evidence['polynomials']]
    for i in range(1,4):polys[0][0][i]=str(-Q(polys[0][0][i]))
    # Strings emitted by the format are decimals, not fractions.
    from decimal import Decimal as D,localcontext
    with localcontext() as ctx:
        ctx.prec=200
        for i in range(1,4):
            q=Q(polys[0][0][i]);polys[0][0][i]=str(D(q.numerator)/D(q.denominator))
    badpiece=replace(p,evidence={**p.evidence,'polynomials':polys})
    result=measure_precision_bridge(m,points,badpiece)
    assert result['status']=='measured' and result['opposed_flow']>0


def test_native_validation_and_condition_budget():
    with pytest.raises(ValueError):_native.precision_bridge(('0',),0,1,'1','1',80,1)
    with pytest.raises(ValueError,match='bit_budget'):_native.precision_bridge(('1','1'),1,1,'1','1e5000',80,1)
    with pytest.raises(ValueError):_native.precision_bridge(('x','0'),1,1,'1','1',80,1)


def test_exact_entry_survives_display_rounding():
    from spong.local import _exact_symmetric_spectral
    from spong.native_arrival import propose
    H=((Q(1),Q(0)),(Q(0),Q(10**30)))
    local=SimpleNamespace(exact_hessian=H,exact_grad=(((Q(0),),(Q(1),)),((Q(0),Q(10**30)),)),spectral=_exact_symmetric_spectral(H))
    center=(Q(1),Q(1));exact=(Q(1)+Q(1,10**40),Q(1)+Q(1,10**50))
    points,e=propose(local,(1.,1.),(1.,1.),center,exact_start=exact)
    actual=tuple(center[d]+sum(Q(t[d+1]) for t in e['terms']) for d in range(2))
    assert max(abs(x-y) for x,y in zip(actual,exact))<Q(1,10**100)
    assert np.array_equal(points[0],(1,1))
    with pytest.raises(ValueError,match='matching_not_converged'):
        propose(local,(1.,1.),(1.,1.),center)


def test_precision_bridge_recovery_preserves_input_and_independent_evidence():
    import importlib.util,random
    from pathlib import Path
    from spong import sturm
    from spong.precision_bridge import checked_precision_path
    from spong.precision_bridge_check import measure_precision_join
    from spong.trajectory_check import inventory
    spec=importlib.util.spec_from_file_location('precision_bridge_qualify',Path(__file__).resolve().parents[1]/'scripts/qualify.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Fixed entry from the 144-case frontier (filled from frozen metadata).
    m,_=module.random_model(random.Random(2090110773),5)
    start=(1.6125943567470656e-08,-89.81012299116381)
    target=(1.606391666801343e-08,-89.8967002117041)
    en=sturm.materialize_stubs(m,sturm.enumerate_critical_points(m))
    q=next(q for q in en.minima if (q.a,q.b)==target)
    points,pieces,checks=checked_precision_path(m,q,start,target,sample_budget=50000)
    assert pieces[0].kind=='precision_bridge' and not checks['certified']
    assert np.array_equal(points[0],start) and np.array_equal(points[-1],target)
    assert inventory(points,pieces)['status']=='complete'
    pre=measure_precision_bridge(m,points,pieces[0],precision=220,sample_budget=50000)
    assert pre['status']=='measured' and pre['max_sin_squared']<1e-6 and pre['opposed_flow']==0
    join=measure_precision_join(points,*pieces)
    assert join['same_orientation'] and join['sin_squared']<1e-6 and join['join_eps']<64
