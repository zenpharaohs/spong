"""Native/oracle parity, independent derivatives, and numerical refusals."""
from decimal import Decimal as D, localcontext
from fractions import Fraction

import pytest

from spong import _native,model,zoo
from spong.wall_continue import SectionContinuation
from spong.wall_native import NativeSectionContinuation


@pytest.fixture(scope='module')
def base():
    family=zoo.get_wall_family('nonnearest-saddle-connection')
    case=zoo.get(family.base_case)
    return model.build(case.f,case.g,model.moments_uniform01(11)),family


def native(base,**kw):
    m,f=base
    return NativeSectionContinuation(m,f.source_b,f.target_b,**kw)


@pytest.mark.parametrize('stages',[2,3,4])
def test_crossing_sensitivity_and_projection_match_independent_decimal_oracle(base,stages):
    m,f=base
    a=native(base,stages=stages,steps=32).evaluate('2.17770956')
    b=SectionContinuation(m,f.source_b,f.target_b,stages=stages,steps=32).evaluate('2.17770956')
    with localcontext() as ctx:
        ctx.prec=60
        assert abs(a.gap-b.gap)<D('1e-40')
        assert abs(a.slope-b.slope)<D('1e-40')
        for x,y in [(a.unstable,b.unstable),(a.stable,b.stable)]:
            assert max(abs(p-q) for p,q in zip(x.point,y.point))<D('1e-40')
            assert max(abs(p-q) for p,q in zip(x.sensitivity,y.sensitivity))<D('1e-40')
            assert abs(x.projection_distance-y.projection_distance)<D('1e-40')


def test_native_sensitivity_matches_finite_differences(base):
    c=native(base,steps=32)
    with localcontext() as ctx:
        ctx.prec=60
        lam,h=D('2.17770956'),D('1e-15')
        middle=c.evaluate(lam);left=c.evaluate(lam-h);right=c.evaluate(lam+h)
        assert abs((right.gap-left.gap)/(2*h)-middle.slope)<D('1e-27')
        assert c.last_diagnostics['max_backward_error']<1e-65


def test_native_roots_converge_in_steps_and_working_precision(base):
    roots=[]
    for steps in (32,64,128):
        r=native(base,steps=steps).find_root('2.1776','2.1778')
        assert r.grade=='NUMERICAL_ORACLE' and r.stage_solves>0
        assert abs(r.evaluation.gap/r.evaluation.slope)<D('1e-25')
        roots.append(r.evaluation.lam)
    assert abs(roots[2]-roots[1])<abs(roots[1]-roots[0])/100
    high=native(base,steps=128,precision_bits=256).find_root('2.1776','2.1778')
    assert abs(high.evaluation.lam-roots[-1])<D('1e-45')


def test_explicit_refusals_and_context_reuse(base):
    c=native(base,steps=32)
    with pytest.raises(ArithmeticError,match='does not change sign'):
        c.find_root('2.17','2.171')
    with pytest.raises(ArithmeticError,match='below working precision floor'):
        c.find_root('2.1776','2.1778',parameter_tol='1e-100')
    with pytest.raises(ArithmeticError,match='root iteration budget'):
        c.find_root('2.1776','2.1778',max_iterations=1)
    with pytest.raises(ArithmeticError,match='Lambda must be positive'):
        c.evaluate(0)
    assert c.find_root('2.1776','2.1778').evaluations>=3


def test_stage_budget_cannot_silently_return_a_crossing(base):
    c=native(base,steps=1,max_newton=1,max_halvings=0)
    with pytest.raises(ArithmeticError,match='step halving budget'):
        c.evaluate('2.1777')


def test_unresolved_small_eigenvalue_is_refused(base):
    m,f=base
    scaled=model.build([x*Fraction(1,10**40) for x in m.f],m.g,m.mu)
    c=NativeSectionContinuation(scaled,f.source_b,f.target_b,steps=32)
    with pytest.raises(ArithmeticError,match='small eigenvalue'):
        c.evaluate('2.1777')


def test_native_rejects_wrong_root_identity(base):
    m,_=base
    with pytest.raises(ValueError,match='saddle interval'):
        _native.rheostat_create(tuple(map(str,m.alpha)),tuple(map(str,m.beta)),str(m.C),
            ('100','101','102','103'),0,1,1,-1,192,4,32,10,24,4,'0.001','0.5')


@pytest.mark.parametrize('options',[{'precision_bits':64},{'steps':0},{'stages':5},
                                    {'launch_order':30},{'source_direction':0}])
def test_native_policy_admission(base,options):
    with pytest.raises(ValueError):
        native(base,**options)


def test_biased_legacy_bracket_is_repaired_using_native_gap(base):
    c=native(base,steps=64)
    direct=c.find_root('2.1776','2.1778')
    repaired=c.find_root_from_brackets([('2.1777095613653','2.1777095613654')],
                                     bounds=('2.17','2.19'))
    assert abs(direct.evaluation.lam-repaired.evaluation.lam)<D('1e-24')
    assert repaired.evaluations>=4
    with pytest.raises(ArithmeticError,match='does not change sign'):
        c.find_root_from_brackets([('2.17','2.171')],bounds=('2.169','2.172'))


@pytest.mark.parametrize('lam', ['1.28', '1.34'])
def test_native_launch_germs_match_independent_oracle(base, lam):
    from spong.wall_continue import Dual
    m, family = base
    c = native(base)
    oracle = SectionContinuation(m, family.source_b, family.target_b)
    with localcontext() as ctx:
        ctx.prec = 60
        launched = c.launch(lam)
        for k, germ in enumerate((launched.unstable, launched.stable)):
            point, residual = oracle._launch(Dual(D(lam), D(1)), k)
            assert max(abs(v - p.v) for v, p in zip(germ.point, point)) < D('1e-40')
            assert max(abs(v - p.d) for v, p in zip(germ.sensitivity, point)) < D('1e-40')
            assert abs(germ.launch_residual - residual) < D('1e-40')
        assert c.level == pytest.approx(oracle.level)
    with pytest.raises(ArithmeticError, match='positive'):
        c.launch('0')
