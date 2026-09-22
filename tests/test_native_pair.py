"""Native two-parameter convergence and fail-closed numerical proposals."""
from decimal import Decimal as D
import pytest
from spong import zoo, model, sturm, _native
from spong.wall_native import locate_connection_pair, NativeSectionContinuation


def locate(name='two-asymmetric-connections', **kwargs):
    case=zoo.get_numerical_connection(name)
    g=list(case.base_g);g[1]='0'
    strong='strongly' in name
    options=dict(initial=('1.31364','-.28365') if strong else ('1.3072','-.01954'),
                 lower=('1.30','-.30') if strong else ('1.29','-.021'),
                 upper=('1.33','-.27') if strong else ('1.33','-.018'),
                 steps=32,scaled_tolerance='1e-20')
    pairs=kwargs.pop('pairs',case.connection_pairs);options.update(kwargs)
    return locate_connection_pair(case.base_f,g,['0','1'],case.moments(21),pairs,**options)


@pytest.mark.parametrize('name',['two-asymmetric-connections',
    # 138 s alone: the strongly asymmetric case runs the 32/64/128-step
    # convergence study where the solver works hardest.  The asymmetric case
    # above exercises the same path in the default run.
    pytest.param('two-strongly-asymmetric-connections', marks=pytest.mark.heavy)])
def test_pair_step_convergence_and_independent_shots(name):
    a=locate(name,steps=32);b=locate(name,steps=64);c=locate(name,steps=128)
    assert abs(c.lam-b.lam)<abs(b.lam-a.lam)/50
    assert abs(c.lam-D(zoo.get_numerical_connection(name).parameter))<D('1e-12')
    assert max(map(abs,c.gaps))<D('1e-20')
    assert c.scaled_correction_norm<D('1e-20')
    assert c.scaled_residual_norm<D('1e-20')
    assert abs(c.scaled_determinant)>D('.01')
    assert c.grade=='NUMERICAL_ORACLE'
    # Independent contexts construct their own section charts and germs.
    from fractions import Fraction as Q
    case=zoo.get_numerical_connection(name);g=list(case.base_g);g[1]=Q(c.eta)
    m=model.build(case.base_f,g,case.moments(21))
    seed=model.build(case.base_f,case.base_g,case.moments(21))
    inventory=sturm.enumerate_critical_points(seed)
    for source,target,sign in case.connection_pairs:
        points=[min((p for p in inventory.points if p.kind=='saddle'),
                    key=lambda p:abs(p.b-reference)) for reference in (source,target)]
        # Deliberately independent boxes and section charts. The high-precision
        # eta trial need not rerun the Python global critical inventory.
        bounds=tuple(str(Q(str(p.b))+offset) for p in points
                     for offset in (Q('-1e-5'),Q('1e-5')))
        context=_native.rheostat_create(tuple(map(str,m.alpha)),tuple(map(str,m.beta)),
            str(m.C),bounds,*(0 if p.source=='B' else 1 for p in points),
            sign,-sign,192,4,128,10,24,4,'.001','.5')
        shot=_native.rheostat_evaluate(context,str(c.lam))
        assert shot['status']==0,shot['reason']
        # A fresh projection chart changes the finite-step shooting map.
        # Agreement is assessed at discretization accuracy, not Newton tolerance.
        assert abs(D(shot['values'][1]))<D('1e-17')


def test_precision_and_difference_step_independence():
    a=locate(steps=32)
    b=locate(steps=32,precision_bits=256,eta_difference_step='1e-6')
    assert abs(a.lam-b.lam)<D('1e-22')
    assert abs(a.eta-b.eta)<D('1e-22')


@pytest.mark.parametrize('options,reason',[
    ({'max_iterations':1},'budget exhausted'),
    ({'initial':('1.29','-.01954')},'strictly inside'),
    ({'scaled_tolerance':'1e-100'},'precision floor'),
    ({'saddle_bounds':[(20,21)]*4},'initialization'),
])
def test_pair_refusals(options,reason):
    with pytest.raises(ArithmeticError,match=reason):locate(**options)


def test_duplicate_constraint_is_not_a_codimension_two_solution():
    case=zoo.get_numerical_connection('two-asymmetric-connections')
    with pytest.raises(ArithmeticError,match='rank unresolved'):
        locate(pairs=[case.connection_pairs[0]]*2)
    assert locate().diagnostics['status']==0
