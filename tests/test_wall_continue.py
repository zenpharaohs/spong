"""Sensitivity, level geometry, and convergence of the experimental shooter."""

from decimal import Decimal as D, localcontext

import pytest

from spong import model, zoo
from spong.wall_continue import Crossing, SectionContinuation, _tableau


@pytest.fixture(scope="module")
def base():
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    case = zoo.get(family.base_case)
    m = model.build(case.f,case.g,model.moments_uniform01(11))
    return m,family


def continuation(base, **options):
    m,f = base
    return SectionContinuation(m,f.source_b,f.target_b,steps=32,**options)


@pytest.mark.parametrize("stages", [2,3,4])
def test_decimal_gauss_tableau_exact_polynomial_quadrature(stages):
    with localcontext() as ctx:
        ctx.prec = 50
        c,a,b = _tableau(stages,50)
        for power in range(2*stages):
            assert abs(sum(w*x**power for w,x in zip(b,c))-D(1)/(power+1)) < D('1e-47')
        for row,node in zip(a,c):
            assert abs(sum(row)-node) < D('1e-47')


def test_parameter_sensitivity_matches_independent_parameter_perturbations(base):
    c = continuation(base)
    with localcontext() as ctx:
        ctx.prec = 50
        lam,h = D('2.17770956'),D('1e-12')
        center = c.evaluate(lam)
        left,right = c.evaluate(lam-h),c.evaluate(lam+h)
        finite_difference = (right.gap-left.gap)/(2*h)
        assert abs(finite_difference-center.slope) < D('1e-20')
        for p in (center.unstable,center.stable):
            assert abs(c.loss(p.point)-c.level) < D('1e-40')
            g = c.gradient(p.point)
            assert abs(sum(x*y for x,y in zip(g,p.sensitivity))) < D('1e-39')


def test_root_is_corrected_on_both_branches_and_converges_with_gl8(base):
    c = continuation(base)
    coarse,history = c.find_root('2.1776','2.1778')
    c.steps = 64
    fine,_ = c.find_root('2.1776','2.1778')
    c.steps = 128
    finer,_ = c.find_root('2.1776','2.1778')
    assert history[0].gap*history[1].gap < 0
    assert len(history) <= 7
    assert abs(finer.lam-fine.lam) < abs(fine.lam-coarse.lam)/100
    assert abs(finer.gap) < D('1e-24')
    assert abs(finer.slope) > 1
    assert max(abs(a-b) for a,b in zip(finer.unstable.point,finer.stable.point)) < D('1e-24')


def test_section_chart_crosses_a_b_turn_without_sheet_aliasing(base):
    c = continuation(base)
    with localcontext() as ctx:
        ctx.prec = 50
        b = D('0.1')
        from spong.wall_continue import polyval
        a = polyval(c.B,b)/polyval(c.A,b)
        c.level = c.loss((a,b))
        sign = D(1) if c.gradient((a,b))[1] > 0 else D(-1)
        normal = (D(0),sign)
        points = []
        for offset in (D('-0.0001'),D('0.0001')):
            raw = Crossing((a+offset,b),(D(1),D(0)),D(0),D(0),0)
            p = c._project(raw,normal)
            assert abs(c.loss(p.point)-c.level) < D('1e-40')
            points.append(p.point)
        c._check_chart([(a,b),*points],normal)
        # Both sheets near the fold are distinguished by the alpha chart.
        assert points[0][0] < a < points[1][0]
        ys = [polyval(c.A,p[1])*p[0]-polyval(c.B,p[1]) for p in points]
        assert ys[0]*ys[1] < 0


def test_wrong_bracket_refuses(base):
    c = continuation(base)
    with pytest.raises(ValueError,match="does not change sign"):
        c.find_root('2.1776','2.17765')


def test_hermite_and_newton_correct_to_the_same_discrete_connection(base):
    c = continuation(base)
    hermite,h = c.find_root('2.1776','2.1778',predictor='hermite')
    newton,n = c.find_root('2.1776','2.1778',predictor='newton')
    assert abs(hermite.lam-newton.lam) < D('1e-25')
    assert len(h) <= len(n)


def test_reuses_one_or_two_coarse_legacy_brackets(base):
    c = continuation(base)
    root,cold = c.find_root('2.1776','2.1778')
    for brackets in ([('2.1777093','2.1777098')],
                     [('2.1777093','2.1777098'),('2.1777094','2.1777099')]):
        warm,history = c.find_root_from_brackets(brackets)
        assert abs(warm.lam-root.lam) < D('1e-25')
        assert history[0].gap*history[1].gap < 0
        assert len(history) < len(cold)
        assert len({e.lam for e in history}) == len(history)


def test_ulp_wide_biased_bracket_is_repaired_using_new_sensitivities(base):
    c = continuation(base)
    root,history = c.find_root_from_brackets(
        [('2.177709561365374','2.177709561365376')],bounds=('2.17','2.18'))
    assert history[0].gap*history[1].gap > 0  # NOT a bracket for the new map.
    assert history[1].gap*history[2].gap < 0  # The repair supplies the other sign.
    assert abs(root.gap/root.slope) < D('1e-25')
    assert len(history) <= 5


def test_warm_start_refuses_if_bounds_prevent_repair(base):
    c = continuation(base)
    with pytest.raises(ValueError,match='did not rebracket'):
        c.find_root_from_brackets([('2.177709561365374','2.177709561365376')],
                                 bounds=('2.17770956','2.177709562'))


@pytest.mark.parametrize('brackets',[[],[('0','1')],[('2','1')]])
def test_invalid_warm_start_brackets_refuse(base,brackets):
    c = continuation(base)
    with pytest.raises(ValueError,match='seed brackets'):
        c.find_root_from_brackets(brackets)
