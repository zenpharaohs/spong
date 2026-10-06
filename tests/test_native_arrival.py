"""Native arrival parity, refusal and checked dispatch against independent evidence."""
from decimal import Decimal as D
from fractions import Fraction as Q
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from spong import _native, model, portrait, sturm, zoo
from spong.local import _exact_symmetric_spectral
from spong.arrival_series import parameterization, _propose
from spong.native_arrival import propose, clear_cache, checked_proposal, refine_branches
from spong.arrival_check import measure_arrival


def linear(ratio):
    H = ((Q(1), Q(0)), (Q(0), Q(ratio)))
    return SimpleNamespace(exact_hessian=H,
        exact_grad=(((Q(0),), (Q(1),)), ((Q(0), Q(ratio)),)),
        spectral=_exact_symmetric_spectral(H))


@pytest.mark.parametrize('ratio', [Q(5, 2), Q(21, 2), 10001, 10**10])
def test_native_two_mode_curve_matches_python_oracle(ratio):
    local = linear(ratio)
    center = (Q(123456789, 100000000), Q(-13, 7))
    start = (float(center[0])+.25, float(center[1])-.125)
    target = tuple(map(float, center))
    chart = parameterization(local)
    chart['center'] = center
    expected, oracle = _propose(local, start, target, chart=chart)
    actual, evidence = propose(local, start, target, center)
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual-expected)/(np.finfo(float).eps*(1+np.abs(expected)))) < 2
    assert np.array_equal(actual[0], start) and np.array_equal(actual[-1], target)
    assert evidence['mode'] == oracle['mode']
    assert all(D(a) != 0 for a in evidence['amplitudes'])
    assert evidence['parameters'][-1] is None
    for native, reference in zip(evidence['terms'], oracle['terms']):
        for x, y in zip(native, reference):
            assert abs(D(x)-D(y)) <= D('1e-60')*(1+abs(D(y)))
    repeated, _ = propose(local, start, target, center)
    assert np.array_equal(actual, repeated)


@pytest.mark.parametrize('ratio', [2, 3, 4])
def test_native_divisors_refuse_exactly_as_oracle(ratio):
    with pytest.raises(ValueError, match='resonance_or_small_divisor'):
        propose(linear(ratio), (.1, .1), (0, 0), (Q(0), Q(0)))


@pytest.mark.parametrize('ratio', [10**13, 10**30, 10**74])
def test_stiff_linear_node_is_not_mistaken_for_resonance(ratio):
    points, evidence = propose(linear(ratio), (.25, -.125), (0, 0), (Q(0), Q(0)))
    info = evidence['conditioning']
    assert info['policy'] == 'componentwise-v1'
    assert D(info['min_relative_divisor']) > D('.1')
    assert D(info['max_matrix_condition']) >= D(ratio)/2
    assert D(info['max_forward_error_indicator']) < D('1e-20')
    assert info['working_bits'] > 330 + int(np.log2(float(ratio)))
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 110
        for point, s in zip(points[:-1], evidence['parameters'][:-1]):
            expected = (.25*float((-D(s)).exp()), -.125*float((-D(ratio)*D(s)).exp()))
            assert np.max(np.abs(point-expected)) < 1e-15


def test_near_resonance_requires_enough_declared_precision():
    local = linear(Q(2)+Q(1,10**70))
    with pytest.raises(ValueError, match='divisor_precision_budget'):
        propose(local, (.1, .1), (0, 0), (Q(0), Q(0)), precision=80)
    _, evidence = propose(local, (.1, .1), (0, 0), (Q(0), Q(0)), precision=100)
    assert D(evidence['conditioning']['min_relative_divisor']) < D('1e-69')
    assert D(evidence['conditioning']['max_forward_error_indicator']) < D('1e-20')


def test_rotated_stiff_homological_system_gets_guard_precision():
    large = Q(10**40)
    H = ((Q(9,25)+Q(16,25)*large, Q(12,25)*(1-large)),
         (Q(12,25)*(1-large), Q(16,25)+Q(9,25)*large))
    local = SimpleNamespace(exact_hessian=H,
        exact_grad=(((Q(0),H[0][1]), (H[0][0],), (Q(1),)),
                    ((Q(0),H[1][1]), (H[1][0],))),
        spectral=_exact_symmetric_spectral(H))
    _, evidence = propose(local, (.001, -.002), (0, 0), (Q(0), Q(0)))
    info = evidence['conditioning']
    assert D(info['max_matrix_condition']) > D('1e39')
    assert D(info['min_relative_divisor']) > D('.1')
    assert D(info['max_forward_error_indicator']) < D('1e-20')
    assert info['working_bits'] >= 465


def test_unaffordable_conditioning_refuses_instead_of_truncating_precision():
    with pytest.raises(ValueError, match='divisor_precision_budget'):
        propose(linear(Q(10)**1500), (.1, .1), (0, 0), (Q(0), Q(0)))


@pytest.mark.parametrize('order,precision', [(1, 80), (13, 80), (8, 49), (8, 101)])
def test_native_adapter_rejects_bad_requests(order, precision):
    with pytest.raises(ValueError):
        _native.arrival_chart([], order, precision)


@pytest.fixture(scope='module')
def example():
    z = zoo.get('minimal-quartet')
    m = model.build(list(z.f), list(z.g), model.moments_uniform01(2*max(len(z.f),len(z.g))-1))
    return portrait.compute(m, _skip_audit=True)


def test_checked_portrait_path_keeps_prefix_and_clears_stale_certificates(example):
    p = example
    branches = refine_branches(p.model, p.enumeration, p.branches)
    for old, new in zip(p.branches, branches):
        if old.term != 'capture':
            assert new is old
            continue
        assert new.term == 'capture' and new.pieces[-1].kind == 'minimum_arrival'
        assert np.array_equal(old.Y[:-1], new.Y[:len(old.Y)-1])
        assert new.certs == {}
        checks = new.diag['arrival_refinement']['checks']
        assert not checks['certified'] and checks['geometry']['max_sin_squared'] < 1e-6
        assert old.pieces[-1].kind == 'capture_connector'


def test_refusal_never_restores_a_capture_chord(example, monkeypatch):
    import spong.arrival_series as series
    def refuse(*a, **kw):
        raise ValueError('arrival_independent_check_failed')
    monkeypatch.setattr(series, 'refine_capture', refuse)
    new = refine_branches(example.model, example.enumeration, example.branches)
    for old, candidate in zip(example.branches, new):
        if old.term != 'capture': continue
        assert candidate.term == 'abort_analytic_arrival'
        assert candidate.certs == {}
        assert all(p.kind != 'capture_connector' for p in candidate.pieces)
        assert np.array_equal(candidate.Y, old.Y[:-1])
        assert candidate.diag['arrival_refusal']['reason'] == 'arrival_independent_check_failed'


def test_checked_dispatch_rejects_false_orientation(example, monkeypatch):
    import spong.arrival_check as check
    old = next(b for b in example.branches if b.term == 'capture')
    q = next(q for q in example.enumeration.minima if np.array_equal((q.a,q.b), old.Y[-1]))
    monkeypatch.setattr(check, 'measure_arrival', lambda *a, **kw: {
        'status': 'measured', 'max_sin_squared': 0, 'binding_eps': 0, 'opposed_flow': 1})
    with pytest.raises(ValueError, match='independent_check_failed'):
        checked_proposal(example.model, q, old.Y[-2], old.Y[-1])


def test_portrait_selection_and_legacy_remain_explicit(example):
    p = portrait.compute(example.model, _enumeration=example.enumeration,
                         _skip_audit=True, arrival_policy='checked_native')
    assert p.ledger['arrival_policy'] == 'checked_native'
    assert p.ledger['topology']['status'] == 'not_audited'
    assert all(b.pieces[-1].kind == 'minimum_arrival' for b in p.branches if b.term == 'capture')
    with pytest.raises(ValueError, match='arrival_policy'):
        portrait.compute(example.model, arrival_policy='typo')


def test_known_bad_chart_is_rejected_by_real_checker():
    # Frozen directed counterexample from the 1,400-entry qualification.
    import importlib.util
    import random
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/qualify.py'
    spec = importlib.util.spec_from_file_location('arrival_test_qualify', path)
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    m, _ = generate.directed_model(random.Random(427028989), 5)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    target = (-0.00017805539375007726, -5.359756261025344)
    start = (-0.0001633612649798915, -5.453048753199056)
    q = next(q for q in e.minima if (q.a, q.b) == target)
    with pytest.raises(ValueError, match='independent_check_failed'):
        checked_proposal(m, q, start, target)


def test_aborted_analytic_arrival_blocks_topology_certificate(example, monkeypatch):
    import spong.arrival_series as series
    def refuse(*a, **kw):
        raise ValueError('arrival_resonance_or_small_divisor')
    monkeypatch.setattr(series, 'refine_capture', refuse)
    p = portrait.compute(example.model, _enumeration=example.enumeration,
                         arrival_policy='checked_native')
    assert p.ledger['topology']['status'] != 'certified'
    assert p.ledger['topology']['resolution_reason'] == 'branch_abort'


def test_large_exact_coefficients_cross_adapter_without_global_digit_override():
    import sys
    from spong.native_arrival import _rational_text
    from decimal import localcontext
    digit_limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)
    before = digit_limit()
    huge = 10**5000
    exact = Q(huge+1, huge+3)
    with localcontext() as context:
        context.prec = 6
        encoded = _rational_text(exact)
    numerator, denominator = encoded.split('/')
    assert D(numerator) == D(huge+1)
    assert D(denominator) == D(huge+3)
    # This exact linear jet is almost diag(1, 2.5); tiny rational corrections
    # exercise GMP parsing without requiring a high-degree fixture.
    local = linear(Q(5, 2))
    local.exact_grad = (((Q(0),), (exact,)), ((Q(0), Q(5, 2)),))
    points, evidence = propose(local, (.1, .2), (0, 0), (Q(0), Q(0)))
    assert points.shape[1] == 2 and evidence['backend'] == 'native_gmp'
    assert digit_limit() == before


def quadratic_node():
    H = ((Q(1), Q(0)), (Q(0), Q(5, 2)))
    return SimpleNamespace(exact_hessian=H,
        exact_grad=(((Q(0),), (Q(1),), (Q(1),)), ((Q(0), Q(5, 2)),)),
        spectral=_exact_symmetric_spectral(H))


def test_damped_inverse_matches_original_point_after_fast_newton_exhaustion():
    from decimal import localcontext
    local = quadratic_node()
    start = (100., .1)
    points, evidence = propose(local, start, (0, 0), (Q(0), Q(0)))
    info = evidence['matching']
    assert info['continuation'] and info['stages'] > 0 and info['backtracks'] > 0
    assert info['iterations'] > 24
    assert D(info['max_solve_forward_error_indicator']) < D('1e-20')
    assert np.array_equal(points[0], start)
    with localcontext() as ctx:
        ctx.prec = 100
        # For x'=-x-x² the order-eight parameterization is sum(t**k).
        t = D(evidence['amplitudes'][0])*local.spectral.decimal_frame[0][0]
        assert abs(sum(t**k for k in range(1,9))-100) < D('1e-55')
        for d in range(2):
            assert abs(sum(D(term[d+1]) for term in evidence['terms'])
                       - D.from_float(start[d])) < D('1e-55')
    repeated, again = propose(local, start, (0,0), (Q(0), Q(0)))
    assert np.array_equal(points, repeated) and evidence == again


def test_outside_polynomial_image_refuses_with_bounded_inverse_diagnostics():
    # sum(t**k,k=1..8) cannot be -1 for real t: adding 1 gives
    # (t**9-1)/(t-1), positive for every real t (including t=1).
    with pytest.raises(ValueError, match='arrival_matching_not_converged') as caught:
        propose(quadratic_node(), (-1., .1), (0,0), (Q(0), Q(0)))
    info = caught.value.matching
    assert info['continuation'] and 0 < info['iterations'] <= 2048
    assert 0 < info['stages'] <= 128 and info['backtracks'] <= 32*info['iterations']
    assert D(info['last_residual']) > 0


def test_recovered_frozen_matching_case_passes_independent_geometry():
    import importlib.util
    import random
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/qualify.py'
    spec = importlib.util.spec_from_file_location('matching_test_qualify', path)
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    m, _ = generate.random_model(random.Random(1889003091), 5)
    en = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    start = (-0.07342521047152911, -1.9473169966897004)
    target = (-0.07747649548555688, -1.9253121193940386)
    q = next(q for q in en.minima if (q.a, q.b) == target)
    points, evidence, checks = checked_proposal(m, q, start, target)
    assert evidence['matching']['continuation']
    assert checks['geometry']['max_sin_squared'] < 1e-6 and not checks['certified']
    assert np.array_equal(points[0], start) and np.array_equal(points[-1], target)
