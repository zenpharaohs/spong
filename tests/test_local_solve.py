"""Native 2x2 arithmetic and the stiff manifold recurrence that exposed it."""
from decimal import Decimal as D, localcontext
from fractions import Fraction
import itertools

import pytest

from spong import _native, local, model, sturm, zoo


def _norm(v):
    return sum(x*x for x in v).sqrt()


def test_stiff_solve_preserves_both_equations_under_permutations_and_scaling():
    # The actual order-two far-saddle system. The former Cramer solve had
    # a stiff-row residual ~5e-11 on an RHS of 46 despite passing its guard.
    original = [[-3888545529.0578413, 1173.8285724040536],
                [1173.8285724040536, -0.00035439001120326396]]
    rhs = [46.31715789212798, -1.3978733393327639e-5]
    for rows, cols, scales in itertools.product(
            ((0, 1), (1, 0)), ((0, 1), (1, 0)),
            ((1, 1), (1e-200, 1e200), (1e200, 1e-200))):
        A = [[original[i][j]*scales[k] for j in cols] for k, i in enumerate(rows)]
        b = [rhs[i]*scales[k] for k, i in enumerate(rows)]
        x = _native.local_solve2(*A[0], *A[1], *b)
        with localcontext() as ctx:
            ctx.prec = 90
            for row, ri in zip(A, b):
                terms = [D.from_float(a)*D.from_float(v) for a, v in zip(row, x)]
                residual = abs(sum(terms)-D.from_float(ri))
                scale = sum(abs(t) for t in terms)+abs(D.from_float(ri))
                assert residual/scale < D('2e-15')


@pytest.mark.parametrize('A,b', [
    ((1, 2, 2, 4), (1, 2)),
    ((0, 0, 1, 1), (0, 1)),
    ((1, 1, 1, 1+1e-14), (1, 1)),
    ((float('nan'), 0, 0, 1), (1, 1)),
    ((1, 0, 0, 1), (float('inf'), 1)),
])
def test_local_solve_refuses_unresolved_or_nonfinite_systems(A, b):
    with pytest.raises(FloatingPointError):
        local._solve2((A[:2], A[2:]), b)


def test_local_solve_pivots_with_zero_leading_diagonal():
    assert _native.local_solve2(0, 2, 3, 4, 4, 11) == (1, 2)


@pytest.mark.parametrize('orientation', [-1, 1])
def test_far_saddle_series_has_small_exact_field_direction_residual(orientation):
    case = zoo.get('dead-neuron-far-saddle-d3')
    m = model.build(list(case.f), list(case.g),
                    model.moments_uniform01(2*max(len(case.f), len(case.g))-1))
    loc = max(sturm.enumerate_critical_points(m).saddles,
              key=lambda p: abs(float(p.b))).local
    a, b = local._manifold_series(loc, 'unstable', orientation, 60)
    # Exact-field evaluation distinguishes recurrence error from cancellation
    # in the native float gradient. No angle acceptance rule is changed here.
    with localcontext() as ctx:
        ctx.prec = 80
        def dec(q):
            if isinstance(q, Fraction):
                return D(q.numerator)/D(q.denominator)
            return D.from_float(float(q))
        def poly(c, t):
            result = D(0)
            for v in reversed(c):
                result = result*t+dec(v)
            return result
        for t in map(D, ('0.025', '0.2', '1.6', '3.2')):
            p = [poly(c, t) for c in (a, b)]
            # Keep Decimal derivative coefficients exact in this diagnostic.
            tangent = []
            for c in (a, b):
                v = D(0)
                for k in range(len(c)-1, 0, -1):
                    v = v*t+k*dec(c[k])
                tangent.append(v)
            gradient = []
            for comp in loc.exact_grad:
                v, power = D(0), D(1)
                for row in comp:
                    v += power*poly(row, p[1])
                    power *= p[0]
                gradient.append(v)
            angle = abs(tangent[0]*gradient[1]-tangent[1]*gradient[0])/(
                _norm(tangent)*_norm(gradient))
            assert angle < D('1e-5'), (orientation, t, angle)
