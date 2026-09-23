from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from spong import _poly as P
from spong.credibility import level_normal_reference


def pencil(alpha=(1,), beta=(0,)):
    return SimpleNamespace(alpha=tuple(map(Q, alpha)), beta=tuple(map(Q, beta)))


@pytest.mark.parametrize("segments", [1, 4, 100])
def test_normalized_angle_does_not_shrink_with_straight_line_resampling(segments):
    t = np.linspace(0, 1, segments+1)
    rows = level_normal_reference(pencil(), np.column_stack((1+t, t)), 1)
    assert len(rows) == segments
    for row in rows:
        assert row["status"] == "measured"
        assert row["sin_squared"] == pytest.approx(0.5)
        assert row["flow_alignment"] == row["loss_direction"] == 1


def test_orthogonality_alone_does_not_admit_reversed_flow():
    Y = [[1., 0.], [2., 0.]]
    for curve, direction, expected in ((Y, 1, 1), (Y, -1, -1),
                                        (Y[::-1], 1, -1), (Y[::-1], -1, 1)):
        row, = level_normal_reference(pencil(), curve, direction)
        assert row["sin_squared"] == 0 and not row["cross_nonzero"]
        assert row["flow_alignment"] == row["loss_direction"] == expected


@pytest.mark.parametrize("scale", [Q(1), Q(1, 2**2000), Q(2**2000)])
def test_exact_coefficient_cancellation_and_scale_are_not_stationary(scale):
    # Binary64 coefficient conversion erases the normal entirely at a=1.
    m = pencil((scale*(1+Q(1, 2**60)),), (scale,))
    row, = level_normal_reference(m, [[1., 0.], [1., 1.]], 1)
    assert row["status"] == "measured"
    assert row["sin_squared"] == 1 and row["cross_nonzero"]
    assert row["flow_alignment"] == row["loss_direction"] == 0


def test_missing_evidence_is_explicit():
    for Y, status in (([[1., 0.], [1., 0.]], "zero_chord"),
                       ([[-1., 0.], [1., 0.]], "stationary_midpoint"),
                       ([[1., 0.], [float("nan"), 0.]], "nonfinite_point")):
        row, = level_normal_reference(pencil(), Y, 1)
        assert row["status"] == status
        assert row["sin_squared"] is None


def test_native_measurements_match_independent_rational_oracle():
    rng = np.random.default_rng(527131)
    for _ in range(30):
        m = pencil((2, 0, 1), [Q(int(x), 17) for x in rng.integers(-20, 21, 5)])
        Y = rng.uniform(-3, 3, (9, 2))
        rows = level_normal_reference(m, Y, -1)
        for row, p, q in zip(rows, Y[:-1], Y[1:]):
            left, right = [tuple(map(Q, z)) for z in (p, q)]
            a, b = [(x+y)/2 for x, y in zip(left, right)]
            da, db = [y-x for x, y in zip(left, right)]
            ga = 2*(a*P.eval_at(m.alpha, b)-P.eval_at(m.beta, b))
            gb = a*a*P.eval_at(P.deriv(m.alpha), b)-2*a*P.eval_at(P.deriv(m.beta), b)
            cross, dot = da*gb-db*ga, da*ga+db*gb
            loss = lambda z: z[0]**2*P.eval_at(m.alpha,z[1])-2*z[0]*P.eval_at(m.beta,z[1])
            delta = loss(right)-loss(left)
            assert row["status"] == "measured"
            assert row["sin_squared"] == pytest.approx(
                float(cross*cross/((da*da+db*db)*(ga*ga+gb*gb))), rel=1e-14)
            assert row["cross_nonzero"] == bool(cross)
            assert row["flow_alignment"] == (dot < 0)-(dot > 0)
            assert row["loss_direction"] == (delta < 0)-(delta > 0)


def test_midpoint_and_chord_do_not_overflow_binary64():
    row, = level_normal_reference(pencil(), [[1e308, 0.], [1.1e308, 1.]], 1)
    assert row["status"] == "measured"
    assert row["flow_alignment"] == row["loss_direction"] == 1
    # Rounded sin^2 underflows, but the exact nonzero witness survives.
    assert row["sin_squared"] == 0 and row["cross_nonzero"]


@pytest.mark.parametrize("direction", [0, 2, True])
def test_invalid_direction_refuses(direction):
    with pytest.raises(ValueError):
        level_normal_reference(pencil(), [[1., 0.], [2., 0.]], direction)
