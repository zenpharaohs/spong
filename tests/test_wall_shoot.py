"""Two-sided shooting for saddle-connection walls (spong.wall_shoot)."""

import math

import numpy as np
import pytest

from spong import sturm, wall_shoot, zoo

FAMILY = "nonnearest-saddle-connection"


def _native_available():
    m = wall_shoot.rheostat_model(zoo.get_wall_family(FAMILY), 3.0)
    kernel = getattr(m, "_native_kernel", None)
    return kernel is not None and hasattr(kernel, "normalized_step")


needs_native = pytest.mark.skipif(
    not _native_available(), reason="normalized_step needs the C core")


def test_rheostat_moves_the_as_but_not_the_bs():
    """f/sqrt(L), g*sqrt(L) leaves B alone and scales A by L, so the
    critical b's are Lambda-independent; only the a's move."""
    family = zoo.get_wall_family(FAMILY)
    bs = {}
    for lam in (family.below_parameter, family.above_parameter):
        e = sturm.enumerate_critical_points(
            wall_shoot.rheostat_model(family, lam))
        bs[lam] = sorted(float(q.b) for q in e.points if q.kind == "saddle")
    lo, hi = bs[family.below_parameter], bs[family.above_parameter]
    assert len(lo) == len(hi) >= 2
    assert np.allclose(lo, hi, rtol=0, atol=1e-9)


@needs_native
def test_shot_is_signed_smooth_and_changes_sign_across_the_wall():
    family = zoo.get_wall_family(FAMILY)
    deltas = []
    for lam in (family.below_parameter, family.wall_parameter,
                family.above_parameter):
        shot = wall_shoot.shoot(
            wall_shoot.rheostat_model(family, lam), family.source_b,
            family.unstable_direction, family.target_b, ds=2e-3)
        assert shot is not None
        # both shots reach the common midlevel, which lies strictly between
        # the two saddle values
        assert shot.unstable[0][1] == pytest.approx(family.source_b, abs=1e-6)
        assert shot.stable[0][1] == pytest.approx(family.target_b, abs=1e-6)
        deltas.append(shot.delta)
    below, wall, above = deltas
    assert below * above < 0
    assert abs(wall) < min(abs(below), abs(above)) * 1e-3


@needs_native
def test_brent_root_is_a_binary64_wall_with_a_glued_candidate():
    family = zoo.get_wall_family(FAMILY)
    root = wall_shoot.find_wall(family, ds=2e-3)
    assert family.below_parameter < root.lam < family.above_parameter
    assert abs(root.delta) < 1e-11
    # superlinear: well under the ~40 bisections binary64 would need
    assert root.evaluations <= 20
    # agrees with the stored fate-bisection coordinate to the tracer's
    # chord error, not to its bracket width (see the probe's ds sweep)
    assert abs(root.lam - family.wall_parameter) < 1e-6
    cand = root.shot.candidate
    assert cand[0][1] == pytest.approx(family.source_b, abs=1e-6)
    assert cand[-1][1] == pytest.approx(family.target_b, abs=1e-6)
    # the two shots meet at the midlevel: b to the last bit by Brent, a to
    # the integrator's accuracy by the refined crossing (not the chord's)
    gap = root.shot.unstable[-1] - root.shot.stable[-1]
    assert abs(gap[1]) < 1e-12
    assert math.hypot(*gap) < 1e-9
    # the candidate carries one merged vertex there, no duplicate
    assert len(cand) == len(root.shot.unstable) + len(root.shot.stable) - 1
    # the history records a sign change and the root's own evaluation
    signs = {math.copysign(1.0, d) for _, d in root.history if d != 0.0}
    assert signs == {-1.0, 1.0}
    assert root.history[-1][0] == root.lam


@needs_native
def test_find_wall_refuses_without_a_sign_change():
    family = zoo.get_wall_family(FAMILY)
    with pytest.raises(ValueError, match="does not change sign"):
        wall_shoot.find_wall(family, lo=family.above_parameter - 0.2,
                             hi=family.above_parameter, ds=2e-3)
