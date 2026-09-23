"""Independent measurements for the four-part portrait credibility contract.

This reference API supplies evidence for scrutiny/calibration. It does not
change the production verdict or prescribe an angular acceptance tolerance.
All polynomial and directional arithmetic is in the native GMP core.
"""

import numpy as np


def level_normal_reference(m, points, direction):
    """Measure each directed chord against the loss normal at its midpoint.

    ``direction`` is +1 for ascent (stable branches stored away from saddle),
    -1 for descent (unstable branches). Exact signs detect reversed flow and
    reversed endpoint loss. ``sin_squared`` is a rounded dimensionless
    directional residual, not the sampling-dependent old angle energy.

    Coordinates are the supplied binary64 values; their midpoint is formed
    exactly in GMP. This measures the represented chords. It cannot resolve
    an invariant curve whose transverse offset is lost on conversion to
    global coordinates, nor establish a bound between sampled midpoints.
    Repeated chords, stationary midpoints and nonfinite points are explicit
    unmeasured entries, never zero-error successes. No endpoints are omitted.
    """
    from . import _native
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("direction must be +1 (ascent) or -1 (descent)")
    Y = np.ascontiguousarray(points, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 2 or len(Y) < 2:
        raise ValueError("at least two (a,b) points are required")
    rows = _native.level_normal_reference(m.alpha, m.beta, Y, int(direction))
    names = ("measured", "nonfinite_point", "zero_chord", "stationary_midpoint")
    return [{"segment": i, "status": names[status],
             "sin_squared": value if status == 0 else None,
             "cross_nonzero": bool(nonzero) if status == 0 else None,
             "flow_alignment": alignment if status == 0 else None,
             "loss_direction": loss if status in (0, 3) else None}
            for i, (status, value, nonzero, alignment, loss) in enumerate(rows)]
