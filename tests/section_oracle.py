"""Legacy Python event algorithm retained only as a differential oracle."""
import numpy as np

def _level_crossing(m, kernel, a0, b0, ascend: bool, c: float,
                    ds: float, cap: int):
    """March until L crosses ``c``; the polyline up to and including the
    crossing, or None if it never crosses.

    The crossing is refined by re-stepping from the bracketing step's start
    with a fractional arclength and secant corrections on L, so it carries
    the integrator's accuracy.  Linear interpolation along the chord would
    get b right to the last bit (the bracket is in L) but leave a with the
    chord's O(ds^2) sagitta error, which is what the two halves of a glued
    candidate would then disagree by.
    """
    sgn = 1.0 if ascend else -1.0
    a, b = float(a0), float(b0)
    prev = (a, b, float(m.L(a, b)))
    pts = [(a, b)]
    for _ in range(cap):
        try:
            a, b = kernel.normalized_step(a, b, sgn * ds, 8)
        except (ArithmeticError, ValueError, OverflowError,
                ZeroDivisionError):
            return None
        a, b = float(a), float(b)
        if not (a == a and b == b):
            return None
        L = float(m.L(a, b))
        if (L - c) * (prev[2] - c) <= 0.0:
            pts.append(_refine_crossing(m, kernel, prev, (a, b, L), sgn * ds,
                                        c))
            return np.asarray(pts, dtype=float)
        pts.append((a, b))
        prev = (a, b, L)
        if abs(a) > 1e4 or abs(b) > 1e4:
            return None
    return None


def _refine_crossing(m, kernel, prev, curr, step, c, iterations: int = 4):
    """Secant on the fractional arclength w in [0, 1] from ``prev`` such
    that L(step(prev, w*step)) = c.  Falls back to the chord interpolation
    if the kernel refuses a fractional step."""
    a0, b0, L0 = prev
    a1, b1, L1 = curr
    w0, f0 = 0.0, L0 - c
    w1, f1 = 1.0, L1 - c
    if f1 != f0:
        w = -f0 / (f1 - f0)
        best = (a0 + w * (a1 - a0), b0 + w * (b1 - b0))
    else:
        best = (a1, b1)
    for _ in range(iterations):
        if f1 == f0:
            break
        w = w1 - f1 * (w1 - w0) / (f1 - f0)
        w = min(max(w, 0.0), 1.0)
        try:
            a, b = kernel.normalized_step(a0, b0, w * step, 8)
        except (ArithmeticError, ValueError, OverflowError,
                ZeroDivisionError):
            return best
        a, b = float(a), float(b)
        if not (a == a and b == b):
            return best
        f = float(m.L(a, b)) - c
        best = (a, b)
        if f == 0.0:
            break
        w0, f0, w1, f1 = w1, f1, w, f
    return best
