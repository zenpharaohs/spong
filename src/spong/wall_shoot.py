"""Two-sided shooting for saddle-connection walls.

A saddle connection is codimension one: binary64 can bracket the wall
parameter but never stand on it.  The old bracketing was a Dedekind cut on
a Boolean -- which minimum did the unstable branch land in -- and that
gives a coordinate only to the width of the last bisection step, with no
candidate orbit at all.  This module replaces it with a ROOT.

SEPARATION.  For a connection s' -> s (needs u(s') > u(s)) pick the rational
midlevel c = (u(s') + u(s))/2.  L is strictly monotone along orbits, so the
unstable branch of s' crosses {L = c} exactly once descending and the stable
branch of s exactly once ascending.  Both crossings lie on one level
component, on which b is a faithful parameter; the signed gap between them,

    delta = sheet * (b_u - b_s),

is smooth in the model and vanishes exactly at connection.  Shooting to a
level BETWEEN the saddles is what keeps this well conditioned: each shot
stops before the lambda-lemma lingering near the far saddle begins.  Raw
nearest-approach distance is not smooth (the nearest point jumps between
trace segments) and is what the explorer used to print.

RHEOSTAT.  Along the family f/sqrt(Lambda), g*sqrt(Lambda), B is unchanged
and A scales by Lambda, so the critical b's do not move -- only the a's do.
Saddles are identified by b alone, and delta(Lambda) is a smooth scalar
function of one real parameter.  Brent's method on it from the family's
[below, above] bracket converges superlinearly to the binary64 root.

BYPRODUCT.  At Lambda* the two shots meet at c to |delta| ~ eps, and glued
end to end they ARE a candidate for the connection -- numerically
indistinguishable from it at binary64.  The same two shots at any other
Lambda, with their gap, are the honest picture of how close the manifolds
come there; the Lambda* candidate can be overlaid on them for comparison.

Nothing here is a certificate.  The shots are the C core's unit-speed
normalized-gradient integrator at GEOMETRIC_IRK_PRIMARY order (the same one
``/trace`` and ``charts`` use), launched from the engine's own Poincare
stubs; the certified statement about a wall remains the bracket with
verified opposite fates, and the certified statement about a NON-connection
remains the trapping-tube exclusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import model as model_module
from . import sturm, zoo

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------
# rheostat members
# --------------------------------------------------------------------------

def rheostat_model(family: zoo.WallFamily, lam: float):
    """The wall-family member at parameter ``lam``, as ``serve._resolve``
    and ``zoo.rheostat_member`` build it."""
    base = zoo.get(family.base_case)
    root = math.sqrt(float(lam))
    f = [float(x) / root for x in base.f]
    g = [float(x) * root for x in base.g]
    n_moments = 2 * max(len(f), len(g)) - 1
    if base.moment_dist == "uniform01":
        mu = model_module.moments_uniform01(n_moments)
    elif base.moment_dist == "normal01":
        mu = model_module.moments_normal01(n_moments)
    else:
        raise ValueError(f"unsupported wall-family moments: "
                         f"{base.moment_dist}")
    return model_module.build(f, g, mu)


# --------------------------------------------------------------------------
# one shot
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Shot:
    """Two shots to a common level and their signed gap."""
    level: float
    source_b: float
    target_b: float
    source_direction: int
    target_direction: int
    unstable: np.ndarray        # (n, 2) from the source saddle to {L = c}
    stable: np.ndarray          # (n, 2) from the target saddle to {L = c}
    unstable_crossing: tuple[float, float]
    stable_crossing: tuple[float, float]
    delta: float
    steps: int

    @property
    def candidate(self) -> np.ndarray:
        """The glued orbit source -> level -> target.  A candidate
        connection only where |delta| is at binary64 resolution.  The two
        crossings are merged into one vertex at the level."""
        meet = 0.5 * (self.unstable[-1] + self.stable[-1])
        return np.vstack((self.unstable[:-1], [meet], self.stable[-2::-1]))

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "source_b": self.source_b, "target_b": self.target_b,
            "source_direction": self.source_direction,
            "target_direction": self.target_direction,
            "delta": self.delta,
            "steps": self.steps,
            "unstable_crossing": list(self.unstable_crossing),
            "stable_crossing": list(self.stable_crossing),
            "unstable_points": self.unstable.tolist(),
            "stable_points": self.stable.tolist(),
        }


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


def _stub_end(enumeration, saddle_b: float, manifold: str, direction: int):
    saddle = min((q for q in enumeration.points if q.kind == "saddle"),
                 key=lambda q: abs(float(q.b) - saddle_b))
    for stub in saddle.stubs:
        if stub.manifold == manifold and stub.b_direction == direction:
            curve = np.asarray(stub.curve, dtype=float)
            return saddle, curve[-1]
    return saddle, None


def shoot(m, source_b: float, source_direction: int, target_b: float,
          target_direction: int | None = None, *, level: float | None = None,
          ds: float = 2e-3, cap: int = 400000, enumeration=None
          ) -> Shot | None:
    """Shoot both branches to a common level and measure the signed gap.

    ``target_direction`` None tries both stable branches of the target and
    keeps the one with the smaller |delta|; the chosen direction is
    recorded so later calls along a family can pin it.
    """
    kernel = getattr(m, "_native_kernel", None)
    if kernel is None or not hasattr(kernel, "normalized_step"):
        return None
    if enumeration is None:
        enumeration = sturm.materialize_stubs(
            m, sturm.enumerate_critical_points(m))
    source, zu = _stub_end(enumeration, source_b, "unstable",
                           source_direction)
    if zu is None:
        return None
    target = min((q for q in enumeration.points if q.kind == "saddle"),
                 key=lambda q: abs(float(q.b) - target_b))
    us, ut = float(m.L(source.a, source.b)), float(m.L(target.a, target.b))
    if not us > ut:
        return None
    c = 0.5 * (us + ut) if level is None else float(level)
    if not ut < c < us:
        return None
    unstable = _level_crossing(m, kernel, zu[0], zu[1], False, c, ds, cap)
    if unstable is None:
        return None
    unstable = np.vstack(([[float(source.a), float(source.b)]], unstable))

    best = None
    directions = ((target_direction,) if target_direction is not None
                  else (+1, -1))
    A = np.asarray([float(x) for x in m.alpha])[::-1]
    B = np.asarray([float(x) for x in m.beta])[::-1]
    for direction in directions:
        _t, zs = _stub_end(enumeration, target_b, "stable", direction)
        if zs is None:
            continue
        stable = _level_crossing(m, kernel, zs[0], zs[1], True, c, ds, cap)
        if stable is None:
            continue
        stable = np.vstack(([[float(target.a), float(target.b)]], stable))
        pu, ps = unstable[-1], stable[-1]
        # Signed gap along the level component: b is a faithful parameter
        # on one sheet; the sheet sign is a - a*(b) at the unstable
        # crossing.
        sheet = math.copysign(
            1.0, pu[0] - np.polyval(B, pu[1]) / np.polyval(A, pu[1]))
        delta = sheet * (float(pu[1]) - float(ps[1]))
        shot = Shot(c, float(source.b), float(target.b),
                    int(source_direction), int(direction), unstable, stable,
                    (float(pu[0]), float(pu[1])),
                    (float(ps[0]), float(ps[1])), float(delta),
                    int(len(unstable) + len(stable)))
        if best is None or abs(shot.delta) < abs(best.delta):
            best = shot
    return best


# --------------------------------------------------------------------------
# Brent on the rheostat
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WallRoot:
    family: str
    lam: float                  # the binary64 root of delta(Lambda)
    delta: float                # delta at lam
    bracket: tuple[float, float]   # final Brent bracket, opposite signs
    evaluations: int
    target_direction: int
    shot: Shot                  # the shot at lam; .candidate is the orbit
    history: tuple[tuple[float, float], ...]   # (lam, delta) per evaluation

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "lam": self.lam,
            "delta": self.delta,
            "bracket": list(self.bracket),
            "evaluations": self.evaluations,
            "target_direction": self.target_direction,
            "history": [list(h) for h in self.history],
            "candidate_points": self.shot.candidate.tolist(),
            "shot": self.shot.as_dict(),
        }


def _brent(fn, a, b, fa, fb, *, xtol: float, ftol: float, max_iter: int):
    """Brent's method (van Wijngaarden-Dekker-Brent) on a bracket."""
    if fa * fb > 0:
        raise ValueError("Brent needs a sign change on the bracket")
    c, fc = a, fa
    d = e = b - a
    evaluations = 0
    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * EPS * abs(b) + 0.5 * xtol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or abs(fb) <= ftol:
            return b, fb, (min(b, c), max(b, c)), evaluations
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else math.copysign(tol1, xm)
        fb = fn(b)
        evaluations += 1
    return b, fb, (min(b, c), max(b, c)), evaluations


def find_wall(family: zoo.WallFamily, *, lo: float | None = None,
              hi: float | None = None, xtol: float = 0.0,
              ftol: float = 1e-13, max_iter: int = 80, ds: float = 2e-3,
              target_direction: int | None = None) -> WallRoot:
    """Brent's method on delta(Lambda) over the family's bracket.

    Returns the binary64 root and the shot there, whose glued polyline is
    the connection candidate.  Raises ValueError if delta does not change
    sign on [lo, hi] for either stable direction of the target -- an
    honest refusal, not a coordinate.
    """
    lo = family.below_parameter if lo is None else float(lo)
    hi = family.above_parameter if hi is None else float(hi)
    history: list[tuple[float, float]] = []
    shots: dict[float, Shot] = {}

    def make(direction):
        def delta(lam):
            m = rheostat_model(family, lam)
            shot = shoot(m, family.source_b, family.unstable_direction,
                         family.target_b, direction, ds=ds)
            if shot is None:
                raise ValueError(
                    f"shot failed at {family.parameter_name} = {lam!r}")
            history.append((float(lam), shot.delta))
            shots[float(lam)] = shot
            return shot.delta
        return delta

    directions = ((target_direction,) if target_direction is not None
                  else (+1, -1))
    failures = []
    for direction in directions:
        delta = make(direction)
        f_lo, f_hi = delta(lo), delta(hi)
        if f_lo * f_hi > 0:
            failures.append((direction, f_lo, f_hi))
            continue
        lam, value, bracket, evaluations = _brent(
            delta, lo, hi, f_lo, f_hi, xtol=xtol, ftol=ftol,
            max_iter=max_iter)
        return WallRoot(family.name, float(lam), float(value),
                        (float(bracket[0]), float(bracket[1])),
                        evaluations + 2, int(direction), shots[float(lam)],
                        tuple(history))
    raise ValueError(
        "delta does not change sign on the family bracket: "
        + "; ".join(f"stable {d:+d}: delta({lo}) = {a:.3e}, "
                    f"delta({hi}) = {b:.3e}" for d, a, b in failures))
