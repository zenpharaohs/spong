#!/usr/bin/env python3
"""Backend for the interactive one-neuron loss explorer.

    python demos/explorer/serve.py        # http://127.0.0.1:8710

Every change of f, g, or the moments in the browser posts to /portrait and
gets back a freshly computed CERTIFIED portrait.  The browser does no
analysis: it evaluates L(a,b) = C - 2a*B(b) + a^2*A(b) for the heatmap and
draws what this endpoint sends.  Critical points, their classification, the
separatrices and the ledger all come from portrait.certified_compute.

The demo endpoints share that exact cached model: /trace continues one
normalized-gradient orbit, while /allocator creates equal-versus-Thompson
optimizer traces from a common design in the visible window.

Nothing here re-implements spong.  Model.alpha, Model.beta and Model.C are
the exact A, B and C; Model.backbone_num / backbone_den are the reduced
B^2/A the constructor already forms; each CriticalPoint carries b, a, kind,
source and the exact sign of u''.  This file only serializes them.

Lives in demos/ because the descent traces it drives are descent methods,
which the founding document keeps out of the library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Default to the accelerated path.  The explorer is a viewer, not the
# certifying CLI, and latency is what it trades in; both settings are
# overridable from the environment.  Must precede the spong import, since
# engine reads SPONG_ENGINE at import time.
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "8")

try:
    from spong import atlas, inverse, merge_tree, model, portrait, sturm, wall_shoot, zoo
except ImportError:                                  # running from a checkout
    sys.path.insert(0, str(REPO / "src"))
    from spong import atlas, inverse, merge_tree, model, portrait, sturm, wall_shoot, zoo

# The allocator experiment remains demo code, but the viewer needs to run it
# against the exact model behind the active portrait.  A script launched as
# ``python demos/explorer/serve.py`` does not otherwise put the repository
# root (and hence the ``demos`` package) on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demos import optimizer_moustaches as optimizer_gallery
from demos import saddle_connection_triptych as saddle_wall
from demos import thompson_moustaches as allocator_demo

try:                                                 # exact C core
    from spong import _native
    NATIVE = {
        "loaded": True,
        "abi": getattr(_native, "SPONG_ABI_VERSION", None),
        "types": [n for n in ("Kernel", "LocalKernel",
                              "SturmPlan", "ContactScan")
                  if hasattr(_native, n)],
    }
except ImportError as _exc:                          # ABI or interpreter skew
    NATIVE = {"loaded": False, "error": str(_exc), "types": []}

PAGE = HERE / "index.html"
PORT = 8710

# Recomputing the same (f, g, mu, view) is common -- flipping between presets,
# re-focusing a coefficient field -- and a degree-11 enumeration plus branch
# tracing is not cheap.  Small LRU on the exact request.
_CACHE: dict = {}
_CACHE_MAX = 24
_TAILS: dict = {}
_TRACED: dict = {}

# Enumeration and materialized stubs are shared between the preview and the
# final stage of the same model: they are upstream of the geometry ladder and
# cost seconds on the hard cases, so recomputing them per stage would be pure
# waste.
_ENUM: dict = {}
_ENUM_MAX = 8

# Models are cached too, so /trace can integrate against the same object the
# portrait was built from without rebuilding it.
_MODELS: dict = {}
_MODELS_MAX = 8

# One Brent solve per wall family (about three seconds at ds = 5e-4),
# computed on the first wall-limit request and kept for the process.
_WALL_ROOTS: dict = {}
WALL_SHOOT_DS = 5e-4


def _wall_root(family):
    """The family's Brent root, or None if shooting is unavailable (no C
    core) or refuses (no sign change on the bracket) -- callers then fall
    back to the stored wall_parameter and the older construction."""
    if family.name in _WALL_ROOTS:
        return _WALL_ROOTS[family.name]
    try:
        root = wall_shoot.find_wall(family, ds=WALL_SHOOT_DS)
    except (ValueError, ArithmeticError, RuntimeError):
        root = None
    _WALL_ROOTS[family.name] = root
    return root


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------

def moment_vector(spec: dict, n: int):
    """Exact moments for the requested input distribution.

    'empirical' takes literal sample points from the client and forms the
    exact rational empirical moments, so a batch portrait is a genuine SPONG
    model rather than an approximation of one.
    """
    kind = spec.get("kind", "uniform01")
    if kind == "absolute_x":
        return model.moments_absolute_x(n)
    if kind == "uniform01":
        return model.moments_uniform01(n)
    if kind == "normal01":
        return model.moments_normal01(n)
    if kind == "rademacher":
        return model.moments_rademacher(n)
    if kind == "empirical":
        xs = [Fraction(float(x)) for x in spec.get("samples", [])]
        if not xs:
            raise ValueError("empirical moments need at least one sample")
        inv = Fraction(1, len(xs))
        return [inv * sum((x ** k for x in xs), Fraction(0)) for k in range(n)]
    raise ValueError(f"unknown moment kind {kind!r}")


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------

def _field_coeffs(m):
    """A(b), B(b), C as ascending float coefficient lists, plus the reduced
    backbone numerator and denominator the model already carries."""
    return {
        "A": [float(c) for c in m.alpha],
        "B": [float(c) for c in m.beta],
        "C": float(m.C),
        "backbone_num": [float(c) for c in m.backbone_num],
        "backbone_den": [float(c) for c in m.backbone_den],
    }


def _critical_points(e, m):
    """The certified inventory, verbatim from sturm.CriticalPoint.

    u2_sign is the EXACT sign of u''; no magnitude is reported, because the
    enumeration certifies the sign rather than a value.  source records which
    polynomial produced the root -- 'B' roots are the a* = 0 saddles that are
    saddles by the identity det H = -4 B'^2.

    'global' follows render.py's convention: minima within a relative
    tolerance of the least loss get an open circle; other minima get the
    equal-size filled version.
    """
    pts = sorted(e.points, key=lambda p: p.b)
    losses = {id(p): float(m.L(p.a, p.b)) for p in pts}
    mins = [losses[id(p)] for p in pts if p.kind == "min"]
    best = min(mins) if mins else None
    out = []
    for p in pts:
        L = losses[id(p)]
        is_global = (p.kind == "min" and best is not None
                     and L <= best + 1e-9 * (1.0 + abs(best)))
        out.append({"b": float(p.b), "a": float(p.a), "kind": p.kind,
                    "source": p.source, "u2_sign": int(p.u2_sign),
                    "loss": L, "global": bool(is_global)})
    return out


def _max_chord(Y) -> float:
    """Longest chord on a traced branch, in model units.

    Interpolating between vertices is licensed by the branch residuals, so the
    chord length is the scale below which the drawn curve stops being a
    certified rendering and becomes a magnified straight line.  The viewer
    reports it rather than silently zooming past it.
    """
    worst = 0.0
    for i in range(1, len(Y)):
        da = float(Y[i][0]) - float(Y[i - 1][0])
        db = float(Y[i][1]) - float(Y[i - 1][1])
        d = (da * da + db * db) ** 0.5
        if d > worst:
            worst = d
    return worst


from spong.stable_tails import (  # noqa: E402
    EPS, _abs_horner, _grow_stable, _horner, _j_floor, _j_invariant, _j_root,
    _j_tails, _j_value, _nullcline_b, _residue, _root_resolution, _traps)


def _branches(p, max_pts: int = 200000):
    """Branch polylines at full traced resolution.

    The tracing is deliberately fine so that the viewer can zoom without
    recomputing -- thinning here would throw that away.  The client decimates
    at draw time against the current pixel scale instead, so the data stays
    complete and only the drawing adapts.  max_pts is a safety valve, not a
    display budget.
    """
    out = []
    for br in p.branches:
        Y = br.Y
        n = len(Y)
        step = max(1, n // max_pts)
        pts = [[float(Y[i][0]), float(Y[i][1])] for i in range(0, n, step)]
        if step > 1 and n:
            pts.append([float(Y[n - 1][0]), float(Y[n - 1][1])])
        # The asymptote certificate travels with the branch.  Interpolation
        # between traced vertices is already certified by the angle and turn
        # residuals, so zooming IN never needs a recompute; extrapolation past
        # the traced box is the only open case, and under genericity it has a
        # certificate of its own -- every separatrix leaves along a diagonal
        # b = +-sqrt(d_eff)*a, and this residual says how close the tail is.
        cert = br.certs.get("asymptote")
        out.append({"kind": br.kind, "term": br.term,
                    "saddle_b": br.diag.get("saddle_b"),
                    "direction": br.diag.get("unstable_direction"),
                    "stable_sign": br.diag.get("stable_sign"),
                    "target": br.diag.get("target"),
                    "connection_ok": bool(br.certs.get("connection_ok", False)),
                    "n_traced": n, "stride": step,
                    "chord_max": _max_chord(Y),
                    "asymptote": (None if cert is None else {
                        "slope_extrapolated": float(
                            cert["slope_extrapolated"]),
                        "target": float(cert["target"]),
                        "residual": float(cert["residual"]),
                    }),
                    "points": pts})
    return out


def _resolve(payload: dict):
    """(f, g, view, moment spec, cache key) from a request."""
    wall = payload.get("wall")
    name = payload.get("zoo")
    separated = payload.get("separated_linear")
    if wall:
        # A rheostat member of a wall family, at an arbitrary Lambda.
        # zoo.rheostat_member materializes only the three named members; the
        # scaling f/sqrt(L), g*sqrt(L) is the same one, opened up so the
        # viewer can move through the family continuously.
        #
        # The wall COORDINATE is not the citable object -- wall_bracket is,
        # an interval whose endpoints have verified opposite landing fates.
        # Inside the bracket the fate is launch-protocol-dependent, so the
        # page must show the bracket rather than imply that any particular
        # Lambda in it is "the" wall.
        w = zoo.get_wall_family(wall)
        base = zoo.get(w.base_case)
        # ``wall_limit`` names the wall itself, not a nearby slider
        # coordinate.  The coordinate used is the Brent root of the
        # two-sided shooting separation (spong.wall_shoot), which is the
        # binary64 wall to the integrator's accuracy; the family's stored
        # ``wall_parameter`` is the older fate-bisection value and can sit
        # a few 1e-9 away from it.  Ordinary slider positions use their
        # literal Lambda.
        wall_root = _wall_root(w) if payload.get("wall_limit") else None
        lam = (wall_root.lam if wall_root is not None else
               w.wall_parameter if payload.get("wall_limit") else
               float(payload.get("lam", w.wall_parameter)))
        scale = math.sqrt(lam)
        f = [float(x) / scale for x in base.f]
        g = [float(x) * scale for x in base.g]
        view = tuple(float(x) for x in w.default_view)
        spec = {"kind": base.moment_dist}
    elif name in zoo.numerical_connection_names():
        z = zoo.get_numerical_connection(name)
        f, g = z.coefficients(payload.get("candidate_lam"))
        view = z.default_view
        spec = {"kind": z.moment_dist}
    elif name:
        z = zoo.get(name)
        f = [float(x) for x in z.f]
        g = [float(x) for x in z.g]
        view = tuple(float(x) for x in z.default_view) \
            if z.default_view else None
        spec = {"kind": z.moment_dist}
    elif separated:
        degree = int(separated.get("degree", 4))
        if not 1 <= degree <= 8:
            raise ValueError(
                "inspector separated-linear degree must lie in [1, 8]")
        raw_separation = separated.get("separation", "30")
        try:
            scale = Fraction(str(raw_separation))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("separation must be a rational number") from exc
        if not Fraction(1) < scale <= Fraction(1000):
            raise ValueError("separation must lie in (1, 1000]")
        family = inverse.separated_linear_case(degree, scale)
        # Keep the rational coefficients through model construction.  They
        # are converted only when serialized for the browser.
        f = list(family.f)
        g = list(family.g)
        view = None
        spec = {"kind": "uniform01"}
    else:
        f = [float(x) for x in payload["f"]]
        g = [float(x) for x in payload["g"]]
        view = payload.get("view")
        if view is not None:
            view = tuple(float(x) for x in view)   # (a_lo, a_hi, b_lo, b_hi)
        spec = payload.get("moments", {})
    key = (tuple(f), tuple(g), view, spec.get("kind", "uniform01"),
           tuple(float(x) for x in spec.get("samples", ())))
    return f, g, view, spec, key


def _wall_pair(payload: dict, p):
    """The two continuations a wall family is about, at ANY Lambda.

    The source saddle's unstable branch in the family's direction and the
    target saddle's stable branch nearest the source, with the closest
    approach of each to the other's critical point.  At the wall these
    coincide in the limit; away from it they miss by an amount the viewer
    should print rather than leave to the eye, since a near-connection at
    ordinary zoom and a clean miss at deep zoom are the same picture.

    Returns None outside a wall family or when the branches cannot be
    identified (a refused portrait may lack them).
    """
    name = payload.get("wall")
    if not name:
        return None
    family = zoo.get_wall_family(name)
    try:
        source = saddle_wall.critical_near(p, family.source_b)
        target = saddle_wall.critical_near(p, family.target_b)
    except (LookupError, ValueError):
        return None
    unstable = [
        (index, branch) for index, branch in enumerate(p.branches)
        if branch.kind == "unstable"
        and abs(branch.diag.get("saddle_b", math.inf)-source.b) < 1e-7
        and branch.diag.get("unstable_direction") ==
        family.unstable_direction]
    stable = [
        (index, branch) for index, branch in enumerate(p.branches)
        if branch.kind == "stable"
        and abs(branch.diag.get("saddle_b", math.inf)-target.b) < 1e-7]
    if not unstable or not stable:
        return None
    su_index, su = unstable[0]
    distance_to_target = ((su.Y[:, 0]-target.a)**2
                          + (su.Y[:, 1]-target.b)**2)
    target_index = int(distance_to_target.argmin())
    closest_target = math.sqrt(float(distance_to_target[target_index]))

    def source_distance(branch):
        return math.sqrt(float(np.min(
            (branch.Y[:, 0]-source.a)**2 + (branch.Y[:, 1]-source.b)**2)))
    ts_index, ts = min(stable, key=lambda item: source_distance(item[1]))
    return {
        "source_b": float(source.b), "target_b": float(target.b),
        "source_a": float(source.a), "target_a": float(target.a),
        "source_unstable": su_index, "target_stable": ts_index,
        "unstable_to_target": closest_target,
        "stable_to_source": source_distance(ts),
        "closest_index": target_index,
        "unstable_direction": family.unstable_direction,
    }


def _geometric_wall_limit(payload: dict, p, root=None):
    """Dedicated display geometry for the selected saddle-connection wall.

    An ordinary portrait at a non-Morse-Smale wall must refuse: its unstable
    and stable numerical continuations are not independently meaningful once
    they coincide.  Remove those two continuations and insert the connection
    candidate.

    With ``root`` (a :class:`wall_shoot.WallRoot`) the candidate is the
    glued two-sided shot at the Brent root: source saddle down to the
    midlevel, target saddle up to it, meeting there to |delta| at binary64
    resolution.  Without it -- the older construction, kept for callers
    that have no root -- the house continuation is truncated at closest
    approach and its last vertex moved onto the target, and the miss is
    reported.
    """
    name = payload.get("wall")
    if not name or not payload.get("wall_limit"):
        return p, [], None

    family = zoo.get_wall_family(name)
    pair = _wall_pair(payload, p)
    if pair is None:
        raise ValueError("wall limit: the source unstable or target stable "
                         "continuation is missing from this portrait")
    if root is not None:
        connection = np.asarray(root.shot.candidate, dtype=float)
        method = "two-sided shooting, Brent root"
        gap = abs(root.delta)
        parameter = root.lam
    else:
        source_unstable = p.branches[pair["source_unstable"]]
        target_index = pair["closest_index"]
        connection = source_unstable.Y[:target_index+1].copy()
        # The house continuation gets within its event/chord tolerance and
        # then continues into one of the adjacent chambers.  At the
        # geometric wall limit the orbit ends at the exact target critical
        # point.
        connection[-1] = (pair["target_a"], pair["target_b"])
        method = "paired house continuations"
        gap = pair["unstable_to_target"]
        parameter = float(family.wall_parameter)

    display, surgery = saddle_wall.wall_limit_portrait(
        p, family, connection)
    points = [[float(q[0]), float(q[1])] for q in connection]
    record = {
        "kind": "stable_unstable",
        "label": "saddle connection (Wu = Ws)",
        "source_b": pair["source_b"],
        "target_b": pair["target_b"],
        "n_traced": len(points),
        "points": points,
        # For the shooting candidate: |delta| at the midlevel where the two
        # shots meet.  For the older construction: how far the numerical
        # continuation missed the target before its last vertex was snapped
        # onto it.  Either way, this number is what binary64 actually did.
        "numerical_miss": float(gap),
    }
    diagnostics = {
        "geometry_method": "geometric wall limit",
        "parameter": parameter,
        "trace": {
            "method": method,
            "source_unstable_closest_to_target": pair["unstable_to_target"],
            "target_stable_closest_to_source": pair["stable_to_source"],
        },
        "surgery": surgery,
    }
    return display, [record], diagnostics


def _model_for(key, f, g, spec):
    m = _MODELS.get(key)
    if m is None:
        n_moments = 2 * max(len(f), len(g)) - 1
        m = model.build(f, g, moment_vector(spec, n_moments))
        _MODELS[key] = m
        if len(_MODELS) > _MODELS_MAX:
            _MODELS.pop(next(iter(_MODELS)))
    return m


def trace(payload: dict) -> dict:
    """Arclength continuation of the gradient field from one point.

    Uses Kernel.normalized_step -- the C core's 2-D normalized-gradient
    integrator at GEOMETRIC_IRK_PRIMARY order, the same one charts falls back
    to when both graph parameterizations go singular.  An explicit method has
    no business on this field: the stiffness that forced the whole certified
    machinery is exactly what makes a descent trajectory crawl.

    Unit speed is the point.  True gradient time never arrives on a stiff
    valley; arclength travels the SAME curve at constant speed, so a bounded
    number of steps answers "where does this initial condition go" instead of
    "how far does it get before you lose patience".
    """
    f, g, _view, spec, key = _resolve(payload)
    m = _model_for(key, f, g, spec)
    kernel = getattr(m, "_native_kernel", None)
    if kernel is None or not hasattr(kernel, "normalized_step"):
        return {"error": "normalized_step unavailable (no C core)"}

    a = float(payload["a"])
    b = float(payload["b"])
    flow = 1 if int(payload.get("flow", 1)) > 0 else -1
    ds = float(payload.get("ds", 0.01))
    steps = max(1, min(200000, int(payload.get("steps", 4000))))
    order = int(payload.get("order", 8))
    box = payload.get("box")
    box = [float(x) for x in box] if box else None

    pts = [a, b]
    term = "steps"
    for _ in range(steps):
        try:
            a_new, b_new = kernel.normalized_step(a, b, -flow * ds, order)
        except (ArithmeticError, ValueError, OverflowError,
                ZeroDivisionError):
            term = "step_failure"
            break
        if not (a_new == a_new and b_new == b_new):      # NaN
            term = "nonfinite"
            break
        if abs(a_new - a) + abs(b_new - b) < 1e-14 * ds:
            a, b = a_new, b_new
            term = "stationary"
            break
        a, b = float(a_new), float(b_new)
        pts.extend((a, b))
        if box and not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
            term = "box_exit"
            break
    return {"points": pts, "term": term, "order": order,
            "arclength": ds * (len(pts) // 2 - 1)}


# Interactive allocator ceilings.  The uniform record is computed in parallel,
# so a long run is bounded by memory rather than patience: every optimizer
# iterate is retained for replay (about 136 bytes per step in the record,
# a few copies at most across the two policies), so 50M steps stays in the
# low tens of GB.  The browser only ever receives thinned traces.
ALLOCATOR_MAX_ROUNDS = 10_000_000
ALLOCATOR_MAX_STEPS = 50_000_000
ALLOCATOR_MAX_SECONDS = 3600.0
ALLOCATOR_HISTORY_POINTS = 384


def _thin_history(history, max_points=ALLOCATOR_HISTORY_POINTS):
    """Log-grid subsample of a per-pull loss history, keeping both ends."""
    n = len(history)
    if n <= max_points:
        return list(history)
    indices = np.unique(np.concatenate((
        [0], np.rint(np.geomspace(1, n-1, max_points-1)).astype(int))))
    return [history[int(i)] for i in indices]


def _bounded_int(payload, name, default, lower, upper):
    value = int(payload.get(name, default))
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _learning_curve_samples(trajectory, m, *, exact=False, max_points=384):
    """Sample optimizer steps on a log grid, retaining every early step."""
    n = len(trajectory)-1
    if n <= 0:
        return []
    indices = (np.arange(1, n+1) if n <= max_points else
               np.unique(np.rint(np.geomspace(1, n, max_points)).astype(int)))
    evaluator = (allocator_demo.opt.ExactFisherGNState(trajectory[0], m)
                 if exact else None)
    result = []
    for step in indices:
        point = trajectory[int(step)]
        value = (float(evaluator._exact_values(*point)) if evaluator is not None
                 else allocator_demo.thompson._raw_loss(m.L, point))
        result.append([int(step), value if math.isfinite(value) else None])
    return result


def _allocator_policy(result, color, max_points=350, *,
                      uniform_steps=None,
                      width_base=0.55, width_gain=1.2,
                      opacity_base=0.08, opacity_gain=0.72,
                      mark_start=False, mark_end=False, model=None,
                      exact_learning_curve=False):
    """Serialize one allocation policy as weighted viewer traces."""
    counts = result.allocations
    largest = max(int(max(counts)), 1)
    traces = []
    for arm, (trajectory, pulls) in enumerate(
            zip(result.trajectories, counts)):
        if uniform_steps is None:
            selected = optimizer_gallery._thin(
                trajectory, max_points=max_points)
        else:
            selected = optimizer_gallery._thin_comparison(
                trajectory, uniform_steps[arm], prefix_points=max_points)
        raw_points = selected.tolist()
        # json.dumps would otherwise emit non-standard NaN/Infinity tokens
        # when a deliberately ill-scaled optimizer escapes.  Stop a drawn
        # trajectory at its last finite point, as the standalone gallery does
        # visually, while the allocation and final observation still record
        # the failed continuation.
        points = []
        for point in raw_points:
            if len(point) != 2 or not all(math.isfinite(float(x))
                                          for x in point):
                break
            points.append([float(point[0]), float(point[1])])
        share = float(pulls) / largest
        traces.append({
            "arm": arm,
            "pulls": int(pulls),
            "executed_steps": int(result.executed_steps[arm]),
            "termination": result.termination_reasons[arm],
            "points": points,
            "plotted_points": len(points),
            "color": color,
            "width": width_base + width_gain * share,
            "opacity": opacity_base + opacity_gain * math.sqrt(share),
            "mark_start": bool(mark_start),
            "mark_end": bool(mark_end),
            "endpoint": (list(map(float, trajectory[-1]))
                         if len(trajectory) and np.all(np.isfinite(trajectory[-1]))
                         else None),
            "last_loss": (float(result.loss_histories[arm][-1][1])
                          if result.loss_histories[arm]
                          and math.isfinite(result.loss_histories[arm][-1][1])
                          else None),
            "loss_history": [
                [int(step), float(value) if math.isfinite(value) else None]
                for step, value in _thin_history(result.loss_histories[arm])],
            "learning_curve": _learning_curve_samples(
                trajectory, model, exact=exact_learning_curve)
                if model is not None else [],
        })
    return {
        **allocator_demo._allocation_summary(result),
        "traces": traces,
        "allocations": [int(x) for x in counts],
        "final_observations": [
            (float(values[-1]) if values else None)
            for values in result.observations],
    }


def allocator(payload: dict) -> dict:
    """Equal-versus-Thompson traces for the active portrait model.

    The request uses the same model selector as ``/portrait`` and ``/trace``.
    ``allocation_view`` is deliberately separate: it is the visible window
    from which the browser asks us to place the common initialization design,
    not a request to rebuild or recertify the portrait.
    """
    def progress(phase):
        if os.environ.get("SPONG_ALLOCATOR_PROGRESS") == "1":
            print(f"SPONG_ALLOCATOR_PHASE={phase}", file=sys.stderr,
                  flush=True)

    progress("model construction")
    f, g, resolved_view, spec, key = _resolve(payload)
    m = _model_for(key, f, g, spec)

    raw_view = payload.get("allocation_view", resolved_view)
    if raw_view is None or len(raw_view) != 4:
        raise ValueError("allocation_view must contain [a0, a1, b0, b1]")
    view = tuple(float(x) for x in raw_view)
    if not all(math.isfinite(x) for x in view):
        raise ValueError("allocation_view must be finite")
    if not (view[0] < view[1] and view[2] < view[3]):
        raise ValueError("allocation_view must have increasing bounds")

    starts = _bounded_int(payload, "starts", 32, 1, 512)
    rounds = _bounded_int(payload, "rounds", 3200, starts,
                          ALLOCATOR_MAX_ROUNDS)
    chunk_steps = _bounded_int(payload, "chunk_steps", 10, 1, 1000)
    batch_size = _bounded_int(payload, "batch_size", 32, 1, 100000)
    seed = _bounded_int(payload, "seed", 1729, 0, 2**32 - 1)
    if rounds * chunk_steps > ALLOCATOR_MAX_STEPS:
        raise ValueError("rounds * chunk_steps must not exceed "
                         f"{ALLOCATOR_MAX_STEPS:,}")

    method = str(payload.get("method", "adam"))
    schedule = str(payload.get("schedule", "inverse-sqrt"))
    design = str(payload.get("design", "low-discrepancy"))
    # "auto" calibrates each arm from its own exact gradient at its own
    # start, so the opening displacement is a fraction of the visible window
    # rather than a fixed multiple of whatever gradient the arm landed on.
    # SGD needs this: it moves lr*|grad L| per step with nothing normalizing
    # it, so an arm starting high on a quartic wall takes an enormous first
    # jump.  A numeric rate is still honoured verbatim.
    raw_lr = payload.get(
        "learning_rate", optimizer_gallery.DEFAULT_LR.get(method, 1e-3))
    if method in {"fisher-gn", "fisher-gn-exact"}:
        learning_rate = None
    elif isinstance(raw_lr, str) and raw_lr.strip().lower() == "auto":
        learning_rate = "auto"
    else:
        learning_rate = float(raw_lr)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
    time_limit_sec = float(payload.get("time_limit_sec", 60.0))
    if (not math.isfinite(time_limit_sec)
            or not 1.0 <= time_limit_sec <= ALLOCATOR_MAX_SECONDS):
        raise ValueError("time_limit_sec must be finite and in "
                         f"[1, {ALLOCATOR_MAX_SECONDS:g}]")
    retire_local_minima = payload.get("retire_local_minima", False)
    if not isinstance(retire_local_minima, bool):
        raise ValueError("retire_local_minima must be boolean")
    if retire_local_minima and method != "fisher-gn-exact":
        raise ValueError("local-minimum retirement requires fisher-gn-exact")

    distribution = spec.get("kind", "uniform01")
    samples = spec.get("samples") if distribution == "empirical" else None
    progress("sampler library")
    cb_library = allocator_demo.cb_sampler.resolve_library(auto_build=True)
    t0 = time.perf_counter()
    deadline = t0 + time_limit_sec
    comparison = allocator_demo.compare_allocators(
        m, f, g, view, starts=starts, rounds=rounds,
        chunk_steps=chunk_steps, batch_size=batch_size, method=method,
        schedule=schedule, design=design, seed=seed,
        distribution=distribution, samples=samples,
        learning_rate=learning_rate, cb_library=cb_library,
        should_stop=lambda: time.perf_counter() >= deadline,
        deadline=deadline, progress=progress,
        retire_local_minima=retire_local_minima)
    elapsed = time.perf_counter() - t0
    progress("response preparation")

    return {
        "format": "spong-equal-thompson-traces-v1",
        "observation": comparison["observation"],
        "selection": "minimum exact continuous-Bernoulli posterior draw",
        "capture_used_for_allocation": False,
        "allocation_view": list(view),
        "elapsed_sec": elapsed,
        "work_loss_histogram": allocator_demo.work_loss_histogram(
            comparison["equal"], comparison["thompson"]),
        "configuration": {
            "starts": starts,
            "rounds": rounds,
            "chunk_steps": chunk_steps,
            "total_optimizer_steps_per_policy": rounds * chunk_steps,
            "batch_size": batch_size,
            "method": method,
            "schedule": schedule,
            "warmup_steps": comparison["warmup_steps"],
            # A list when calibrated per arm; a scalar when fixed.
            "learning_rate": comparison["learning_rate"],
            "learning_rate_mode": (
                "line-search" if method in {"fisher-gn", "fisher-gn-exact"} else
                "auto" if learning_rate == "auto" else "fixed"),
            "time_limit_sec": time_limit_sec,
            "design": design,
            "seed": seed,
            "distribution": distribution,
            "observation_calibration": comparison["observation_calibration"],
            "cb_library": comparison["library_path"],
            "optimizer_backend": comparison["optimizer_backend"],
            "fisher_damping": (
                allocator_demo.opt.DEFAULT_FISHER_DAMPING
                if method in {"fisher-gn", "fisher-gn-exact"} else None),
            "line_search_scope": (
                "minibatch" if method == "fisher-gn" else
                "population" if method == "fisher-gn-exact" else None),
            "retire_local_minima": retire_local_minima,
            "uniform_workers": comparison["uniform_workers"],
            "replayed_bandit_chunks": comparison["replayed_chunks"],
            "extended_bandit_chunks": comparison["extended_chunks"],
        },
        # Equal is deliberately a broad, half-transparent underlay.  The
        # narrower saturated Thompson trace then remains legible while the
        # common initial portion still reads as an overlap.
        "equal": _allocator_policy(
            comparison["equal"], "#66727b",
            width_base=0.85, width_gain=1.30,
            opacity_base=0.14, opacity_gain=0.28, model=m,
            exact_learning_curve=(method == "fisher-gn-exact")),
        "thompson": _allocator_policy(
            comparison["thompson"], "#5b21b6",
            uniform_steps=comparison["equal"].executed_steps,
            opacity_base=0.10, opacity_gain=0.80,
            mark_start=True, mark_end=True, model=m,
            exact_learning_curve=(method == "fisher-gn-exact")),
    }


def allocator_endpoint_audit(payload: dict) -> dict:
    """Resolve optimizer endpoints by the exact merge tree, then IRK8-GL.

    A bounded saddle-free sublevel component containing one minimum already
    proves the eventual gradient-flow destination.  Only undecided endpoints
    need numerical continuation.  Unresolved integrations have no final loss
    in the returned scatter data; a bounded step/time budget is not evidence
    of convergence.
    """
    f, g, _view, spec, key = _resolve(payload)
    m = _model_for(key, f, g, spec)
    endpoints = payload.get("endpoints", [])
    if not isinstance(endpoints, list) or len(endpoints) > 1024:
        raise ValueError("endpoints must be a list of at most 1024 entries")
    limit = float(payload.get("audit_time_limit_sec", 60.0))
    if not math.isfinite(limit) or not 1 <= limit <= ALLOCATOR_MAX_SECONDS:
        raise ValueError("audit_time_limit_sec must lie in "
                         f"[1, {ALLOCATOR_MAX_SECONDS:g}]")
    deadline = time.perf_counter()+limit
    enumeration = _ENUM.get(key)
    if enumeration is None:
        enumeration = sturm.enumerate_critical_points(m)
    tree = merge_tree.build(m, enumeration)
    ultimate_losses = [float(merge_tree.exact_loss(m, q.a, q.b))
                       for q in enumeration.minima]
    infinity_loss = merge_tree.backbone_level_at_infinity(m)
    if infinity_loss is not None:
        ultimate_losses.append(float(infinity_loss))
    ultimate_losses = [value for value in ultimate_losses
                       if math.isfinite(value) and value > 0]
    ultimate_loss_range = ([min(ultimate_losses), max(ultimate_losses)]
                           if ultimate_losses else None)
    kernel = getattr(m, "_native_kernel", None)
    results, memo = [], {}

    def fate(a, b):
        found = merge_tree.fate_from_tree(m, enumeration, tree, a, b)
        if (found is not None and found["forced"]
                and found["fate"] == "capture"):
            q = enumeration.points[found["minima"][0]]
            return {"final_loss": float(merge_tree.exact_loss(m, q.a, q.b)),
                    "minimum": [float(q.a), float(q.b)], "fate": "minimum"}
        if (found is not None and found["forced"]
                and found["fate"] == "escape"):
            limit = merge_tree.backbone_level_at_infinity(m)
            if limit is not None:
                return {"final_loss": float(limit), "minimum": None,
                        "fate": "escape"}
        return None

    for item in endpoints:
        policy = str(item.get("policy", ""))
        arm = int(item.get("arm", -1))
        point = item.get("point")
        if policy not in {"initial", "equal", "thompson"} or arm < 0:
            raise ValueError("endpoint policy or arm is invalid")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("endpoint point must be [a, b]")
        a, b = map(float, point)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("endpoint coordinates must be finite")
        key_point = (a, b)
        if key_point not in memo:
            resolved = fate(a, b)
            if resolved is not None:
                memo[key_point] = dict(
                    resolved, method="exact_merge_tree",
                    irk_steps=0,
                    term="capture" if resolved["fate"] == "minimum" else "escape")
            elif kernel is None or not hasattr(kernel, "normalized_step"):
                memo[key_point] = {"final_loss": None, "minimum": None,
                                   "fate": "unresolved", "method": "unavailable", "irk_steps": 0,
                                   "term": "native_irk_unavailable"}
            else:
                step_size = 0.2
                steps = 0
                term = "step_budget"
                current = float(m.L(a, b))
                while steps < 3000:
                    if time.perf_counter() >= deadline:
                        term = "time_limit"
                        break
                    moved = False
                    for _ in range(60):
                        try:
                            aa, bb = kernel.normalized_step(a, b, -step_size, 8)
                            value = float(m.L(aa, bb))
                        except (ArithmeticError, ValueError, OverflowError,
                                ZeroDivisionError):
                            value = float("inf")
                        if (math.isfinite(value) and value < current
                                and math.isfinite(aa) and math.isfinite(bb)):
                            moved = True
                            break
                        step_size *= 0.5
                    if not moved:
                        term = "step_failure"
                        break
                    a, b, current = float(aa), float(bb), value
                    steps += 1
                    if abs(a) > 1e12 or abs(b) > 1e12:
                        term = "escape_bound"
                        break
                    if steps % 8 == 0:
                        resolved = fate(a, b)
                        if resolved is not None:
                            term = "capture"
                            break
                    step_size = min(1.0, step_size*1.25)
                if resolved is not None:
                    memo[key_point] = dict(
                        resolved, method="irk8_gl_merge_tree",
                        irk_steps=steps,
                        term="capture" if resolved["fate"] == "minimum"
                        else "escape")
                else:
                    memo[key_point] = {"final_loss": None, "minimum": None,
                                       "fate": "unresolved", "method": "irk8_gl",
                                       "irk_steps": steps,
                                       "term": term}
        results.append(dict(memo[key_point], policy=policy, arm=arm))
    return {"format": "spong-allocator-endpoint-audit-v1",
            "order": 8, "outcomes": results,
            "ultimate_loss_range": ultimate_loss_range}


def isolated_allocator(payload: dict) -> dict:
    """Run the interactive allocator beyond a killable process boundary.

    The cooperative deadline inside :func:`allocator` keeps ordinary runs
    cheap to stop.  This outer deadline also covers a native sampler call
    that never returns to Python.
    """
    raw_limit = payload.get("time_limit_sec", 60.0)
    try:
        requested_limit = float(raw_limit)
    except (TypeError, ValueError):
        requested_limit = 60.0
    if not math.isfinite(requested_limit):
        requested_limit = 60.0
    # Allow worker startup, library discovery and pool teardown outside the
    # cooperative allocation clock before enforcing the hard process limit.
    hard_limit = min(ALLOCATOR_MAX_SECONDS, max(1.0, requested_limit)) + 15.0

    command = [sys.executable, str(Path(__file__).resolve()),
               "--allocator-worker"]
    worker_env = dict(os.environ, SPONG_ALLOCATOR_PROGRESS="1")
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        start_new_session=(os.name == "posix"), env=worker_env)
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload), timeout=hard_limit)
    except subprocess.TimeoutExpired:
        # The worker may own a parallel uniform-prefix pool.  End the whole
        # session so a hard timeout cannot strand those child processes.
        if os.name == "posix" and getattr(process, "pid", None) is not None:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        _, stderr = process.communicate()
        phases = [line.partition("=")[2] for line in stderr.splitlines()
                  if line.startswith("SPONG_ALLOCATOR_PHASE=")]
        phase = phases[-1] if phases else "worker startup"
        raise TimeoutError(
            "allocation stopped by hard interactive time limit "
            f"(last phase: {phase})") from None

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        detail = stderr.strip() or stdout.strip() or "allocator worker failed"
        raise RuntimeError(detail) from exc
    if envelope.get("ok"):
        return envelope["result"]
    message = envelope.get("error", "allocator worker failed")
    if envelope.get("kind") == "allocation_timeout":
        raise TimeoutError(message)
    if envelope.get("kind") == "value_error":
        raise ValueError(message)
    raise RuntimeError(message)


def allocator_worker_main() -> int:
    """Private JSON transport used by :func:`isolated_allocator`."""
    try:
        payload = json.load(sys.stdin)
        envelope = {"ok": True, "result": allocator(payload)}
    except TimeoutError as exc:
        envelope = {"ok": False, "kind": "allocation_timeout",
                    "error": str(exc)}
    except ValueError as exc:
        envelope = {"ok": False, "kind": "value_error",
                    "error": str(exc)}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        envelope = {"ok": False, "kind": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}"}
    json.dump(envelope, sys.stdout)
    return 0


def _numerical_connection_portrait(payload):
    """Full numerical portrait, with blue idealized connections at the stored wall."""
    f, g, view, spec, key = _resolve(payload)
    cache_key = (key, "numerical_connection")
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key], cached=True, stage=payload.get("stage", "final"))
    t0 = time.perf_counter()
    case = zoo.get_numerical_connection(payload["zoo"])
    m = _model_for(key, f, g, spec)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    mu = case.moments(2 * max(len(f), len(g)) - 1)
    mean = sum((x * mu[i] for i, x in enumerate(f)), Fraction(0))
    variance = m.C - mean * mean
    from decimal import Decimal
    parameter = str(payload.get("candidate_lam", case.parameter))
    at_connection = Decimal(parameter) == Decimal(case.parameter)
    # Use the house portrait and stable-tail extensions, as for ordinary cases.
    # Its geometry-only ledger must not turn the wall proposal into a verdict.
    p = portrait.compute(m, view=view, _enumeration=e, _skip_audit=True)
    pairs, connections = [], []
    remove = set()
    for pair in case.trace_pairs(m, parameter=parameter):
        gap = math.hypot(*(pair["unstable"][-1] - pair["stable"][-1]))
        pairs.append({"side": pair["side"], "level": pair["level"], "gap": gap,
                      "delta_b": float(pair["unstable"][-1, 1] - pair["stable"][-1, 1])})
        if not at_connection:
            continue
        source = saddle_wall.critical_near(p, pair["source_b"])
        target = saddle_wall.critical_near(p, pair["target_b"])
        # House stable signs refer to displacement from the backbone, not b.
        a0, b0 = pair["stable"][0]
        stable_sign = 1 if a0 - float(m.a_star(b0)) > 0 else -1
        for kind, saddle_b, direction in (("unstable", source.b, pair["sign"]),
                                         ("stable", target.b, stable_sign)):
            matches = [i for i, br in enumerate(p.branches)
                       if br.kind == kind and abs(br.diag["saddle_b"] - saddle_b) < 1e-8
                       and br.diag["unstable_direction" if kind == "unstable"
                                   else "stable_sign"] == direction]
            if len(matches) != 1:
                raise ValueError("connection display could not identify its house branch")
            remove.add(matches[0])
        points = np.vstack(((source.a, source.b), pair["unstable"],
                            pair["stable"][::-1], (target.a, target.b)))
        connections.append({"kind": "stable_unstable",
            "label": "saddle connection (numerical wall limit)",
            "source_b": source.b, "target_b": target.b,
            "n_traced": len(points), "points": points.tolist(),
            "numerical_miss": gap})
    if remove:
        p = portrait.Portrait(p.model, p.enumeration,
            [br for i, br in enumerate(p.branches) if i not in remove],
            p.box, p.view, dict(p.ledger))
    branches = _branches(p)
    tails = _j_tails(m, p, float(atlas.effective_degree(m)), enumeration=e)
    out = {"f": list(map(float, f)), "g": list(map(float, g)),
        "stage": payload.get("stage", "final"), "cached": False,
        "elapsed_sec": time.perf_counter() - t0, "timing": {},
        "native": NATIVE, "varf": float(variance), "field": _field_coeffs(m),
        "critical": _critical_points(e, m), "branches": branches,
        "n_branches": len(branches),
        "n_branch_points": sum(b["n_traced"] for b in branches),
        "chord_max": max(b["chord_max"] for b in branches),
        "box": list(p.box), "view": list(p.view), "frame_view": list(view),
        "legal_box": list(p.box), "d_eff": atlas.effective_degree(m),
        "enumeration": {"n_critical": len(e.points), "n_min": len(e.minima),
            "n_saddle": len(e.saddles), "psi_positive": bool(e.psi_positive),
            "morse": bool(e.morse), "alternates": bool(e.alternates)},
        "status": "numerical_connection_candidate",
        "reason": ("Full portrait with two blue numerical wall-limit connections; topology not certified"
                   if at_connection else "Full numerical portrait away from the connection value; topology not audited"),
        "candidate": {"parameter": parameter, "wall_parameter": case.parameter, "at_connection": at_connection, "pairs": pairs,
                      "description": case.description, "grade": "NUMERICAL_ORACLE"},
        "tails": tails, "wall_connections": connections, "wall_limit": None,
        "wall_pair": None, "wall_shot": None, "wall_root": None,
        "ledger_summary": {}, "ledger_timing": {}, "attempts": []}
    _CACHE[cache_key] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    return out


def _tails_for(key, m, p, enumeration):
    """_j_tails, once per distinct set of stable branches: the inspector's two
    stages ordinarily trace the same curves, and the tails depend on no others."""
    h = hashlib.sha256(repr(tuple(p.box)).encode())
    for i, br in enumerate(p.branches):
        if br.kind == "stable":
            h.update(repr((i, br.term)).encode())
            h.update(np.asarray(br.Y, dtype=float).tobytes())
    entry = (key, h.hexdigest())
    tails = _TAILS.get(entry)
    if tails is None:
        tails = _j_tails(m, p, float(atlas.effective_degree(m)),
                         enumeration=enumeration)
        _TAILS[entry] = tails
        if len(_TAILS) > _CACHE_MAX:
            _TAILS.pop(next(iter(_TAILS)))
    return tails


def compute(payload: dict) -> dict:
    if payload.get("zoo") in zoo.numerical_connection_names():
        return _numerical_connection_portrait(payload)

    # A zoo case supplies its own f, g, moment distribution and default_view.
    # Using them is not a convenience: default_view is tuned per case, and
    # _trace_box widens whatever view it is handed, so an invented box costs
    # real tracing time.  This is the cli.zoo_phase_portrait path.
    f, g, view, spec, key = _resolve(payload)

    # Two stages.  'preview' is geometry_level 0 only -- the picture, with no
    # escalation ladder behind it; 'final' runs certified_compute to a verdict.
    # On linear-target-d17-thrash level 0 is about 95s of an 808s portrait, and
    # levels 1 and 2 refine the VERDICT rather than the curves, so blocking the
    # display on them makes the viewer unusable on exactly the cases it is most
    # wanted for.
    stage = payload.get("stage", "final")

    display_mode = "wall_limit" if payload.get("wall_limit") else "ordinary"
    # A response computed before the family's Brent root existed lacks the
    # root and the candidate overlay; once the root exists, recompute.
    if _WALL_ROOTS.get(payload.get("wall")) is not None:
        display_mode += "+root"
    hit = _CACHE.get((key, stage, display_mode))
    if hit is not None:
        return dict(hit, cached=True)

    t0 = time.perf_counter()
    n_moments = 2 * max(len(f), len(g)) - 1     # exactly mu_0..mu_2D, as cli
    mu = moment_vector(spec, n_moments)
    t1 = time.perf_counter()

    m = _model_for(key, f, g, spec)
    # The box contract is view subset of compute box subset of legal max, and
    # the repo policy is a compute box as big as is sensible.
    # atlas.legal_max_b is that bound: past the Cauchy bounds of N and B every
    # finite critical point is enclosed and the far field owns the dynamics.
    # But compute_box only CLAMPS to it -- it builds the skeleton's bounding
    # box plus a margin, unioned with whatever view it is handed -- so an
    # interactive request sized to the current view gets a box sized to the
    # view, and stable separatrices are cut off long before their diagonal
    # asymptotes take hold.
    #
    # Taking the legal box costs nothing in steps: ds = span/30000 is derived
    # from the box, so the count to cross it is fixed whatever its size.  A
    # larger box buys reach at proportionally coarser chords, and 30000 chords
    # across the legal box is still thousands across a typical view.  The view
    # the client asked for travels separately, as frame_view, for framing.
    bmax = atlas.legal_max_b(m)
    amax = bmax / max(1.0, math.sqrt(max(1, atlas.effective_degree(m))))
    trace_view = (-amax, amax, -bmax, bmax)
    enumeration = _ENUM.get(key)
    if enumeration is None:
        enumeration = sturm.materialize_stubs(
            m, sturm.enumerate_critical_points(m))
        _ENUM[key] = enumeration
        if len(_ENUM) > _ENUM_MAX:
            _ENUM.pop(next(iter(_ENUM)))
    t2 = time.perf_counter()

    if stage == "preview":
        # Geometry only, no audit.  The audit is the cost -- 714s of an 808s
        # level-0 portrait on linear-target-d17-thrash -- and the viewer draws
        # curves, not certificates.  The status comes back `not_audited` and
        # the page must say so rather than imply a verdict.
        display_view = atlas.compute_box(m, enumeration, view=trace_view)
        p = portrait.compute(
            m, view=trace_view, geometry_level=0, _enumeration=enumeration,
            _display_view=display_view, _genericity=atlas.genericity(m),
            _skip_audit=True)
        if not payload.get("wall_limit"):
            _TRACED[key] = p.branches
            if len(_TRACED) > _ENUM_MAX:
                _TRACED.pop(next(iter(_TRACED)))
    else:
        # The preview's geometry is level 0 of this ladder: same model, same
        # enumeration, same box.  Only the audit is left to do.
        p = portrait.certified_compute(
            m, view=trace_view, _enumeration=enumeration,
            _branches=(None if payload.get("wall_limit") else _TRACED.pop(key, None)))
    # The pair a wall family is about, identified on the ORDINARY portrait
    # before any wall-limit surgery removes one of them.
    wall_pair = _wall_pair(payload, p)
    wall_name = payload.get("wall")
    family = zoo.get_wall_family(wall_name) if wall_name else None
    root = (_wall_root(family)
             if family is not None and payload.get("wall_limit") else
             _WALL_ROOTS.get(wall_name) if wall_name else None)
    # The two-sided shot at THIS Lambda: source unstable down and target
    # stable up to the common midlevel, with the signed gap.  Smooth in
    # Lambda and zero exactly at the wall -- the honest "how close do they
    # come" number, and what the Brent root is a root of.
    wall_shot = None
    if family is not None:
        shot = wall_shoot.shoot(
            m, family.source_b, family.unstable_direction, family.target_b,
            root.target_direction if root is not None else None,
            ds=WALL_SHOOT_DS, enumeration=enumeration)
        wall_shot = shot.as_dict() if shot is not None else None
    p, wall_connections, wall_limit = _geometric_wall_limit(payload, p, root)
    if wall_limit:
        # After surgery the branch indices no longer apply; the connection
        # record carries the distances instead.
        wall_pair = None
    wall_root = None if root is None else {
        "lam": root.lam, "delta": root.delta,
        "evaluations": root.evaluations, "bracket": list(root.bracket),
        "stored_wall": float(family.wall_parameter),
        "stored_bracket": (list(family.wall_bracket)
                           if family.wall_bracket else None),
        "ds": WALL_SHOOT_DS,
        "level": root.shot.level,
        "candidate_points": root.shot.candidate.tolist(),
    }
    t3 = time.perf_counter()
    elapsed = t3 - t0
    timing = {"moments": t1 - t0, "build": t2 - t1, "portrait": t3 - t2}

    mean = sum((Fraction(f[i]) * mu[i] for i in range(len(f))), Fraction(0))
    csq = sum((Fraction(f[i]) * Fraction(f[j]) * mu[i + j]
               for i in range(len(f)) for j in range(len(f))), Fraction(0))
    varf = float(csq - mean * mean)

    e = p.enumeration
    out = {
        "f": [float(x) for x in f], "g": [float(x) for x in g],
        "separated_linear": ({
            "degree": int(payload["separated_linear"].get("degree", 4)),
            "separation": str(payload["separated_linear"].get(
                "separation", "30")),
            "algebraic_ceiling": 4 * int(
                payload["separated_linear"].get("degree", 4)) - 2,
        } if payload.get("separated_linear") else None),
        "stage": stage,
        "varf": varf if varf > 0 else 1.0,
        "elapsed_sec": elapsed,
        "timing": timing,
        "n_branch_points": sum(len(br.Y) for br in p.branches),
        "n_branches": len(p.branches),
        "chord_max": max((_max_chord(br.Y) for br in p.branches), default=0.0),
        "native": NATIVE,
        "cached": False,
        "field": _field_coeffs(m),
        "critical": _critical_points(e, m),
        "branches": _branches(p),
        # Closed-form stable tails (J level set or b-nullcline), each with
        # the growth that licensed it, for drawing only.
        "tails": _tails_for(key, m, p, enumeration),
        "wall_connections": wall_connections,
        "wall_limit": wall_limit,
        "wall_pair": wall_pair,
        "wall_shot": wall_shot,
        "wall_root": wall_root,
        "box": [float(x) for x in p.box],
        "view": None if p.view is None else [float(x) for x in p.view],
        "enumeration": {
            "n_critical": len(e.points),
            "n_min": len(e.minima),
            "n_saddle": len(e.saddles),
            "psi_positive": bool(e.psi_positive),
            "morse": bool(e.morse),
            "alternates": bool(e.alternates),
        },
        "ledger_summary": (p.ledger or {}).get("summary", {}),
        "status": ("geometric_wall_limit" if wall_limit else
                   (p.ledger or {}).get("topology", {}).get("status")),
        "ordinary_status": (p.ledger or {}).get(
            "topology", {}).get("status") if wall_limit else None,
        "d_eff": atlas.effective_degree(m),
        # What the client asked to LOOK at, distinct from what was traced.
        "frame_view": (list(view) if view is not None else None),
        "legal_box": list(trace_view),
        "reason": ("saddle_connection_wall" if wall_limit else
                   (p.ledger or {}).get(
                       "topology", {}).get("resolution_reason")),
        # The library measures itself: enumeration / stub / geometry split,
        # and one entry per geometry escalation with its status and reason.
        "ledger_timing": (p.ledger or {}).get("timing", {}),
        "attempts": (p.ledger or {}).get("topology", {}).get("attempts", []),
        "geometry_level": (p.ledger or {}).get(
            "topology", {}).get("geometry_level"),
    }
    _CACHE[(key, stage, display_mode)] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    return out


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # the page reloaded or abandoned the request while we were
            # still computing; nothing to report
            pass

    def do_GET(self):
        if self.path == "/walls":
            out = []
            for nm in zoo.wall_family_names():
                w = zoo.get_wall_family(nm)
                out.append({
                    "name": nm, "base_case": w.base_case,
                    "parameter_name": w.parameter_name,
                    "below": w.below_parameter, "wall": w.wall_parameter,
                    "above": w.above_parameter,
                    "bracket": (list(w.wall_bracket) if w.wall_bracket
                                else None),
                    "bracket_protocol": w.bracket_protocol,
                    "description": w.description,
                })
            self._send(200, json.dumps(out), "application/json")
            return
        if self.path == "/zoo":
            cases = []
            for nm in zoo.names():
                z = zoo.get(nm)
                cases.append({"name": nm,
                              "description": getattr(z, "description", ""),
                              "moment_dist": getattr(z, "moment_dist", ""),
                              "deg_f": len(z.f) - 1, "deg_g": len(z.g) - 1})
            for nm in zoo.numerical_connection_names():
                z = zoo.get_numerical_connection(nm)
                cases.append({"name": nm, "description": z.description,
                              "moment_dist": z.moment_dist,
                              "deg_f": len(z.f) - 1, "deg_g": len(z.g) - 1,
                              "candidate_parameter": z.parameter})
            self._send(200, json.dumps(cases), "application/json")
            return
        if self.path in ("/", "/index.html"):
            if not PAGE.exists():
                self._send(404, f"missing {PAGE.name}", "text/plain")
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/portrait", "/trace", "/allocator",
                             "/allocator/endpoints"):
            self._send(404, "not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            result = (trace(payload) if self.path == "/trace"
                      else allocator_endpoint_audit(payload)
                      if self.path == "/allocator/endpoints"
                      else isolated_allocator(payload)
                      if self.path == "/allocator"
                      else compute(payload))
            self._send(200, json.dumps(result), "application/json")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError as exc:
            self._send(408, json.dumps({
                "error": str(exc), "kind": "allocation_timeout",
            }), "application/json")
        except Exception as exc:
            traceback.print_exc()
            self._send(400, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}), "application/json")


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    core = (f"C core abi={NATIVE['abi']} "
            f"[{', '.join(NATIVE['types'])}]" if NATIVE["loaded"]
            else f"NO C CORE -- {NATIVE['error']} "
                 f"(pure-Python fallback: expect this to be slow)")
    print(f"spong explorer on http://127.0.0.1:{port}\n  {core}\n"
          f"  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(allocator_worker_main()
                     if "--allocator-worker" in sys.argv else main())
