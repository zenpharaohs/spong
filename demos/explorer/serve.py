#!/usr/bin/env python3
"""Backend for the interactive one-neuron loss explorer.

    python demos/explorer/serve.py        # http://127.0.0.1:8710

Every change of f, g, or the moments in the browser posts to /portrait and
gets back a freshly computed CERTIFIED portrait.  The browser does no
analysis: it evaluates L(a,b) = C - 2a*B(b) + a^2*A(b) for the heatmap and
draws what this endpoint sends.  Critical points, their classification, the
separatrices and the ledger all come from portrait.certified_compute.

Nothing here re-implements spong.  Model.alpha, Model.beta and Model.C are
the exact A, B and C; Model.backbone_num / backbone_den are the reduced
B^2/A the constructor already forms; each CriticalPoint carries b, a, kind,
source and the exact sign of u''.  This file only serializes them.

Lives in demos/ because the descent traces it drives are descent methods,
which the founding document keeps out of the library.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Default to the accelerated path.  The explorer is a viewer, not the
# certifying CLI, and latency is what it trades in; both settings are
# overridable from the environment.  Must precede the spong import, since
# engine reads SPONG_ENGINE at import time.
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "8")

try:
    from spong import atlas, model, portrait, sturm, zoo
except ImportError:                                  # running from a checkout
    sys.path.insert(0, str(REPO / "src"))
    from spong import atlas, model, portrait, sturm, zoo

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
    if kind == "uniform01":
        return model.moments_uniform01(n)
    if kind == "normal01":
        return model.moments_normal01(n)
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
    tolerance of the least loss get the open-circle glyph.
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
        lam = float(payload.get("lam", w.wall_parameter))
        root = math.sqrt(lam)
        f = [float(x) / root for x in base.f]
        g = [float(x) * root for x in base.g]
        view = tuple(float(x) for x in w.default_view)
        spec = {"kind": base.moment_dist}
    elif name:
        z = zoo.get(name)
        f = [float(x) for x in z.f]
        g = [float(x) for x in z.g]
        view = tuple(float(x) for x in z.default_view) \
            if z.default_view else None
        spec = {"kind": z.moment_dist}
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


def compute(payload: dict) -> dict:
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

    hit = _CACHE.get((key, stage))
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
    else:
        p = portrait.certified_compute(m, view=trace_view,
                                       _enumeration=enumeration)
    t3 = time.perf_counter()
    elapsed = t3 - t0
    timing = {"moments": t1 - t0, "build": t2 - t1, "portrait": t3 - t2}

    mean = sum((Fraction(f[i]) * mu[i] for i in range(len(f))), Fraction(0))
    csq = sum((Fraction(f[i]) * Fraction(f[j]) * mu[i + j]
               for i in range(len(f)) for j in range(len(f))), Fraction(0))
    varf = float(csq - mean * mean)

    e = p.enumeration
    out = {
        "f": f, "g": g,
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
        "status": (p.ledger or {}).get("topology", {}).get("status"),
        "d_eff": atlas.effective_degree(m),
        # What the client asked to LOOK at, distinct from what was traced.
        "frame_view": (list(view) if view is not None else None),
        "legal_box": list(trace_view),
        "reason": (p.ledger or {}).get(
            "topology", {}).get("resolution_reason"),
        # The library measures itself: enumeration / stub / geometry split,
        # and one entry per geometry escalation with its status and reason.
        "ledger_timing": (p.ledger or {}).get("timing", {}),
        "attempts": (p.ledger or {}).get("topology", {}).get("attempts", []),
        "geometry_level": (p.ledger or {}).get(
            "topology", {}).get("geometry_level"),
    }
    _CACHE[(key, stage)] = out
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
        if self.path not in ("/portrait", "/trace"):
            self._send(404, "not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            result = (trace(payload) if self.path == "/trace"
                      else compute(payload))
            self._send(200, json.dumps(result), "application/json")
        except (BrokenPipeError, ConnectionResetError):
            pass
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
    raise SystemExit(main())
