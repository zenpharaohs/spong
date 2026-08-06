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
        out.append({"kind": br.kind, "term": br.term,
                    "saddle_b": br.diag.get("saddle_b"),
                    "n_traced": n, "stride": step,
                    "chord_max": _max_chord(Y),
                    "points": pts})
    return out


def compute(payload: dict) -> dict:
    # A zoo case supplies its own f, g, moment distribution and default_view.
    # Using them is not a convenience: default_view is tuned per case, and
    # _trace_box widens whatever view it is handed, so an invented box costs
    # real tracing time.  This is the cli.zoo_phase_portrait path.
    name = payload.get("zoo")
    if name:
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

    # Two stages.  'preview' is geometry_level 0 only -- the picture, with no
    # escalation ladder behind it; 'final' runs certified_compute to a verdict.
    # On linear-target-d17-thrash level 0 is about 95s of an 808s portrait, and
    # levels 1 and 2 refine the VERDICT rather than the curves, so blocking the
    # display on them makes the viewer unusable on exactly the cases it is most
    # wanted for.
    stage = payload.get("stage", "final")

    key = (tuple(f), tuple(g), view, spec.get("kind", "uniform01"),
           tuple(float(x) for x in spec.get("samples", ())))
    hit = _CACHE.get((key, stage))
    if hit is not None:
        return dict(hit, cached=True)

    t0 = time.perf_counter()
    n_moments = 2 * max(len(f), len(g)) - 1     # exactly mu_0..mu_2D, as cli
    mu = moment_vector(spec, n_moments)
    t1 = time.perf_counter()

    m = model.build(f, g, mu)
    enumeration = _ENUM.get(key)
    if enumeration is None:
        enumeration = sturm.materialize_stubs(
            m, sturm.enumerate_critical_points(m))
        _ENUM[key] = enumeration
        if len(_ENUM) > _ENUM_MAX:
            _ENUM.pop(next(iter(_ENUM)))
    t2 = time.perf_counter()

    if stage == "preview":
        display_view = atlas.compute_box(m, enumeration, view=view)
        p = portrait.compute(
            m, view=view, geometry_level=0, _enumeration=enumeration,
            _display_view=display_view, _genericity=atlas.genericity(m))
    else:
        p = portrait.certified_compute(m, view=view,
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
        if self.path != "/portrait":
            self._send(404, "not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            result = compute(json.loads(self.rfile.read(n) or b"{}"))
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
