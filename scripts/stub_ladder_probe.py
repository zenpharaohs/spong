#!/usr/bin/env python3
"""Per-stub accounting of the invariant-graph reach ladder.

    python scripts/stub_ladder_probe.py linear-target-d17-thrash
    python scripts/stub_ladder_probe.py tricky-d11 --workers 1

QUESTION.  build_stubs starts each stub at desired_reach (a guess from the
nearest REAL critical point), halves until the graph certificate passes,
then doubles until the continuation is ready -- solving the coarse and
fine Hadamard fixed points from scratch at every rung.  The complex
portrait supplies a certified launch radius for free: R_complex = distance
from the saddle's b to the nearest OTHER root of B*N (real or complex) or
zero of A.  If the reach the ladder finally accepts is a stable fraction
of R_complex, the ladder can be seeded from it and most rungs disappear
without changing what is certified.

OUTPUT.  One row per stub: saddle b, manifold, orientation, verdict,
desired reach, accepted reach, halvings, extension steps, coarse/fine
fixed-point iterations, graph solves and seconds attributed to the stub,
and accepted_reach / R_complex.  Then totals.  R_complex is probe grade
(numpy roots on max-normalised coefficients), as in pole_portrait.py.
"""

from __future__ import annotations

import os
import sys
import time

if "--workers" in sys.argv:
    i = sys.argv.index("--workers")
    os.environ["SPONG_WORKERS"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
os.environ.setdefault("SPONG_ENGINE", "native")

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from potential_corpus import model as model_mod, sturm, zoo   # noqa: E402
from spong import local                                        # noqa: E402


def _roots(coeffs):
    c = np.asarray([float(x) for x in coeffs], dtype=float)
    nz = np.nonzero(np.abs(c) > 0)[0]
    if len(nz) < 2:
        return np.array([], dtype=complex)
    c = c[:nz[-1] + 1]
    c = c / np.max(np.abs(c))
    return np.roots(c[::-1])


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "linear-target-d17-thrash"
    zz = zoo.get(name)
    f, g = list(zz.f), list(zz.g)
    mu = (model_mod.moments_normal01 if zz.moment_dist == "normal01"
          else model_mod.moments_uniform01)(2 * max(len(f), len(g)) - 1)
    m = model_mod.build(f, g, mu)
    e = sturm.enumerate_critical_points(m)

    # complex launch radii from the model's own polynomials (ascending
    # binary64 coefficient arrays _ca, _cb, _cn: A, B, N = A'B - 2B'A)
    others = np.concatenate([_roots(m._ca), _roots(m._cb), _roots(m._cn)])

    def r_complex(b):
        if others is None or len(others) == 0:
            return float("nan")
        d = np.abs(others - b)
        d = d[d > 1e-9 * max(1.0, abs(b))]       # exclude the saddle's own root
        return float(np.min(d)) if len(d) else float("nan")

    # count and time the graph solves, attributed per stub
    calls = {"n": 0, "sec": 0.0}
    per: dict = {}
    orig_graph = local.PoincareData.graph
    orig_ug = local.LocalJet.unstable_graph

    def note(key, dt):
        calls["n"] += 1
        calls["sec"] += dt
        n, sec = per.get(key, (0, 0.0))
        per[key] = (n + 1, sec + dt)

    def timed_graph(chart, jet, sign, *a, **k):
        t = time.perf_counter()
        try:
            return orig_graph(chart, jet, sign, *a, **k)
        finally:
            note((float(jet.b), chart.manifold, int(sign)),
                 time.perf_counter() - t)

    def timed_ug(jet, reach, *a, **k):
        t = time.perf_counter()
        try:
            return orig_ug(jet, reach, *a, **k)
        finally:
            note((float(jet.b), k.get("manifold", "unstable"),
                  int(k.get("sign", 1))), time.perf_counter() - t)
    local.PoincareData.graph = timed_graph
    local.LocalJet.unstable_graph = timed_ug
    t0 = time.perf_counter()
    try:
        e = sturm.materialize_stubs(m, e)
    finally:
        local.PoincareData.graph = orig_graph
        local.LocalJet.unstable_graph = orig_ug
    wall = time.perf_counter() - t0

    print(f"{name}: materialize_stubs {wall:.1f}s, "
          f"{calls['n']} graph solves, {calls['sec']:.1f}s in graphs "
          f"({1000*calls['sec']/max(calls['n'],1):.0f} ms each)\n")
    hdr = (f"{'saddle b':>12} {'man':>8} {'or':>3} {'verdict':>22} "
           f"{'desired':>9} {'accepted':>9} {'halv':>4} {'ext':>3} "
           f"{'it_c':>4} {'it_f':>4} {'solves':>6} {'sec':>6} "
           f"{'R_cx':>9} {'acc/R':>7}")
    print(hdr)
    ratios = []
    for q in e.points:
        if not q.stubs:
            continue
        R = r_complex(float(q.b))
        for s in q.stubs:
            c = dict(s.certificates)
            acc = c.get("reach", float("nan"))
            ratio = acc / R if R == R and R > 0 else float("nan")
            if ratio == ratio:
                ratios.append(ratio)
            desired = float("nan")
            for ch in (q.local.poincare if q.local else ()):
                if ch.manifold == s.manifold:
                    desired = ch.desired_reach
            ns, secs = per.get((float(q.b), s.manifold, int(s.orientation)),
                               (0, 0.0))
            print(f"{float(q.b):>12.6g} {s.manifold:>8} {s.orientation:>3} "
                  f"{s.preferred_chart:>22} {desired:>9.3g} {acc:>9.3g} "
                  f"{int(c.get('reach_halvings', 0)):>4} "
                  f"{int(c.get('extension_steps', 0)):>3} "
                  f"{int(c.get('graph_iterations_coarse', 0)):>4} "
                  f"{int(c.get('graph_iterations_fine', 0)):>4} "
                  f"{ns:>6} {secs:>6.2f} "
                  f"{R:>9.3g} {ratio:>7.3g}")
    if ratios:
        r = np.asarray(ratios)
        print(f"\naccepted/R_complex: min {r.min():.3g} median "
              f"{np.median(r):.3g} max {r.max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
