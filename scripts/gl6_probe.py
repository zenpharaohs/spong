#!/usr/bin/env python3
"""Why does GL6 refuse a step that GL4 takes and the descent test accepts?

    python scripts/gl6_probe.py                       # crawl vertex of br9
    python scripts/gl6_probe.py --vertex 199999 --chords 1.59e-5,1.27e-4,1.02e-3

Stands at a vertex of the recorded crawl segment (replayed through the C),
forms the fast-chart step the loop would take, and runs the REFERENCE
gl6_scalar / gl4_scalar with the stage internals exposed: h*J at the three
stage points, the equilibrated determinant (the Hadamard ratio the guard
tests), the Newton residual per iteration, and the exception the reference
raises.  Also evaluates Q3(z) = 1 - z/2 + z^2/10 - z^3/120 at z = h*J: the
(3,3) Pade denominator whose real root at z = 4.6444 is where the GL6
stage matrix is exactly singular on an EXPANDING step, and Q2(z) = 1 - z/2
+ z^2/12 for GL4, which has no real root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import charts, gauss                        # noqa: E402
from segment_corpus import context                     # noqa: E402
from crawl_probe import entry_for, replay_native       # noqa: E402


def q3(z):
    return 1.0 - z/2.0 + z*z/10.0 - z**3/120.0


def q2(z):
    return 1.0 - z/2.0 + z*z/12.0


def gl6_exposed(f, j, x, y, h, tol=1e-13, maxit=30):
    """gauss.gl6_scalar with its internals printed.  Same arithmetic."""
    c1, c2, c3 = gauss._GL3_C
    (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = gauss._GL3_A
    x1, x2, x3 = x + c1*h, x + c2*h, x + c3*h
    k = f(x, y)
    K1 = K2 = K3 = k
    print(f"      f(x,y)={k:+.6e}  h*J(x,y)={h*j(x, y):+.4e}"
          f"  Q3={q3(h*j(x, y)):+.4e}  Q2={q2(h*j(x, y)):+.4e}")
    for it in range(maxit):
        Y1 = y + h*(a11*K1 + a12*K2 + a13*K3)
        Y2 = y + h*(a21*K1 + a22*K2 + a23*K3)
        Y3 = y + h*(a31*K1 + a32*K2 + a33*K3)
        r1, r2, r3 = K1 - f(x1, Y1), K2 - f(x2, Y2), K3 - f(x3, Y3)
        m_ = max(abs(K1), abs(K2), abs(K3))
        rm = max(abs(r1), abs(r2), abs(r3))
        if rm < tol*(1.0 + m_):
            print(f"      it={it:2d} residual={rm:.2e} CONVERGED")
            return y + h*(gauss._GL3_B[0]*K1 + gauss._GL3_B[1]*K2
                          + gauss._GL3_B[2]*K3), "converged"
        J1, J2, J3 = j(x1, Y1), j(x2, Y2), j(x3, Y3)
        z = (h*J1, h*J2, h*J3)
        M = np.array([[1.0 - h*a11*J1, -h*a12*J1, -h*a13*J1],
                      [-h*a21*J2, 1.0 - h*a22*J2, -h*a23*J2],
                      [-h*a31*J3, -h*a32*J3, 1.0 - h*a33*J3]])
        n = np.max(np.abs(M), axis=1)
        det_eq = float(np.linalg.det(M/n[:, None]))
        cond = float(np.linalg.cond(M/n[:, None]))
        print(f"      it={it:2d} residual={rm:.2e} |K|={m_:.3e}"
              f" h*J=({z[0]:+.3e},{z[1]:+.3e},{z[2]:+.3e})"
              f" Q3(hJ)={q3(z[1]):+.3e} det_eq={det_eq:+.3e} cond={cond:.2e}"
              f"{'  <-- guard 1e-6' if abs(det_eq) < gauss._STAGE_GUARD else ''}")
        if abs(det_eq) < gauss._STAGE_GUARD:
            return float("nan"), f"guard: |det|={abs(det_eq):.2e}"
        try:
            d = np.linalg.solve(M/n[:, None], np.array([r1, r2, r3])/n)
        except np.linalg.LinAlgError:
            return float("nan"), "singular"
        K1 -= d[0]; K2 -= d[1]; K3 -= d[2]
        if max(abs(d)) < tol*(1.0 + m_):
            print(f"      it={it:2d} step={max(abs(d)):.2e} CONVERGED")
            return y + h*(gauss._GL3_B[0]*K1 + gauss._GL3_B[1]*K2
                          + gauss._GL3_B[2]*K3), "converged"
    return float("nan"), "no convergence"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="directed:1414065525")
    ap.add_argument("--flow", type=int, default=1)
    ap.add_argument("--vertex", type=int, default=199999)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--chords", default="1.59e-5,1.27e-4,1.02e-3")
    ap.add_argument("--chart", default=None, choices=(None, "slow", "fast"))
    ap.add_argument("--state", default=None,
                    help="b,w to stand at instead of a replay vertex")
    args = ap.parse_args(argv)

    m, e, _z = context(args.case)
    try:
        entry = entry_for(args.case, args.flow, None)
        i = entry["input"]
        flow = i["flow"]
    except SystemExit:
        if not args.state:
            raise
        i, flow = None, args.flow          # --state needs no corpus entry
    if args.state:
        b, w = (float(s) for s in args.state.split(","))
        a = m.s_a_star(b) + w
        k = -1
    else:
        pts, term, taken, rejected, _diag = replay_native(m, i, args.steps)
        k = min(args.vertex, len(pts) - 1)
        a, b = float(pts[k, 0]), float(pts[k, 1])
        w = a - m.s_a_star(b)
    vb, vw = charts._s_velocities(m, b, w)
    vb, vw = flow*vb, flow*vw
    chart = args.chart or ("slow" if abs(vw) <= charts.R_SWITCH*abs(vb) else "fast")
    print(f"vertex {k}: b={b:.10g} a={a:.6e} w={w:.4e} chart={chart}"
          f" vb={vb:+.3e} vw={vw:+.3e}")

    sf, sj = charts.slow_rhs_s(m)
    ff, fj = charts.fast_rhs_s(m)
    s_floor, f_floor = charts.slow_floor_s(m), charts.fast_floor_s(m)
    native = m._native_kernel

    for cur in [float(c) for c in args.chords.split(",")]:
        va = flow*(-2.0*m.sA(b)*w)
        vp = (vb*vb + va*va)**0.5
        if chart == "slow":
            h = cur*vb/vp
            f, j, x, y, fl = sf, sj, b, w, s_floor
            nat6 = lambda: native.slow_step(b, w, h)
            nat4 = lambda: native.slow_step_gl4(b, w, h)
            ref4 = lambda: gauss.gl4_scalar(sf, sj, b, w, h, floor=s_floor)
        else:
            h = cur*vw/vp
            f, j, x, y, fl = ff, fj, w, b, f_floor
            nat6 = lambda: native.fast_step(w, b, h)
            nat4 = lambda: native.fast_step_gl4(w, b, h)
            ref4 = lambda: gauss.gl4_scalar(ff, fj, w, b, h, floor=f_floor)
        print(f"\n  chord={cur:.3e}  h={h:+.4e}  ({chart} chart: "
              f"{'w(b)' if chart == 'slow' else 'b(w)'})"
              f"  evaluation floor of f at (x,y) = {fl(x, y):.3e}"
              f"  (x {gauss._NOISE_C:g} = {gauss._NOISE_C*fl(x, y):.3e})")
        print(f"    native gl6 -> {nat6():+.12e}")
        print(f"    native gl4 -> {nat4():+.12e}")
        try:
            print(f"    reference gl4 -> {ref4():+.12e}")
        except FloatingPointError as ex:
            print(f"    reference gl4 RAISED: {ex}")
        try:
            val = gauss.gl6_scalar(f, j, x, y, h, floor=fl)
            print(f"    reference gl6 -> {val:+.12e}")
        except FloatingPointError as ex:
            print(f"    reference gl6 RAISED: {ex}")
        print("    gl6 exposed:")
        val, why = gl6_exposed(f, j, x, y, h)
        print(f"    -> {val!r} ({why})")
        # the plane step at the same chord, with the descent test's numbers
        a_prev = m.s_a_star(b) + w
        for order in (8, 6, 4):
            try:
                a_try, b_try = native.normalized_step(a_prev, b, -flow*cur, order)
            except Exception as ex:                       # noqa: BLE001
                print(f"    plane gl{order}: raised {type(ex).__name__}: {ex}")
                continue
            if not (np.isfinite(a_try) and np.isfinite(b_try)):
                print(f"    plane gl{order}: NaN")
                continue
            da, db = a_try - a_prev, b_try - b
            expected = flow*float(m.gradL(a_prev, b) @ np.array([da, db]))
            actual = flow*float(m.L(a_try, b_try) - m.L(a_prev, b))
            slack = 64.0*np.finfo(float).eps*(1.0 + abs(float(m.L(a_prev, b))))
            ok = not (expected >= 0.0 or actual > 1e-4*expected + slack)
            print(f"    plane gl{order}: {'PASS' if ok else 'FAIL'} da={da:+.3e} db={db:+.3e}"
                  f" |chord|={float(np.hypot(da, db)):.3e} expected={expected:+.3e}"
                  f" actual={actual:+.3e} slack={slack:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
