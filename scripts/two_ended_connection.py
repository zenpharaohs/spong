#!/usr/bin/env python3
"""Locate a saddle connection by meeting two invariant-manifold shots.

The selected unstable half-branch is shot down from the higher saddle and a
selected stable half-branch is shot upward (backward in descent time) from the
lower saddle.  On every regular meeting level ``c`` the two endpoints lie on
the same exactly-known level curve, so their oriented b-coordinate difference

    D_c(Lambda) = b_down(Lambda; c) - b_up(Lambda; c)

is a signed scalar shooting residual.  D_c = 0 glues the two halves into one
gradient trajectory by ODE uniqueness.  Several meeting levels give an
overdetermined check: their inferred roots should agree.

Both shots use the same *pure* GL order on the regular constant-loss ODE

    dz/dL = grad(L) / |grad(L)|^2.

There is no order fallback and no destination-dependent step schedule.  A
full step and two half steps control both position and the collocation-stage
proper-time quadrature.  The accepted endpoint and time are the two-half
composition.

The clocks start on fixed regular sections below/above the two saddles.  At a
genuine connection the split sum

    S_c = T_down(c) + T_up(c)

is independent of c.  This is the clock witness; S_c need not have an
extremum as a function of Lambda at any one arbitrarily chosen split.

This is a development diagnostic, not yet a portrait or zoo certificate.

Examples:

    python scripts/two_ended_connection.py
    python scripts/two_ended_connection.py --orders=-4,-6 \
        --launch-levels 4,5 --step-levels 0,1
    python scripts/two_ended_connection.py --case near-slide-d2 \
        --branch 6 --target-b 0.984296 --stable-sign 1 \
        --lo 1.0798784 --hi 1.07987855
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait, sturm, zoo                         # noqa: E402
from level_crossing import rheostat                            # noqa: E402


@dataclass
class StepStats:
    accepted: int = 0
    rejected: int = 0
    clock_steps: int = 0
    max_space_ratio: float = 0.0
    max_clock_ratio: float = 0.0
    max_loss_defect: float = 0.0
    max_section_error: float = 0.0


@dataclass
class Shot:
    points: dict[float, np.ndarray]
    times: dict[float, float]
    stats: StepStats
    # Every accepted step endpoint and its loss, in order.  The separation
    # certificate bounds INT ||J|| dL along the shot, and the step points are
    # the decomposition it needs: sampling extra meeting levels instead costs
    # a whole extra pair of shots per level and is far coarser -- 127 sampled
    # levels still gave a bound of 17.6 against a true 3.6.
    path: list[np.ndarray] | None = None
    path_levels: list[float] | None = None


def _loss(m, z: np.ndarray) -> float:
    # Use the same Horner evaluator as the native stages.
    return float(m._native_kernel.loss(float(z[0]), float(z[1]), float(m.C)))


def _full_and_two_half(m, z: np.ndarray, h: float, order: int,
                       timed: bool):
    kernel = m._native_kernel
    if timed:
        f = kernel.potential_step_clock
        full = np.asarray(f(float(z[0]), float(z[1]), h, order), float)
        mid = np.asarray(f(float(z[0]), float(z[1]), 0.5*h, order), float)
        end = np.asarray(f(float(mid[0]), float(mid[1]), 0.5*h, order), float)
        tau_full = float(full[2])
        tau_half = float(mid[2] + end[2])
        full, mid, end = full[:2], mid[:2], end[:2]
    else:
        f = kernel.potential_step
        full = np.asarray(f(float(z[0]), float(z[1]), h, order), float)
        mid = np.asarray(f(float(z[0]), float(z[1]), 0.5*h, order), float)
        end = np.asarray(f(float(mid[0]), float(mid[1]), 0.5*h, order), float)
        tau_full = tau_half = 0.0
    if not (np.all(np.isfinite(full)) and np.all(np.isfinite(mid))
            and np.all(np.isfinite(end))
            and np.isfinite(tau_full) and np.isfinite(tau_half)):
        return None
    return full, mid, end, tau_full, tau_half


def _attempt(m, z: np.ndarray, h: float, order: int, timed: bool,
             space_rtol: float, space_atol: float,
             clock_rtol: float, clock_atol: float):
    trial = _full_and_two_half(m, z, h, order, timed)
    if trial is None:
        return None
    full, _mid, end, tau_full, tau_half = trial
    divisor = float(2**order - 1)
    chord = float(np.hypot(*(end-z)))
    space_error = float(np.hypot(*(end-full)))/divisor
    space_tol = space_atol + space_rtol*max(chord, 1e-300)
    space_ratio = space_error/max(space_tol, np.finfo(float).tiny)
    clock_error = abs(tau_half-tau_full)/divisor
    clock_tol = clock_atol + clock_rtol*abs(tau_half)
    clock_ratio = (clock_error/max(clock_tol, np.finfo(float).tiny)
                   if timed else 0.0)
    level0 = _loss(m, z)
    level1 = _loss(m, end)
    loss_defect = abs((level1-level0)-h)
    loss_tol = max(
        512.0*np.finfo(float).eps*(1.0+abs(level0)),
        2e-9*abs(h))
    monotone = h*(level1-level0) > 0.0
    ok = (space_ratio <= 1.0 and (not timed or clock_ratio <= 1.0)
          and loss_defect <= loss_tol and monotone)
    return {
        "ok": ok, "end": end, "tau": tau_half,
        "space_ratio": space_ratio, "clock_ratio": clock_ratio,
        "loss_defect": loss_defect,
    }


def _shoot(m, start, boundaries, clock_start: float, order: int,
           base_step: float, space_rtol: float, space_atol: float,
           clock_rtol: float, clock_atol: float) -> Shot:
    """Shoot monotonically through loss ``boundaries`` in their given order."""
    z = np.asarray(start, float)
    level = _loss(m, z)
    if not boundaries:
        return Shot({}, {}, StepStats())
    direction = math.copysign(1.0, boundaries[-1]-level)
    if any(direction*(b-level) <= 0.0 for b in boundaries):
        raise ValueError("shot boundaries must lie strictly ahead of the stub")
    if any(direction*(b1-b0) <= 0.0
           for b0, b1 in zip(boundaries, boundaries[1:])):
        raise ValueError("shot boundaries must be strictly monotone")

    stats = StepStats()
    points: dict[float, np.ndarray] = {}
    times: dict[float, float] = {}
    path: list[np.ndarray] = [z.copy()]
    path_levels: list[float] = [_loss(m, z)]
    tau = 0.0
    timing = False
    current_step = base_step
    section_tol_scale = 2048.0*np.finfo(float).eps

    for boundary in boundaries:
        while True:
            level = _loss(m, z)
            section_tol = section_tol_scale*(1.0+abs(boundary))
            gap = direction*(boundary-level)
            if gap <= section_tol:
                stats.max_section_error = max(
                    stats.max_section_error, abs(level-boundary))
                break
            h = direction*min(current_step, gap)
            landing = gap <= current_step
            accepted = None
            for _retry in range(24):
                # On a section step, compensate the tiny collocation loss
                # defect by solving for the loss increment itself.  Since
                # dL/dh = 1 for the exact ODE, h <- h-residual is Newton with
                # unit derivative and normally converges in one correction.
                h_trial = h
                for _correction in range(4 if landing else 1):
                    trial = _attempt(
                        m, z, h_trial, order, timing,
                        space_rtol, space_atol, clock_rtol, clock_atol)
                    if trial is None or not trial["ok"]:
                        break
                    if not landing:
                        accepted = trial
                        h = h_trial
                        break
                    residual = _loss(m, trial["end"])-boundary
                    if abs(residual) <= section_tol:
                        accepted = trial
                        h = h_trial
                        break
                    h_trial -= residual
                if accepted is not None:
                    break
                h *= 0.5
                landing = False
                stats.rejected += 1
            if accepted is None:
                raise RuntimeError(
                    f"pure GL{order} step failed at L={level:.17g}")
            z = accepted["end"]
            stats.accepted += 1
            path.append(z.copy())
            path_levels.append(_loss(m, z))
            stats.max_space_ratio = max(
                stats.max_space_ratio, accepted["space_ratio"])
            stats.max_clock_ratio = max(
                stats.max_clock_ratio, accepted["clock_ratio"])
            stats.max_loss_defect = max(
                stats.max_loss_defect, accepted["loss_defect"])
            if timing:
                tau += accepted["tau"]
                stats.clock_steps += 1
            current_step = min(base_step, 1.5*abs(h))

        points[boundary] = z.copy()
        if boundary == clock_start:
            timing = True
            tau = 0.0
        elif timing:
            times[boundary] = tau
    return Shot(points, times, stats, path, path_levels)


def _stub_error(stub) -> float:
    return float(dict(stub.certificates).get("grid_error", math.nan))


def measure(base, lam: float, branch_index: int, target_b: float,
            stable_sign: int, fractions, order: int, launch_level: int,
            n_steps: int, source_delta: float, target_delta: float,
            space_rtol: float, space_atol: float,
            clock_rtol: float, clock_atol: float):
    m = rheostat(base, lam)
    e = sturm.materialize_stubs(
        m, sturm.enumerate_critical_points(m),
        resolution_level=launch_level)
    unstable_offset = branch_index-2*len(e.saddles)
    if not 0 <= unstable_offset < 2*len(e.saddles):
        raise ValueError("--branch does not select an unstable portrait branch")
    source = e.saddles[unstable_offset//2]
    direction = 1 if unstable_offset % 2 == 0 else -1
    target = min((q for q in e.saddles),
                 key=lambda q: abs(float(q.b)-target_b))
    source_stub = next(
        s for s in source.stubs
        if s.manifold == "unstable" and s.b_direction == direction)
    target_stub = portrait._stable_stub(target, stable_sign, m)
    if target_stub is None:
        raise RuntimeError("selected target stable half-branch has no stub")

    c_high = _loss(m, np.array([source.a, source.b], float))
    c_low = _loss(m, np.array([target.a, target.b], float))
    if not c_high > c_low:
        raise ValueError("selected source is not above the target saddle")
    levels = [c_low+f*(c_high-c_low) for f in fractions]
    c_source = c_high-source_delta
    c_target = c_low+target_delta
    source_start = _loss(m, np.asarray(source_stub.curve[-1], float))
    target_start = _loss(m, np.asarray(target_stub.curve[-1], float))
    if not (source_start > max(levels) and target_start < min(levels)):
        raise ValueError("a materialized stub has already crossed a meeting level")
    # A small saddle-level gap can put the requested clock section inside the
    # materialized graph stub.  The clock must start *after* the stub but
    # before every meeting section.  Clamp it into that regular interval;
    # this changes no trajectory endpoint and retains one fixed clock origin
    # shared by all meeting levels of the shot.
    if not max(levels) < c_source < source_start:
        c_source = 0.5*(max(levels)+source_start)
    if not target_start < c_target < min(levels):
        c_target = 0.5*(target_start+min(levels))
    if not (c_high > c_source > max(levels)
            and c_low < c_target < min(levels)):
        raise ValueError("clock sections must lie between saddles and meetings")

    base_step = (c_high-c_low)/n_steps
    down_boundaries = [c_source] + sorted(levels, reverse=True)
    up_boundaries = [c_target] + sorted(levels)
    down = _shoot(
        m, source_stub.curve[-1], down_boundaries, c_source, order,
        base_step, space_rtol, space_atol, clock_rtol, clock_atol)
    up = _shoot(
        m, target_stub.curve[-1], up_boundaries, c_target, order,
        base_step, space_rtol, space_atol, clock_rtol, clock_atol)

    rows = []
    for fraction, level in zip(fractions, levels):
        xd, xu = down.points[level], up.points[level]
        rows.append({
            "fraction": fraction, "level": level,
            "down": xd, "up": xu,
            "D_b": float(xd[1]-xu[1]),
            "distance": float(np.hypot(*(xd-xu))),
            "T_down": down.times[level], "T_up": up.times[level],
            "T_sum": down.times[level]+up.times[level],
        })
    return {
        "rows": rows, "down_stats": down.stats, "up_stats": up.stats,
        "down_shot": down, "up_shot": up,
        "source_stub_error": _stub_error(source_stub),
        "target_stub_error": _stub_error(target_stub),
        "source_b": float(source.b), "target_b": float(target.b),
        "c_high": c_high, "c_low": c_low,
        "clock_source_level": c_source, "clock_target_level": c_target,
    }


def _csv_ints(value: str):
    return [int(x) for x in value.split(",")]


def _csv_floats(value: str):
    return [float(x) for x in value.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--family")
    source.add_argument("--case", help="ordinary zoo case used as rheostat base")
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--target-b", type=float, default=0.640274)
    ap.add_argument("--stable-sign", type=int, choices=(-1, 1), default=-1)
    ap.add_argument("--lo", type=float, default=2.17770946083)
    ap.add_argument("--hi", type=float, default=2.17770966083)
    ap.add_argument("--fractions", default="0.35,0.5,0.65",
                    help="meeting levels as fractions from low to high loss")
    ap.add_argument("--orders", default="-6",
                    help="pure GL orders; signs are accepted and ignored")
    ap.add_argument("--launch-levels", default="4")
    ap.add_argument("--step-levels", default="0",
                    help="level k multiplies --n-steps by 2^k")
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--source-delta", type=float, default=0.01)
    ap.add_argument("--target-delta", type=float, default=0.01)
    ap.add_argument("--space-rtol", type=float, default=1e-8)
    ap.add_argument("--space-atol", type=float, default=1e-13)
    ap.add_argument("--clock-rtol", type=float, default=1e-8)
    ap.add_argument("--clock-atol", type=float, default=1e-12)
    ap.add_argument("--root-rounds", type=int, default=2,
                    help="shared secant corrections after the endpoint fit")
    args = ap.parse_args(argv)

    fractions = _csv_floats(args.fractions)
    if not fractions or any(not 0.0 < f < 1.0 for f in fractions):
        ap.error("--fractions must lie strictly between 0 and 1")
    fractions = sorted(fractions)
    if not args.lo < args.hi:
        ap.error("--lo must be below --hi")
    launch_levels = _csv_ints(args.launch_levels)
    step_levels = _csv_ints(args.step_levels)
    if any(level < 0 for level in launch_levels):
        ap.error("--launch-levels must be nonnegative")
    if any(level < 0 for level in step_levels):
        ap.error("--step-levels must be nonnegative")
    if args.case is not None:
        base = zoo.get(args.case)
        source_name = args.case
    else:
        family_name = args.family or "nonnearest-saddle-connection"
        base = zoo.get(zoo.get_wall_family(family_name).base_case)
        source_name = family_name
    print(f"two-ended connection proposal: {source_name}")
    summaries = []

    for signed_order in _csv_ints(args.orders):
        order = abs(signed_order)
        if order not in (4, 6, 8):
            ap.error("orders must be 4, 6, or 8")
        for launch_level in launch_levels:
            for step_level in step_levels:
                n_steps = args.n_steps*2**step_level
                cache = {}

                def probe(lam):
                    key = float(lam)
                    if key not in cache:
                        cache[key] = measure(
                            base, key, args.branch, args.target_b,
                            args.stable_sign, fractions, order, launch_level,
                            n_steps, args.source_delta, args.target_delta,
                            args.space_rtol, args.space_atol,
                            args.clock_rtol, args.clock_atol)
                    return cache[key]

                left, right = probe(args.lo), probe(args.hi)
                dl = np.array([r["D_b"] for r in left["rows"]])
                dr = np.array([r["D_b"] for r in right["rows"]])
                sl = np.array([r["T_sum"] for r in left["rows"]])
                sr = np.array([r["T_sum"] for r in right["rows"]])
                if np.any(dl*dr >= 0.0):
                    print(f"\n=== GL{order} launch {launch_level} "
                          f"n_steps {n_steps}: NOT BRACKETED")
                    for f, a, b in zip(fractions, dl, dr):
                        print(f"   f={f:.3f}  D(lo)={a:+.6e} "
                              f"D(hi)={b:+.6e}")
                    continue

                slopes = (dr-dl)/(args.hi-args.lo)
                roots = args.lo-dl/slopes
                last_x = None
                last_d = None
                at_root = None
                for _ in range(max(0, args.root_rounds)):
                    x = float(np.median(roots))
                    at_root = probe(x)
                    d = np.array([r["D_b"] for r in at_root["rows"]])
                    if last_x is not None and x != last_x:
                        local = (d-last_d)/(x-last_x)
                        good = np.isfinite(local) & (local != 0.0)
                        slopes = np.where(good, local, slopes)
                    roots = x-d/slopes
                    last_x, last_d = x, d
                x = float(np.median(roots))
                at_root = probe(x)

                print(f"\n=== pure GL{order}  launch_level={launch_level} "
                      f"n_steps={n_steps}")
                print(f"   bracket [{args.lo:.17g}, {args.hi:.17g}]")
                print(f"   D ranges {dl.min():+.3e}..{dl.max():+.3e} "
                      f"at lo; {dr.min():+.3e}..{dr.max():+.3e} at hi")
                print(f"   off-root split-time spreads: "
                      f"lo={float(np.ptp(sl)):.3e}, "
                      f"hi={float(np.ptp(sr)):.3e}")
                print(f"   source stub error "
                      f"{at_root['source_stub_error']:.3e}; target stub error "
                      f"{at_root['target_stub_error']:.3e}")
                print(f"   common evaluation Lambda={x:.17g}")
                print(f"   {'f':>6} {'root Lambda':>22} {'D at common':>14} "
                      f"{'dD/dLam':>12} {'distance':>12} {'T_down':>15} "
                      f"{'T_up':>15} {'T_sum':>15}")
                sums = []
                for root, slope, row in zip(roots, slopes, at_root["rows"]):
                    sums.append(row["T_sum"])
                    print(f"   {row['fraction']:>6.3f} {root:>22.17g} "
                          f"{row['D_b']:>+14.6e} {slope:>+12.5e} "
                          f"{row['distance']:>12.3e} "
                          f"{row['T_down']:>15.12f} "
                          f"{row['T_up']:>15.12f} "
                          f"{row['T_sum']:>15.12f}")
                print(f"   meeting-level root spread: "
                      f"{float(np.ptp(roots)):.3e}")
                print(f"   split-time spread at common Lambda: "
                      f"{float(np.ptp(sums)):.3e}")
                for label, stats in (("down", at_root["down_stats"]),
                                     ("up", at_root["up_stats"])):
                    print(f"   {label}: accepted={stats.accepted} "
                          f"rejected={stats.rejected} "
                          f"clock_steps={stats.clock_steps} "
                          f"max_space_ratio={stats.max_space_ratio:.3g} "
                          f"max_clock_ratio={stats.max_clock_ratio:.3g} "
                          f"max_loss_defect={stats.max_loss_defect:.3e} "
                          f"max_section_error={stats.max_section_error:.3e}")
                summaries.append({
                    "order": order, "launch_level": launch_level,
                    "step_level": step_level, "n_steps": n_steps,
                    "root": float(np.median(roots)),
                    "level_spread": float(np.ptp(roots)),
                    "time_spread": float(np.ptp(sums)),
                })

    if len(summaries) > 1:
        print("\n=== convergence summary")
        for r in summaries:
            print(f"   GL{r['order']} launch={r['launch_level']} "
                  f"n_steps={r['n_steps']}: Lambda={r['root']:.17g} "
                  f"level_spread={r['level_spread']:.3e} "
                  f"time_spread={r['time_spread']:.3e}")

        # The materialized graph stubs use a centered second-order grid and
        # their recorded grid errors fall by about four on each refinement.
        # Report, rather than silently apply, the matching Richardson limit.
        groups = {}
        for r in summaries:
            groups.setdefault((r["order"], r["step_level"]), []).append(r)
        extrapolants = []
        for (order, step_level), rows in groups.items():
            rows.sort(key=lambda r: r["launch_level"])
            for coarse, fine in zip(rows, rows[1:]):
                gap = fine["launch_level"]-coarse["launch_level"]
                if gap <= 0:
                    continue
                ratio = 4.0**gap
                extrap = ((ratio*fine["root"]-coarse["root"])
                          /(ratio-1.0))
                extrapolants.append(extrap)
                print(f"   launch Richardson GL{order} "
                      f"L{coarse['launch_level']}->L{fine['launch_level']} "
                      f"(n_steps={fine['n_steps']}): {extrap:.17g}")
        if len(extrapolants) > 1:
            print(f"   Richardson-extrapolant spread: "
                  f"{float(np.ptp(extrapolants)):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
