#!/usr/bin/env python3
"""Screen an ensemble for near saddle connections, then shoot them.

This is the floating exploration/proposal layer.  It deliberately separates
two jobs:

1. Reproduce ensemble seeds and rank late approaches of unstable branches to
   lower N-saddles.  The screen is the distance to the saddle where the
   ordinary branch crosses that saddle's loss level, reported together with
   the crossing segment's sagitta allowance.  The unrestricted geometric
   closest approach is retained as a diagnostic but does not rank candidates:
   it can occur at entirely the wrong loss.
2. For the strongest candidates, shoot the unstable branch downward and each
   target stable half-branch upward on regular loss sections.  Search the
   canonical Lambda rheostat in t = log(Lambda), and report a proposed root
   only when every meeting level changes sign in one common bracket.

The screen is not a distance-to-wall and the proposed root is not a
certificate.  Validation belongs to launch refinement, pure-order comparison,
and ultimately interval propagation.

Examples:

    python scripts/explore_non_smale.py --screen-cases 40 --shoot-top 5
    python scripts/explore_non_smale.py --mode directed --screen-only --jobs 6
    python scripts/explore_non_smale.py --jobs 6 --shoot-top 12 --force
    python scripts/explore_non_smale.py --screen-only --out out/screen.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("SPONG_ENGINE", "native")

from qualify import directed_model, random_model              # noqa: E402
from spong import portrait, sturm, topology, wall_shoot       # noqa: E402
from two_ended_connection import measure                      # noqa: E402


def _critical_loss(m, point) -> float:
    return float(m.L(float(point.a), float(point.b)))


def _closest_segment(Y: np.ndarray, point: np.ndarray):
    """Closest point on a polyline, its segment, and local coordinate."""
    if len(Y) < 2:
        return None
    delta = Y[1:] - Y[:-1]
    denom = np.sum(delta * delta, axis=1)
    numer = np.sum((point - Y[:-1]) * delta, axis=1)
    t = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
    t = np.clip(t, 0.0, 1.0)
    nearest = Y[:-1] + t[:, None] * delta
    distance = np.hypot(nearest[:, 0] - point[0],
                        nearest[:, 1] - point[1])
    k = int(np.argmin(distance))
    return float(distance[k]), k, float(t[k]), nearest[k]


def _level_crossing(m, Y: np.ndarray, level: float, first_segment: int):
    """First descending crossing of ``level`` after the launch prefix.

    This is a screen, so refinement stays on the already computed chord.  Its
    local sagitta is carried with the result and decides whether the geometric
    reading is finer than its own representation.
    """
    losses = np.asarray([
        float(m.L(float(point[0]), float(point[1]))) for point in Y],
        dtype=float)
    start = max(1, int(first_segment)+1)
    below = np.flatnonzero(losses[start:] <= level)
    if not len(below):
        return None
    k = start+int(below[0])
    if losses[k-1] < level:
        return None
    left, right = Y[k-1].copy(), Y[k].copy()
    for _ in range(80):
        middle = 0.5*(left+right)
        if float(m.L(float(middle[0]), float(middle[1]))) > level:
            left = middle
        else:
            right = middle
        if np.hypot(*(right-left)) <= 4*np.finfo(float).eps*(
                1.0+np.hypot(*middle)):
            break
    point = 0.5*(left+right)
    return k-1, point


def _generator(mode: str):
    return directed_model if mode == "directed" else random_model


def _screen_seed_inner(mode, case, seed, max_degree):
    started = time.perf_counter()
    try:
        m, spec = _generator(mode)(random.Random(seed), max_degree)
        if m is None:
            raise RuntimeError(f"{mode} generator declined seed {seed}")
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
        if len(e.saddles) < 2:
            return {"case": case, "seed": seed, "spec": spec,
                    "candidates": [], "seconds": time.perf_counter()-started}

        # Stable traces are not inputs to this screen.  Keeping them inside
        # the display box avoids paying the exact level-bar retrace while
        # leaving every unstable branch bit-for-bit on its ordinary path.
        old_display = os.environ.get("SPONG_STABLE_DISPLAY_BOX")
        os.environ["SPONG_STABLE_DISPLAY_BOX"] = "1"
        try:
            p = portrait.compute(m, _enumeration=e, _skip_audit=True)
        finally:
            if old_display is None:
                os.environ.pop("SPONG_STABLE_DISPLAY_BOX", None)
            else:
                os.environ["SPONG_STABLE_DISPLAY_BOX"] = old_display

        candidates = []
        for branch_index, branch in enumerate(p.branches):
            if branch.kind != "unstable" or len(branch.Y) < 2:
                continue
            source = min(
                e.saddles,
                key=lambda q: abs(float(q.b)
                                  - float(branch.diag.get("saddle_b", q.b))))
            source_loss = _critical_loss(m, source)
            Y = np.asarray(branch.Y, dtype=float)
            sagitta = np.asarray(topology._sagitta_bounds(Y), dtype=float)
            launch_segments = int(branch.diag.get("critical_steps", 0))

            for target in e.saddles:
                # A descending saddle connection can terminate only at an
                # N-saddle.  B-saddles all have the common level C, so they
                # cannot be the lower endpoint.  Random generic models use
                # the factorized B/N enumeration; an H-root is not silently
                # guessed to be N here.
                if target is source or target.source != "N":
                    continue
                target_loss = _critical_loss(m, target)
                loss_gap = source_loss - target_loss
                loss_floor = 512.0*np.finfo(float).eps*max(
                    1.0, abs(source_loss), abs(target_loss))
                if loss_gap <= loss_floor:
                    continue

                target_xy = np.array((float(target.a), float(target.b)))
                closest = _closest_segment(Y, target_xy)
                if closest is None:
                    continue
                raw_distance, raw_segment, raw_segment_t, nearest = closest

                # Reject proximity at the source launch itself.  That is a
                # near-Morse critical-spacing observation, not a returning
                # unstable manifold approaching a lower saddle.
                if raw_segment <= launch_segments:
                    continue
                crossing = _level_crossing(
                    m, Y, target_loss, launch_segments)
                if crossing is None:
                    continue
                segment, section_point = crossing
                distance = float(np.hypot(*(section_point-target_xy)))

                others = [
                    math.hypot(float(q.a)-float(target.a),
                               float(q.b)-float(target.b))
                    for q in e.points if q is not target]
                target_spacing = min(others, default=1.0)
                local_sagitta = (float(sagitta[segment])
                                  if segment < len(sagitta) else math.inf)
                nearest_loss = float(m.L(float(nearest[0]), float(nearest[1])))
                relative_distance = distance/max(target_spacing, 1e-300)
                raw_relative_loss_mismatch = (
                    abs(nearest_loss-target_loss)/loss_gap)
                candidates.append({
                    "candidate_id": (
                        f"{int(case)}:{int(branch_index)}:"
                        f"{float(target.b).hex()}"),
                    "case": int(case), "seed": int(seed), "spec": spec,
                    "branch": int(branch_index),
                    "source_b": float(source.b),
                    "source_type": source.source,
                    "unstable_direction": int(
                        branch.diag.get("unstable_direction", 0)),
                    "target_b": float(target.b),
                    "target_type": target.source,
                    "source_loss": source_loss,
                    "target_loss": target_loss,
                    "loss_gap": loss_gap,
                    "distance": distance,
                    "target_spacing": target_spacing,
                    "relative_distance": relative_distance,
                    "sagitta": local_sagitta,
                    "distance_over_sagitta": (
                        distance/local_sagitta if local_sagitta > 0.0
                        else math.inf),
                    "section_point": section_point.tolist(),
                    "section_loss_error": abs(float(m.L(
                        float(section_point[0]), float(section_point[1])))
                        - target_loss),
                    "raw_closest_distance": raw_distance,
                    "raw_closest_segment": raw_segment,
                    "raw_closest_fraction": raw_segment_t,
                    "raw_closest_loss": nearest_loss,
                    "raw_relative_loss_mismatch": raw_relative_loss_mismatch,
                    "closest_segment": segment,
                    "branch_segments": len(Y)-1,
                    "launch_segments": launch_segments,
                    "branch_term": branch.term,
                })

        candidates.sort(key=lambda x: x["relative_distance"])
        return {
            "case": int(case), "seed": int(seed), "spec": spec,
            "n_critical": len(e.points), "n_saddle": len(e.saddles),
            "candidates": candidates,
            "seconds": time.perf_counter()-started,
        }
    except Exception as exc:                                  # noqa: BLE001
        return {
            "case": int(case), "seed": int(seed),
            "error": f"{type(exc).__name__}: {exc}",
            "candidates": [], "seconds": time.perf_counter()-started,
        }


def _screen_seed(task):
    """Screen one seed, with an optional per-case wall-clock alarm.

    The directed ensemble deliberately contains extreme-radius cases.  A
    diagnostic screen must record those as timeouts rather than allowing a
    handful of stiffness probes to prevent the tractable ensemble results
    from being written.
    """
    mode, case, seed, max_degree, timeout = task
    if timeout <= 0.0 or not hasattr(signal, "setitimer"):
        return _screen_seed_inner(mode, case, seed, max_degree)

    def expired(_signum, _frame):
        raise TimeoutError(f"screen exceeded {timeout:g}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _screen_seed_inner(mode, case, seed, max_degree)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _row_vector(reading, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in reading["rows"]], dtype=float)


def _json_clean(value):
    """Strict JSON has no NaN/Infinity; preserve those diagnostics as null."""
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    """Atomically replace a strict-JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        _json_clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _shoot_candidate(candidate, max_degree: int, *, mode: str, fractions,
                     order: int, launch_level: int, n_steps: int,
                     initial_log_step: float, max_log_radius: float,
                     root_iterations: int):
    started = time.perf_counter()
    m, spec = _generator(mode)(
        random.Random(candidate["seed"]), max_degree)
    if m is None:
        raise RuntimeError(
            f"{mode} generator declined seed {candidate['seed']}")
    loss_gap0 = float(candidate["loss_gap"])
    branch = int(candidate["branch"])
    target_b = float(candidate["target_b"])
    cache = {}

    def read(t, stable_sign):
        key = (float(t), int(stable_sign))
        if key not in cache:
            lam = math.exp(float(t))
            # Every critical value and hence the saddle-level gap scales as
            # 1/Lambda under this rheostat.  Use fixed *fractional* clock
            # sections even while the bracket expands.
            section_delta = 0.05*loss_gap0/lam
            cache[key] = measure(
                m, lam, branch, target_b, stable_sign, fractions,
                order, launch_level, n_steps,
                section_delta, section_delta,
                1e-8, 1e-13, 1e-8, 1e-12)
        return cache[key]

    try:
        at_zero = {}
        for stable_sign in (+1, -1):
            try:
                at_zero[stable_sign] = read(0.0, stable_sign)
            except Exception:                                 # noqa: BLE001
                pass
        if not at_zero:
            raise RuntimeError("neither target stable half-branch was shootable")
        stable_sign, zero = min(
            at_zero.items(),
            key=lambda item: float(np.median(
                _row_vector(item[1], "distance"))))

        values = {0.0: zero}
        radius = initial_log_step
        bracket = None
        while radius <= max_log_radius*(1.0+1e-12):
            for t in (-radius, radius):
                if t not in values:
                    values[t] = read(t, stable_sign)
            ordered = sorted(values)
            possible = []
            for a, b in zip(ordered, ordered[1:]):
                da = _row_vector(values[a], "D_b")
                db = _row_vector(values[b], "D_b")
                if np.all(da*db <= 0.0) and np.any(da*db < 0.0):
                    possible.append((a, b))
            if possible:
                bracket = min(possible, key=lambda ab: min(abs(ab[0]),
                                                            abs(ab[1])))
                break
            radius *= 2.0

        zero_D = _row_vector(zero, "D_b")
        result = {
            **candidate,
            "stable_sign": int(stable_sign),
            "D_at_one": zero_D.tolist(),
            "distance_at_one": _row_vector(zero, "distance").tolist(),
            "clock_spread_at_one": float(np.ptp(
                _row_vector(zero, "T_sum"))),
            "bracketed": bracket is not None,
            "search_log_radius": min(radius, max_log_radius),
            "order": int(order), "launch_level": int(launch_level),
            "n_steps": int(n_steps),
        }
        if bracket is None:
            samples = []
            for t in sorted(values):
                D = _row_vector(values[t], "D_b")
                samples.append({"log_lambda": t, "lambda": math.exp(t),
                                "D": D.tolist()})
            result["samples"] = samples
            result["seconds"] = time.perf_counter()-started
            return result

        left, right = bracket
        initial_left, initial_right = left, right
        left_read, right_read = values[left], values[right]
        f_left = float(np.median(_row_vector(left_read, "D_b")))
        f_right = float(np.median(_row_vector(right_read, "D_b")))
        if f_left*f_right > 0.0:
            raise RuntimeError("common component bracket lost its median sign")

        def scalar(t):
            return float(np.median(_row_vector(
                read(float(t), stable_sign), "D_b")))

        root_t, _root_value, final_bracket, _evaluations = wall_shoot._brent(
            scalar, left, right, f_left, f_right,
            xtol=8*np.finfo(float).eps, ftol=1e-12,
            max_iter=max(root_iterations, 8))
        root_t = float(root_t)
        root_read = read(root_t, stable_sign)
        root_D = _row_vector(root_read, "D_b")
        # Use the original common bracket for section-wise root estimates.
        # Brent's final bracket may be only a few ulps wide, where differencing
        # D would turn a clean derivative into a rounding diagnostic.
        left_D = _row_vector(values[initial_left], "D_b")
        right_D = _row_vector(values[initial_right], "D_b")
        slopes = (right_D-left_D)/(initial_right-initial_left)
        good = np.isfinite(slopes) & (slopes != 0.0)
        section_roots = np.full(len(slopes), np.nan)
        section_roots[good] = root_t-root_D[good]/slopes[good]

        result.update({
            "root_log_lambda": root_t,
            "root_lambda": math.exp(root_t),
            "log_rheostat_margin": abs(root_t),
            "final_log_bracket": list(final_bracket),
            "root_D": root_D.tolist(),
            "root_distance": _row_vector(root_read, "distance").tolist(),
            "section_root_log_lambda": section_roots.tolist(),
            "section_root_spread": float(np.nanmax(section_roots)
                                         - np.nanmin(section_roots)),
            "root_clock_sums": _row_vector(root_read, "T_sum").tolist(),
            "root_clock_spread": float(np.ptp(
                _row_vector(root_read, "T_sum"))),
            "source_stub_error": float(root_read["source_stub_error"]),
            "target_stub_error": float(root_read["target_stub_error"]),
            "seconds": time.perf_counter()-started,
        })
        return result
    except Exception as exc:                                  # noqa: BLE001
        return {
            **candidate, "bracketed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter()-started,
        }


def _load_ensemble(path: Path, limit: int):
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    rows.sort(key=lambda row: int(row["case"]))
    rows = [row for row in rows if int(row.get("n_saddle", 0)) >= 2]
    return rows if limit <= 0 else rows[:limit]


def _generated_tasks(master_seed: int, start_case: int, count: int,
                     max_degree: int):
    """The same per-case seed schedule as scripts/ensemble.py."""
    rng = random.Random(master_seed)
    stop = start_case+count
    seeds = [rng.randrange(2**31) for _ in range(stop)]
    return [(case, seeds[case], max_degree)
            for case in range(start_case, stop)]


def _rank_candidates(screened, per_case: int):
    candidates = []
    for result in screened:
        rows = result.get("candidates", [])
        for row in rows:
            row.setdefault(
                "candidate_id",
                f"{int(row['case'])}:{int(row['branch'])}:"
                f"{float(row['target_b']).hex()}")
        candidates.extend(rows[:per_case])
    candidates.sort(key=lambda row: row["relative_distance"])
    return candidates


def _print_candidate(prefix: str, row):
    print(
        f"{prefix} case={row['case']:3d} seed={row['seed']:10d} "
        f"br={row['branch']:2d} {row['source_type']}:{row['source_b']:+.6g}"
        f"/{row['unstable_direction']:+d} -> "
        f"N:{row['target_b']:+.6g}  d={row['distance']:.3e} "
        f"d/spacing={row['relative_distance']:.3e} "
        f"d/sag={row['distance_over_sagitta']:.3e} "
        f"raw-d={row['raw_closest_distance']:.3e} "
        f"raw-dL/gap={row['raw_relative_loss_mismatch']:.3e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("random", "directed"), default=None,
                    help="ensemble constructor; inferred from a reused artifact")
    ap.add_argument("--ensemble", type=Path,
                    help="ensemble ledger (default: ensemble-MODE-dN.jsonl)")
    ap.add_argument("--reuse-screen", type=Path,
                    help="reuse the screened rows from a previous JSON artifact")
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--generated-cases", type=int, default=0,
                    help="screen this many generator cases instead of a ledger")
    ap.add_argument("--start-case", type=int, default=0,
                    help="first generator case used with --generated-cases")
    ap.add_argument("--master-seed", type=int, default=20260806,
                    help="ensemble.py-compatible generator master seed")
    ap.add_argument("--screen-cases", type=int, default=0,
                    help="number of multi-saddle ledger cases; 0 means all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--case-timeout", type=float, default=60.0,
                    help="seconds per screen case; 0 disables the watchdog")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="write a valid partial artifact every N cases")
    ap.add_argument("--per-case", type=int, default=2,
                    help="maximum candidates from one seed in global ranking")
    ap.add_argument("--show-top", type=int, default=20)
    ap.add_argument("--multiplicity-threshold", type=float, default=1.0,
                    help="report portraits with at least two candidates "
                         "below this normalized screen distance")
    ap.add_argument("--shoot-top", type=int, default=10)
    ap.add_argument("--shoot-cases",
                    help="comma-separated case ids, overriding --shoot-top")
    ap.add_argument("--shoot-ranks",
                    help="comma-separated one-based candidate ranks, "
                         "overriding --shoot-top")
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--fractions", default="0.35,0.5,0.65")
    ap.add_argument("--order", type=int, choices=(4, 6, 8), default=6)
    ap.add_argument("--orders",
                    help="comma-separated pure orders; overrides --order")
    ap.add_argument("--launch-level", type=int, default=3)
    ap.add_argument("--launch-levels",
                    help="comma-separated launch levels; overrides singular")
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--step-levels", default="0",
                    help="comma-separated k values multiplying n-steps by 2^k")
    ap.add_argument("--initial-log-step", type=float, default=0.01)
    ap.add_argument("--max-log-radius", type=float, default=0.32)
    ap.add_argument("--root-iterations", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    help="JSON artifact (default: non_smale_MODE_probe.json)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    previous = None
    if args.reuse_screen is not None:
        previous = json.loads(args.reuse_screen.read_text())
        previous_mode = previous.get("configuration", {}).get("mode", "random")
        if args.mode is not None and args.mode != previous_mode:
            ap.error(f"--mode {args.mode} disagrees with reused "
                     f"artifact mode {previous_mode}")
        args.mode = previous_mode
    elif args.mode is None:
        args.mode = "random"
    if args.ensemble is None:
        args.ensemble = (REPO / "out" /
                         f"ensemble-{args.mode}-d{args.max_degree}.jsonl")
    if args.out is None:
        args.out = (REPO / "out" /
                    f"non_smale_{args.mode}_probe.json")

    if args.jobs < 1:
        ap.error("--jobs must be positive")
    if args.case_timeout < 0.0:
        ap.error("--case-timeout must be nonnegative")
    if args.checkpoint_every < 1:
        ap.error("--checkpoint-every must be positive")
    if args.multiplicity_threshold <= 0.0:
        ap.error("--multiplicity-threshold must be positive")
    if args.shoot_cases and args.shoot_ranks:
        ap.error("--shoot-cases and --shoot-ranks are mutually exclusive")
    if args.out.exists() and not args.force:
        ap.error(f"{args.out} exists; use --force or choose --out")
    fractions = tuple(float(x) for x in args.fractions.split(","))
    if not fractions or any(not 0.0 < x < 1.0 for x in fractions):
        ap.error("--fractions must lie strictly between zero and one")
    fractions = tuple(sorted(fractions))
    orders = ([int(x) for x in args.orders.split(",")]
              if args.orders else [args.order])
    launch_levels = ([int(x) for x in args.launch_levels.split(",")]
                     if args.launch_levels else [args.launch_level])
    step_levels = [int(x) for x in args.step_levels.split(",")]
    if any(order not in (4, 6, 8) for order in orders):
        ap.error("--orders entries must be 4, 6, or 8")
    if any(level < 0 for level in launch_levels+step_levels):
        ap.error("launch and step levels must be nonnegative")

    if args.reuse_screen is not None:
        screened = previous["screened"]
        tasks = []
        source_description = f"screen artifact {args.reuse_screen}"
    elif args.generated_cases > 0:
        tasks = [(args.mode, *task, args.case_timeout)
                 for task in _generated_tasks(
            args.master_seed, args.start_case, args.generated_cases,
            args.max_degree)]
        source_description = (
            f"generator seed={args.master_seed} cases "
            f"[{args.start_case},{args.start_case+args.generated_cases})")
    else:
        ledger = _load_ensemble(args.ensemble, args.screen_cases)
        tasks = [(args.mode, int(row["case"]), int(row["seed"]),
                  args.max_degree, args.case_timeout)
                 for row in ledger]
        source_description = str(args.ensemble)
    if args.reuse_screen is None:
        print(f"screening {len(tasks)} {args.mode} cases from "
              f"{source_description}, "
              f"jobs={args.jobs}")
    else:
        print(f"reusing {len(screened)} screened cases from "
              f"{args.reuse_screen}")
    started = time.perf_counter()
    if args.reuse_screen is None:
        screened = []

    def checkpoint():
        partial = {
            "format": "spong-non-smale-exploration-v1",
            "scope": "partial floating screen; not a certificate",
            "complete": False,
            "ensemble": source_description,
            "configuration": {
                "mode": args.mode, "max_degree": args.max_degree,
                "case_timeout": args.case_timeout,
            },
            "screened": sorted(screened, key=lambda row: row["case"]),
            "ranked": _rank_candidates(screened, args.per_case),
            "shots": [],
        }
        _write_json(args.out, partial)

    # Do not multiply process workers by portrait branch threads.  The screen
    # is embarrassingly parallel over seeds.
    if args.reuse_screen is not None:
        pass
    elif args.jobs > 1:
        os.environ["SPONG_WORKERS"] = "1"
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_map = {
                pool.submit(_screen_seed, task): task for task in tasks}
            for completed, future in enumerate(cf.as_completed(future_map), 1):
                result = future.result()
                screened.append(result)
                if completed % args.checkpoint_every == 0:
                    checkpoint()
                if (completed <= 5 or completed % 10 == 0
                        or completed == len(tasks)):
                    best = result.get("candidates", [])
                    detail = (f" best={best[0]['relative_distance']:.3e}"
                              if best else "")
                    print(f"  [{completed:3d}/{len(tasks):3d}] "
                          f"case={result['case']:3d} "
                          f"{result['seconds']:.2f}s{detail}", flush=True)
    else:
        for completed, task in enumerate(tasks, 1):
            result = _screen_seed(task)
            screened.append(result)
            if completed % args.checkpoint_every == 0:
                checkpoint()
            best = result.get("candidates", [])
            detail = (f" best={best[0]['relative_distance']:.3e}"
                      if best else "")
            print(f"  [{completed:3d}/{len(tasks):3d}] case={result['case']:3d} "
                  f"{result['seconds']:.2f}s{detail}", flush=True)

    ranked = _rank_candidates(screened, args.per_case)
    errors = [row for row in screened if row.get("error")]
    print(f"screen complete in {time.perf_counter()-started:.1f}s: "
          f"{len(ranked)} ranked candidates, {len(errors)} errors")
    for i, row in enumerate(ranked[:args.show_top], 1):
        _print_candidate(f"  #{i:02d}", row)
    multiplicity = []
    for result in screened:
        close = sorted(
            (row for row in result.get("candidates", [])
             if row["relative_distance"] < args.multiplicity_threshold),
            key=lambda row: row["relative_distance"])
        if len(close) >= 2:
            multiplicity.append({
                "case": int(result["case"]),
                "candidate_ids": [row["candidate_id"] for row in close],
                "relative_distances": [
                    row["relative_distance"] for row in close],
            })
    if multiplicity:
        print(f"multiple-candidate portraits below normalized distance "
              f"{args.multiplicity_threshold:g}:")
        for row in multiplicity:
            print(f"  case={row['case']} "
                  f"relative distances={row['relative_distances']}")

    shots = []
    if not args.screen_only:
        if args.shoot_ranks:
            wanted = [int(x) for x in args.shoot_ranks.split(",")]
            if (not wanted or any(rank < 1 or rank > len(ranked)
                                  for rank in wanted)):
                ap.error(f"--shoot-ranks must lie in [1,{len(ranked)}]")
            selected = [ranked[rank-1] for rank in wanted]
        elif args.shoot_cases:
            wanted = [int(x) for x in args.shoot_cases.split(",")]
            by_case = {}
            for row in ranked:
                by_case.setdefault(row["case"], row)
            missing = [case for case in wanted if case not in by_case]
            if missing:
                ap.error(f"shoot cases absent from ranking: {missing}")
            selected = [by_case[case] for case in wanted]
        else:
            selected = ranked[:args.shoot_top]
        axes = [(order, launch_level, args.n_steps*2**step_level)
                for order in orders for launch_level in launch_levels
                for step_level in step_levels]
        print(f"shooting {len(selected)} candidates "
              f"on {len(axes)} pure-integrator/refinement axes")
        for i, candidate in enumerate(selected, 1):
            _print_candidate(f"  [{i:02d}]", candidate)
            for order, launch_level, n_steps in axes:
                result = _shoot_candidate(
                    candidate, args.max_degree, mode=args.mode,
                    fractions=fractions,
                    order=order, launch_level=launch_level,
                    n_steps=n_steps,
                    initial_log_step=args.initial_log_step,
                    max_log_radius=args.max_log_radius,
                    root_iterations=args.root_iterations)
                shots.append(result)
                axis = f"GL{order} launch={launch_level} n={n_steps}"
                if result.get("error"):
                    print(f"       {axis}: ERROR {result['error']}")
                elif not result["bracketed"]:
                    print(f"       {axis}: no common bracket within "
                          f"|log Lambda| <= {args.max_log_radius:g}; "
                          f"D(1)={result['D_at_one']}")
                else:
                    print(f"       {axis}: "
                          f"Lambda*={result['root_lambda']:.17g} "
                          f"|log Lambda*|="
                          f"{result['log_rheostat_margin']:.3e} "
                          f"level spread="
                          f"{result['section_root_spread']:.3e} "
                          f"clock spread={result['root_clock_spread']:.3e}")

    artifact = {
        "format": "spong-non-smale-exploration-v1",
        "scope": "floating screen and proposal; not a certificate",
        "complete": True,
        "screen_complete": (
            args.reuse_screen is None
            or bool(previous.get(
                "screen_complete", previous.get("complete", True)))),
        "ensemble": source_description,
        "configuration": {
            "mode": args.mode,
            "case_timeout": args.case_timeout,
            "multiplicity_threshold": args.multiplicity_threshold,
            "max_degree": args.max_degree,
            "generated_cases": args.generated_cases,
            "start_case": args.start_case,
            "master_seed": args.master_seed,
            "screen_cases": args.screen_cases,
            "per_case": args.per_case,
            "fractions": list(fractions),
            "orders": orders,
            "launch_levels": launch_levels,
            "step_levels": step_levels,
            "n_steps": args.n_steps,
            "initial_log_step": args.initial_log_step,
            "max_log_radius": args.max_log_radius,
        },
        "screened": sorted(screened, key=lambda row: row["case"]),
        "ranked": ranked,
        "multiplicity": multiplicity,
        "shots": shots,
    }
    _write_json(args.out, artifact)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
