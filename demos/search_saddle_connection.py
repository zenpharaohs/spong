#!/usr/bin/env python3
"""Continuous-Bernoulli search for an exceptional SPONG saddle connection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demos.cb_sampler import ContinuousBernoulliBank
from demos import saddle_connections as sc
from demos.thompson import transformed_loss


def _evaluate(payload):
    f, g, kwargs = payload
    return sc.evaluate(f, g, **kwargs)


class _SerialMap:
    def map(self, function, iterable):
        return map(function, iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _json_coefficients(values):
    return [[x.numerator, x.denominator] for x in values]


def _record(index, f, g, evaluation, pulls):
    return {
        "arm": index,
        "pulls": pulls,
        "f": _json_coefficients(f),
        "g": _json_coefficients(g),
        "evaluation": asdict(evaluation),
    }


def _load_coefficients(path):
    data = json.loads(Path(path).read_text())
    if "arms" in data:
        if not data["arms"]:
            raise ValueError("resume search contains no arms")
        data = min(
            data["arms"], key=lambda arm: arm["evaluation"]["score"])
    return (
        tuple(Fraction(n, d) for n, d in data["f"]),
        tuple(Fraction(n, d) for n, d in data["g"]))


def _jitter_coefficients(f, g, rng, amount, bits):
    f0 = np.asarray(list(map(float, f)))
    g0 = np.asarray(list(map(float, g)))
    return (
        sc.quantized_unit_coefficients(
            f0 + amount*rng.normal(size=len(f0)), bits),
        sc.quantized_unit_coefficients(
            g0 + amount*rng.normal(size=len(g0)), bits))


def _bracket_endpoint(coefficients, evaluation):
    return {
        "f": _json_coefficients(coefficients[0]),
        "g": _json_coefficients(coefficients[1]),
        "evaluation": asdict(evaluation),
    }


def _configuration(args):
    values = vars(args).copy()
    for key, value in tuple(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    values["cb_library"] = str(Path(args.cb_library).resolve())
    values["central_strip_width"] = sc.central_strip_width(args.degree)
    return values


def _write_checkpoint(path, args, coefficients, evaluations, pulls, bracket):
    records = [
        _record(i, *coefficients[i], evaluations[i], int(pulls[i]))
        for i in range(args.arms)]
    records.sort(key=lambda x: x["evaluation"]["score"])
    output = {
        "configuration": _configuration(args),
        "arms": records,
        "bracket": bracket,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(path)
    return output


def _advance_arm(f, g, incumbent, rng, args, kwargs, pool,
                 round_index, arm):
    """One minimization step of a persistent bandit arm."""
    tracked = (sc.tracked_pair(incumbent.pair)
               if incumbent.valid and incumbent.pair is not None else None)
    arm_kwargs = kwargs | {"tracked": tracked}
    plus, minus, df, dg = sc.perturb_spsa(
        f, g, rng, args.epsilon, args.coefficient_bits)
    ep, em = list(pool.map(
        _evaluate,
        [(plus[0], plus[1], arm_kwargs),
         (minus[0], minus[1], arm_kwargs)]))
    sp = ep.pair.shooting_mismatch if ep.pair else None
    sm = em.pair.shooting_mismatch if em.pair else None
    s0 = incumbent.pair.shooting_mismatch if incumbent.pair else None
    if sp is not None and sm is not None and sp*sm <= 0.0:
        return f, g, incumbent, {
            "round": round_index + 1, "arm": arm, "tracked": tracked,
            "left": _bracket_endpoint(minus, em),
            "right": _bracket_endpoint(plus, ep),
        }
    if not (np.isfinite(ep.score) and np.isfinite(em.score)):
        return f, g, incumbent, None
    if (s0 is not None and sp is not None and sm is not None
            and ep.objective_kind == em.objective_kind == "shooting"):
        candidate = sc.shooting_secant_update(
            f, g, s0, sp, sm, df, dg, args.epsilon,
            args.coefficient_bits, args.max_coefficient_step)
    else:
        candidate = sc.spsa_update(
            f, g, ep.score, em.score, df, dg,
            args.epsilon, args.learning_rate,
            args.coefficient_bits, args.max_coefficient_step)
    if candidate is None:
        return f, g, incumbent, None
    ec = _evaluate((candidate[0], candidate[1], arm_kwargs))
    if (ec.valid and ec.pair is not None and s0 is not None
            and ec.pair.shooting_mismatch is not None
            and s0*ec.pair.shooting_mismatch <= 0.0):
        return f, g, incumbent, {
            "round": round_index + 1, "arm": arm, "tracked": tracked,
            "left": _bracket_endpoint((f, g), incumbent),
            "right": _bracket_endpoint(candidate, ec),
        }
    if ec.valid and ec.score < incumbent.score:
        return candidate[0], candidate[1], ec, None
    return f, g, incumbent, None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Target a codimension-one SPONG saddle connection")
    parser.add_argument("--degree", type=int, default=11)
    parser.add_argument("--arms", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=36)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallel-backend",
                        choices=("auto", "process", "thread"),
                        default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--coefficient-bits", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=2e-3)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--max-coefficient-step", type=float, default=5e-3)
    parser.add_argument("--strip-factor", type=float, default=3.0)
    parser.add_argument("--maximum-vertices", type=int, default=512)
    parser.add_argument("--shooting-switch", type=float, default=0.5)
    parser.add_argument("--geometry-level", type=int, default=0)
    parser.add_argument("--respawn-attempts", type=int, default=8)
    parser.add_argument("--resume-candidate", type=Path)
    parser.add_argument("--resume-jitter", type=float, default=2e-3)
    parser.add_argument(
        "--screen-only", action="store_true",
        help="evaluate and checkpoint the initial arms without SPSA pulls")
    parser.add_argument(
        "--chunk-steps", type=int, default=3,
        help="minimization steps performed whenever an arm is pulled")
    parser.add_argument("--moment-dist", choices=("uniform01", "normal01"),
                        default="uniform01")
    parser.add_argument("--cb-library", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("out/saddle_connection_search.json"))
    args = parser.parse_args(argv)
    if args.rounds < args.arms:
        parser.error("--rounds must allow one initial pull per arm")

    seed_sequence = np.random.SeedSequence(args.seed)
    rngs = [np.random.default_rng(s) for s in
            seed_sequence.spawn(args.arms)]
    if args.resume_candidate is None:
        coefficients = [
            (sc.random_coefficients(rng, args.degree, args.coefficient_bits),
             sc.random_coefficients(rng, args.degree, args.coefficient_bits))
            for rng in rngs]
    else:
        base = _load_coefficients(args.resume_candidate)
        if max(len(base[0]), len(base[1])) - 1 != args.degree:
            parser.error("--degree does not match --resume-candidate")
        coefficients = [base] + [
            _jitter_coefficients(
                *base, rngs[i], args.resume_jitter, args.coefficient_bits)
            for i in range(1, args.arms)]
    kwargs = {
        "moment_dist": args.moment_dist,
        "geometry_level": args.geometry_level,
        "strip_factor": args.strip_factor,
        "maximum": args.maximum_vertices,
        "shooting_switch": args.shooting_switch,
    }
    pulls = np.zeros(args.arms, dtype=np.int64)

    if args.workers == 1:
        pool_context = _SerialMap()
    elif args.parallel_backend == "thread":
        pool_context = ThreadPoolExecutor(max_workers=args.workers)
    elif args.parallel_backend == "process":
        pool_context = ProcessPoolExecutor(max_workers=args.workers)
    else:
        try:
            pool_context = ProcessPoolExecutor(max_workers=args.workers)
        except (OSError, PermissionError):
            # Some sandboxed installation validators deny POSIX semaphores.
            # Threads retain correctness and still overlap native exact/C
            # kernels, although ordinary installations should use processes.
            pool_context = ThreadPoolExecutor(max_workers=args.workers)
    with pool_context as pool:
        evaluations = list(pool.map(
            _evaluate, [(f, g, kwargs) for f, g in coefficients]))
        # Invalid coefficient draws are not useful bandit arms.  In
        # particular, many draws have only equal-level B-root saddles and
        # therefore no energy-feasible ordered saddle pair.
        for arm in range(args.arms):
            attempts = 0
            while not evaluations[arm].valid \
                    and attempts < args.respawn_attempts:
                coefficients[arm] = (
                    sc.random_coefficients(
                        rngs[arm], args.degree, args.coefficient_bits),
                    sc.random_coefficients(
                        rngs[arm], args.degree, args.coefficient_bits))
                evaluations[arm] = _evaluate(
                    (coefficients[arm][0], coefficients[arm][1], kwargs))
                attempts += 1
        if args.screen_only:
            output = _write_checkpoint(
                args.output, args, coefficients, evaluations, pulls, None)
            best = next(
                (arm for arm in output["arms"]
                 if arm["evaluation"]["valid"]), None)
            if best is not None:
                pair = best["evaluation"]["pair"]
                print(json.dumps({
                    "screen_only": True,
                    "best_arm": best["arm"],
                    "best_score": best["evaluation"]["score"],
                    "objective": best["evaluation"]["objective_kind"],
                    "distance": pair["distance"] if pair else None,
                    "normalized_distance": (
                        pair["normalized_distance"] if pair else None),
                    "shooting_mismatch": (
                        pair["shooting_mismatch"] if pair else None),
                    "source_b": pair["source_b"] if pair else None,
                    "target_b": pair["target_b"] if pair else None,
                }), flush=True)
            print(f"wrote {args.output}")
            return 0
        with ContinuousBernoulliBank(
                args.arms, seed=args.seed, library=args.cb_library) as posterior:
            # Forced initial pull per arm, then posterior-directed pulls.
            # A pull is a short minimization continuation, not merely an
            # observation of the arm's starting state.
            bracket = None
            for arm in range(args.arms):
                for _ in range(args.chunk_steps):
                    f, g, evaluation, bracket = _advance_arm(
                        *coefficients[arm], evaluations[arm], rngs[arm],
                        args, kwargs, pool, arm, arm)
                    coefficients[arm] = (f, g)
                    evaluations[arm] = evaluation
                    if bracket is not None:
                        break
                posterior.update(
                    arm, transformed_loss(evaluations[arm].score))
                pulls[arm] += 1
                _write_checkpoint(
                    args.output, args, coefficients, evaluations, pulls,
                    bracket)
                if bracket is not None:
                    break

            for round_index in range(args.arms, args.rounds):
                if bracket is not None:
                    break
                arm = int(np.argmin(posterior.draw_all()))
                for _ in range(args.chunk_steps):
                    f, g, evaluation, bracket = _advance_arm(
                        *coefficients[arm], evaluations[arm], rngs[arm],
                        args, kwargs, pool, round_index, arm)
                    coefficients[arm] = (f, g)
                    evaluations[arm] = evaluation
                    if bracket is not None:
                        print(json.dumps({
                            "round": round_index + 1, "arm": arm,
                            "bracket": True,
                            "left_shooting": bracket["left"]["evaluation"]
                                ["pair"]["shooting_mismatch"],
                            "right_shooting": bracket["right"]["evaluation"]
                                ["pair"]["shooting_mismatch"],
                        }), flush=True)
                        break
                if bracket is not None:
                    break
                posterior.update(
                    arm, transformed_loss(evaluations[arm].score))
                pulls[arm] += 1
                best = min(
                    (i for i, e in enumerate(evaluations) if e.valid),
                    key=lambda i: evaluations[i].score, default=None)
                if best is not None:
                    pair = evaluations[best].pair
                    print(json.dumps({
                        "round": round_index + 1,
                        "arm": arm,
                        "best_arm": best,
                        "best_score": evaluations[best].score,
                        "objective": evaluations[best].objective_kind,
                        "distance": pair.distance if pair else None,
                        "shooting_mismatch": (
                            pair.shooting_mismatch if pair else None),
                        "source_b": pair.source_b if pair else None,
                        "target_b": pair.target_b if pair else None,
                    }), flush=True)
                _write_checkpoint(
                    args.output, args, coefficients, evaluations, pulls,
                    bracket)

    _write_checkpoint(
        args.output, args, coefficients, evaluations, pulls, bracket)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
