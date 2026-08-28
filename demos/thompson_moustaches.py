"""Compare equal and exact continuous-Bernoulli allocation on a portrait."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from demos import cb_sampler
from demos import initializers
from demos import optimizer_moustaches as gallery
from demos import optimizers as opt
from demos import thompson
from spong import portrait, render, zoo


def _schedule(name, base_lr, horizon, warmup_steps=100):
    if name == "constant":
        return float(base_lr)
    if name == "cosine":
        return opt.cosine_schedule(base_lr, horizon)
    # A rested arm must have the same optimizer prefix no matter how many
    # allocation rounds the caller later requests.  Deriving warmup from the
    # experiment horizon changed the arm itself when only the budget changed.
    return opt.inverse_sqrt_schedule(base_lr, warmup_steps)


def _state(method, start, learning_rate):
    if method == "sgd":
        return opt.SGDState(start, learning_rate)
    if method == "sgd-momentum":
        return opt.SGDState(
            start, learning_rate, momentum=0.9, nesterov=True)
    if method == "adam":
        return opt.AdamState(start, learning_rate)
    raise ValueError(method)


def _problem(f, g, starts, method, batch_size, schedule, horizon, seed,
             design, distribution="uniform01", samples=None,
             learning_rate=None, warmup_steps=100):
    states = []
    gradients = []
    base_lr = (gallery.DEFAULT_LR[method] if learning_rate is None
               else float(learning_rate))
    if not np.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    lr = _schedule(schedule, base_lr, horizon, warmup_steps)
    design_code = 0 if design == "low-discrepancy" else 1
    for index, start in enumerate(starts):
        states.append(_state(method, start, lr))
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, index, batch_size, design_code]))
        gradients.append(opt.BatchGradient(
            f, g, batch_size=batch_size, rng=rng,
            distribution=distribution, samples=samples))
    return states, gradients


def _equal(states, gradients, loss, rounds, chunk_steps, should_stop=None):
    class RoundRobin:
        def __init__(self, n):
            self.n = n
            self.next = 0

        def draw_all(self):
            draws = np.ones(self.n)
            draws[self.next] = 0.0
            self.next = (self.next+1) % self.n
            return draws

        def update(self, arm, observation):
            pass

    # The allocator itself supplies the first round-robin sweep.
    bank = RoundRobin(len(states))
    return thompson.allocate(
        states, gradients, loss, bank, rounds, chunk_steps,
        should_stop=should_stop)


def _overlays(result, color):
    largest = max(int(np.max(result.allocations)), 1)
    overlays = []
    for trajectory, pulls in zip(result.trajectories, result.allocations):
        overlays.append({
            "Y": gallery._thin(trajectory),
            "color": color,
            "width": 0.55+1.2*pulls/largest,
            "opacity": 0.08+0.72*np.sqrt(pulls/largest),
            "mark_start": True,
            "mark_end": True,
        })
    return overlays


def _allocation_summary(result):
    counts = result.allocations
    order = np.argsort(counts)[::-1]
    total = int(np.sum(counts))
    successes = sum(reason == "zero_loss"
                    for reason in result.termination_reasons)
    return {
        "pulls": total,
        "executed_optimizer_steps": int(np.sum(result.executed_steps)),
        "terminated_arms": int(np.count_nonzero(result.terminated)),
        "zero_loss_arms": int(successes),
        "allocation_quantiles": dict(zip(
            ("0", "0.25", "0.5", "0.75", "1"),
            map(float, np.quantile(counts, [0, .25, .5, .75, 1])))),
        "largest_share": float(counts[order[0]]/total),
        "top_5_share": float(np.sum(counts[order[:5]])/total),
        "most_allocated_arms": list(map(int, order[:10])),
    }


def work_loss_histogram(equal, adaptive, bins=24):
    """Shared log-loss bins weighted by optimizer steps actually executed.

    Each allocation event contributes its executed chunk steps at the exact
    post-chunk loss.  Zero and nonfinite losses are reported separately since
    neither belongs on a logarithmic axis.  Common edges make policy heights
    directly comparable.
    """
    bins = int(bins)
    if bins < 1:
        raise ValueError("histogram bins must be positive")
    positive = []
    for result in (equal, adaptive):
        losses = np.asarray(result.allocation_losses, dtype=float)
        steps = np.asarray(result.allocation_steps, dtype=np.int64)
        mask = (steps > 0) & np.isfinite(losses) & (losses > 0.0)
        positive.extend(losses[mask].tolist())
    if positive:
        log_lo = float(np.log10(min(positive)))
        log_hi = float(np.log10(max(positive)))
        if not log_hi > log_lo:
            log_lo -= 0.5
            log_hi += 0.5
    else:
        log_lo, log_hi = -1.0, 0.0
    edges = np.power(10.0, np.linspace(log_lo, log_hi, bins+1))

    def policy(result):
        losses = np.asarray(result.allocation_losses, dtype=float)
        steps = np.asarray(result.allocation_steps, dtype=np.int64)
        finite = np.isfinite(losses)
        counts, _ = np.histogram(
            losses[(steps > 0) & finite & (losses > 0.0)],
            bins=edges,
            weights=steps[(steps > 0) & finite & (losses > 0.0)])
        return {
            "steps": [int(x) for x in counts],
            "zero_steps": int(np.sum(steps[(steps > 0) & finite
                                            & (losses <= 0.0)])),
            "nonfinite_steps": int(np.sum(steps[(steps > 0) & ~finite])),
            "total_steps": int(np.sum(steps)),
        }

    return {
        "weight": "executed_optimizer_steps",
        "loss_sample": "exact_post_chunk_loss",
        "scale": "log10",
        "edges": [float(x) for x in edges],
        "equal": policy(equal),
        "thompson": policy(adaptive),
    }


def compare_allocators(m, f, g, view, *, starts=100, rounds=20000,
                       chunk_steps=10, batch_size=32, method="adam",
                       schedule="inverse-sqrt", design="low-discrepancy",
                       seed=1729, distribution="uniform01", samples=None,
                       learning_rate=None, cb_library=None,
                       should_stop=None, warmup_steps=100):
    """Run equal and exact Thompson allocation on an arbitrary portrait.

    ``m`` supplies only the exact loss used for the posterior observation.
    The minibatch oracle is reconstructed from ``f``, ``g`` and the same
    input law that supplied the portrait moments.  The two policies receive
    identical starts and per-arm random streams.
    """
    starts = int(starts)
    rounds = int(rounds)
    chunk_steps = int(chunk_steps)
    batch_size = int(batch_size)
    if starts <= 0 or rounds < starts:
        raise ValueError("rounds must be at least starts > 0")
    if chunk_steps <= 0 or batch_size <= 0:
        raise ValueError("chunk_steps and batch_size must be positive")
    warmup_steps = int(warmup_steps)
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")
    if method not in {"sgd", "sgd-momentum", "adam"}:
        raise ValueError(f"unsupported optimizer {method!r}")
    if schedule not in {"constant", "cosine", "inverse-sqrt"}:
        raise ValueError(f"unsupported schedule {schedule!r}")
    if design not in gallery.DESIGNS:
        raise ValueError(f"unsupported initialization design {design!r}")

    points = (initializers.low_discrepancy(starts, view)
              if design == "low-discrepancy"
              else initializers.blue_noise(starts, view, seed=seed))
    # Schedules are measured in an arm's own optimizer steps.  Calibrate the
    # horizon to its equal-allocation budget; a favored Thompson arm may run
    # beyond it, for which inverse-sqrt remains well defined.
    horizon = int(np.ceil(rounds/starts))*chunk_steps

    common = dict(
        f=f, g=g, starts=points, method=method, batch_size=batch_size,
        schedule=schedule, horizon=horizon, seed=seed, design=design,
        distribution=distribution, samples=samples,
        learning_rate=learning_rate, warmup_steps=warmup_steps)
    equal_states, equal_gradients = _problem(**common)
    equal = _equal(
        equal_states, equal_gradients, m.L, rounds, chunk_steps,
        should_stop=should_stop)

    ts_states, ts_gradients = _problem(**common)
    with cb_sampler.ContinuousBernoulliBank(
            starts, seed=seed, library=cb_library) as bank:
        adaptive = thompson.allocate(
            ts_states, ts_gradients, m.L, bank, rounds, chunk_steps,
            should_stop=should_stop)
        library_path = str(bank.library_path)

    return {
        "starts": points,
        "equal": equal,
        "thompson": adaptive,
        "horizon": horizon,
        "warmup_steps": warmup_steps,
        "learning_rate": (gallery.DEFAULT_LR[method]
                          if learning_rate is None else float(learning_rate)),
        "library_path": library_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact continuous-Bernoulli Thompson moustaches")
    parser.add_argument("--zoo", choices=zoo.names(), default="quadratic-stiff")
    parser.add_argument("--method",
                        choices=("sgd", "sgd-momentum", "adam"),
                        default="adam")
    parser.add_argument("--design",
                        choices=gallery.DESIGNS, default="low-discrepancy")
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--schedule",
                        choices=("constant", "cosine", "inverse-sqrt"),
                        default="inverse-sqrt")
    parser.add_argument(
        "--learning-rate", type=float,
        help="base learning rate (default depends on optimizer)")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--cb-library", type=Path,
        help="shared cb_core library (or set CB_CORE_LIBRARY)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/thompson_moustaches"))
    args = parser.parse_args(argv)
    if args.starts <= 0 or args.rounds < args.starts:
        parser.error("--rounds must be at least --starts > 0")
    if args.chunk_steps <= 0 or args.batch_size <= 0:
        parser.error("--chunk-steps and --batch-size must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    case = zoo.get(args.zoo)
    m = gallery._zoo_model(case)
    p = portrait.certified_compute(m, view=case.default_view)
    comparison = compare_allocators(
        m, case.f, case.g, p.view, starts=args.starts,
        rounds=args.rounds, chunk_steps=args.chunk_steps,
        batch_size=args.batch_size, method=args.method,
        schedule=args.schedule, design=args.design, seed=args.seed,
        distribution=case.moment_dist, learning_rate=args.learning_rate,
        cb_library=args.cb_library)
    equal = comparison["equal"]
    adaptive = comparison["thompson"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.name}_{args.design}_{args.method}"
    panels = []
    outcome_summaries = {}
    for name, result, color in (
            ("equal", equal, "#707070"),
            ("thompson", adaptive, gallery.COLORS[args.method])):
        filename = f"{stem}_{name}.svg"
        render.save(render.plane_view(
            p, view=p.view, width=800, height=600, n_levels=32, n_grid=801,
            overlays=_overlays(result, color),
            title=f"{case.name}: {args.method}, {name} allocation"),
            str(args.output_dir/filename))
        summary = _allocation_summary(result)
        outcome = gallery._summary(m, p, result.trajectories, p.box)
        outcome_summaries[name] = outcome
        loss_quantiles = outcome["final_loss_quantiles"]
        best_loss = loss_quantiles["0.0"] if loss_quantiles else float("inf")
        median_loss = loss_quantiles["0.5"] if loss_quantiles else float("inf")
        panels.append(
            f"<section><h2>{html.escape(name.title())} allocation</h2>"
            f'<object data="{html.escape(filename)}" '
            f'type="image/svg+xml"></object>'
            f"<p>largest share={summary['largest_share']:.3f}; "
            f"top-five share={summary['top_5_share']:.3f}; "
            f"best final loss={best_loss:.5g}; "
            f"median final loss={median_loss:.5g}</p></section>")

    report = {
        "format": "spong-thompson-moustaches-v1",
        "zoo": case.name,
        "method": args.method,
        "design": args.design,
        "observation": "L/(1+L)",
        "selection": "minimum exact continuous-Bernoulli posterior draw",
        "capture_used_for_allocation": False,
        "configuration": {
            "starts": args.starts,
            "rounds": args.rounds,
            "chunk_steps": args.chunk_steps,
            "total_optimizer_steps": args.rounds*args.chunk_steps,
            "batch_size": args.batch_size,
            "schedule": args.schedule,
            "warmup_steps": comparison["warmup_steps"],
            "learning_rate": comparison["learning_rate"],
            "seed": args.seed,
            "cb_library": comparison["library_path"],
        },
        "equal": _allocation_summary(equal),
        "thompson": _allocation_summary(adaptive),
        "work_loss_histogram": work_loss_histogram(equal, adaptive),
        "posthoc_outcomes": outcome_summaries,
        "thompson_allocations": list(map(int, adaptive.allocations)),
        "thompson_final_observations": [
            (float(values[-1]) if values else None)
            for values in adaptive.observations],
    }
    (args.output_dir/f"{stem}.json").write_text(
        json.dumps(report, indent=2)+"\n")
    index = args.output_dir/f"{stem}.html"
    index.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG Thompson allocation: {html.escape(case.name)}</title>
<style>
body {{ font-family:system-ui,sans-serif; margin:24px; background:#fafafa; }}
main {{ display:grid; grid-template-columns:repeat(2,minmax(520px,1fr)); gap:20px; }}
section {{ background:white; border:1px solid #ddd; padding:12px; }}
object {{ width:100%; aspect-ratio:4/3; }}
</style>
<h1>Equal versus Thompson allocation on {html.escape(case.name)}</h1>
<p>Both panels spend exactly {args.rounds*args.chunk_steps:,} optimizer
steps. The Thompson scheduler sees only L/(1+L) after each short continuation;
capture and the certified geometry are used only to interpret the result.</p>
<main>{''.join(panels)}</main>
""")
    print(index)
    print(args.output_dir/f"{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
