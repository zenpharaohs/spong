"""Large random ensembles: does the new certification hold up at scale?

    python scripts/ensemble.py --cases 400 --mode random
    python scripts/ensemble.py --cases 200 --mode directed
    python scripts/ensemble.py --cases 400 --mode random --resume

WHY THIS AND NOT qualify.py
---------------------------
`qualify.py` is the SOUNDNESS harness: invariants, worker determinism, and
agreement with the Python reference.  Its reference leg is deliberately
expensive, which caps the case count.  This is the OUTCOME harness: no
oracle, no reference, just certified_compute on many models, recording what
happened.  Run both -- qualify at modest N for soundness, this at large N
for statistics.  Neither replaces the other.

WHAT IT MEASURES
----------------
1. Refusal taxonomy.  How many certify, and for those that do not, WHICH
   condition refused.  The specific thing to watch is whether
   `unstable_endpoint_unresolved` reappears: it was the whole capture-class
   refusal and the directed run at N=20 showed zero.
2. Route census.  Every certified capture should read `exact_merge_tree`.
   A `exact_level_tube` or `strictly_convex_ball` anywhere is the slack
   ladder firing, and it is the one thing that would block deleting it.
3. Escalation.  geometry_level and attempts per case -- escalation is the
   dominant cost, so its frequency is the cost story.
4. Cost.  Per-case seconds against degree and critical-point count.

Results stream to JSONL as they complete, so a long run is inspectable
mid-flight and `--resume` skips seeds already recorded.  Seeds follow the
same protocol as qualify.py, so any case is reproducible on its own.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("SPONG_ENGINE", "native")

from spong import portrait, sturm                            # noqa: E402
from qualify import directed_model, random_model             # noqa: E402


def one_case(m, spec, seed, case):
    started = time.perf_counter()
    record = {"case": case, "seed": seed, "spec": spec}
    try:
        p = portrait.certified_compute(m)
    except Exception as exc:                    # noqa: BLE001 - recorded
        record.update(status="ERROR", error=f"{type(exc).__name__}: {exc}",
                      seconds=time.perf_counter() - started)
        return record

    top = p.ledger["topology"]
    e = p.enumeration
    captures = collections.Counter()
    escapes = collections.Counter()
    fallback = []
    for end in top.get("unstable_ends", ()):
        label = (end.get("method") if end.get("certified")
                 else f"UNCERTIFIED:{end.get('reason')}")
        if end["kind"] == "finite_capture":
            captures[label] += 1
            if end.get("certified") and label != "exact_merge_tree":
                fallback.append({"branch": end["branch"], "method": label})
        elif end["kind"] == "infinity_escape":
            escapes[label] += 1

    record.update(
        status=top.get("status"),
        reason=top.get("resolution_reason"),
        geometry_level=top.get("geometry_level"),
        attempts=len(top.get("attempts", ()) or ()),
        forbidden=top.get("forbidden_count"),
        ambiguous=top.get("ambiguous_count"),
        n_critical=len(e.points),
        n_min=len(e.minima),
        n_saddle=len(e.saddles),
        morse=bool(e.morse),
        psi_positive=bool(e.psi_positive),
        alternates=bool(e.alternates),
        terms=dict(collections.Counter(
            f"{br.kind}/{br.term}" for br in p.branches)),
        captures=dict(captures),
        escapes=dict(escapes),
        capture_fallbacks=fallback,
        seconds=time.perf_counter() - started,
    )
    return record


def _run_case(task):
    """Build and run one case in a worker process.

    Takes the SEED, not the model: spong models carry Fractions and the
    generators are deterministic, so shipping four ints and rebuilding is
    cheaper and safer than pickling a model across a process boundary.

    ``stub`` selects SPONG_STUB_MODE for this task, so the two arms of an
    A/B can share one pool: local.stub_mode() reads the environment on every
    call, and a reused worker process is re-pointed before each task.  None
    leaves the inherited environment alone.
    """
    mode, max_degree, seed, case, stub = task
    if stub is not None:
        os.environ["SPONG_STUB_MODE"] = stub
    generate = directed_model if mode == "directed" else random_model
    built = generate(random.Random(seed), max_degree)
    if built is None or built[0] is None:
        return None
    m, spec = built
    record = one_case(m, spec, seed, case)
    record["stub_mode"] = os.environ.get("SPONG_STUB_MODE", "grid")
    return record


def _predicted_seconds(paths):
    """Recorded seconds per SEED, the maximum over every file given.

    Seeds are deterministic, so a previous run predicts each case's cost.
    Keyed by seed rather than case index so a run with a different --cases
    still matches; the maximum over files lets a grid run and a jet run
    predict together.
    """
    cost = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seconds = float(r.get("seconds", 0.0))
                cost[r["seed"]] = max(cost.get(r["seed"], 0.0), seconds)
    return cost


def _outputs(args, stubs):
    """One results file per stub arm.

    With a single arm the path is exactly as before.  With several, a
    '{stub}' in --out is filled in, or else '-<stub>' is appended to the
    stem, so the arms can never resume over one another.
    """
    base = Path(args.out or (REPO / "out" /
                             f"ensemble-{args.mode}-d{args.max_degree}.jsonl"))
    if stubs == [None]:
        return {None: base}
    outs = {}
    for stub in stubs:
        if "{stub}" in str(base):
            outs[stub] = Path(str(base).replace("{stub}", stub))
        else:
            outs[stub] = base.with_name(f"{base.stem}-{stub}{base.suffix}")
    return outs


def _summarize(records, out, elapsed):
    print(f"\n=== {len(records)} cases in {elapsed:.0f}s   ->  {out}")
    status = collections.Counter(r["status"] for r in records)
    for key, count in status.most_common():
        print(f"    {key:<20s} {count}")
    reasons = collections.Counter(
        r.get("reason") for r in records if r["status"] != "certified")
    if reasons:
        print("  refusal reasons:")
        for key, count in reasons.most_common():
            print(f"    {str(key):<32s} {count}")
    captures = collections.Counter()
    escapes = collections.Counter()
    for r in records:
        captures.update(r.get("captures", {}))
        escapes.update(r.get("escapes", {}))
    print("  capture routes:")
    for key, count in captures.most_common():
        print(f"    {key:<40s} {count}")
    print("  escape routes:")
    for key, count in escapes.most_common():
        print(f"    {key:<40s} {count}")
    levels = collections.Counter(
        r.get("geometry_level") for r in records)
    print(f"  geometry levels: {dict(sorted(levels.items(), key=str))}")
    fallbacks = [r for r in records if r.get("capture_fallbacks")]
    if fallbacks:
        print(f"\n  CAPTURE FALLBACKS in {len(fallbacks)} cases -- the merge "
              "tree declined where it should have decided:")
        for r in fallbacks[:20]:
            print(f"    case {r['case']} seed {r['seed']}: "
                  f"{r['capture_fallbacks']}")
    else:
        print("\n  No capture fallback anywhere: the slack ladder and the "
              "convex ball are dead on this ensemble.")
    bad = [r for r in records
           if not (r.get("morse", True) and r.get("psi_positive", True)
                   and r.get("alternates", True))]
    if bad:
        print(f"  NON-MORSE / NON-POSITIVE / NON-ALTERNATING: "
              f"{[r['seed'] for r in bad]}")
    slowest = sorted(records, key=lambda r: -r["seconds"])[:5]
    print("  slowest cases:")
    for r in slowest:
        print(f"    {r['seconds']:8.1f}s  seed {r['seed']}  {r['spec']}  "
              f"{r['status']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="random")
    ap.add_argument("--seed", type=int, default=20260806,
                    help="master seed; per-case seeds derive from it")
    ap.add_argument("--case-seed", type=int, default=None,
                    help="run ONE case from its own recorded seed "
                         "(as printed in the JSONL) and write nothing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing results file")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many cases in parallel processes; "
                         "cases are independent, so this scales far better "
                         "than spong's per-portrait threading")
    ap.add_argument("--order-from", action="append", default=[],
                    metavar="JSONL",
                    help="schedule longest-first using the seconds recorded "
                         "per seed in a previous run (repeatable).  Wall "
                         "time is floored by the slowest case; starting it "
                         "first guarantees the floor instead of hoping for "
                         "it")
    ap.add_argument("--stub-modes", default=None, metavar="grid,jet",
                    help="run these SPONG_STUB_MODE arms in ONE pool, each "
                         "to its own file.  Each arm is floored by its own "
                         "slowest case, so concurrent arms cost about the "
                         "max of the floors rather than their sum")
    args = ap.parse_args()

    if args.jobs > 1:
        # Do NOT stack process parallelism on top of thread parallelism:
        # jobs x SPONG_WORKERS threads on a machine with fewer cores thrashes
        # and the exact arithmetic is already the bottleneck.  Across cases
        # the work is embarrassingly parallel and load-balances itself, which
        # per-portrait threading cannot do (one dominant branch caps it).
        os.environ["SPONG_WORKERS"] = "1"

    generate = directed_model if args.mode == "directed" else random_model

    if args.case_seed is not None:
        # Reproduce a single case from the seed printed in the JSONL.  No
        # file is touched: a one-case rerun must never be able to clobber
        # an ensemble that took half an hour to produce.
        built = generate(random.Random(args.case_seed), args.max_degree)
        if built is None or built[0] is None:
            print("generator declined this seed")
            return 1
        m, spec = built
        record = one_case(m, spec, args.case_seed, -1)
        print(json.dumps(record, indent=2))
        return 0

    stubs = ([s.strip() for s in args.stub_modes.split(",") if s.strip()]
             if args.stub_modes else [None])
    outs = _outputs(args, stubs)
    done = {}
    for stub, out in outs.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        done[stub] = set()
        if args.resume and out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    done[stub].add(json.loads(line)["case"])
            print(f"resuming: {len(done[stub])} cases already recorded "
                  f"in {out}")
        elif out.exists():
            if not args.force:
                print(f"{out} already exists.  Use --resume to continue it, "
                      f"--force to overwrite, or --out to write elsewhere.")
                return 1
            out.unlink()

    rng = random.Random(args.seed)
    started = time.perf_counter()
    records = {stub: [] for stub in stubs}
    tasks = []
    for case in range(args.cases):
        seed = rng.randrange(2 ** 31)
        for stub in stubs:
            if case not in done[stub]:
                tasks.append((args.mode, args.max_degree, seed, case, stub))

    if args.order_from:
        # LONGEST FIRST.  Parallel wall time is floored by the slowest case:
        # the directed ensemble at N=200 took 887 s with a single 842 s case,
        # i.e. it was one case long and the other 199 finished in its shadow.
        # Case order only met that floor by luck -- the slow case happened to
        # start early.  Seeds are deterministic, so a previous run predicts
        # every cost; unrecorded seeds get the median, a neutral guess.  The
        # sort is stable, so ties keep case order.
        cost = _predicted_seconds(args.order_from)
        known = sorted(cost.values())
        guess = known[len(known)//2] if known else 0.0
        tasks.sort(key=lambda t: -cost.get(t[2], guess))
        print(f"longest-first from {len(cost)} recorded seeds; first up: "
              + ", ".join(f"case {t[3]}" + (f"/{t[4]}" if t[4] else "")
                          + f" ({cost.get(t[2], guess):.0f}s)"
                          for t in tasks[:3]))

    def emit(record):
        if record is None:
            return
        stub = record.get("stub_mode") if stubs != [None] else None
        records[stub].append(record)
        with outs[stub].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        flag = ""
        if record.get("capture_fallbacks"):
            flag = "   <-- CAPTURE FALLBACK"
        if record["status"] == "ERROR":
            flag = "   <-- ERROR"
        reason = record.get("reason") or (
            "" if record["status"] == "certified" else "(no reason given)")
        tag = f"{record.get('stub_mode', ''):>4s} " if stubs != [None] else ""
        print(f"[{record['case']:4d}] {tag}{record['status']:<16s} "
              f"{reason:<32s} "
              f"{record['seconds']:7.1f}s  {record['spec']}{flag}",
              flush=True)

    if args.jobs > 1:
        import concurrent.futures as cf
        # submit + as_completed, NOT pool.map: map yields in SUBMISSION
        # order, so one slow case blocks the display and the JSONL write
        # for every case behind it -- the workers stay busy but nothing is
        # visible or recoverable until the straggler lands.  Records carry
        # their own case index and seed, so completion order is fine.
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_case, task) for task in tasks]
            for future in cf.as_completed(futures):
                emit(future.result())
    else:
        for task in tasks:
            emit(_run_case(task))

    elapsed = time.perf_counter() - started
    for stub in stubs:
        if stub is not None:
            print(f"\n##### stub mode: {stub}")
        _summarize(records[stub], outs[stub], elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
