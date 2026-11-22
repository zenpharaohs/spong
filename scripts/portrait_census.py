"""Census of portrait classes: which have we seen, and which are new?

    python scripts/portrait_census.py --source zoo
    python scripts/portrait_census.py --source random --cases 400 --jobs 8
    python scripts/portrait_census.py --source inspector --deg-g 13 \
        --moments normal01 --cases 200 --jobs 8 --known out/census-random.jsonl

Each case gets the two-layer signature of spong.signature: the backbone
(merge tree, exact, no tracing) and the attachments (where each unstable
branch ends).  The census then says how many classes there are, how often
each occurs, and which cases belong to classes not seen before -- in this
run or in the ``--known`` files of earlier runs.

WHAT GETS TRACED

A portrait is computed only when the backbone leaves a branch undecided
(``--trace open``, the default).  When every branch is forced the class
follows from the exact tree alone and the case is passed over in
milliseconds.  ``--trace always`` computes every portrait and checks the
forced predictions against its certified ends; that is the audit of the
shortcut, and any disagreement is printed and recorded.  ``--trace never``
is layer 1 only.

``--assume-funnel`` also passes over the funnel branches (one minimum
beyond the saddle, then an open backbone end).  They have always been
captures, but that is the far-field funnel's theorem and not the merge
tree's, so the default traces them.  Run ``--trace always --assume-funnel``
to measure the assumption instead of taking it.

SEEDS

random/directed use scripts/qualify.py's generators with ensemble.py's
seed protocol, so a seed printed here reproduces with
``scripts/ensemble.py --case-seed``.  inspector replays the page's own
"draw random" (same generator, same rounding), so a case found here opens
in the inspector by typing its seed, degrees and distributions.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spong import model, portrait, signature, sturm, zoo      # noqa: E402

_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------
# the inspector's "draw random", replayed
# --------------------------------------------------------------------------

def _imul(a, b):
    return (a * b) & _M32


def _mulberry32(seed):
    state = [seed & _M32]

    def rnd():
        state[0] = (state[0] + 0x6D2B79F5) & _M32
        x = state[0]
        x = _imul(x ^ (x >> 15), x | 1)
        x = (x ^ ((x + _imul(x ^ (x >> 7), x | 61)) & _M32)) & _M32
        return ((x ^ (x >> 14)) & _M32) / 4294967296
    return rnd


def _js_fixed6(x):
    # JavaScript's +x.toFixed(6): round half away from zero on the decimal
    # expansion of the double.  Python's format rounds the same expansion.
    return float(f"{x:.6f}")


def _inspector_coeffs(rnd, deg, dist):
    def draw():
        if dist == "normal":
            u, v = max(rnd(), 1e-12), rnd()
            return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)
        if dist == "integer":
            return float(math.floor(rnd() * 8 + 0.5) - 4)
        return 2 * rnd() - 1
    c = [draw() for _ in range(deg + 1)]
    if abs(c[deg]) < 0.05:
        c[deg] = -1.0 if c[deg] < 0 else 1.0
    return c if dist == "integer" else [_js_fixed6(x) for x in c]


def inspector_case(seed, deg_f, deg_g, dist):
    """(f, g) exactly as the inspector's draw-random button produces them.

    ``deg_f = 0`` means the page's "same f and g" box is ticked.
    """
    same = deg_f == 0
    df = deg_g if same else deg_f
    rnd = _mulberry32(int(seed * 2654435761 + df * 1000003 + deg_g * 10007
                          + len(dist)))
    g = _inspector_coeffs(rnd, deg_g, dist)
    if abs(g[0]) < 0.05:
        g[0] = 1.0 if dist == "integer" else 0.5
    f = list(g) if same else _inspector_coeffs(rnd, df, dist)
    return f, g


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------

def _build(task):
    source = task["source"]
    if source == "zoo":
        z = zoo.get(task["name"])
        f, g = [float(x) for x in z.f], [float(x) for x in z.g]
        mu = zoo.moments_for(z, 2 * max(len(f), len(g)) - 1)
        return model.build(f, g, mu), f"zoo {task['name']}"
    if source == "inspector":
        f, g = inspector_case(task["seed"], task["deg_f"], task["deg_g"],
                              task["dist"])
        mu = getattr(model, "moments_" + task["moments"])(
            2 * max(len(f), len(g)) - 1)
        same = "f=g" if task["deg_f"] == 0 else f"deg f {task['deg_f']}"
        return model.build(f, g, mu), (
            f"inspector seed {task['seed']} deg g {task['deg_g']} {same} "
            f"{task['dist']} coefficients, {task['moments']}")
    from qualify import directed_model, random_model
    generate = directed_model if source == "directed" else random_model
    return generate(random.Random(task["seed"]), task["max_degree"])


def run_case(task):
    started = time.perf_counter()
    record = {k: task[k] for k in ("source", "case", "seed", "name")
              if k in task}
    try:
        m, spec = _build(task)
        if m is None:
            return dict(record, status="generator_declined", seconds=0.0)
        record["spec"] = spec
        e = sturm.enumerate_critical_points(m)
        if not e.psi_positive or not e.morse:
            return dict(record, status=("psi_not_positive"
                                        if not e.psi_positive
                                        else "non_morse"),
                        seconds=time.perf_counter() - started)
        funnel = task["assume_funnel"]
        sig = signature.of_model(m, e, funnel=funnel)
        trace = task["trace"]
        if trace == "always" or (trace == "open" and sig.key2 is None):
            p = portrait.certified_compute(m)
            top = p.ledger["topology"]
            traced = signature.of_portrait(p)
            record.update(
                portrait_status=top.get("status"),
                portrait_reason=top.get("resolution_reason"),
                forced_disagreements=signature.forced_disagreements(
                    p, funnel=funnel))
            if sig.key2 is not None and sig.key2 != traced.key2 \
                    and not traced.grades.get("unknown"):
                record["forced_key2"] = sig.key2
            sig = traced
        record.update(sig.as_dict(), status="ok")
    except Exception as exc:                    # noqa: BLE001 - recorded
        record.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    record["seconds"] = time.perf_counter() - started
    return record


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _load(path):
    out = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _class_key(r):
    # Layer 2 when it is known and complete, otherwise the backbone alone
    # with a mark saying so: an undecided portrait is not the same class as
    # a decided one that happens to share its backbone.
    if r.get("key2") and not r.get("grades", {}).get("unknown"):
        return r["key2"]
    return r["key1"] + "+?"


def summarize(records, known, top=25):
    ok = [r for r in records if r.get("status") == "ok"]
    n = len(ok)
    print(f"\n{len(records)} cases, {n} with a signature")
    other = collections.Counter(r.get("status") for r in records
                                if r.get("status") != "ok")
    for status, count in other.most_common():
        print(f"    {count:5d}  {status}")
    if not ok:
        return
    bad = [r for r in ok if r.get("forced_disagreements")]
    if bad:
        print(f"\n!!! {len(bad)} cases where a certified end contradicts a "
              "forced prediction:")
        for r in bad:
            print(f"    {r.get('spec')}: {r['forced_disagreements']}")
    traced = sum(bool(r.get("traced")) for r in ok)
    print(f"portraits computed: {traced} of {n} "
          f"({n - traced} passed over on the backbone alone)")

    layer1 = collections.Counter(r["key1"] for r in ok)
    classes = collections.defaultdict(list)
    for r in ok:
        classes[_class_key(r)].append(r)
    singletons = sum(len(v) == 1 for v in classes.values())
    print(f"backbone classes: {len(layer1)}    "
          f"portrait classes: {len(classes)}")
    # Good-Turing: the share of the sampler's mass on classes not yet seen
    # is estimated by the share of cases whose class occurred exactly once.
    print(f"classes seen once: {singletons}   "
          f"estimated unseen mass: {singletons / n:.3f}")

    known_keys = {_class_key(r) for r in known if r.get("status") == "ok"}
    if known:
        fresh = [k for k in classes if k not in known_keys]
        print(f"classes not in --known ({len(known_keys)} classes): "
              f"{len(fresh)}")

    def interest(item):
        key, members = item
        g = members[0].get("grades", {})
        return (key in known_keys, len(members),
                -g.get("nonadjacent", 0), -g.get("open_branches", 0),
                -g.get("eligible_connections", 0), -g.get("n_critical", 0))

    print(f"\nrarest first (up to {top}); "
          "count, [open/funnel/eligible/nonadjacent], class, an example")
    for key, members in sorted(classes.items(), key=interest)[:top]:
        r = members[0]
        g = r.get("grades", {})
        mark = " " if key in known_keys else "*" if known else " "
        text = r["backbone"] + (" | " + r["attachments"]
                                if r.get("attachments") else "")
        print(f" {mark}{len(members):4d}  [{g.get('open_branches', 0)}/"
              f"{g.get('funnel_branches', 0)}/"
              f"{g.get('eligible_connections', 0)}/"
              f"{g.get('nonadjacent', '-')}]  {key}  {text}")
        print(f"            e.g. {r.get('spec')}"
              + (f"  (seed {r['seed']})" if "seed" in r
                 and "seed" not in str(r.get("spec")) else ""))
    print("\nmost common")
    for key, members in sorted(classes.items(),
                               key=lambda kv: -len(kv[1]))[:8]:
        r = members[0]
        text = r["backbone"] + (" | " + r["attachments"]
                                if r.get("attachments") else "")
        print(f"  {len(members):4d}  {key}  {text}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="random",
                    choices=("zoo", "random", "directed", "inspector"))
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--max-degree", type=int, default=5,
                    help="random/directed: qualify.py's degree ceiling")
    ap.add_argument("--seed", type=int, default=20260806,
                    help="random/directed: master seed, as ensemble.py; "
                         "inspector: the FIRST page seed, then seed+1, ...")
    ap.add_argument("--deg-g", type=int, default=5)
    ap.add_argument("--deg-f", type=int, default=0,
                    help="inspector: degree of f, or 0 for f = g")
    ap.add_argument("--dist", default="uniform",
                    choices=("uniform", "normal", "integer"),
                    help="inspector: coefficient distribution")
    ap.add_argument("--moments", default="uniform01",
                    choices=("uniform01", "normal01"))
    ap.add_argument("--trace", default="open",
                    choices=("never", "open", "always"))
    ap.add_argument("--assume-funnel", action="store_true")
    ap.add_argument("--known", action="append", default=[], metavar="JSONL",
                    help="earlier census output; its classes count as seen "
                         "(repeatable)")
    ap.add_argument("--summarize", action="append", default=[],
                    metavar="JSONL",
                    help="summarize existing output instead of running")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    known = [r for path in args.known for r in _load(path)]
    if args.summarize:
        summarize([r for path in args.summarize for r in _load(path)], known)
        return 0

    if args.jobs > 1:
        os.environ["SPONG_WORKERS"] = "1"     # as ensemble.py: do not stack

    common = {"source": args.source, "trace": args.trace,
              "assume_funnel": args.assume_funnel}
    tasks = []
    if args.source == "zoo":
        tasks = [dict(common, case=i, name=name)
                 for i, name in enumerate(zoo.names())]
    elif args.source == "inspector":
        tasks = [dict(common, case=i, seed=args.seed + i, deg_f=args.deg_f,
                      deg_g=args.deg_g, dist=args.dist, moments=args.moments)
                 for i in range(args.cases)]
    else:
        rng = random.Random(args.seed)
        tasks = [dict(common, case=i, seed=rng.randrange(2 ** 31),
                      max_degree=args.max_degree)
                 for i in range(args.cases)]

    out = Path(args.out) if args.out else (
        ROOT / "out" / f"census-{args.source}.jsonl")
    records = []
    if out.exists() and args.resume:
        records = _load(out)
        done = {r["case"] for r in records}
        tasks = [t for t in tasks if t["case"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with open(out, "a" if args.resume else "w") as handle:
        def emit(record):
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            tag = record.get("key2") or (record.get("key1", "") + "+?"
                                         if record.get("key1") else "")
            print(f"[{record['case']:4d}] {record['status']:<10s} "
                  f"{tag:<14s} {record['seconds']:7.1f}s  "
                  f"{record.get('spec', record.get('error', ''))}",
                  flush=True)

        if args.jobs > 1:
            import concurrent.futures as cf
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for future in cf.as_completed(
                        [pool.submit(run_case, t) for t in tasks]):
                    emit(future.result())
        else:
            for task in tasks:
                emit(run_case(task))

    print(f"\nwrote {out}  ({time.perf_counter() - started:.1f}s)")
    summarize(records, known)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
