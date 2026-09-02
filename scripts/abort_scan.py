#!/usr/bin/env python3
"""Scan a random population for aborting branches, in parallel across seeds.

    python scripts/abort_scan.py --n 500 --workers 10
    python scripts/abort_scan.py --n 500 --workers 10 --box default
    python scripts/abort_scan.py --df 1 --dg 13 --dist uniform01 --n 2000

Each seed builds f (df+1 coefficients) and g (dg+1) from U(-1,1) via
random.Random(seed) -- f drawn first, then g -- and computes the portrait
at geometry level 0 with no audit (branch fates are the question, not the
topology certificate).  --box legal (default) traces on the explorer's
legal box, which is where the abort classes live; --box default uses the
skeleton box, which masks most of them.

Output: one JSON line per aborting case in out/abort_scan-<tag>.jsonl,
with seed, the abort list (branch, kind, term, n, saddle_b), and seconds.
Resumable: seeds already in the output are skipped.  A summary histogram
of abort terms prints at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ["SPONG_WORKERS"] = "1"          # parallelism is across seeds

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ARGS = None


def _worker_init(args_dict):
    """macOS spawns workers by re-importing this module, so the parent's
    ARGS never arrives on its own -- each worker gets it here."""
    global ARGS
    ARGS = argparse.Namespace(**args_dict)


def scan(seed: int):
    from spong import atlas, model as mm, portrait
    rng = random.Random(seed)
    f = [rng.uniform(-1, 1) for _ in range(ARGS.df + 1)]
    g = [rng.uniform(-1, 1) for _ in range(ARGS.dg + 1)]
    mu = (mm.moments_uniform01 if ARGS.dist == "uniform01"
          else mm.moments_normal01)(2 * max(len(f), len(g)) - 1)
    t = time.perf_counter()
    try:
        m = mm.build(f, g, mu)
        if ARGS.box == "legal":
            bmax = atlas.legal_max_b(m)
            amax = bmax / math.sqrt(max(1, atlas.effective_degree(m)))
            view = (-amax, amax, -bmax, bmax)
        else:
            view = None
        p = portrait.compute(m, view=view, _skip_audit=True)
    except Exception as ex:
        return {"seed": seed, "error": f"{type(ex).__name__}: {ex}"[:120],
                "seconds": round(time.perf_counter() - t, 2)}
    aborts = [
        {"branch": i, "kind": br.kind, "term": br.term, "n": len(br.Y),
         "saddle_b": round(float(br.diag.get("saddle_b", 0.0)), 6),
         "end": [round(float(br.Y[-1][0]), 6), round(float(br.Y[-1][1]), 6)]}
        for i, br in enumerate(p.branches) if br.term.startswith("abort")]
    return {"seed": seed, "aborts": aborts,
            "seconds": round(time.perf_counter() - t, 2)}


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--df", type=int, default=1)
    ap.add_argument("--dg", type=int, default=13)
    ap.add_argument("--dist", choices=["uniform01", "normal01"],
                    default="uniform01")
    ap.add_argument("--box", choices=["legal", "default"], default="legal")
    ARGS = ap.parse_args()

    tag = f"d{ARGS.df}-{ARGS.dg}-{ARGS.dist}-{ARGS.box}"
    out = REPO / "out" / f"abort_scan-{tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["seed"])
    todo = [s for s in range(ARGS.start, ARGS.start + ARGS.n)
            if s not in seen]
    print(f"{tag}: {len(todo)} seeds to scan "
          f"({len(seen)} already in {out.name}), {ARGS.workers} workers")

    t0 = time.time()
    n_bad = 0
    terms: dict = {}
    with out.open("a") as fh, Pool(
            ARGS.workers, initializer=_worker_init,
            initargs=(vars(ARGS),)) as pool:
        for row in pool.imap_unordered(scan, todo):
            interesting = row.get("aborts") or row.get("error")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if interesting:
                n_bad += 1
                for ab in row.get("aborts", []):
                    key = (ab["kind"], ab["term"])
                    terms[key] = terms.get(key, 0) + 1
                print(json.dumps(row))
    print(f"\n{len(todo)} seeds in {time.time()-t0:.0f}s; "
          f"{n_bad} with aborts/errors")
    for (kind, term), n in sorted(terms.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {kind}/{term}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
