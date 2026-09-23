#!/usr/bin/env python3
"""Census of the pieces the tracer actually uses to build invariant manifolds.

WHY.  The planned checker needs a step for every construction the tracer can
use to piece a manifold together.  A static scan gives the candidates; this
counts which actually FIRE, so the checker's vocabulary is fixed by evidence.
The vocabulary is open by design -- there is always a worse example -- so the
census is a standing monitor, not a one-off inventory: its job is to see a new
construction before a user does.

TWO MODES.

  certified (default)   certified_compute on every case: terminations, zones,
                        launches, AND the endpoint certificate that closed
                        each branch, plus the verdict.  Expensive; the slowest
                        directed case dominates the wall time.

  --trace-only          portrait.compute with the audit skipped, on DIFFICULT
                        models only.  The dispatch decisions -- which zone,
                        which launch -- happen in the tracing, so certification
                        is not needed to see them, and rare constructions live
                        where the landscape is hard.  Difficulty is screened
                        first from EXACT data, before anything is traced:

                          stiffness   max over saddles of log10 |lam+/lam-|
                          A-range     log10 of max/min A(b) across the span of
                                      the critical points

                        A model below both thresholds costs one enumeration
                        and is skipped, so thousands can be surveyed.  Why A:
                        the scale of A is the villain in both directions --
                        large A makes the fast direction stiff, small A makes
                        a* = B/A huge (directed seed 953953598 has
                        a*(0) = -4.45e7).

Tallies, per branch or stub:
  termination   branch.term
  zones         piece labels in branch.diag["zones"]; "(unzoned)" for a
                branch the stiff dispatcher never touched
  launch        how each stub was built, and whether it was ready
  endpoint      (certified mode) the certificate that closed the branch
  verdict       (certified mode) status and reason
  difficulty    (trace-only) the stiffness and A-range bins surveyed

Seeds follow scripts/ensemble.py's protocol, so a case here is the same case
there.

    python scripts/piece_census.py --random 100 --directed 100 --jobs 16
    python scripts/piece_census.py --trace-only --random 2000 --directed 2000 \\
        --max-degree 7 --jobs 16
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

TALLIES = ("termination", "zones", "launch", "endpoint", "verdict",
           "difficulty")


def _build(spec, max_degree):
    from spong import model as mm, zoo
    from qualify import directed_model, random_model
    kind, key = spec
    if kind == "zoo":
        case = zoo.get(key)
        n = 2*max(len(case.f), len(case.g)) - 1
        moments = (mm.moments_normal01 if case.moment_dist == "normal01"
                   else mm.moments_uniform01)(n)
        return mm.build(list(case.f), list(case.g), moments)
    generate = directed_model if kind == "directed" else random_model
    built = generate(random.Random(key), max_degree)
    if built is None or built[0] is None:
        return None
    return built[0]


def _difficulty(m, e):
    """(log10 stiffness, log10 A-range) from exact data, before any tracing."""
    import numpy as np
    stiff = 0.0
    for q in e.saddles:
        lam = np.asarray(q.local.spectral.eigenvalues, dtype=float)
        if lam[0] < 0.0 < lam[1]:
            stiff = max(stiff, math.log10(abs(lam[1]/lam[0])))
    bs = [float(q.b) for q in e.points]
    lo, hi = min(bs), max(bs)
    pad = 0.05*max(1.0, hi - lo)
    A = np.array([float(m.A(float(b)))
                  for b in np.linspace(lo - pad, hi + pad, 256)])
    A = A[np.isfinite(A) & (A > 0.0)]
    arange = math.log10(A.max()/A.min()) if len(A) else 0.0
    return stiff, arange


def _launch_label(c):
    if "stub_jet" not in c and "continuation_ready" not in c:
        return "refused"
    if c.get("stub_jet") == 1.0:
        base = "jet"
    elif c.get("poincare_conditioned") == 1.0:
        base = "grid-poincare"
    else:
        base = "grid-centered"
    mods = []
    if c.get("centered_extension"):
        mods.append("centered_ext")
    if (c.get("flow_extension_arclength") or 0.0) > 0.0:
        mods.append("flow_ext")
    ready = "ready" if c.get("continuation_ready") == 1.0 else "unready"
    return "+".join([base] + mods) + f" [{ready}]"


def _census(job):
    """One case -> (spec, {tally: dict} | None, error | None, seconds)."""
    spec, opts = job
    from spong import portrait, sturm
    t0 = time.perf_counter()
    try:
        m = _build(spec, opts["max_degree"])
        if m is None:
            return spec, None, "skip: generator returned no model", 0.0
        out = {k: collections.Counter() for k in TALLIES}
        if opts["trace_only"]:
            e = sturm.enumerate_critical_points(m)
            stiff, arange = _difficulty(m, e)
            if (stiff < opts["min_stiffness"]
                    and arange < opts["min_arange"]):
                return spec, None, "skip: not difficult", 0.0
            out["difficulty"][f"stiffness 1e{int(stiff)}"] += 1
            out["difficulty"][f"A-range   1e{int(arange)}"] += 1
            e = sturm.materialize_stubs(m, e)
            p = portrait.compute(m, _enumeration=e, _skip_audit=True)
            points = e.points
        else:
            p = portrait.certified_compute(m)
            points = p.enumeration.points
    except Exception as exc:                      # a census never aborts
        return spec, None, f"{type(exc).__name__}: {exc}"[:120], 0.0

    for br in p.branches:
        out["termination"][f"{br.kind}:{br.term}"] += 1
        seen = {z[0] for z in (br.diag or {}).get("zones", [])
                if isinstance(z, (tuple, list)) and z}
        # Zones are recorded only by the stiff dispatcher.  A branch with
        # none went through ordinary continuation -- the commonest
        # construction of all, which must not be invisible here.
        for label in (seen or {"(unzoned)"}):
            out["zones"][f"{br.kind}:{label}"] += 1
    for q in points:
        for s in getattr(q, "stubs", ()) or ():
            out["launch"][f"{s.manifold}:{_launch_label(dict(s.certificates))}"] += 1
    if not opts["trace_only"]:
        top = p.ledger.get("topology", {})
        for e_ in top.get("unstable_ends", []) or []:
            tag = e_.get("method") or e_.get("tier") or e_.get("reason")
            state = "ok" if e_.get("certified") else "UNCERT"
            out["endpoint"][f"unstable:{e_.get('kind')}:{tag}:{state}"] += 1
        for e_ in top.get("stable_tails", []) or []:
            tag = e_.get("method") or e_.get("reason")
            state = "ok" if e_.get("certified") else "UNCERT"
            out["endpoint"][f"stable:{tag}:{state}"] += 1
        out["verdict"][f"{top.get('status')}:{top.get('resolution_reason')}"] += 1
    return spec, {k: dict(v) for k, v in out.items()}, None, \
        time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--directed", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260806,
                    help="master seed; matches scripts/ensemble.py")
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--zoo-only", action="store_true")
    ap.add_argument("--no-zoo", action="store_true")
    ap.add_argument("--trace-only", action="store_true",
                    help="trace without certifying, difficult models only")
    ap.add_argument("--min-stiffness", type=float, default=8.0,
                    help="trace-only: log10 eigenvalue ratio threshold")
    ap.add_argument("--min-arange", type=float, default=6.0,
                    help="trace-only: log10 A max/min threshold")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rare", type=int, default=3,
                    help="list the cases for any kind seen in <= this many")
    args = ap.parse_args(argv)

    from spong import zoo
    specs = []
    if not args.no_zoo and hasattr(zoo, "CASES"):
        specs += [("zoo", name) for name in sorted(
            c if isinstance(c, str) else c.name for c in zoo.CASES)]
    if not args.zoo_only:
        for mode, count in (("random", args.random),
                            ("directed", args.directed)):
            rng = random.Random(args.seed)
            specs += [(mode, rng.randrange(2**31)) for _ in range(count)]
    opts = {"max_degree": args.max_degree, "trace_only": args.trace_only,
            "min_stiffness": args.min_stiffness,
            "min_arange": args.min_arange}

    os.environ["SPONG_WORKERS"] = "1"
    totals = {k: collections.Counter() for k in TALLIES}
    where = {k: collections.defaultdict(list) for k in TALLIES}
    failures, skipped, surveyed = [], 0, 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_census, (s, opts)) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            spec, tallies, err, secs = fut.result()
            if err:
                if err.startswith("skip:"):
                    skipped += 1
                else:
                    failures.append((spec, err))
                continue
            surveyed += 1
            for k in TALLIES:
                for label, n in tallies[k].items():
                    totals[k][label] += n
                    where[k][label].append(spec)
            print(f"[{i:5d}/{len(specs)}] {spec[0]:8s} {str(spec[1]):>12s}  "
                  f"{secs:6.1f}s", flush=True)

    print(f"\n=== {surveyed} cases surveyed, {skipped} skipped, "
          f"{len(failures)} failed to run, of {len(specs)} "
          f"in {time.perf_counter()-started:.0f}s")
    for k in TALLIES:
        if not totals[k]:
            continue
        print(f"\n{k}:")
        for label, n in totals[k].most_common():
            cases = where[k][label]
            rare = (f"   <- {', '.join(f'{a}:{b}' for a, b in cases[:6])}"
                    if len(cases) <= args.rare else "")
            print(f"   {label:<58s} {n:6d} in {len(cases):4d} cases{rare}")
    for spec, err in failures[:10]:
        print(f"  FAILED {spec}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
