#!/usr/bin/env python3
"""Qualify the accelerated engine on its own terms.

    python scripts/qualify.py                      # 40 random models
    python scripts/qualify.py --cases 500 --workers 8
    python scripts/qualify.py --reference-rate 0.25

WHY
---
The Python engine is the reference, but it is too slow to be the routine
oracle: a single tricky-d11 portrait costs minutes through it.  If every check
has to go through the reference, bulk testing never happens.  This qualifies
the accelerated path so the reference can be an AUDITING instrument -- run on
a sample and on every interesting case -- rather than a dependency of ordinary
work.

THREE INDEPENDENT LEGS
----------------------
1. INVARIANTS.  Properties a certified portrait must satisfy whatever computed
   it: index balance, Morse alternation, branch inventory, the relationship
   between status and the certificates behind it.  These need no oracle, so
   they scale to as many random models as you care to run.  A violation is a
   bug regardless of which engine produced it.

2. DETERMINISM.  The same model recomputed at one worker and at many must
   agree exactly.  map_ordered assembles by submission order precisely so this
   holds; the check is what makes that claim testable rather than asserted.

3. REFERENCE AGREEMENT.  The Python engine on a sample -- every refusal, plus
   a random fraction of the rest.  Refusals are sampled at 100% deliberately:
   a subtle engine difference shows up as a changed verdict far more readily
   than as a changed certified skeleton, so the cases that refuse are exactly
   the ones worth spending reference time on.

Random models are drawn from a seeded generator, so a failure is reproducible
from the printed seed alone.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    from spong import engine, inverse, model, portrait
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import engine, inverse, model, portrait

# Every termination the engine can legitimately produce, taken from the source
# rather than from memory:
#   grep -ohE '"(abort_[a-z_]+|capture|box_exit|enter_shallow|step_failure|
#               sounding_[a-z_]+)"' src/spong/charts.py | sort -u
# An unknown term means either a real bug or that this list has fallen behind
# the code, and the suite cannot tell which -- so it is worth keeping exact.
# abort_conditioning_handoff is trace_stable refusing to launch when the
# materialized stub's global field is not ready: a legitimate refusal, not a
# failure.
TERMS = {"capture", "box_exit", "enter_shallow",
         "abort_conditioning_handoff", "abort_max_steps",
         "abort_nonfinite", "abort_not_graph", "abort_stationary",
         "abort_step_failure", "abort_switch_limit", "abort_zone_limit",
         "sounding_interval_exhausted", "step_failure"}


# --------------------------------------------------------------------------
# random models
# --------------------------------------------------------------------------

def random_model(rng, max_degree: int):
    df = rng.randint(1, max_degree)
    dg = rng.randint(1, max_degree)
    f = [round(rng.uniform(-2.0, 2.0), 6) for _ in range(df + 1)]
    g = [round(rng.uniform(-2.0, 2.0), 6) for _ in range(dg + 1)]
    if abs(g[-1]) < 0.05:
        g[-1] = 0.05
    if abs(f[-1]) < 0.05:
        f[-1] = 0.05
    dist = rng.choice(("uniform01", "normal01"))
    mu = (model.moments_normal01 if dist == "normal01"
          else model.moments_uniform01)(2 * max(len(f), len(g)) - 1)
    return model.build(f, g, mu), f"random d{df}/{dg} {dist}"


# Stiffness lives at large |b|, and undirected sampling does not go there:
# twenty random models at degree <= 4 certified every time, so the reference
# leg checked nothing.  inverse.straddle_case prescribes a critical point at a
# chosen radius and solves exactly over Q for the f that puts it there, which
# is the only way to reach the shallow-water regime on purpose.  The ladder
# mirrors inverse.straddling_suite's default radii.
_RADII = ([2 ** k for k in range(1, 20)]
          + [3 * 2 ** k for k in range(1, 17)]
          + [5 * 2 ** k for k in range(1, 15)])


def directed_model(rng, max_degree: int):
    """A model with a critical point at a prescribed large |b|."""
    dg = rng.randint(2, max_degree)
    g = [round(rng.uniform(-2.0, 2.0), 6) for _ in range(dg + 1)]
    if abs(g[-1]) < 0.05:
        g[-1] = 0.05
    dist = rng.choice(("uniform01", "normal01"))
    mu = (model.moments_normal01 if dist == "normal01"
          else model.moments_uniform01)(4 * len(g) + 3)
    radius = Fraction(rng.choice(_RADII))
    if rng.random() < 0.5:
        radius = -radius
    case = inverse.straddle_case([radius], tuple(
        Fraction(c).limit_denominator(10**6) for c in g), mu)
    if case is None:
        return None, None
    return case.design.model, f"directed d{dg} |b|={abs(radius)} {dist}"


# --------------------------------------------------------------------------
# leg 1: invariants
# --------------------------------------------------------------------------

def invariants(p) -> list[str]:
    """Everything a portrait must satisfy regardless of what produced it."""
    bad = []
    led, e = p.ledger, p.enumeration
    en = led["enumeration"]
    top = led["topology"]

    if en["n_min"] + en["n_saddle"] != en["n_critical"]:
        bad.append(f"critical counts: {en['n_min']}+{en['n_saddle']} "
                   f"!= {en['n_critical']}")
    if len(e.points) != en["n_critical"]:
        bad.append("enumeration.points disagrees with the ledger count")

    for q in e.points:
        if q.kind == "min" and q.u2_sign <= 0:
            bad.append(f"minimum at b={float(q.b):.6g} has u2_sign "
                       f"{q.u2_sign}")
        if q.kind == "saddle" and q.u2_sign >= 0:
            bad.append(f"saddle at b={float(q.b):.6g} has u2_sign "
                       f"{q.u2_sign}")

    # Every critical point lies on the backbone -- a defining identity, not a
    # numerical coincidence, so a violation means the enumeration and the
    # model have parted company.
    for q in e.points:
        a_star = float(m_of(p).a_star(float(q.b)))
        if abs(a_star - float(q.a)) > 1e-6 * (1 + abs(a_star)):
            bad.append(f"critical point at b={float(q.b):.6g} is off the "
                       f"backbone by {abs(a_star-float(q.a)):.3e}")

    if len(p.branches) != 4 * en["n_saddle"]:
        bad.append(f"{len(p.branches)} branches for {en['n_saddle']} saddles "
                   f"(expected {4*en['n_saddle']})")
    for i, br in enumerate(p.branches):
        if br.term not in TERMS:
            bad.append(f"branch {i}: unknown term {br.term!r}")
        if len(br.Y) < 1:
            bad.append(f"branch {i}: empty polyline")

    if led["index_balance[EXACT]"].get("balanced") is False:
        bad.append("index balance is not satisfied")

    if top["status"] == "certified":
        if top["forbidden_count"]:
            bad.append("certified with forbidden intersections")
        if not led["summary"].get("all_branches_clean"):
            bad.append("certified with unclean branches")
        if not en["psi_positive[EXACT]"]:
            bad.append("certified without psi positivity")
        if not en["morse[EXACT]"]:
            bad.append("certified without a Morse skeleton")
    return bad


def m_of(p):
    return p.model if hasattr(p, "model") else p.m


# --------------------------------------------------------------------------
# snapshot for determinism and reference comparison
# --------------------------------------------------------------------------

def snapshot(p) -> dict:
    """The assertions of a portrait, independent of how it was computed."""
    led = p.ledger
    return {
        "enumeration": led["enumeration"],
        "critical": [(str(q.b), str(q.a), q.kind, q.source, int(q.u2_sign))
                     for q in sorted(p.enumeration.points, key=lambda q: q.b)],
        "branches": [(br.kind, br.term, len(br.Y)) for br in p.branches],
        "status": led["topology"]["status"],
        "reason": led["topology"]["resolution_reason"],
        "forbidden": led["topology"]["forbidden_count"],
        "ambiguous": led["topology"]["ambiguous_count"],
        "level": led["topology"]["geometry_level"],
    }


def differences(a: dict, b: dict) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            out.append(f"{key}: {a.get(key)!r} != {b.get(key)!r}")
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reference-rate", type=float, default=0.1,
                    help="fraction of CERTIFYING cases also run through the "
                         "Python reference; every refusal is checked anyway")
    ap.add_argument("--determinism-rate", type=float, default=0.2)
    ap.add_argument("--mode", choices=("random", "directed", "mixed"),
                    default="mixed",
                    help="directed cases prescribe a critical point at a "
                         "large |b|, which is where the stiffness is")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.environ["SPONG_WORKERS"] = str(args.workers)

    tally = {"ok": 0, "invariant": 0, "determinism": 0, "reference": 0,
             "error": 0, "certified": 0, "refused": 0}
    # Why the refusals refuse.  The classes need different work and are worth
    # separating: branch_abort is a LAUNCH failure and yields to routing (the
    # dead-neuron stub going to the fixed point instead of refusing);
    # unstable_endpoint_unresolved and topology_contact are CERTIFICATE
    # limits, which no amount of launch work touches; certification_segment
    # _budget is a resource constant.
    reasons: dict = {}
    # branch_abort fires for ANY branch terminating outside capture/box_exit,
    # so the reason alone does not say what to fix.  The terminations do:
    # abort_conditioning_handoff is a launch that found no downstream owner,
    # abort_zone_limit is the dispatcher thrashing between charts,
    # abort_max_steps is a budget, abort_not_graph a degenerate frame.
    terms: dict = {}
    ref_checked = det_checked = 0
    t_native = t_reference = 0.0

    for case in range(args.cases):
        seed = rng.randrange(2**31)
        sub = random.Random(seed)
        directed = (args.mode == "directed"
                    or (args.mode == "mixed" and case % 2 == 1))
        try:
            if directed:
                m, spec = directed_model(sub, args.max_degree)
                if m is None:
                    continue
            else:
                m, spec = random_model(sub, args.max_degree)
        except Exception as exc:
            print(f"[{case:4d}] seed {seed}: model rejected ({exc})")
            continue

        engine.use("native")
        try:
            t0 = time.perf_counter()
            p = portrait.certified_compute(m)
            t_native += time.perf_counter() - t0
        except Exception as exc:
            tally["error"] += 1
            print(f"[{case:4d}] seed {seed}: FAILED {type(exc).__name__}: "
                  f"{exc}  [{spec}]")
            continue

        status = p.ledger["topology"]["status"]
        tally["certified" if status == "certified" else "refused"] += 1
        if status != "certified":
            why = p.ledger["topology"].get("resolution_reason") or "unstated"
            reasons[why] = reasons.get(why, 0) + 1
            for br in p.branches:
                if br.term in ("capture", "box_exit"):
                    continue
                tag = f"{br.kind}/{br.term}"
                terms[tag] = terms.get(tag, 0) + 1

        bad = invariants(p)
        if bad:
            tally["invariant"] += 1
            print(f"[{case:4d}] seed {seed}: INVARIANT  [{spec}]")
            for line in bad:
                print(f"        {line}")
            continue

        base = snapshot(p)

        if sub.random() < args.determinism_rate:
            det_checked += 1
            os.environ["SPONG_WORKERS"] = "1"
            solo = snapshot(portrait.certified_compute(m))
            os.environ["SPONG_WORKERS"] = str(args.workers)
            delta = differences(base, solo)
            if delta:
                tally["determinism"] += 1
                print(f"[{case:4d}] seed {seed}: WORKER-DEPENDENT")
                for line in delta:
                    print(f"        {line}")
                continue

        if status != "certified" or sub.random() < args.reference_rate:
            ref_checked += 1
            engine.use("python")
            t0 = time.perf_counter()
            ref = snapshot(portrait.certified_compute(m))
            t_reference += time.perf_counter() - t0
            engine.use("native")
            delta = differences(base, ref)
            if delta:
                tally["reference"] += 1
                print(f"[{case:4d}] seed {seed}: ENGINE DISAGREEMENT")
                for line in delta:
                    print(f"        {line}")
                continue

        tally["ok"] += 1

    print(f"\n{args.cases} models, max degree {args.max_degree}, "
          f"{args.workers} workers, seed {args.seed}")
    print(f"  passed              {tally['ok']}")
    print(f"  certified / refused {tally['certified']} / {tally['refused']}")
    print(f"  invariant failures  {tally['invariant']}")
    print(f"  worker-dependent    {tally['determinism']}  "
          f"({det_checked} checked)")
    print(f"  engine disagreement {tally['reference']}  "
          f"({ref_checked} checked)")
    print(f"  errors              {tally['error']}")
    if reasons:
        print("  refusal reasons:")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"     {n:4d}  {why}")
    if terms:
        print("  unfinished branch terminations:")
        for tag, n in sorted(terms.items(), key=lambda kv: -kv[1]):
            print(f"     {n:4d}  {tag}")
    print(f"  native {t_native:.1f}s total"
          + (f", reference {t_reference:.1f}s on {ref_checked} cases"
             if ref_checked else ""))
    failures = (tally["invariant"] + tally["determinism"]
                + tally["reference"] + tally["error"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
