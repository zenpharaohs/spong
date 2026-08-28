#!/usr/bin/env python3
"""Compare the current chord audit with the experimental order sweep.

Examples:

    python scripts/order_sweep_probe.py 1495454581 directed 5
    python scripts/order_sweep_probe.py 1495454581 directed 5 \
        --proposer arc-forward-euler --euler-stride 32
    python scripts/order_sweep_probe.py quadratic-stiff

The comparison consumes the contact examples retained in the current ledger.
If the current event count exceeds its sample limit, the output explicitly
marks the order-sweep result as a sample rather than an exhaustive comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

from qualify import directed_model, random_model  # noqa: E402
from spong import comparison, model, order_sweep, portrait, zoo  # noqa: E402


def _case(spec, mode, degree):
    if spec.isdigit():
        generate = directed_model if mode == "directed" else random_model
        m, description = generate(random.Random(int(spec)), degree)
        return m, None, f"{mode} seed {spec} ({description})"
    z = zoo.get(spec)
    n = 2*max(len(z.f), len(z.g))-1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return (model.build(list(z.f), list(z.g), mu), z.default_view, spec)


def _preterminal_pair_contacts(p, top, limit=50000):
    """Production pair candidates before certified terminal discharges."""
    return order_sweep.pair_contact_candidates(
        p.branches, p.enumeration.points, p.box,
        float(top["predicate_tolerance"]), limit=limit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="1495454581")
    ap.add_argument("mode", nargs="?", choices=("directed", "random"),
                    default="directed")
    ap.add_argument("degree", nargs="?", type=int, default=5)
    ap.add_argument("--geometry-level", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--candidate-source",
                    choices=("preterminal", "retained"),
                    default="preterminal")
    ap.add_argument("--candidate-limit", type=int, default=50000)
    ap.add_argument("--proposer",
                    choices=("production", "arc-forward-euler"),
                    default="production")
    ap.add_argument("--euler-stride", type=int, default=32,
                    help="reference chord count per arc-length Euler step")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    m, view, label = _case(args.case, args.mode, args.degree)
    p = portrait.compute(m, view=view, geometry_level=args.geometry_level)
    top = p.ledger["topology"]
    if args.proposer == "arc-forward-euler":
        if args.candidate_source != "preterminal":
            ap.error("arc-forward-euler requires --candidate-source "
                     "preterminal")
        proposed_branches = comparison.arc_forward_euler(
            m, p.branches, p.box, args.euler_stride)
        proposed = portrait.Portrait(
            m, p.enumeration, proposed_branches, p.box, p.view, {})
        terminal_suffixes = ()
    else:
        proposed_branches = p.branches
        proposed = p
        terminal_suffixes = top.get("terminal_suffixes", ())
    if args.candidate_source == "preterminal":
        contacts, candidate_limit_hit = _preterminal_pair_contacts(
            proposed, top, args.candidate_limit)
    else:
        contacts = (list(top.get("forbidden_intersections", ()))
                    + list(top.get("ambiguous_contacts", ())))
        candidate_limit_hit = False
    current_total = int(top.get("forbidden_count", 0)) + int(
        top.get("ambiguous_count", 0))
    result = order_sweep.classify_contacts(
        m, p.enumeration, proposed_branches, contacts,
        float(top["predicate_tolerance"]), threshold=args.threshold,
        terminal_suffixes=terminal_suffixes)
    report = {
        "case": label,
        "geometry_level": args.geometry_level,
        "proposer": args.proposer,
        "proposer_vertices": sum(len(branch.Y)
                                 for branch in proposed_branches),
        "euler_stride": (args.euler_stride
                         if args.proposer == "arc-forward-euler" else None),
        "candidate_source": args.candidate_source,
        "candidate_limit_hit": candidate_limit_hit,
        "current": {
            "status": top["status"],
            "reason": top["resolution_reason"],
            "forbidden": top["forbidden_count"],
            "ambiguous": top["ambiguous_count"],
            "retained_examples": len(contacts),
            "examples_exhaustive": (
                args.candidate_source == "retained"
                and len(contacts) == current_total),
        },
        "order_sweep": result,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(label)
    print("current: "
          f"{top['status']} reason={top['resolution_reason']} "
          f"forbidden={top['forbidden_count']} "
          f"ambiguous={top['ambiguous_count']}")
    if args.proposer == "arc-forward-euler":
        print("proposer: uncertified arc-length Forward Euler "
              f"stride={args.euler_stride} "
              f"vertices={report['proposer_vertices']}")
    qualifier = ("limited" if candidate_limit_hit else
                 "retained" if args.candidate_source == "retained" else
                 "preterminal")
    print(f"order sweep ({qualifier} {len(contacts)} candidates, "
          f"R>={args.threshold:g}): decision={result['decision']} "
          f"roots={result['roots']} "
          f"same_order={result['same_order']} "
          f"terminal={result['terminal']} "
          f"critical_transition={result['critical_transition']} "
          f"unresolved={result['unresolved']}")
    for pair in result["pairs"]:
        print(f"  branches={pair['branches']} candidates="
              f"{pair['candidate_count']} roots={pair['root_count']} "
              f"same_order={pair['same_order_count']} "
              f"terminal={pair['terminal_count']} "
              f"unresolved={pair['unresolved_count']} "
              f"profile_levels={pair['profile_levels']} "
              f"loss_drops=({pair['first_dropped_nonmonotone']},"
              f"{pair['second_dropped_nonmonotone']}) "
              f"valid_fraction={pair['profile_valid_fraction']}")
        clusters = pair.get("clusters", ())
        if clusters:
            kinds = {}
            for cluster in clusters:
                kinds[cluster["kind"]] = kinds.get(cluster["kind"], 0)+1
            extents = [cluster["normalized_extent"] for cluster in clusters]
            print(f"    contact_clusters={len(clusters)} kinds={kinds} "
                  f"extent_range=({min(extents):.3g},{max(extents):.3g})")
        for root in pair["roots"][:8]:
            print(f"    root loss={root['loss_bracket']} "
                  f"margin={root['resolution_margin']:.3g} "
                  f"candidates={root['candidate_count']}")
        if len(pair["roots"]) > 8:
            print(f"    ... {len(pair['roots'])-8} more root clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
