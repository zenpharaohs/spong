#!/usr/bin/env python3
"""Freeze and check golden zoo portraits — the regression suite.

    python scripts/golden_zoo.py freeze            # write goldens
    python scripts/golden_zoo.py check             # compare against them
    python scripts/golden_zoo.py check tricky-d11  # one case

Purpose: the C dispatcher migration will change the geometry engine.  Without
a snapshot taken BEFORE it, a later difference cannot be told apart from an
improvement.  This captures what a portrait asserts, not how long it took.

WHAT IS COMPARED, AND HOW STRICTLY
----------------------------------
EXACT (any difference is a regression) — these are the claims:
    the Morse skeleton: count, kind, source, sign of u'' at every point
    psi_positive, morse, u2_alternation, index balance, genericity
    per-branch kind and termination
    the topology verdict: status, resolution reason, forbidden/ambiguous
    the escalation ladder: which levels ran and what each concluded

TOLERANT — these are measurements of the same claims:
    critical point coordinates (a, b)          rel 1e-9
    RESIDUAL certificates and worst-case rollups   rel 1e-6
    branch vertex counts                       rel 5% (a resolution choice,
                                                not an assertion)
    angles the certificate resolved            rel 5% (one per resolvable
                                                vertex; tracks the count)

angle_unresolved stays EXACT: how many angles binary64 could NOT resolve is
a claim about the certificate, and it has not moved across any rewrite.

Timing is excluded entirely: it is the thing the migration is meant to change.

A tolerant mismatch is reported as DRIFT and an exact one as BREAK, because
they mean different things.  Drift on a rewrite is expected; a break is not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The reference (python) engine is the skeptic's tool, not the routine
# configuration: a check asserts what portraits claim, and the two engines
# are pinned to agree at every tier (step parity, segment corpus, portrait
# goldens, sampling qualification).  Default to the fast configuration; an
# explicit SPONG_ENGINE=python still selects the oracle, and the engine
# line printed at startup reports which one actually ran.
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLDEN = REPO / "tests" / "golden"

try:
    from spong import model, portrait, sturm, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import model, portrait, sturm, zoo

REL_COORD = 1e-9
REL_RESIDUAL = 1e-6
REL_POINTS = 0.05


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def _moments(name: str, n: int):
    if name == "normal01":
        return model.moments_normal01(n)
    return model.moments_uniform01(n)


def _plain(x):
    """Coerce NumPy scalars and arrays to plain Python types.

    Certificates like connection_ok come back as numpy bools because they are
    formed from numpy comparisons; np.bool is not JSON serializable and is not
    a subclass of Python bool, so it would also compare wrongly against a
    golden loaded from JSON.  Normalizing at capture keeps both honest.
    """
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item") and getattr(x, "shape", None) == ():
        return _plain(x.item())
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return _plain(x.tolist())
    if isinstance(x, float) and x != x:
        return "nan"                      # JSON has no NaN; keep it comparable
    return x


def capture(name: str) -> dict:
    z = zoo.get(name)
    f, g = list(z.f), list(z.g)
    mu = _moments(z.moment_dist, 2 * max(len(f), len(g)) - 1)
    m = model.build(f, g, mu)
    p = portrait.certified_compute(m, view=z.default_view)
    led, e = p.ledger, p.enumeration
    top = led.get("topology", {})

    return _plain({
        "case": {
            "name": z.name,
            "f": [float(x) for x in z.f],
            "g": [float(x) for x in z.g],
            "moment_dist": z.moment_dist,
            "default_view": (None if z.default_view is None
                             else [float(x) for x in z.default_view]),
        },
        "enumeration": led["enumeration"],
        "genericity": led["genericity[EXACT]"],
        "index_balance": led["index_balance[EXACT]"],
        "critical": [
            {"b": float(q.b), "a": float(q.a), "kind": q.kind,
             "source": q.source, "u2_sign": int(q.u2_sign)}
            for q in sorted(e.points, key=lambda q: q.b)
        ],
        "branches": led["branches"],
        "summary": led["summary"],
        "topology": {
            "status": top.get("status"),
            "resolution_reason": top.get("resolution_reason"),
            "forbidden_count": top.get("forbidden_count"),
            "ambiguous_count": top.get("ambiguous_count"),
            "geometry_level": top.get("geometry_level"),
            # attempts without elapsed_sec: the ladder is an assertion,
            # its duration is not
            "attempts": [
                {k: v for k, v in att.items() if k != "elapsed_sec"}
                for att in top.get("attempts", [])
            ],
        },
    })


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

EXACT_KEYS = {
    "kind", "source", "u2_sign", "term", "status", "resolution_reason",
    "forbidden_count", "ambiguous_count", "geometry_level", "n_critical",
    "n_min", "n_saddle", "psi_positive[EXACT]", "morse[EXACT]",
    "u2_alternation[EXACT]", "balanced", "all_branches_clean",
    "angle_unresolved", "connection[RESIDUAL]",
    "reason", "uncertified_ends", "uncertified_tails", "name",
    "moment_dist", "sublevel_unique",
}
POINT_KEYS = {"n_points", "angle_resolved"}


def _rel(x, y) -> float:
    if x == y:
        return 0.0
    scale = max(abs(x), abs(y), 1e-300)
    return abs(x - y) / scale


def diff(old, new, path="", out=None):
    out = [] if out is None else out
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k not in old or k not in new:
                out.append(("BREAK", f"{path}.{k}",
                            "missing" if k not in new else "added", ""))
                continue
            diff(old[k], new[k], f"{path}.{k}", out)
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append(("BREAK", path, f"len {len(old)}", f"len {len(new)}"))
            return out
        for i, (a, b) in enumerate(zip(old, new)):
            diff(a, b, f"{path}[{i}]", out)
        return out

    key = path.rsplit(".", 1)[-1].split("[")[0]
    if isinstance(old, bool) or isinstance(new, bool) or key in EXACT_KEYS:
        if old != new:
            out.append(("BREAK", path, repr(old), repr(new)))
        return out
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if key in POINT_KEYS:
            tol = REL_POINTS
        elif key in ("a", "b") or "view" in path:
            tol = REL_COORD
        else:
            tol = REL_RESIDUAL
        r = _rel(float(old), float(new))
        if r > tol:
            out.append(("DRIFT" if tol > 0 else "BREAK", path,
                        f"{old!r}", f"{new!r} (rel {r:.2e} > {tol:g})"))
        return out
    if old != new:
        out.append(("BREAK", path, repr(old), repr(new)))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def names(argv) -> list[str]:
    return list(argv) if argv else list(zoo.names())


def do_freeze(argv) -> int:
    import time
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in names(argv):
        print(f"{name:<38s} computing…", flush=True)
        t0 = time.perf_counter()
        data = capture(name)
        elapsed = time.perf_counter() - t0
        path = GOLDEN / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        en = data["enumeration"]
        print(f"froze {name:<38s} {en['n_critical']} crit "
              f"({en['n_min']}m/{en['n_saddle']}s), "
              f"{len(data['branches'])} branches, "
              f"topology={data['topology']['status']}  ({elapsed:.1f}s)")
    print(f"\n{len(names(argv))} goldens in {GOLDEN}")
    return 0


def do_check(argv) -> int:
    import time
    bad = 0
    for name in names(argv):
        path = GOLDEN / f"{name}.json"
        if not path.exists():
            print(f"{name:<38s} NO GOLDEN — run freeze first")
            bad += 1
            continue
        print(f"{name:<38s} computing…", flush=True)
        t0 = time.perf_counter()
        deltas = diff(json.loads(path.read_text()), capture(name))
        elapsed = time.perf_counter() - t0
        breaks = [d for d in deltas if d[0] == "BREAK"]
        drifts = [d for d in deltas if d[0] == "DRIFT"]
        if not deltas:
            print(f"{name:<38s} ok  ({elapsed:.1f}s)")
            continue
        bad += bool(breaks)
        print(f"{name:<38s} {len(breaks)} break, {len(drifts)} drift"
              f"  ({elapsed:.1f}s)")
        # BREAKs first: the verdict-bearing lines (.topology.status,
        # resolution_reason, geometry_level) sort after .branches and were
        # being pushed past the display cap by branch-level drift.
        for kind, where, a, b in (breaks + drifts)[:40]:
            print(f"    {kind:<5s} {where}\n          was {a}\n          now {b}")
        if len(deltas) > 40:
            print(f"    … {len(deltas) - 40} more")
    print("\nFAIL" if bad else "\nall assertions hold")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("freeze", "check"):
        print(__doc__)
        return 2
    try:
        from spong import engine
        reason = engine.native_error()
        if engine.active_name() == "native":
            print("engine: native (C core)" if reason is None else
                  "engine: native requested, C core UNAVAILABLE — "
                  f"python fallback ({reason})")
        else:
            print("engine: python (reference implementation)")
    except ImportError:
        pass
    return (do_freeze if sys.argv[1] == "freeze" else do_check)(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
