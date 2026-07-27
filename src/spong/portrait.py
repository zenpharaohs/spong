"""Assembly: enumeration → branches → skeleton, plus the certificate ledger.

SPONG_FOUNDING Part II, section 11.  A Portrait is a measurement: every
drawn object carries residuals a skeptic can recompute without trusting
the code that produced it.

Branch targeting honors the corrected Theorem 2: unstable branches aim at
the nearest MINIMUM on their side of the backbone (B-root saddles are not
attractors and are skipped); saddles with no minimum on a side send that
branch to the compute-box edge along the valley (pseudo-target on the
backbone).  Stable branches (separatrices) run to compute-box exit and
carry the √d_eff asymptote certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import atlas, charts, sturm
from .model import Model


@dataclass
class Portrait:
    model: Model
    enumeration: sturm.Enumeration
    branches: list
    box: tuple                    # compute box (a_lo, a_hi, b_lo, b_hi)
    view: tuple | None
    ledger: dict = field(default_factory=dict)


def _target_for(e: sturm.Enumeration, s, direction: int):
    """Nearest minimum strictly on the given b-side of saddle s, or None."""
    side = [p for p in e.minima
            if (p.b > s.b if direction > 0 else p.b < s.b)]
    if not side:
        return None
    return min(side, key=lambda p: abs(p.b - s.b))


def _box_with_points(box, Y, margin: float = 0.02):
    if len(Y) == 0:
        return box
    a0, a1, b0, b1 = box
    ya0, ya1 = float(np.nanmin(Y[:, 0])), float(np.nanmax(Y[:, 0]))
    yb0, yb1 = float(np.nanmin(Y[:, 1])), float(np.nanmax(Y[:, 1]))
    da = margin * max(a1 - a0, ya1 - ya0, 1.0)
    db = margin * max(b1 - b0, yb1 - yb0, 1.0)
    return (min(a0, ya0 - da), max(a1, ya1 + da),
            min(b0, yb0 - db), max(b1, yb1 + db))


def _expand_a_box(box, side: int):
    a0, a1, b0, b1 = box
    width = max(a1 - a0, 1.0)
    if side < 0:
        a0 -= width
    else:
        a1 += width
    return (a0, a1, b0, b1)


def _trace_box(m: Model, display_box, scale: float = 1.35):
    """Larger integration box around the display view.

    The default view keeps the critical skeleton readable, but separatrices
    may leave that view and later pass through it again.  Trace against a
    larger box so rendering has those visible re-entry chords.
    """
    a0, a1, b0, b1 = display_box
    ac, bc = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    ah = 0.5 * scale * max(a1 - a0, 1.0)
    bh = 0.5 * scale * max(b1 - b0, 1.0)
    bmax = atlas.legal_max_b(m)
    return (ac - ah, ac + ah, max(bc - bh, -bmax), min(bc + bh, bmax))


def _trace_finite_unstable(m: Model, s, t, box):
    # Finite branches can be nearly horizontal in the (a,b)-plane: choosing
    # the chord budget from |Δb| alone can over-resolve the branch until the
    # engine hits max_steps before reaching a perfectly finite target.
    db = abs(t.b - s.b)
    da = abs(t.a - s.a)
    ds = max(db / 4000.0, float(np.hypot(da, db)) / 8000.0)
    cur_box = box
    br = None
    for _ in range(6):
        br = charts.trace_unstable(m, s.b, (t.a, t.b), box=cur_box, ds=ds)
        if br.term == "capture":
            return br, _box_with_points(cur_box, br.Y)
        a_end = float(br.Y[-1, 0])
        side = -1 if a_end <= cur_box[0] else (1 if a_end >= cur_box[1] else 0)
        if br.term != "box_exit" or side == 0:
            return br, _box_with_points(cur_box, br.Y)
        cur_box = _expand_a_box(cur_box, side)
    return br, _box_with_points(cur_box, br.Y)


def compute(m: Model, view=None,
            trace_stable_branches: bool = True) -> Portrait:
    """Compute the certified portrait inside the §8b box contract."""
    e = sturm.enumerate_critical_points(m)
    display_view = atlas.compute_box(m, e, view=view)
    box = _trace_box(m, display_view)
    gen = atlas.genericity(m)

    branches = []
    unbounded: list[tuple] = []

    for s in e.saddles:
        # ---- unstable branches (descent), one per side ---------------- #
        for direction in (+1, -1):
            t = _target_for(e, s, direction)
            if t is not None:
                br, box = _trace_finite_unstable(m, s, t, box)
                br.diag["saddle_b"] = s.b
                br.diag["target"] = (t.a, t.b)
                br.certs["adjacency_ok"] = (
                    br.term == "capture"
                    and abs(br.Y[-1, 1] - t.b) < 1e-9)
                branches.append(br)
            else:
                unbounded.append((s, direction))

    for s, direction in unbounded:
        b_exit = box[3] if direction > 0 else box[2]
        if abs(b_exit - s.b) > 100.0:
            b_local = None
            if view is not None:
                vspan = view[3] - view[2]
                b_edge = view[3] if direction > 0 else view[2]
                b_local = b_edge + direction * max(0.25 * vspan, 1.0)
            br = charts.trace_valley_exit(
                m, s.b, b_exit, box=box, local_until=b_local)
        else:
            a_exit = float(m.a_star(b_exit))
            br = charts.trace_unstable(m, s.b, (a_exit, b_exit),
                                       box=box)
        br.diag["saddle_b"] = s.b
        br.diag["target"] = None
        br.certs["adjacency_ok"] = br.term in ("capture", "box_exit")
        branches.append(br)

    # ---- stable branches (ascent separatrices) ------------------------- #
    if trace_stable_branches:
        span_scale = max(display_view[3] - display_view[2],
                         display_view[1] - display_view[0])
        for s in e.saddles:
            for sign in (+1, -1):
                br = charts.trace_stable(m, s.b, sign, box=box,
                                         ds=span_scale / 30000.0)
                br.diag["saddle_b"] = s.b
                if br.term == "box_exit" and len(br.Y) > 50:
                    br.certs["asymptote"] = atlas.asymptote_certificate(
                        m, br.Y)
                branches.append(br)

    p = Portrait(m, e, branches, box,
                 view if view is not None else display_view)
    p.ledger = build_ledger(p, gen)
    return p


def _max_turn_deg(Y: np.ndarray) -> float:
    if len(Y) < 3:
        return 0.0
    d = np.diff(Y, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    ok = (seg[:-1] > 1e-13) & (seg[1:] > 1e-13)
    if not ok.any():
        return 0.0
    ct = ((d[:-1, 0] * d[1:, 0] + d[:-1, 1] * d[1:, 1])
          / (seg[:-1] * seg[1:]))[ok]
    return float(np.degrees(np.arccos(np.clip(ct, -1.0, 1.0)).max()))


def build_ledger(p: Portrait, gen: dict) -> dict:
    """The certificate ledger (SPONG_FOUNDING §11), machine-checkable."""
    e = p.enumeration
    balance = atlas.index_balance(p.model, e)
    led = {
        "enumeration": {
            "n_critical": len(e.points),
            "n_min": len(e.minima),
            "n_saddle": len(e.saddles),
            "psi_positive[EXACT]": e.psi_positive,
            "morse[EXACT]": e.morse,
            "u2_alternation[EXACT]": e.alternates,
        },
        "genericity[EXACT]": gen,
        "index_balance[EXACT]": balance,
        "branches": [],
    }
    for br in p.branches:
        entry = {
            "kind": br.kind,
            "saddle_b": br.diag.get("saddle_b"),
            "term": br.term,
            "n_points": int(len(br.Y)),
            "angle_energy[RESIDUAL]": br.certs.get("angle_energy"),
            "max_turn_deg[RESIDUAL]": _max_turn_deg(br.Y),
        }
        if "seam_residual" in br.certs:
            entry["seam_residual[RESIDUAL]"] = br.certs["seam_residual"]
        if "adjacency_ok" in br.certs:
            entry["adjacency[RESIDUAL,thm-backed]"] = br.certs["adjacency_ok"]
        if "asymptote" in br.certs:
            entry["asymptote_residual[RESIDUAL]"] = \
                br.certs["asymptote"]["residual"]
        led["branches"].append(entry)

    led["summary"] = {
        "all_branches_clean": all(
            b["term"] in ("capture", "box_exit") for b in led["branches"]),
        "worst_angle_energy": max(
            (b["angle_energy[RESIDUAL]"] or 0.0) for b in led["branches"])
        if led["branches"] else 0.0,
        "worst_max_turn_deg": max(
            b["max_turn_deg[RESIDUAL]"] for b in led["branches"])
        if led["branches"] else 0.0,
        "balanced": balance["balanced"],
    }
    return led
