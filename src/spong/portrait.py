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


def compute(m: Model, view=None,
            trace_stable_branches: bool = True) -> Portrait:
    """Compute the certified portrait inside the §8b box contract."""
    e = sturm.enumerate_critical_points(m)
    box = atlas.compute_box(m, e, view=view)
    gen = atlas.genericity(m)

    branches = []
    span_scale = max(box[3] - box[2], box[1] - box[0])

    for s in e.saddles:
        # ---- unstable branches (descent), one per side ---------------- #
        for direction in (+1, -1):
            t = _target_for(e, s, direction)
            if t is not None:
                br = charts.trace_unstable(m, s.b, (t.a, t.b), box=box,
                                           ds=abs(t.b - s.b) / 4000.0)
                br.diag["saddle_b"] = s.b
                br.diag["target"] = (t.a, t.b)
                br.certs["adjacency_ok"] = (
                    br.term == "capture"
                    and abs(br.Y[-1, 1] - t.b) < 1e-9)
            else:
                b_exit = box[3] if direction > 0 else box[2]
                a_exit = float(m.a_star(b_exit))
                br = charts.trace_unstable(m, s.b, (a_exit, b_exit), box=box)
                br.diag["saddle_b"] = s.b
                br.diag["target"] = None
                br.certs["adjacency_ok"] = br.term in ("capture", "box_exit")
            branches.append(br)

        # ---- stable branches (ascent separatrices) --------------------- #
        if trace_stable_branches:
            for sign in (+1, -1):
                br = charts.trace_stable(m, s.b, sign, box=box,
                                         ds=span_scale / 30000.0)
                br.diag["saddle_b"] = s.b
                if br.term == "box_exit" and len(br.Y) > 50:
                    br.certs["asymptote"] = atlas.asymptote_certificate(
                        m, br.Y)
                branches.append(br)

    p = Portrait(m, e, branches, box, view)
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
