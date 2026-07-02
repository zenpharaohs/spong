"""Phase-5 gates: full ledger emitted; parity on the tricky portrait;
rendering guarantees; the d=2 oracle end-to-end."""

import numpy as np
import pytest

from spong import model, portrait, render
from tests.test_enumeration import TRICKY_F


@pytest.fixture(scope="module")
def tricky_portrait():
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    return portrait.compute(m, view=(-1.5, 2.5, -4.0, 3.0))


def test_ledger_complete_and_clean(tricky_portrait):
    p = tricky_portrait
    led = p.ledger
    e = led["enumeration"]
    assert e["n_critical"] == 8 and e["n_min"] == 4 and e["n_saddle"] == 4
    assert e["psi_positive[EXACT]"] and e["morse[EXACT]"]
    assert e["u2_alternation[EXACT]"]
    assert led["index_balance[EXACT]"]["balanced"]
    assert led["genericity[EXACT]"]["generic"]
    assert len(led["branches"]) == 16          # 4 saddles x 4 branches
    assert led["summary"]["all_branches_clean"]
    assert led["summary"]["worst_angle_energy"] < 1e-6
    assert led["summary"]["worst_max_turn_deg"] < 3.0


def test_tricky_branch_parity(tricky_portrait):
    """The founding MATLAB-parity gate, now through the full assembly."""
    p = tricky_portrait
    tricky = [b for b in p.ledger["branches"]
              if b["kind"] == "unstable"
              and abs(b["saddle_b"] - (-2.738230515199397)) < 1e-6]
    assert len(tricky) == 2
    for b in tricky:
        assert b["term"] == "capture"
        assert b["adjacency[RESIDUAL,thm-backed]"]
        assert b["angle_energy[RESIDUAL]"] < 1e-12


def test_adjacency_everywhere(tricky_portrait):
    for b in tricky_portrait.ledger["branches"]:
        if b["kind"] == "unstable":
            assert b["adjacency[RESIDUAL,thm-backed]"]


def test_asymptote_certificates_present(tricky_portrait):
    """Certificates exist on all separatrices; residuals are meaningful
    (small) only where the exit radius is genuinely asymptotic — the
    certificate must be honest about pre-asymptotic exits, not silent."""
    p = tricky_portrait
    stables = [(br, b) for br, b in zip(p.branches, p.ledger["branches"])
               if b["kind"] == "stable"]
    assert len(stables) == 8
    n_asymptotic = 0
    for br, b in stables:
        assert "asymptote_residual[RESIDUAL]" in b
        r_exit = float(np.hypot(*br.Y[-1]))
        if r_exit > 8.0:                       # genuinely toward the rim
            assert b["asymptote_residual[RESIDUAL]"] < 0.2
            n_asymptotic += 1
    assert n_asymptotic >= 2                   # the tricky saddle's pair


def test_render_svg(tricky_portrait, tmp_path):
    svg = render.plane_view(tricky_portrait, title="test")
    assert svg.startswith("<svg") and svg.count("<path") > 100
    disk = render.disk_view(tricky_portrait)
    assert disk.startswith("<svg") and disk.count("<path") > 50
    f = render.save(svg, str(tmp_path / "p.svg"))
    assert (tmp_path / "p.svg").stat().st_size > 10000


def test_d2_end_to_end():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    p = portrait.compute(m)
    led = p.ledger
    assert led["enumeration"]["n_critical"] == 4
    assert len(led["branches"]) == 8
    assert led["summary"]["all_branches_clean"]
    assert led["summary"]["balanced"]
    assert led["summary"]["worst_angle_energy"] < 1e-6
    svg = render.plane_view(p)
    assert svg.startswith("<svg")
