"""The two-certificate scheme for branch validity.

`angle_energy` is GEOMETRIC and decays outward: it needs the direction of
grad L, which loses digits as |grad L| ~ C_inf/b^2 falls toward its own
evaluation floor.  `backbone_residual` is ALGEBRAIC and improves outward: far
out the branch IS the backbone a* = B/A, an exact rational function.  Together
they span a branch that the geometric one alone cannot certify.
"""

import numpy as np
import pytest

from spong import charts, model, portrait
from tests.test_enumeration import TRICKY_F


@pytest.fixture(scope="module")
def tricky():
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    return m, portrait.compute(m, view=(-1.5, 2.5, -4.0, 3.0))


def _escaping(p):
    for L, br in zip(p.ledger["branches"], p.branches):
        if L["kind"] == "unstable" and br.Y[:, 1].min() < -9:
            return L, br
    raise AssertionError("escaping branch not found")


# --------------------------------------------------------- the graded guard --


def test_guard_is_graded_not_a_cliff():
    """The old guard fired only at R < 1, which never triggers: |grad L|'s
    direction degrades continuously, so vertices at R ~ 6 were reporting
    evaluation noise as curve geometry."""
    assert charts.ANGLE_DIGIT_BUDGET > 1.0


def test_energy_falls_monotonically_as_the_budget_tightens(tricky):
    """Raising the digit budget must remove noise, not signal: E should fall
    monotonically as unresolved vertices are excluded."""
    m, p = tricky
    _, br = _escaping(p)
    prev = None
    for K in (1.0, 1e1, 1e2, 1e3, 1e4):
        E, used, skipped = charts.angle_energy_detail(m, br.Y, digits=K)
        assert used + skipped == len(br.Y) - 2
        if prev is not None:
            assert E < prev
        prev = E


def test_budget_reproduces_the_historical_box(tricky):
    """ANGLE_DIGIT_BUDGET was chosen by measurement: at 1e3 the certificate
    cuts where the pre-inflation compute box happened to end (b ~ -7), and
    returns essentially the value that box reported (3.70e-14)."""
    m, p = tricky
    _, br = _escaping(p)
    E, used, _ = charts.angle_energy_detail(m, br.Y)
    assert 1e-14 < E < 1e-13
    assert used > 1500


# ------------------------------------------------------- non-vacuous passing --


def test_the_gate_cannot_pass_by_measuring_nothing(tricky):
    """A budget strict enough to silence all noise would skip every vertex and
    return E = 0.  That is a vacuous pass, so the counts must be reported and
    a real fraction of vertices must survive."""
    m, p = tricky
    L, br = _escaping(p)
    assert L["angle_resolved"] is not None
    assert L["angle_unresolved"] is not None
    assert L["angle_resolved"] > 0.4 * (L["angle_resolved"] + L["angle_unresolved"])

    E_all, used_all, _ = charts.angle_energy_detail(m, br.Y, digits=1e9)
    assert used_all == 0 and E_all == 0.0      # the vacuous case is reachable...
    assert charts.ANGLE_DIGIT_BUDGET < 1e9     # ...and we are nowhere near it


def test_most_of_the_portrait_is_geometrically_resolved(tricky):
    """The guard must not be over-skipping: only the far valley should be
    unresolved, not the working part of the portrait."""
    _, p = tricky
    used = sum(b["angle_resolved"] or 0 for b in p.ledger["branches"])
    skipped = sum(b["angle_unresolved"] or 0 for b in p.ledger["branches"])
    assert skipped / (used + skipped) < 0.10


# ------------------------------------------------ the algebraic certificate --


def test_unresolved_tail_is_certified_algebraically(tricky):
    """Where angle_energy runs out of digits the branch has become the
    backbone, and that IS checkable — against an exact rational function."""
    L, _ = _escaping(tricky[1])
    assert L["angle_unresolved"] > 0            # it does have a far tail
    assert L["backbone_residual[RESIDUAL]"] < 1e-8


def test_the_two_certificates_run_in_opposite_directions(tricky):
    """Geometric decays outward, algebraic improves outward — which is why
    truncating is not needed and the pair spans the branch."""
    m, p = tricky
    _, br = _escaping(p)
    Y = br.Y
    order = np.argsort(-Y[:, 1])                # inward -> outward
    Y = Y[order]
    inner, outer = Y[:len(Y) // 3], Y[2 * len(Y) // 3:]
    eps = np.finfo(float).eps

    def digits(seg):
        out = []
        for a, b in seg:
            g = m.gradL(a, b)
            ng = float(np.hypot(g[0], g[1]))
            sa = 2.0 * (abs(a) * m.A(b) + abs(m.B(b)))
            sb = 2.0 * abs(a) * abs(m.Bp(b)) + a * a * abs(m.Ap(b))
            gf = 16.0 * eps * float(np.hypot(sa, sb))
            if gf > 0:
                out.append(ng / gf)
        return float(np.median(out))

    def wrel(seg):
        out = []
        for a, b in seg:
            astar = float(m.a_star(b))
            if abs(astar) > 1e-300:
                out.append(abs(float(a) - astar) / abs(astar))
        return float(np.median(out))

    assert digits(inner) > digits(outer) * 1e3          # geometric decays
    assert wrel(outer) < wrel(inner) * 1e-3             # algebraic improves
    # (measured on |w/a*| directly: backbone_residual is now scoped to the
    # UNRESOLVED vertices, so it makes no claim about the inner segment)


def test_backbone_residual_makes_no_claim_where_it_is_not_relied_on(tricky):
    """Scoped to the vertices angle_energy could NOT resolve.

    Measured over the whole polyline instead, it reports where the branch is
    legitimately far from the backbone and then 'fails' on branches it was
    never asked about — out of sample that read as 37 of 75 branches
    uncertified when their unresolved tails were in fact fine.
    """
    m, p = tricky
    for L, br in zip(p.ledger["branches"], p.branches):
        if L["kind"] == "unstable" and br.Y[:, 1].min() > -4:
            assert L["angle_unresolved"] == 0            # fully resolved...
            assert L["backbone_residual[RESIDUAL]"] == 0.0   # ...so no claim
            # forced to answer, it must still refuse to certify a genuine curve
            assert charts.backbone_residual(m, br.Y, digits=1e300) > 1e-3
            return
    pytest.skip("no finite-range unstable branch in this portrait")
