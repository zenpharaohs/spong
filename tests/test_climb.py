"""Gates and the climb test: closed form, end rule, exact proofs."""
from fractions import Fraction

import numpy as np

from spong import climb, model, portrait, signature, sturm, zoo


def _zoo_model(name):
    z = zoo.get(name)
    f, g = [float(x) for x in z.f], [float(x) for x in z.g]
    return model.build(f, g, zoo.moments_for(z, 2*max(len(f), len(g)) - 1))


def test_work_closed_form_matches_quadrature():
    # W = INT 4A/|A'| from the partial-fraction form of J, against
    # Gauss-Legendre on the stretches of near-slide-d2's backbone.
    m = _zoo_model("near-slide-d2")
    e = sturm.enumerate_critical_points(m)
    an = climb.Analysis(m, e)
    x, w = np.polynomial.legendre.leggauss(200)
    A = np.asarray([float(c) for c in m.alpha][::-1])
    Ap = np.polyder(A)
    checked = 0
    for i in range(len(e.points) - 1):
        if an.gate_between(i, i + 1):
            continue                      # A' may vanish inside: not a stretch
        lo, hi = float(e.points[i].b), float(e.points[i + 1].b)
        t = 0.5*(hi - lo)*x + 0.5*(hi + lo)
        quad = float(np.sum(w*4*np.polyval(A, t)/np.abs(np.polyval(Ap, t)))
                     * 0.5*(hi - lo))
        assert abs(an.work(lo, hi) - quad) <= 1e-9*max(1.0, quad)
        checked += 1
    assert checked >= 2


def test_backbone_end_is_reachable_only_past_an_outermost_saddle():
    # Linear regression: one roof saddle, one minimum.
    m = model.build([1.213452, 1.96381], [0.670652, 0.204005],
                    model.moments_uniform01(3))
    e = sturm.enumerate_critical_points(m)
    an = climb.Analysis(m, e)
    bb = an.bb
    s = next(i for i in range(len(bb.letters)) if bb.is_saddle(i))
    toward_min = 1 if bb.letters[-1] == "g" else -1
    minima, saddles, end, _ = an.reachable(s, toward_min)
    assert len(minima) == 1 and not saddles and end is None
    minima, saddles, end, _ = an.reachable(s, -toward_min)
    assert not minima and not saddles and end is not None


def test_near_slide_nonadjacent_attachment_passes_and_nothing_is_excluded():
    m = _zoo_model("near-slide-d2")
    p = portrait.certified_compute(m)
    report = climb.report(m, p.enumeration, signature.attachments(p))
    assert report["contradicted"] == []
    assert report["checked"] == 6
    # The merge tree leaves 3 branches open; the gates and the climb test
    # leave only the roof saddle's inward branch, the one that does skip.
    assert report["open_tree"] == 3 and report["open_climb"] == 1
    assert report["type1_pass"] == 1 and 0.0 < report["tightest_pass"] < 1.0


def test_uphill_chords_respect_the_lemma_on_a_traced_portrait():
    m = _zoo_model("near-slide-d2")
    p = portrait.compute(m, _skip_audit=True)
    seen = 0
    for br in p.branches:
        if br.kind != "unstable":
            continue
        got = climb.uphill_chords(m, br)
        seen += got["chords"]
        assert got["below"] == 0
    assert seen > 100


def test_sqrt_upper_is_an_upper_bound_and_tight():
    for q in (Fraction(0), Fraction(2), Fraction(1, 3), Fraction(10**40, 7)):
        r = climb._sqrt_upper(q)
        assert r*r >= q
        assert r == Fraction(1, 2**64) or (r - Fraction(1, 2**63))**2 < q


def test_tricky_d11_blocks_are_proved_in_exact_arithmetic():
    # Both climbs the binary64 proposal blocks on tricky-d11 have an exact
    # proof, so every branch of the portrait is decided before any trace.
    m = _zoo_model("tricky-d11")
    e = sturm.enumerate_critical_points(m)
    report = climb.report(m, e)
    assert report["proved_work"] == 2 and report["blocked_unproved"] == 0
    assert report["open_tree"] == 4 and report["open_proved"] == 0
    assert report["connections_proved"] == 0 and report["alarms"] == 0


def test_a_proof_never_excludes_what_the_portrait_certifies():
    # The exact test is stronger than the proposal (node chains and the
    # level), so it is checked against certified ends on its own account.
    for name in ("near-slide-d2", "nonnearest-attachment", "minimal-quartet"):
        m = _zoo_model(name)
        p = portrait.certified_compute(m)
        report = climb.report(m, p.enumeration, signature.attachments(p))
        assert report["contradicted"] == [], name
        assert report["alarms"] == 0, name


def test_level_above_is_at_or_above_the_saddle_value():
    m = _zoo_model("near-slide-d2")
    e = sturm.enumerate_critical_points(m)
    an = climb.Analysis(m, e)
    for i in range(len(an.bb.letters)):
        if an.bb.is_saddle(i):
            assert float(an.level_above(i)) >= an.u(i) - 1e-12
            if an.bb.letters[i] == "B":
                assert an.level_above(i) == Fraction(m.C)


def test_exact_lower_sum_is_below_the_closed_form_and_converges():
    m = _zoo_model("near-slide-d2")
    e = sturm.enumerate_critical_points(m)
    an = climb.Analysis(m, e)
    checked = 0
    for i in range(len(e.points) - 1):
        if an.gate_between(i, i + 1):
            continue
        lo, hi = an._inner(i, i + 1)
        closed = an.work(float(lo), float(hi))
        coarse, fine = an.work_lower(lo, hi, 16), an.work_lower(lo, hi, 256)
        assert 0 <= coarse <= fine
        assert float(fine) <= closed*(1 + 1e-12)
        # First-order convergence: the cell bound uses one global sup|A''|.
        assert float(fine) >= 0.9*closed
        checked += 1
    assert checked >= 2
