"""Inverse construction: prescribe critical points, gate on the depth gauge."""

from fractions import Fraction as F

import pytest

from spong import _poly as P
from spong import charts, inverse, model, sturm


G = [F(1), F(1), F(1, 2)]          # g(0) != 0, so A(0) = g_0^2 mu_0 > 0


@pytest.fixture(scope="module")
def mu():
    return model.moments_uniform01(41)


# ---------------------------------------------------------------- exactness

@pytest.mark.parametrize("target", [F(1, 2), F(3), F(-7, 2), F(25)])
def test_prescribed_point_is_an_exact_root(target, mu):
    """Not 'close to a root' -- the residual is exactly zero over Q."""
    d = inverse.design([target], G, mu)
    assert P.eval_at(d.model.N, target) == 0


def test_b_family_prescription_is_an_exact_root_of_B(mu):
    d = inverse.design([F(2)], G, mu, families=["B"])
    assert P.eval_at(d.model.beta, F(2)) == 0


@pytest.mark.parametrize("target", [F(1, 2), F(3), F(-7, 2), F(25)])
def test_enumeration_contains_the_prescription(target, mu):
    r = inverse.report(inverse.design([target], G, mu))
    assert r.realised == (target,)
    assert r.missing == ()


def test_multiple_points_simultaneously(mu):
    """Needs deg(g) >= number of points -- see the cap test below."""
    pts = [F(1, 2), F(-2), F(5)]
    g4 = [F(1), F(1), F(1, 2), F(1, 6), F(1, 24)]
    d = inverse.design(pts, g4, mu)
    assert d.model.N != ()            # not the vacuous B == 0 solution
    for b in pts:
        assert P.eval_at(d.model.N, b) == 0
    r = inverse.report(d)
    assert set(r.realised) == set(pts) and r.missing == ()


def test_prescription_is_capped_by_the_degree_of_g(mu):
    """Every condition is a functional on B, which has only deg(g)+1
    coefficients; deg(g)+1 of them force B == 0, satisfying the prescription
    vacuously with no critical points at all.  Raising deg_f does not help."""
    with pytest.raises(ValueError, match=r"at most deg\(g\)=2"):
        inverse.design([F(1), F(2), F(3)], G, mu, deg_f=12)


def test_never_returns_the_vacuous_zero_B_solution(mu):
    """B == 0 satisfies any prescription but has no critical points."""
    for pts in ([F(2)], [F(1, 2), F(4)]):
        d = inverse.design(pts, G, mu)
        assert d.model.beta != (), pts
        assert d.model.N != (), pts


# ------------------------------------------------------------ preconditions

def test_rejects_activation_with_vanishing_constant_term(mu):
    """A(0) = g_0^2 mu_0; a vanishing g(0) puts a pole of a* = B/A at b = 0."""
    with pytest.raises(ValueError, match=r"g\(0\) must be nonzero"):
        inverse.design([F(1)], [F(0), F(1), F(1, 2)], mu)


def test_rejects_when_f_has_too_few_coefficients(mu):
    g4 = [F(1), F(1), F(1, 2), F(1, 6), F(1, 24)]
    with pytest.raises(ValueError, match="no nonzero f|B identically zero"):
        inverse.design([F(1), F(2), F(3)], g4, mu, deg_f=1)


def test_rejects_mismatched_families(mu):
    with pytest.raises(ValueError, match="families must match"):
        inverse.design([F(1), F(2)], G, mu, families=["N"])


# ------------------------------------------------------------- depth gauge

def test_gauge_is_the_dispatcher_gauge_not_the_spectral_one(mu):
    """`sounding` is diagnostic only and lies at u-inflections; the gate must
    read what `trace_unstable` branches on."""
    d = inverse.design([F(16)], G, mu)
    b = 16.0
    off = 1e-6 * abs(b)
    expected = min(float(charts.depth_gauge_floor(d.model, b + off)),
                   float(charts.depth_gauge_floor(d.model, b - off)))
    assert inverse.depth_gauge_at(d.model, F(16)) == pytest.approx(expected,
                                                                  rel=1e-12)


def test_gauge_climbs_with_backbone_position(mu):
    """Stiffness is a position effect: the whole point of prescribing |b|."""
    gauges = [inverse.depth_gauge_at(inverse.design([F(r)], G, mu).model, F(r))
              for r in (1, 2, 4, 16, 64, 256)]
    assert all(a < b for a, b in zip(gauges, gauges[1:])), gauges
    assert gauges[-1] / gauges[0] > 1e10          # measured ~2e14


def test_gate_selects_the_shallow_water_zone(mu):
    mild = inverse.design([F(1, 2)], G, mu)
    stiff = inverse.design([F(64)], G, mu)
    assert not inverse.is_shallow(mild.model, F(1, 2))
    assert inverse.is_shallow(stiff.model, F(64))
    assert inverse.depth_gauge_at(stiff.model, F(64)) >= charts.KAPPA_HI


# ------------------------------------------------------------------ tracing

@pytest.mark.parametrize("target", [F(4), F(64), F(256)])
def test_designed_stiff_branches_still_trace_cleanly(target, mu):
    """The ladder these designs exist to build: at gauges up to ~1e16 the
    engine must still terminate cleanly, not abort."""
    d = inverse.design([target], G, mu)
    assert inverse.is_shallow(d.model, target)
    e = sturm.enumerate_critical_points(d.model)
    assert e.morse and e.psi_positive
    traced = 0
    for s in e.saddles:
        side = [p for p in e.minima if p.b > s.b] or \
               [p for p in e.minima if p.b < s.b]
        if not side:
            continue
        t = min(side, key=lambda p: abs(p.b - s.b))
        br = charts.trace_unstable(d.model, s.b, (t.a, t.b),
                                   box=(-1e4, 1e4, -1e4, 1e4),
                                   ds=abs(t.b - s.b) / 2000.0)
        assert br.term in ("capture", "box_exit"), br.term
        traced += 1
    assert traced >= 1


def test_extras_are_reported_not_hidden(mu):
    """The construction contains the prescription; surplus roots are visible."""
    r = inverse.report(inverse.design([F(3)], G, mu))
    assert set(r.realised) == {F(3)}
    assert len(r.critical) == len(r.realised) + len(r.extras)
    assert r.morse and r.alternates


def test_stiffness_ladder_is_monotone_in_gauge(mu):
    rungs = inverse.stiffness_ladder([F(1), F(4), F(16), F(64)], G, mu)
    gauges = []
    for radius, d, rep in rungs:
        assert d is not None, rep
        assert rep.missing == ()
        gauges.append(rep.gauges[0])
    assert all(a < b for a, b in zip(gauges, gauges[1:])), gauges


# ------------------------------------------------- the knob, turned till it breaks

def test_extreme_stiffness_is_easier_than_the_transition(mu):
    """Difficulty is NOT monotone in the gauge.

    At the mild/shallow boundary a branch straddles both zones and pays for
    the handoff; past it the branch is entirely shallow water, the Hadamard
    fixed point owns all of it, and the seam falls back to roundoff.  So the
    instrument is stressed by the HANDOFF, not by the magnitude of kappa --
    which is why pushing kappa alone finds nothing.
    """
    def trace(target):
        d = inverse.design([target], G, mu)
        e = sturm.enumerate_critical_points(d.model)
        s = e.saddles[0]
        side = [p for p in e.minima if p.b > s.b] or \
               [p for p in e.minima if p.b < s.b]
        t = min(side, key=lambda p: abs(p.b - s.b))
        br = charts.trace_unstable(d.model, s.b, (t.a, t.b),
                                   box=(-1e9, 1e9, -1e9, 1e9),
                                   ds=abs(t.b - s.b) / 2000.0)
        seams = br.certs.get("seam_residuals", [])
        return br, max((abs(float(x)) for x in seams), default=0.0)

    transition, seam_mid = trace(F(2) ** 14)
    extreme, seam_far = trace(F(2) ** 18)

    assert transition.term == "capture" and extreme.term == "capture"
    # the far case is uniformly shallow: one zone, no chart handoffs
    assert len(extreme.diag.get("zones", [])) == 1
    assert extreme.diag.get("switches", 0) == 0
    # and is orders of magnitude cleaner than the straddling case
    assert seam_far < seam_mid


def test_the_depth_gauge_saturates_and_the_cap_is_known(mu):
    """depth_gauge_floor divides by max(|w1'|, 1e-16 |a*'|), so it cannot
    report more than 2e16.  A reading AT the cap means 'at least', not 'equal'
    -- anything gating on the magnitude beyond that needs the raw ratio."""
    d = inverse.design([F(2) ** 20], G, mu)
    assert inverse.depth_gauge_at(d.model, F(2) ** 20) == pytest.approx(2e16,
                                                                       rel=1e-9)
