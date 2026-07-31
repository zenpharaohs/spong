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


def test_repeated_N_constraint_prescribes_exact_saddle_node_wall(mu):
    r = F(-3)
    d = inverse.design(
        [r, r], G, mu, deg_f=5, families=["N", "N'"])
    assert P.eval_at(d.model.N, r) == 0
    assert P.eval_at(P.deriv(d.model.N), r) == 0


def test_reference_recovers_existing_nullspace_representative(mu):
    target = F(2)
    original = inverse.design([target], G, mu, deg_f=5)
    recovered = inverse.design(
        [target], G, mu, deg_f=5, reference=original.f)
    assert recovered.f == original.f


def test_reference_and_combo_are_mutually_exclusive(mu):
    with pytest.raises(ValueError, match="mutually exclusive"):
        inverse.design(
            [F(2)], G, mu, deg_f=5, combo=[1], reference=[1])


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

def test_extreme_stiffness_handoff_remains_resolved(mu):
    """Difficulty is not inferred from one coarse gauge sample.

    A far target can have a mild saddle followed by an overwhelmingly shallow
    tail.  The exact pointwise gauge must own that short engine prefix, while
    the Hadamard graph owns the enormous remainder.  Difficulty is governed
    by the handoff, not by the eventual magnitude of kappa.
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
    # The saddle itself is mild even though the far tail is overwhelmingly
    # shallow.  A coarse target-spanning sounding grid used to hide this.
    assert extreme.diag["kappa_spectral_saddle"] < charts.KAPPA_HI
    assert any(zone[0] == "shallow"
               for zone in extreme.diag.get("zones", []))
    assert extreme.diag.get("switches", 0) == 0
    # Both independently constructed seams remain at a negligible physical
    # scale; their ordering is not a correctness property.
    assert seam_mid < 1e-8 and seam_far < 1e-8


def test_the_depth_gauge_saturates_and_the_cap_is_known(mu):
    """depth_gauge_floor divides by max(|w1'|, 1e-16 |a*'|), so it cannot
    report more than 2e16.  A reading AT the cap means 'at least', not 'equal'
    -- anything gating on the magnitude beyond that needs the raw ratio."""
    d = inverse.design([F(2) ** 20], G, mu)
    assert inverse.depth_gauge_at(d.model, F(2) ** 20) == pytest.approx(2e16,
                                                                       rel=1e-9)


# ------------------------------------------ transition-straddling suite

def test_raw_crossings_do_not_predict_switches__hysteresis_is_required(mu):
    """The engine enters shallow at KAPPA_HI and leaves only below KAPPA_EXIT.
    A gauge that pokes above 1e4 and falls back to 5e3 never leaves the fixed
    point, so counting raw threshold crossings predicts nothing."""
    gauges = [1e2, 2e4, 5e3, 2e4, 1e2]          # two raw crossings of KAPPA_HI
    raw = sum(1 for a, b in zip(gauges, gauges[1:])
              if (a >= charts.KAPPA_HI) != (b >= charts.KAPPA_HI))
    assert raw == 4          # F,T,F,T,F -> four raw transitions
    # hysteretically: enter at 2e4, the 5e3 dip does NOT exit, leave at the end
    assert inverse.hysteretic_zones(gauges) == 2


def test_suite_finds_straddling_cases(mu):
    suite = inverse.straddling_suite(G, mu, min_zones=1)
    assert suite, "no straddling cases generated"
    assert all(c.predicted_zones >= 1 for c in suite)
    # ranked hardest first: most transitions, then most evenly split
    assert suite == sorted(
        suite, key=lambda c: (-c.predicted_zones,
                              -min(c.shallow_fraction, 1 - c.shallow_fraction)))


def test_verified_suite_catches_what_screening_misses(mu):
    """The cheap screen samples the gauge on a straight b-grid; the engine
    follows the trajectory.  g4 at b* = 20480 screens as zero transitions and
    the engine finds several zones, so a suite built on prediction alone still
    omits the case it most needs -- which is why verify_all exists."""
    g4 = [F(1), F(1), F(1, 2), F(1, 6), F(1, 24)]
    c = inverse.verify(inverse.straddle_case([F(20480)], g4, mu))
    assert c.predicted_zones == 0            # screening says: nothing to see
    assert c.actual_zones >= 3               # the engine says otherwise
    assert c.mispredicted
    assert c.term == "capture"


def test_no_shallow_zone_runs_backward(mu):
    """Regression: grid_index rounds to NEAREST, so bg[i_cur] can lie behind
    b_cur.  When the KAPPA_EXIT walk did not advance, the shallow zone ran
    BACKWARD -- at g4/b*=20480 it entered at 4.674 and ended at 3.381, the
    engine re-ran the same ground, and the re-entry seam was 3.42 with an angle
    energy of 32.  Every shallow zone must advance toward the target."""
    g4 = [F(1), F(1), F(1, 2), F(1, 6), F(1, 24)]
    d = inverse.design([F(20480)], g4, mu)
    s, t, _ = inverse.branch_span(d.model)
    sgn = 1.0 if float(t.b) > float(s.b) else -1.0
    br = charts.trace_unstable(d.model, s.b, (t.a, t.b),
                               box=(-1e9, 1e9, -1e9, 1e9),
                               ds=abs(t.b - s.b) / 2000.0)
    for zone in br.diag.get("zones", []):
        if zone[0] == "shallow":
            assert (zone[2] - zone[1]) * sgn > 0, f"backward zone: {zone}"
    seams = [abs(float(x)) for x in br.certs.get("seam_residuals", [])]
    assert max(seams, default=0.0) < 1e-3, seams
    assert abs(float(br.certs.get("angle_energy") or 0.0)) < 1e-1


def test_suite_dedupes_radii(mu):
    radii = [F(2) ** 12, F(4096), F(2) ** 13]      # first two are equal
    suite = inverse.straddling_suite(G, mu, radii=radii, min_zones=0)
    got = [c.design.prescribed[0] for c in suite]
    assert len(got) == len(set(got))
