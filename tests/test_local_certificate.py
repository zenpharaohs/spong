"""Validated local graph launches and their holonomy handoff."""

from dataclasses import replace
from fractions import Fraction

import pytest

from spong import hyperelliptic, local_certificate, portrait, sturm


@pytest.fixture(scope="module")
def validated_d2(d2):
    m, enumeration = d2
    return m, sturm.materialize_validated_launches(m, enumeration)


def test_every_degree_two_saddle_branch_gets_an_exact_launch(validated_d2):
    _m, enumeration = validated_d2
    launches = [stub.validated_launch
                for point in enumeration.saddles for stub in point.stubs]

    assert len(launches) == 8
    assert all(launch.validated for launch in launches)
    assert {launch.time_direction for launch in launches
            if launch.manifold == "stable"} == {1}
    assert {launch.time_direction for launch in launches
            if launch.manifold == "unstable"} == {-1}
    for launch in launches:
        assert launch.status_code == int(local_certificate.LocalCertificateStatus.OK)
        assert launch.flow_margin > 0
        assert launch.lower_face_margin > 0
        assert launch.upper_face_margin > 0
        assert launch.b_interval.width > 0
        assert launch.y_interval.width > 0
        assert launch.work.cone_tests <= 25*97
        assert launch.work.section_bisections == 80
        assert launch.work.peak_endpoint_bits <= 16384
        assert launch.as_dict()["method"] == \
            "exact-rational invariant-cone graph transform"


def test_singular_rational_frame_is_an_explicit_refusal(validated_d2):
    m, enumeration = validated_d2
    point = enumeration.saddles[0]
    chart = replace(point.local.poincare[0],
                    frame=((1.0, 1.0), (1.0, 1.0)))

    launch = local_certificate.certify_poincare_launch(
        m, point, chart, orientation=1)

    assert not launch.validated
    assert launch.status_code == int(
        local_certificate.LocalCertificateStatus.FRAME_SINGULAR)
    assert "singular" in launch.reason


def test_endpoint_swell_budget_is_an_explicit_refusal(validated_d2):
    m, enumeration = validated_d2
    point = enumeration.saddles[0]
    chart = point.local.poincare[0]

    launch = local_certificate.certify_poincare_launch(
        m, point, chart, orientation=1, max_endpoint_bits=64)

    assert not launch.validated
    assert launch.status_code == int(
        local_certificate.LocalCertificateStatus.WORK_BUDGET)
    assert launch.work.peak_endpoint_bits > 64


def test_ledger_distinguishes_capability_from_materialization(validated_d2):
    m, enumeration = validated_d2
    result = portrait.Portrait(m, enumeration, [], (-2, 2, -2, 2), None)

    ledger = portrait.build_ledger(result, {}, certify_complex=True)
    pencil = ledger["hyperelliptic_pencil"]

    assert pencil["engine_capabilities[STATIC]"][
        "interval_local_launch_certifier"]
    assert pencil["portrait_local_launches_materialized[VALIDATED]"]
    assert pencil["validated_local_launch_count"] == 8
    assert pencil["status"] == \
        "portrait local launches validated; holonomy engine available"
    assert all(len(row["graph_launches"]) == 4
               for row in ledger["local_launches"])


def _lifted_tail(m, vertices, launch, cap: int):
    """Vertices lifted EXACTLY, kept only past the section in the launch's
    level direction with strictly monotone levels.  The tail is cut near the
    saddle: the handoff is what is under test, not far-field holonomy."""
    direction = launch.time_direction
    tail, last = [], launch.section_level
    for a, b in vertices:
        lifted = hyperelliptic.lift_exact(
            m, Fraction(float(a)), Fraction(float(b)))
        if direction*(lifted.level-last) > 0:
            tail.append(lifted)
            last = lifted.level
    if len(tail) > 8*cap:
        tail = tail[:8*cap:8]
    elif len(tail) > cap:
        tail = tail[:cap]
    return tail


def _launch_distance(launch, lifted):
    return max(abs(lifted.b-launch.b_interval.midpoint),
               abs(lifted.y-launch.y_interval.midpoint))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "handoff gap found 2026-08-29: 5/8 close. Both unstable launches at "
        "the far saddle b=-9.445 fail at every bisection depth because their "
        "section rectangles contain y=0 (printed: y in [-0.042,0.018] with "
        "K=2.4e-4, so it is not transverse slack). The branch departs along "
        "the backbone, so the true y=A(a-a*) is O(0.01) there, the same "
        "order as the rectangle's interval width; |grad L|^2 then cannot "
        "exclude the critical point at any slab depth. Fix in "
        "local_certificate: a tighter y evaluation on the section slab, or a "
        "section pushed out until |y| dominates its width. The stable +1 "
        "launch at b=-0.517 hands off and closes 25 slabs from the traced "
        "branch before the inflating tube meets a critical point: intrinsic "
        "growth along an increasing-loss (repelling) direction, a tail-length "
        "choice rather than a certificate defect. Strict: fixing the launch "
        "must flip this to a pass."))
def test_d2_launch_hands_off_to_a_real_model_tube(validated_d2):
    """The fake ``A=1, B=0`` tests in test_hyperelliptic replay the tube
    mechanics on a model whose flow is known in closed form.  This test
    exercises the handoff on a real model: validated invariant-cone launch,
    then floating vertices lifted EXACTLY to their own fibres as centre
    proposals, then the rational trapping-tube replay.  The stub is tried
    first; where it is shorter than the cone reach, the traced separatrix
    of the same saddle (matched by nearness to the launch box) supplies the
    proposal instead.  Nothing floating is trusted: every face inequality
    is re-proved in intervals, and failing slabs are bisected.

    On failure ``reasons`` names the mode per launch.
    """
    m, enumeration = validated_d2
    traced = portrait.compute(m, _enumeration=enumeration, _skip_audit=True)
    closed = 0
    reasons = []
    for point in enumeration.saddles:
        for stub in point.stubs:
            launch = stub.validated_launch
            assert launch.validated
            tail = _lifted_tail(m, stub.curve, launch, cap=24)
            source = "stub"
            if not tail:
                candidates = []
                for br in traced.branches:
                    if br.diag.get("saddle_b") != point.b:
                        continue
                    branch_tail = _lifted_tail(m, br.Y, launch, cap=24)
                    if branch_tail:
                        candidates.append(
                            (_launch_distance(launch, branch_tail[0]),
                             branch_tail))
                if candidates:
                    tail = min(candidates, key=lambda c: c[0])[1]
                    source = "branch"
            if not tail:
                reasons.append((float(point.b), stub.manifold,
                                stub.orientation, "none", 0, 0, 0,
                                "no proposal leaves the section slab"))
                continue
            cert = hyperelliptic.certify_flow_tube_from_launch(
                m, launch, tail, max_inflations=48)
            if cert.status == "validated":
                closed += 1
                assert cert.level_direction == launch.time_direction
                assert cert.knots[0].b_interval == launch.b_interval
                assert cert.knots[0].y_interval == launch.y_interval
                assert cert.slabs[0].level_lo <= launch.section_level \
                    <= cert.slabs[0].level_hi
            else:
                reasons.append((float(point.b), stub.manifold,
                                stub.orientation, source, len(tail),
                                cert.slab_bisections, len(cert.slabs),
                                cert.reason))
    if closed != 8:
        pytest.fail(f"closed {closed}/8 launches; refusals:\n"
                    + "\n".join(
                        "  b=%.4f %-8s %+d source=%s tail=%d bisections=%d "
                        "slabs=%d: %s" % row for row in reasons))
