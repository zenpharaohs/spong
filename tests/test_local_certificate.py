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
    m, enumeration = validated_d2
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
        # The section rectangle is a flow box for the lifted transport: this
        # is the handoff contract, checked here independently of any tube.
        hyperelliptic.level_transport_interval(
            m, launch.b_interval, launch.y_interval)
        assert not launch.y_interval.contains_zero()
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


def _handoff(m, enumeration, traced, caps: dict):
    """Run every validated launch into a rational trapping tube.

    Centres are floating vertices lifted EXACTLY to their own fibres: the
    stub first, else the traced separatrix of the same saddle matched by
    nearness to the launch box.  Nothing floating is trusted; every face
    inequality is re-proved in intervals and failing slabs are bisected.
    Returns ``(closed, refusal_lines)``.
    """
    closed = 0
    reasons = []
    for point in enumeration.saddles:
        for stub in point.stubs:
            launch = stub.validated_launch
            assert launch.validated
            cap = caps[stub.manifold]
            tail = _lifted_tail(m, stub.curve, launch, cap=cap)
            source = "stub"
            if not tail:
                candidates = []
                for br in traced.branches:
                    if br.diag.get("saddle_b") != point.b:
                        continue
                    branch_tail = _lifted_tail(m, br.Y, launch, cap=cap)
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
    lines = ["  b=%.4f %-8s %+d source=%s tail=%d bisections=%d slabs=%d: %s"
             % row for row in reasons]
    return closed, lines


@pytest.fixture(scope="module")
def traced_d2(validated_d2):
    m, enumeration = validated_d2
    return portrait.compute(m, _enumeration=enumeration, _skip_audit=True)


def test_every_d2_launch_hands_off_to_a_real_model_tube(validated_d2,
                                                         traced_d2):
    """The fake ``A=1, B=0`` tests in test_hyperelliptic replay the tube
    mechanics on a model whose flow is known in closed form.  This is the
    handoff on a real model: each validated cone launch must seed a tube
    that closes through its first few downstream centres.  Short tails on
    purpose: the handoff is under test, not how far a level-parametrised
    tube can follow a separatrix (see the continuation test below).

    History: 2026-08-29 this closed 5/8; the two far-saddle unstable
    rectangles contained y=0 because the cone's transverse slack, thin in
    (a,b), is stretched by A into a y-width of the same order as the
    branch's second-order departure from the backbone.  Halving the reach
    when the rectangle fails the flow-box contract fixed it (slack ~R^2,
    signal ~R).
    """
    m, enumeration = validated_d2
    closed, lines = _handoff(m, enumeration, traced_d2,
                             {"unstable": 3, "stable": 3})
    if closed != 8:
        pytest.fail(f"closed {closed}/8 launches; refusals:\n"
                    + "\n".join(lines))


def test_d2_tubes_continue_along_the_separatrices(validated_d2, traced_d2):
    """Same launches, longer tails: the level-parametrised tube follows
    each separatrix well past the launch neighbourhood.

    History: with the flow-box contract alone this closed 7/8; the unstable
    -1 branch at b=-9.445 closed six slabs and then inflated onto a critical
    point, because its rectangle still straddled y=0 and the 2Ay/|grad L|^2
    term of dy/dlevel straddled with it.  The one-sheet contract (y
    one-signed on the rectangle) closed all eight, long tails included.
    """
    m, enumeration = validated_d2
    closed, lines = _handoff(m, enumeration, traced_d2,
                             {"unstable": 24, "stable": 6})
    if closed != 8:
        pytest.fail(f"closed {closed}/8 launches; refusals:\n"
                    + "\n".join(lines))
