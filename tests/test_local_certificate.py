"""Validated local graph launches and their holonomy handoff."""

from dataclasses import replace
from fractions import Fraction

import pytest

from spong import local_certificate, portrait, sturm


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

    ledger = portrait.build_ledger(result, {})
    pencil = ledger["hyperelliptic_pencil"]

    assert pencil["implemented"][
        "generic_interval_local_launch_certifier[VALIDATED]"]
    assert pencil["implemented"][
        "portrait_local_launches_materialized[VALIDATED]"]
    assert pencil["validated_local_launch_count"] == 8
    assert pencil["status"] == \
        "portrait local launches validated; holonomy engine available"
    assert all(len(row["graph_launches"]) == 4
               for row in ledger["local_launches"])
