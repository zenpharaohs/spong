"""Validated complex divisor of the reduced meromorphic backbone."""

from fractions import Fraction

import pytest

from spong import _poly as P, model, portrait, sturm
from spong.complex_structure import (certify_backbone,
                                     certify_polynomial_roots,
                                     schur_cohn_count_disk)


def common_factor_model():
    # A=(1+2b²)(1+6b²), B=1+2b².  The first factor cancels from
    # B²/A, so ±i/sqrt(2) are removable rather than poles.
    return model.build([1], [1, 2, 2], model.moments_normal01(7))


def test_complex_certificate_uses_the_reduced_backbone():
    m = common_factor_model()
    certificate = certify_backbone(m)

    assert certificate.complete
    assert certificate.transverse.degree == 4
    assert certificate.valley_denominator.degree == 2
    assert certificate.denominator.degree == 2
    assert len(certificate.denominator.disks) == 2
    assert certificate.critical.degree == 1
    assert len(certificate.critical.disks) == 1
    assert certificate.denominator.real_axis_clearance() == pytest.approx(
        1/6**0.5, rel=1e-13)

    # The cancelled ±i/sqrt(2) pair is recorded algebraically, but does not
    # contaminate either the pole or critical divisor.
    ledger = certificate.as_dict()
    assert ledger["cancelled_factor_degree[EXACT]"] == 2
    assert ledger["transverse_zero_divisor"]["degree[EXACT]"] == 4
    assert ledger["valley_chart_poles"]["degree[EXACT]"] == 2
    assert ledger["backbone_poles"]["degree[EXACT]"] == 2
    assert ledger["critical_divisor"]["degree[EXACT]"] == 1


def test_every_reported_disk_has_an_exact_positive_rouche_margin():
    certificate = certify_backbone(common_factor_model())
    for divisor in (certificate.transverse, certificate.valley_denominator,
                    certificate.denominator, certificate.critical):
        assert divisor.complete
        assert len(divisor.disks) == divisor.degree
        for disk in divisor.disks:
            assert isinstance(disk.centre_re, Fraction)
            assert isinstance(disk.centre_im, Fraction)
            assert isinstance(disk.radius, Fraction)
            assert disk.rouche_margin > 0
        assert all(a.disjoint(b) for i, a in enumerate(divisor.disks)
                   for b in divisor.disks[i+1:])


def test_exact_lehmer_schur_disk_count_and_boundary_refusal():
    # (z-.2)(z-.3): both roots are in |z|<1/2.
    inside = P.poly([Fraction(3, 50), Fraction(-1, 2), 1])
    counted = schur_cohn_count_disk(
        inside, (Fraction(0), Fraction(0)), Fraction(1, 2))
    assert counted is not None and counted[0] == 2
    assert all(sign in (-1, 1) for _, sign in counted[1])

    # (z-2)(z-3): no root is in |z|<1, while radius 3 puts a root
    # exactly on the boundary and must decline rather than guess.
    outside = P.poly([6, -5, 1])
    counted = schur_cohn_count_disk(
        outside, (Fraction(0), Fraction(0)), Fraction(1))
    assert counted is not None and counted[0] == 0
    assert schur_cohn_count_disk(
        outside, (Fraction(0), Fraction(0)), Fraction(3)) is None


def test_high_degree_simple_divisor_uses_exact_rouche_witnesses():
    certificate = certify_polynomial_roots(
        P.poly([1] + [0]*11 + [1]))  # z^12+1

    assert certificate.complete
    assert len(certificate.disks) == 12
    assert all(disk.root_count == 1 and disk.rouche_margin > 0
               and not disk.schur_trace for disk in certificate.disks)


def test_ledger_uses_the_actual_poincare_spectral_ratio():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    enumeration = sturm.enumerate_critical_points(m)
    p = portrait.Portrait(m, enumeration, [], (-2, 2, -2, 2), None)
    ledger = portrait.build_ledger(p, {}, certify_complex=True)

    assert ledger["complex_backbone"]["status"] == "validated"
    assert ledger["hyperelliptic_pencil"]["generic_genus[EXACT]"] >= 1
    pencil = ledger["hyperelliptic_pencil"]
    assert pencil["status"] == \
        "certificate engine present; portrait launch pending"
    capabilities = pencil["engine_capabilities[STATIC]"]
    assert capabilities["genus_zero_residue_log_reduction"]
    assert capabilities["rational_flow_tubes"]
    assert capabilities["interval_local_launch_certifier"]
    assert not capabilities["positive_genus_unwrapped_period_transport"]
    # Static capabilities never carry a per-portrait epistemic tag.
    assert not any("[" in key for key in capabilities)
    assert not pencil["portrait_local_launches_materialized[VALIDATED]"]
    assert pencil["validated_local_launch_count"] == 0
    assert len(ledger["local_launches"]) == len(enumeration.saddles)
    for row, saddle in zip(ledger["local_launches"], enumeration.saddles):
        lm, lp = saddle.local.spectral.eigenvalues
        assert row[
            "poincare_transverse_departure_ratio[HIGH_PRECISION]"
        ] == pytest.approx(lp/lm, rel=1e-15)
        assert row["backbone_pole_clearance_lower[VALIDATED]"] > 0
        assert row["valley_chart_pole_clearance_lower[VALIDATED]"] > 0
        assert "Frobenius convergence radius" in row["scope"]
