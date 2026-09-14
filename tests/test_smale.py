"""Portrait-wide, fail-closed finite-plane Smale orchestration."""

from spong import hyperelliptic, model, portrait, smale, sturm


def test_one_saddle_has_no_connection_pairs_and_needs_no_trace():
    m = model.build([1, 2], [1, 1], model.moments_uniform01(5))
    enumeration = sturm.enumerate_critical_points(m)
    assert len(enumeration.saddles) == 1
    measured = portrait.Portrait(m, enumeration, [], (-2, 2, -2, 2), None)

    certificate = smale.certify_finite_plane(m, measured)

    assert certificate.status == "certified_finite_plane_morse_smale"
    assert certificate.validated
    assert not certificate.records
    assert not certificate.ordering_refusals
    # Structural pruning returned before exact local-launch materialization.
    assert all(not point.stubs for point in enumeration.saddles)


def test_degree_two_orchestrator_certifies_every_candidate_pair(d2):
    m, enumeration = d2
    measured = portrait.compute(
        m, _enumeration=enumeration, _skip_audit=True)

    certificate = smale.certify_finite_plane(
        m, measured, enumeration=enumeration, centre_limit=64)

    assert certificate.source_target_pairs == 1
    assert len(certificate.records) == 4
    component = [record for record in certificate.records
                 if record.witness == "launch_sublevel_component"]
    assert len(component) == 2
    assert all(record.connection_excluded for record in component)
    propagated = [record for record in certificate.records
                  if record.witness == "common_fibre_constrained_intervals"]
    assert len(propagated) == 2
    assert all(record.connection_excluded for record in propagated)
    assert any(isinstance(
        record.decision.unstable,  # connection decision retains both tubes
        hyperelliptic.NativeBParameterHandoffCertificate)
        or isinstance(record.decision.stable,
                      hyperelliptic.NativeBParameterHandoffCertificate)
        for record in propagated)
    assert any(isinstance(record.decision.unstable,
                          hyperelliptic.NativeSheetFlowTubeCertificate)
               or isinstance(record.decision.stable,
                             hyperelliptic.NativeSheetFlowTubeCertificate)
               for record in propagated)
    assert certificate.status == "certified_finite_plane_morse_smale"
    assert certificate.validated
    assert certificate.as_dict()["unresolved"] == 0
    assert certificate.as_dict()["finite_plane_morse_smale[VALIDATED]"]
