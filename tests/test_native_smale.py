"""Differential boundary for the native exact Morse--Smale kernels."""

from fractions import Fraction
from types import SimpleNamespace

from spong import _native, hyperelliptic, local_certificate, sturm


def _synthetic_case():
    # A=1, B=0, N=1 makes the complementary chart exactly dy/db=2.
    # It is a deliberate independent-vector-field test, so N verification is
    # disabled; production Model inputs always enable the identity check.
    model = SimpleNamespace(
        alpha=(Fraction(1),), beta=(), N=(Fraction(1),), C=Fraction(0))
    centres = []
    for index in range(4):
        b = Fraction(index, 8)
        y = 1+2*b
        centres.append(hyperelliptic.LiftedPoint(y*y, b, y, y))
    radius = Fraction(1, 2**40)
    target = Fraction(121, 100)
    return model, centres, radius, target


def test_native_b_parameter_handoff_matches_fraction_oracle_exactly():
    model, centres, radius, target = _synthetic_case()
    tube = hyperelliptic.certify_b_parameter_flow_tube(
        model, centres,
        initial_y_interval=hyperelliptic.RationalInterval(
            1-radius, 1+radius),
        launch_validated=True, max_inflations=8)
    oracle = hyperelliptic.project_b_parameter_tube_to_level(
        model, tube, target, max_bisection_depth=10)

    native = _native.b_parameter_handoff(
        model.alpha, (Fraction(0),), model.N, model.C,
        [(point.level, point.b, point.y) for point in centres],
        (1-radius, 1+radius), target,
        max_inflations=8, max_projection_bisections=10,
        verify_n_identity=False)
    as_fraction = lambda value: Fraction(*value)

    assert native["status"] == _native.SPONG_SMALE_OK
    assert native["tube_validated"]
    assert native["projection_validated"]
    assert tuple(as_fraction(value) for value in native["target_b"]) == (
        oracle.b_interval.lo, oracle.b_interval.hi)
    assert tuple(as_fraction(value) for value in native["target_y"]) == (
        oracle.y_interval.lo, oracle.y_interval.hi)
    prefix_minimum = min(
        tube.slabs[0].lower_face_margin,
        tube.slabs[0].upper_face_margin)
    assert as_fraction(native["minimum_face_margin"]) == prefix_minimum
    assert as_fraction(native["minimum_gradient_norm_squared"]) == \
        tube.slabs[0].gradient_norm_squared_lower


def test_native_b_parameter_handoff_checks_model_identity_and_work_cap():
    model, centres, radius, target = _synthetic_case()
    arguments = (
        model.alpha, (Fraction(0),), model.N, model.C,
        [(point.level, point.b, point.y) for point in centres],
        (1-radius, 1+radius), target)

    identity = _native.b_parameter_handoff(*arguments)
    assert identity["status"] == _native.SPONG_SMALE_MODEL_IDENTITY_FAILURE
    assert not identity["projection_validated"]

    limited = _native.b_parameter_handoff(
        *arguments, verify_n_identity=False, max_rational_bits=64,
        radius_round_bits=32)
    assert limited["status"] == _native.SPONG_SMALE_WORK_LIMIT
    assert limited["reason_name"] == "endpoint_bits"
    assert not limited["projection_validated"]


def test_native_fixed_sheet_tube_matches_fraction_oracle_exactly():
    model = SimpleNamespace(
        alpha=(Fraction(1),), beta=(Fraction(0), Fraction(1)),
        N=(Fraction(-2),),
        C=Fraction(0))
    step = Fraction(1, 1024)
    centres = (
        hyperelliptic.HolonomyCentre(
            Fraction(1), Fraction(0), Fraction(1)),
        hyperelliptic.HolonomyCentre(
            Fraction(1)+step, Fraction(-1, 4096), Fraction(1)),
        hyperelliptic.HolonomyCentre(
            Fraction(1)+2*step, Fraction(-2, 4096), Fraction(1)))
    radius = Fraction(1, 2**30)
    initial_b = hyperelliptic.RationalInterval(-radius, radius)
    initial_y = hyperelliptic.RationalInterval(1-radius, 1+radius)
    projection = hyperelliptic.certify_fibre_projection(
        model, centres[0].level, initial_b, initial_y)
    oracle = hyperelliptic.certify_sheet_flow_tube(
        model, centres, initial_b_interval=projection.projected_b,
        sheet=1, launch_validated=True, launch_projection=projection,
        max_inflations=8)
    native = hyperelliptic.certify_sheet_flow_tube_native(
        model, centres, initial_b_interval=initial_b,
        initial_y_interval=initial_y, sheet=1, max_inflations=8)

    assert oracle.status == native.status == "validated"
    assert native.projection_validated
    assert native.b_interval == oracle.knots[-1].b_interval
    assert native.y_interval == oracle.knots[-1].y_interval
    assert native.minimum_face_margin == min(
        margin for slab in oracle.slabs for margin in
        (slab.lower_face_margin, slab.upper_face_margin))
    assert native.minimum_gradient_norm_squared == min(
        slab.gradient_norm_squared_lower for slab in oracle.slabs)


def test_native_fixed_sheet_tube_refuses_wrong_sheet_and_work_cap():
    model = SimpleNamespace(
        alpha=(Fraction(1), Fraction(0), Fraction(-1)),
        beta=(Fraction(1),), N=(Fraction(0), Fraction(-2)),
        C=Fraction(0))
    centres = (
        hyperelliptic.HolonomyCentre(
            Fraction(0), Fraction(0), Fraction(1)),
        hyperelliptic.HolonomyCentre(
            Fraction(5, 4), Fraction(0), Fraction(3, 2)))
    radius = Fraction(1, 2**20)
    arguments = (
        model.alpha, model.beta, model.N, model.C,
        tuple((point.level, point.b, point.y) for point in centres),
        (-radius, radius), (1-radius, 1+radius))
    wrong = _native.sheet_flow_tube(*arguments, -1)
    assert wrong["status"] == _native.SPONG_SMALE_PROJECTION_UNRESOLVED
    assert wrong["reason_name"] == "target_sheet"
    limited = _native.sheet_flow_tube(
        *arguments, 1, max_rational_bits=64, radius_round_bits=32)
    assert limited["status"] == _native.SPONG_SMALE_WORK_LIMIT
    assert not limited["tube_validated"]


def test_native_local_launch_checks_root_and_returns_one_sheet():
    # L=a^2(1-b^2)-2a has an N-saddle at (1,0); the identity frame is its
    # exact eigenframe and b=0 is its unstable invariant manifold.
    arguments = (
        (Fraction(1), Fraction(0), Fraction(-1)),
        (Fraction(1),), (Fraction(0), Fraction(-2)), Fraction(0),
        "N", (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(0),)*6, Fraction(2), Fraction(1, 16), 1)
    result = _native.local_launch(*arguments, require_frobenius=True)
    assert result["status"] == _native.SPONG_SMALE_OK
    assert result["validated"]
    assert result["cone_power"] == 2
    assert tuple(Fraction(*value) for value in result["tangent_slope"]) == (
        Fraction(0), Fraction(0))
    assert Fraction(*result["section_y"][0]) > 0

    bad = list(arguments)
    bad[5] = (Fraction(1, 2), Fraction(1, 2))
    refused = _native.local_launch(*bad, require_frobenius=False)
    assert not refused["validated"]
    assert refused["reason_name"] == "root_interval"

    not_saddle = list(arguments)
    not_saddle[0] = (Fraction(1), Fraction(0), Fraction(1))
    not_saddle[2] = (Fraction(0), Fraction(2))
    refused = _native.local_launch(*not_saddle, require_frobenius=True)
    assert not refused["validated"]
    assert refused["reason_name"] == "critical_point"


def test_native_local_launch_matches_fraction_oracle_exactly():
    model = SimpleNamespace(
        alpha=(Fraction(1), Fraction(0), Fraction(-1)),
        beta=(Fraction(1),), N=(Fraction(0), Fraction(-2)),
        C=Fraction(0))
    root = sturm.RootInterval(Fraction(0), Fraction(0), True)
    point = SimpleNamespace(
        source="N", interval=root,
        local=SimpleNamespace(center_interval=root))
    chart = SimpleNamespace(
        manifold="unstable", frame=((1.0, 0.0), (0.0, 1.0)),
        selected_map=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        eigenvalues=(2.0, -2.0), desired_reach=1/16)

    oracle = local_certificate.certify_poincare_launch(
        model, point, chart, 1, frobenius_graph=True)
    native = local_certificate.certify_poincare_launch_native(
        model, point, chart, 1)
    assert oracle.validated and native.validated
    for field in (
            "reach", "cone_slope", "flow_margin", "lower_face_margin",
            "upper_face_margin", "tangent_slope_interval", "section_level",
            "b_interval", "y_interval", "cone_power"):
        assert getattr(native, field) == getattr(oracle, field)


def test_native_frobenius_launch_closes_on_degree_two(d2):
    model, enumeration = d2
    point = next(point for point in enumeration.points
                 if point.kind == "saddle")
    chart = point.local.poincare[0]
    launch = local_certificate.certify_poincare_launch_native(
        model, point, chart, -1)
    assert launch.validated
    assert launch.engine == "native_gmp"
    assert launch.cone_power == 2
    assert launch.reach == Fraction(float(chart.desired_reach))/32
    assert launch.y_interval is not None
    assert not launch.y_interval.contains_zero()
    assert launch.b_section is not None
    assert launch.b_section.validated
    assert not launch.b_section.y_interval.contains_zero()
