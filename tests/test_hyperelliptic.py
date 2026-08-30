"""Exact identities for the hyperelliptic loss pencil and its holonomy."""

from fractions import Fraction
from math import log
from types import SimpleNamespace

import pytest

from spong import _poly as P, hyperelliptic, model, sturm


@pytest.fixture
def m():
    return model.build([1, -2, 3], [2, 1, -1],
                       model.moments_uniform01(5))


def test_exact_lift_lies_on_the_loss_pencil(m):
    point = hyperelliptic.lift_exact(m, Fraction(3, 5), Fraction(2, 7))
    S = hyperelliptic.level_polynomial(m, point.level)

    assert point.y*point.y == P.eval_at(S, point.b)
    assert hyperelliptic.reconstruct_a_exact(m, point.b, point.y) == point.a


def test_curve_gradient_is_the_physical_gradient(m):
    point = hyperelliptic.lift_exact(m, Fraction(-4, 9), Fraction(3, 8))
    La, Lb = hyperelliptic.curve_gradient_exact(m, point.b, point.y)
    A = P.eval_at(m.alpha, point.b)
    B = P.eval_at(m.beta, point.b)
    Ap = P.eval_at(P.deriv(m.alpha), point.b)
    Bp = P.eval_at(P.deriv(m.beta), point.b)

    assert La == 2*(A*point.a-B)
    assert Lb == Ap*point.a**2-2*Bp*point.a


def test_level_transport_is_tangent_to_the_hyperelliptic_surface(m):
    point = hyperelliptic.lift_exact(m, Fraction(5, 6), Fraction(-2, 5))
    transport = hyperelliptic.level_transport_exact(
        m, point.level, point.b, point.y)
    S = hyperelliptic.level_polynomial(m, point.level)
    A = P.eval_at(m.alpha, point.b)
    Sb = P.eval_at(P.deriv(S), point.b)

    assert transport.db_dlevel == (
        transport.loss_b/transport.gradient_norm_squared)
    assert (2*point.y*transport.dy_dlevel
            - A-Sb*transport.db_dlevel) == 0


def test_abel_basis_and_level_derivative_are_exact(m):
    point = hyperelliptic.lift_exact(m, Fraction(7, 10), Fraction(1, 3))
    values = hyperelliptic.abel_basis_values_exact(m, point.b, point.y)
    derivatives = hyperelliptic.abel_level_derivative_values_exact(
        m, point.b, point.y)
    A = P.eval_at(m.alpha, point.b)

    assert len(values) == hyperelliptic.generic_genus(m)
    assert len(derivatives) == len(values)
    for k, (value, derivative) in enumerate(zip(values, derivatives)):
        assert value == point.b**k/point.y
        assert derivative == -A*point.b**k/(2*point.y**3)


def test_regular_lifted_transport_crosses_an_ordinary_branch_point():
    # L=a^2-2ab has y=a-b and y=0 is a regular branch point when b != 0.
    fake = SimpleNamespace(
        alpha=(Fraction(1),), beta=(Fraction(0), Fraction(1)),
        N=(Fraction(-2),), C=Fraction(0))
    point = hyperelliptic.LiftedPoint(
        Fraction(-1), Fraction(1), Fraction(0), Fraction(1))

    transport = hyperelliptic.level_transport_exact(
        fake, point.level, point.b, point.y)

    assert transport.loss_a == 0
    assert transport.loss_b == -2
    assert transport.db_dlevel == Fraction(-1, 2)
    assert transport.dy_dlevel == Fraction(1, 2)


def test_fibre_certificate_counts_every_complex_branch_point(m):
    fibre = hyperelliptic.certify_fibre(m, Fraction(7, 5))

    assert fibre.regular
    assert sum(disk.root_count for disk in fibre.roots.disks) == \
        P.degree(hyperelliptic.level_polynomial(m, fibre.level))
    assert fibre.real_branch_points == sturm.count_roots(
        hyperelliptic.level_polynomial(m, fibre.level))


def test_genus_zero_residue_log_integral_is_validated():
    # Integral_0^1 dx/(1+x) = log(2).  The exact certificate records the
    # simple algebraic pole -1 and the residue-log root sum; its numerical
    # enclosure is obtained independently with rational midpoint remainders.
    cert = hyperelliptic.certify_genus_zero_integral(
        (Fraction(1),), (Fraction(1), Fraction(1)),
        Fraction(0), Fraction(1), tolerance=Fraction(1, 10**6))

    assert cert.validated
    assert cert.logarithmic_root_sum
    assert cert.interval.lo < Fraction(log(2)) < cert.interval.hi
    assert cert.interval.width <= Fraction(1, 10**6)
    assert sum(d.root_count for d in cert.pole_divisor.disks) == 1


def test_genus_zero_integral_refuses_a_real_pole():
    cert = hyperelliptic.certify_genus_zero_integral(
        (Fraction(1),), (Fraction(-1, 2), Fraction(1)),
        Fraction(0), Fraction(1))

    assert not cert.validated
    assert cert.interval is None
    assert cert.reason == "integration interval contains a pole"


def test_validated_lifted_flow_tube_and_conditional_launch_status():
    # A=1, B=0 gives b'=0 and y'=1/(2y), hence y^2=ell.  Rational samples
    # taken from the exact solution make a replayable inward-face test.
    fake = SimpleNamespace(
        alpha=(Fraction(1),), beta=(), N=(), C=Fraction(0))
    centres = []
    for i in range(5):
        y = Fraction(4+i, 4)
        centres.append(hyperelliptic.LiftedPoint(
            y*y, Fraction(0), y, y))

    cert = hyperelliptic.certify_flow_tube(
        fake, centres, initial_b_radius=Fraction(1, 1024),
        initial_y_radius=Fraction(1, 1024), max_radius=Fraction(1),
        launch_validated=False, max_inflations=8)

    assert cert.tube_validated
    assert cert.status == "conditional_on_launch"
    assert len(cert.slabs) == 4
    assert all(margin >= 0 for slab in cert.slabs
               for margin in (*slab.lower_face_margins,
                              *slab.upper_face_margins))


def test_validated_local_launch_hands_off_to_decreasing_loss_holonomy():
    # The local launch midpoint is only a tube proposal; its exact rectangle
    # contains the true y=2 crossing on ell=4.  Holonomy then follows the
    # unstable branch toward decreasing loss.
    fake = SimpleNamespace(
        alpha=(Fraction(1),), beta=(), N=(), C=Fraction(0))
    radius = Fraction(1, 1024)
    launch = SimpleNamespace(
        validated=True, section_level=Fraction(4), time_direction=-1,
        b_interval=hyperelliptic.RationalInterval(-radius, radius),
        y_interval=hyperelliptic.RationalInterval(2-radius, 2+radius))
    y1 = Fraction(15, 8)
    tail = [hyperelliptic.LiftedPoint(y1*y1, Fraction(0), y1, y1)]

    cert = hyperelliptic.certify_flow_tube_from_launch(
        fake, launch, tail, max_radius=Fraction(1), max_inflations=32)

    assert cert.status == "validated"
    assert cert.level_direction == -1
    assert cert.knots[0].b_interval == launch.b_interval
    assert cert.knots[0].y_interval == launch.y_interval
    assert cert.slabs[0].level_lo == y1*y1
    assert cert.slabs[0].level_hi == Fraction(4)


def test_same_sheet_abel_gap_excludes_a_connection():
    fake = SimpleNamespace(
        alpha=(Fraction(1),), beta=(), N=(), C=Fraction(0))
    first_b = hyperelliptic.RationalInterval(Fraction(0), Fraction(1, 10))
    second_b = hyperelliptic.RationalInterval(Fraction(1), Fraction(11, 10))
    y = hyperelliptic.RationalInterval(Fraction(19, 10), Fraction(21, 10))

    gap = hyperelliptic.certify_abel_gap(
        fake, Fraction(4), first_b, y, second_b, y)

    assert gap.zero_excluded
    assert gap.abel_gap == hyperelliptic.RationalInterval(
        Fraction(9, 20), Fraction(11, 20))


def test_validated_tubes_compose_to_a_smale_connection_exclusion():
    fake = SimpleNamespace(
        alpha=(Fraction(1),), beta=(), N=(), C=Fraction(0))

    def tube(b):
        centres = []
        for i in range(5):
            y = Fraction(4+i, 4)
            centres.append(hyperelliptic.LiftedPoint(y*y, b, y, y))
        return hyperelliptic.certify_flow_tube(
            fake, centres, initial_b_radius=Fraction(1, 1024),
            initial_y_radius=Fraction(1, 1024), max_radius=Fraction(1),
            launch_validated=True, max_inflations=8)

    decision = hyperelliptic.certify_connection_exclusion(
        fake, tube(Fraction(0)), tube(Fraction(1)))

    assert decision.status == "connection_excluded"
    assert decision.gap.zero_excluded
    assert decision.as_dict()["connection_excluded[VALIDATED]"]
