"""Loss-level order-sweep semantics, independent of chord scanning."""

from fractions import Fraction
import numpy as np
import pytest
from types import SimpleNamespace

from spong import (comparison, inverse, model, order_sweep, portrait, sturm,
                   topology, zoo)
from spong.charts import Branch


class _LinearLoss:
    """L(a,b)=a: vertical regular levels, oriented upward."""

    @staticmethod
    def L(a, b):
        return np.asarray(a)

    @staticmethod
    def gradL(a, b):
        shape = np.broadcast(np.asarray(a), np.asarray(b)).shape
        return np.stack((np.ones(shape), np.zeros(shape)))


def test_resolved_simple_root_has_opposite_order_witnesses():
    levels = np.linspace(-1.0, 1.0, 2001)
    delta = levels.copy()
    error = np.full_like(levels, 1e-3)
    result = order_sweep.classify_signed_profile(
        levels, delta, error, [0.0], threshold=4.0)
    assert result["root_count"] == 1
    assert result["event_classes"] == ["resolved_root"]
    root = result["roots"][0]
    assert root["delta_bracket"][0] < 0 < root["delta_bracket"][1]
    assert root["resolution_margin"] >= 4.0


def test_tangent_touch_keeps_the_same_resolved_order():
    levels = np.linspace(-1.0, 1.0, 2001)
    delta = levels**2
    error = np.full_like(levels, 1e-3)
    result = order_sweep.classify_signed_profile(
        levels, delta, error, [0.0], threshold=4.0)
    assert result["root_count"] == 0
    assert result["same_order_count"] == 1
    assert result["event_classes"] == ["same_order"]


def test_unresolved_asymptotic_tail_needs_a_terminal_certificate():
    levels = np.linspace(0.0, 12.0, 3001)
    delta = np.exp(-levels)*np.sin(20.0*levels)
    error = np.full_like(levels, 1e-3)
    declined = order_sweep.classify_signed_profile(
        levels, delta, error, [10.0], threshold=4.0)
    discharged = order_sweep.classify_signed_profile(
        levels, delta, error, [10.0], threshold=4.0,
        terminal_ok=[True])
    assert declined["unresolved_count"] == 1
    assert discharged["terminal_count"] == 1
    assert discharged["root_count"] == 0


def test_order_change_across_a_critical_level_is_not_a_crossing_witness():
    levels = np.linspace(-1.0, 1.0, 2001)
    delta = levels.copy()
    error = np.full_like(levels, 1e-3)
    result = order_sweep.classify_signed_profile(
        levels, delta, error, [0.0], threshold=4.0,
        critical_levels=[0.0])
    assert result["root_count"] == 0
    assert result["critical_transition_count"] == 1


def test_critical_level_contact_remains_unresolved_without_local_certificate():
    m = _LinearLoss()
    x = np.linspace(-1.0, 1.0, 100)
    first = Branch("unstable", np.column_stack((x, x)), "box_exit")
    second = Branch("unstable", np.column_stack((x, -x)), "box_exit")
    event = next(topology._pair_events(
        first.Y, topology._tree(first.Y),
        second.Y, topology._tree(second.Y), 1e-12))
    contact = {"branches": (0, 1), "segments": event[:2],
               "kind": event[2], "point": event[3]}
    critical = SimpleNamespace(a=0.0, b=0.0)

    result = order_sweep.classify_contacts(
        m, SimpleNamespace(points=[critical]), [first, second],
        [contact], 1e-12)

    assert result["decision"] == "unresolved"
    assert result["roots"] == 0
    assert result["critical_transition"] == 1
    assert result["pairs"][0]["event_classes"] == ["critical_transition"]


def test_events_with_one_witness_pair_collapse_to_one_root():
    levels = np.linspace(-1.0, 1.0, 2001)
    delta = levels.copy()
    error = np.full_like(levels, 1e-2)
    result = order_sweep.classify_signed_profile(
        levels, delta, error, [-0.001, 0.0, 0.001], threshold=4.0)
    assert result["candidate_count"] == 3
    assert result["root_count"] == 1
    assert result["roots"][0]["candidate_count"] == 3


def test_straight_polyline_has_zero_interpolation_allowance():
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(order_sweep.sagitta_allowance(Y), 0.0)


def test_transverse_chord_crossing_is_a_resolved_order_root():
    m = _LinearLoss()
    x = np.linspace(-1.0, 1.0, 100)  # even: the root is between two knots
    first = Branch("unstable", np.column_stack((x, x)), "box_exit")
    second = Branch("unstable", np.column_stack((x, -x)), "box_exit")
    events = list(topology._pair_events(
        first.Y, topology._tree(first.Y),
        second.Y, topology._tree(second.Y), 1e-12))
    assert len(events) == 1 and events[0][2] == "cross"
    contact = {"branches": (0, 1), "segments": events[0][:2],
               "kind": events[0][2], "point": events[0][3]}
    result = order_sweep.classify_contacts(
        m, SimpleNamespace(points=[]), [first, second], [contact], 1e-12)
    assert result["decision"] == "fault"
    assert result["roots"] == 1
    assert result["pairs"][0]["roots"][0]["resolution_margin"] > 1e6


def test_nonisolated_tail_band_requires_and_uses_terminal_certificate():
    m = _LinearLoss()
    x = np.linspace(20.0, 0.0, 201)  # stored toward the lower-loss terminal
    first = Branch("unstable", np.column_stack((x, np.zeros_like(x))),
                   "capture")
    second = Branch("unstable", np.column_stack((
        x, 1e-4*(-1.0)**np.arange(len(x)))), "capture")
    events = list(topology._pair_events(
        first.Y, topology._tree(first.Y),
        second.Y, topology._tree(second.Y), 1e-12))
    contacts = [
        {"branches": (0, 1), "segments": event[:2],
         "kind": event[2], "point": event[3]}
        for event in events]
    assert len(contacts) > 100
    enumeration = SimpleNamespace(points=[])
    declined = order_sweep.classify_contacts(
        m, enumeration, [first, second], contacts, 1e-12)
    suffixes = [
        {"kind": "minimum_sublevel", "start": 190,
         "terminal": (0.0, 0.0)},
        {"kind": "minimum_sublevel", "start": 190,
         "terminal": (0.0, 0.0)},
    ]
    discharged = order_sweep.classify_contacts(
        m, enumeration, [first, second], contacts, 1e-12,
        terminal_suffixes=suffixes)
    assert declined["roots"] == 0
    assert declined["decision"] == "unresolved"
    assert declined["unresolved"] == len(contacts)
    assert discharged["roots"] == 0
    assert discharged["decision"] == "accepted"
    assert discharged["terminal"] == len(contacts)
    assert discharged["pairs"][0]["contact_cluster_count"] == 1


def test_isolated_root_is_not_hidden_by_a_later_terminal_band():
    m = _LinearLoss()
    x = np.linspace(10.0, -10.0, 2000)
    separation = np.where(x >= 4.0, 0.05*(x-5.003), -0.05015)
    tail = np.flatnonzero(x <= -6.0)
    separation[tail] = 1e-5*(-1.0)**(np.arange(len(tail))+1)
    first = Branch(
        "unstable", np.column_stack((x, np.zeros_like(x))), "capture")
    second = Branch(
        "unstable", np.column_stack((x, separation)), "capture")
    events = list(topology._pair_events(
        first.Y, topology._tree(first.Y),
        second.Y, topology._tree(second.Y), 1e-12))
    contacts = [
        {"branches": (0, 1), "segments": event[:2],
         "kind": event[2], "point": event[3]}
        for event in events]
    suffixes = [
        {"kind": "minimum_sublevel", "start": int(tail[0]),
         "terminal": (-10.0, 0.0)},
        {"kind": "minimum_sublevel", "start": int(tail[0]),
         "terminal": (-10.0, 0.0)},
    ]

    result = order_sweep.classify_contacts(
        m, SimpleNamespace(points=[]), [first, second], contacts, 1e-12,
        terminal_suffixes=suffixes)

    assert result["decision"] == "fault"
    assert result["roots"] == 1
    assert result["terminal"] == len(contacts)-1
    assert result["unresolved"] == 0
    clusters = result["pairs"][0]["clusters"]
    assert [cluster["kind"] for cluster in clusters] == [
        "terminal", "resolved_root"]


def test_near_saddle_connection_euler_crossing_is_a_resolved_fault():
    """A bad proposer reaches the isolated-simple-root production path.

    Forward Euler is not a production option.  At the registered B-to-N
    saddle-connection wall it is a useful negative control: its independently
    traced W^u(B) and W^s(N) cross once, away from either saddle, instead of
    agreeing on the wall orbit.  Unlike an asymptotic contact band, this event
    must be reported as an isolated resolved construction fault.
    """
    family = zoo.get_wall_family("nonnearest-saddle-connection")
    case = zoo.rheostat_member(family, "wall")
    degree = len(case.g)-1
    m = model.build(
        case.f, case.g, model.moments_uniform01(2*degree+1))
    enumeration = sturm.enumerate_critical_points(m)
    p = comparison.casual_portrait(
        m, "forward-euler", reference_enumeration=enumeration,
        view=family.default_view, step_size=0.025,
        max_steps=40000, time_horizon=200.0,
        capture_saddles=False)

    source = min(
        enumeration.saddles, key=lambda q: abs(q.b-family.source_b))
    target = min(
        enumeration.saddles, key=lambda q: abs(q.b-family.target_b))
    unstable = next(
        branch for branch in p.branches
        if branch.kind == "unstable"
        and abs(branch.diag["saddle_b"]-source.b) < 1e-7
        and branch.diag["unstable_direction"]
        == family.unstable_direction)
    stable = [
        branch for branch in p.branches
        if branch.kind == "stable"
        and abs(branch.diag["saddle_b"]-target.b) < 1e-7]
    branches = [unstable, *stable]
    scale = max(1.0, *(abs(x) for x in p.box))
    tolerance = 128*np.finfo(float).eps*scale
    sagittae = [topology._sagitta_bounds(branch.Y) for branch in branches]
    trees = [topology._tree(np.asarray(branch.Y)) for branch in branches]
    contacts = []
    for j in range(1, len(branches)):
        events = topology._pair_contact_events(
            np.asarray(branches[0].Y), trees[0],
            np.asarray(branches[j].Y), trees[j],
            tolerance, sagittae[0], sagittae[j])
        contacts.extend({
            "branches": (0, j), "segments": (si, sj),
            "kind": kind, "point": point,
        } for si, sj, kind, point in events)

    assert len(contacts) == 1
    result = order_sweep.classify_contacts(
        m, enumeration, branches, contacts, tolerance)
    assert result["decision"] == "fault"
    assert result["roots"] == 1
    assert result["unresolved"] == 0
    pair = result["pairs"][0]
    assert pair["candidate_count"] == 1
    assert pair["contact_cluster_count"] == 1
    assert pair["clusters"][0]["isolated"]
    assert pair["clusters"][0]["normalized_extent"] == 1.0
    assert pair["first_dropped_nonmonotone"] == 0
    assert pair["second_dropped_nonmonotone"] == 0
    root = pair["roots"][0]
    assert root["delta_bracket"][0]*root["delta_bracket"][1] < 0.0
    assert root["resolution_margin"] >= result["threshold"]

    # Exercise the geometry-engine integration as well as the mathematical
    # classifier.  Shadow mode must preserve the established chord verdict;
    # active mode must retain this resolved crossing as a fault.
    shadow = topology.audit(
        m, enumeration, p.branches, p.box,
        pair_contact_policy="order_sweep_shadow")
    active = topology.audit(
        m, enumeration, p.branches, p.box,
        pair_contact_policy="order_sweep")
    assert shadow["forbidden_count"] == 1
    assert active["forbidden_count"] == 1
    for audit in (shadow, active):
        sweep = audit["pair_order_sweep"]
        assert sweep["decision"] == "fault"
        assert sweep["candidates"] == 1
        assert sweep["roots"] == 1
        assert sweep["unresolved"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("radius, expected_status", [
    (-1536, "certified"),
    (-2048, "fp64_unresolved"),
])
def test_far_radius_terminal_band_boundary(radius, expected_status):
    """Real dense-contact family: discharge only the certified suffix.

    The directed seed 1495454581 prescribes a minimum at b=-1536.  Its two
    stable branches agree asymptotically and produce hundreds of interpolated
    chord crossings.  Moving the prescribed point to -2048 creates a second
    contact band which does not reach the certified terminal suffix.  The
    active policy must certify the former while retaining the latter.
    """
    g = (0.060689, 0.119698, -1.132917,
         -0.34751, 1.424413, 1.06735)
    exact_g = tuple(Fraction(c).limit_denominator(10**6) for c in g)
    moments = model.moments_uniform01(4*len(g)+3)
    case = inverse.straddle_case([Fraction(radius)], exact_g, moments)
    assert case is not None

    p = portrait.compute(
        case.design.model, geometry_level=0,
        pair_contact_policy="order_sweep")
    top = p.ledger["topology"]
    sweep = top["pair_order_sweep"]

    assert top["status"] == expected_status
    assert sweep["candidates"] > 200
    assert sweep["roots"] == 0
    assert sweep["terminal"] > 200
    if radius == -1536:
        assert top["resolution_reason"] is None
        assert top["forbidden_count"] == 0
        assert sweep["decision"] == "accepted"
        assert sweep["terminal"] == sweep["candidates"]
        assert sweep["unresolved"] == 0
    else:
        assert top["resolution_reason"] == "topology_contact"
        assert top["forbidden_count"] > 0
        assert sweep["decision"] == "unresolved"
        assert sweep["unresolved"] == top["forbidden_count"]
