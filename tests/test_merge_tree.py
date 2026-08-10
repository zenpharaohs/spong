"""Exact merge tree: invariants, and agreement with the fates inventory.

These assert the mathematics, not the previous implementation.  The two
routes to a component -- merge_tree's rational levels and topology's
slack-derived inventory -- are independent, so agreement between them is
evidence; disagreement indicts one of them.
"""

import pytest

from fractions import Fraction

from spong import merge_tree, model, sturm, zoo


CASES = ["near-slide-d2", "dead-neuron-far-saddle-d3", "quadratic-stiff"]


def _model(name):
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu)


@pytest.fixture(scope="module", params=CASES)
def portrait(request):
    m = _model(request.param)
    e = sturm.enumerate_critical_points(m)
    return request.param, m, e, merge_tree.build(m, e)


def test_levels_are_strictly_between_critical_values(portrait):
    _, m, e, tree = portrait
    points = [p for p in e.points if p.kind != "degenerate"]
    for c in tree.levels:
        signs = [merge_tree.value_sign(m, p, c) for p in points]
        assert all(s in (-1, 1) for s in signs), "level hits a critical value"
        assert 0 < sum(1 for s in signs if s < 0) < len(points)
    assert list(tree.levels) == sorted(tree.levels)


def test_every_critical_value_gap_is_separated(portrait):
    """Distinct critical values must end up in distinct classes.

    Coincident values are expected -- every B-root has u = C exactly -- so
    a class with several members must be a genuine coincidence, which for
    B-roots is checkable: u = C means the level polynomial at c = C
    vanishes there.
    """
    _, m, e, tree = portrait
    points = [p for p in e.points if p.kind != "degenerate"]
    for group in tree.sequence.unseparated:
        sources = {points[i].source for i in group}
        assert sources == {"B"}, (
            f"unseparated non-B critical values: {group}")


def test_components_are_contiguous_and_euler_exact(portrait):
    """Contiguity in b, and #minima = #saddles + 1 on bounded components.

    A component with NO critical points is legitimate and expected: it is
    an unbounded escape sector (near-slide-d2 has one beyond its outer
    B-saddle).  Contiguity is vacuous there.
    """
    _, m, e, tree = portrait
    order = {i: rank for rank, i in enumerate(
        sorted((i for i, p in enumerate(e.points)
                if p.kind != "degenerate"),
               key=lambda i: e.points[i].interval.lo))}
    for comps in tree.components:
        for comp in comps:
            enclosed = sorted(order[i] for i in comp.minima + comp.saddles)
            if enclosed:
                assert enclosed == list(
                    range(enclosed[0], enclosed[-1] + 1)), (
                        "enclosed critical set is not backbone-contiguous")
            else:
                assert not comp.bounded, (
                    "a bounded component must contain a minimum")
            if comp.bounded:
                assert len(comp.minima) == len(comp.saddles) + 1


def test_same_level_components_are_disjoint(portrait):
    _, _, _, tree = portrait
    for comps in tree.components:
        spans = [(c.lo, c.hi) for c in comps]
        for i in range(len(spans) - 1):
            _, hi = spans[i]
            lo, _ = spans[i + 1]
            assert hi is not None and lo is not None and hi <= lo


def test_components_nest_upward(portrait):
    """Each component sits inside exactly one component at the next level.

    Tested through the exact primitives, not the reported endpoints: the
    child's interior sample must land in the parent's gap, and the parent
    must enclose every critical point the child does.  Comparing the
    reported lo/hi across levels is NOT a valid nesting test -- those are
    outer enclosures from the root isolation, and their slack can invert
    the comparison even when the true intervals nest strictly.
    """
    _, m, _, tree = portrait
    for k, comps in enumerate(tree.components[:-1]):
        above = tree.components[k + 1]
        R_above = merge_tree.level_polynomial(m, tree.levels[k + 1])
        for comp, parent in zip(comps, tree.parents[k]):
            assert parent is not None, "component has no parent one level up"
            up = above[parent]
            assert sturm.count_roots(R_above, None, comp.sample) == up.gap
            assert set(comp.minima) <= set(up.minima)
            assert set(comp.saddles) <= set(up.saddles)


def test_minima_are_forced_at_their_own_lowest_level(portrait):
    """A minimum sits alone in some component, so its own fate is forced."""
    name, m, e, tree = portrait
    assert tree.levels, f"{name}: no separating levels"
    for i, p in enumerate(e.points):
        if p.kind != "min":
            continue
        found = merge_tree.fate_from_tree(m, e, tree, p.a, p.b)
        assert found is not None, f"minimum {i} located in no component"
        assert i in found["minima"]


def test_bounded_components_trap_their_saddles(portrait):
    """Escape eligibility: a saddle in a bounded component cannot escape.

    Forward invariance keeps both unstable branches inside, and a bounded
    component with k saddles holds k+1 minima, so all 2k branch ends land
    there.  The eligible set is the complement, and it must contain every
    saddle that no bounded component encloses.
    """
    _, m, e, tree = portrait
    eligible = set(merge_tree.escape_eligible(m, e, tree))
    trapped = {i for comps in tree.components for comp in comps
               if comp.bounded for i in comp.saddles}
    assert not (eligible & trapped)
    saddles = {i for i, p in enumerate(e.points) if p.kind == "saddle"}
    assert eligible | trapped == saddles


def test_locate_agrees_with_exact_loss(portrait):
    _, m, e, tree = portrait
    c = tree.levels[0]
    for p in e.points:
        if p.kind == "degenerate":
            continue
        inside = merge_tree.locate(m, e, c, p.a, p.b) is not None
        assert inside == (merge_tree.exact_loss(m, p.a, p.b) < c)


def test_level_polynomial_sign_is_the_sign_of_u_minus_c(portrait):
    """A > 0, so sign((C-c)A - B^2) = sign(u - c) -- checked numerically."""
    _, m, e, tree = portrait
    c = tree.levels[-1]
    R = merge_tree.level_polynomial(m, c)
    for b in (Fraction(-3), Fraction(-1, 2), Fraction(0), Fraction(1),
              Fraction(7, 2)):
        A = merge_tree.P.eval_at(m.alpha, b)
        B = merge_tree.P.eval_at(m.beta, b)
        u = Fraction(m.C) - B * B / A
        assert (merge_tree.P.eval_at(R, b) < 0) == (u < c)
