"""Lehmer-filter (fates) soundness on the simple portraits.

Easy cases first -- degree 2 and 3, seconds each -- before the directed
dead-neuron models.  The assertions are soundness, not forcing: the
filter must certify on these clean examples; the Euler equality
#minima = #saddles + 1 is enforced inside component_fates, so a violation
surfaces here as a decline; and a traced capture target must lie inside
its own candidate set at launch AND at the terminal sample.  The target
invariant is not a heuristic: the orbit from the launch sample to the
minimum has loss below any tested level, so the target minimum is in the
launch point's component whenever the inventory certifies -- a target
outside candidates indicts the filter or the trace.

Forcing is a property of the example, not of the filter, and is asserted
only where the geometry guarantees it: the certified dead-neuron zoo
case, whose audit already proves a bounded one-minimum terminal tube.
"""

from spong import fates, model, portrait, zoo


def _compute(m):
    try:
        return portrait.compute(m, _skip_audit=True)
    except TypeError:                            # older compute signature
        return portrait.compute(m)


def _unstable(m, p):
    return [e for e in fates.fate_report(m, p.enumeration, p.branches)
            if e["kind"] == "unstable"]


def _assert_sound(entries):
    assert entries
    for e in entries:
        assert e["certified"], (e["branch"], e.get("reason"))
        if e["forced"]:
            assert e["bounded"] and not e["saddles"]
            assert len(e["minima"]) == 1
        if e["term"] == "capture":
            assert e["target_in_candidates"] is True, e["branch"]
            terminal = e["terminal_fates"]
            assert terminal["certified"], (
                e["branch"], terminal.get("reason"))
            assert terminal["target_in_candidates"] is True, e["branch"]
            if terminal["forced"]:
                assert len(terminal["minima"]) == 1


def test_d2_quadratic():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    _assert_sound(_unstable(m, _compute(m)))


def test_d2_far_lower_minimum():
    f = [1.0511595983436535, 2.207740477509359, -0.3128201040655276]
    m = model.build(f, f, model.moments_uniform01(5))
    _assert_sound(_unstable(m, _compute(m)))


def test_dead_neuron_far_saddle_d3():
    z = zoo.get("dead-neuron-far-saddle-d3")
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    m = model.build(list(z.f), list(z.g), mu)
    entries = _unstable(m, _compute(m))
    _assert_sound(entries)
    captures = [e for e in entries if e["term"] == "capture"]
    assert captures
    # The audit certifies a bounded one-minimum tube on this case, so the
    # filter must force at least one capture by its terminal sample.
    assert any(e["forced"] or e["terminal_fates"]["forced"]
               for e in captures)
