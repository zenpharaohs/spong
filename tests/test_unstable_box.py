"""Unstable branches are not truncated by the display view.

deg g = 1 is ordinary linear regression seen through (a, b) -> (a g0,
a b g1): one roof saddle on a = 0, one minimum, nothing else.  A portrait
that simple has no business being refused, yet 21 of 88 random ones were,
because the branch to the minimum left the DISPLAY box and was handed to
the escape certificate.  These are two of them.
"""
import numpy as np
import pytest

from spong import model, portrait, signature


def _regression_model(f, g):
    return model.build(f, g, model.moments_uniform01(2*max(len(f), len(g))-1))


@pytest.mark.parametrize("f, g, why", [
    # Saddle at b = -5.72, minimum at (1.81, 5.32): the branch swings out to
    # a = 2.8 across the flat strip and left through an a side.
    ([1.213452, 1.96381], [0.670652, 0.204005], "a side"),
    # Minimum at (5.96, 0.012): the branch overshoots it in b and left
    # through the view's upper b edge at a = 1.2.
    ([1.969574, 1.259616, -1.145343], [0.362525, 1.661619], "b side"),
])
def test_linear_regression_portrait_certifies(f, g, why):
    p = portrait.certified_compute(_regression_model(f, g))
    top = p.ledger["topology"]
    assert top["status"] == "certified", (why, top["resolution_reason"])
    sig = signature.of_portrait(p)
    assert sig.backbone_text == "(e B^ g_)"
    assert sig.attachment_text == "[L,+1]"
    retraced = [b for b in p.branches if "unstable_box_retrace" in b.diag]
    assert len(retraced) == 1 and retraced[0].term == "capture"


def test_unstable_box_contains_the_roof_sublevel_set():
    m = _regression_model([1.213452, 1.96381], [0.670652, 0.204005])
    display = (-1.0, 1.0, -8.0, 8.0)
    lo, hi, b_lo, b_hi = portrait._unstable_trace_box(m, display)
    # {L <= C} lies between a = 0 and a = 2 a*(b).
    b = np.linspace(-200.0, 200.0, 20001)
    edge = 2.0*np.array([float(m.a_star(x)) for x in b])
    assert lo <= min(0.0, edge.min()) and hi >= max(0.0, edge.max())
    assert lo <= display[0] and hi >= display[1]
    assert b_lo <= display[2] and b_hi >= display[3]
    # And the bound is attained, not merely safe: within a part in 10^3.
    assert hi <= max(display[1], edge.max())*(1 + 1e-3) + 1e-9
