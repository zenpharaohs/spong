"""Portrait signatures: canonical form, forcing, and one certified case."""
import sys
from pathlib import Path

import pytest

from spong import model, portrait, signature, zoo
from spong.signature import Backbone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _zoo_model(name):
    z = zoo.get(name)
    f, g = [float(x) for x in z.f], [float(x) for x in z.g]
    return model.build(f, g, zoo.moments_for(z, 2*max(len(f), len(g)) - 1))


# near-slide-d2's backbone, written out: m B m N m B with both B-saddles on
# the roof, the N-saddle and all minima below u(infinity).
NEAR_SLIDE = Backbone(("m", "B", "m", "N", "g", "B"), (2, 5, 1, 3, 0, 5), 4)


def test_tree_string_splits_at_the_highest_saddles_together():
    assert signature.tree_string(NEAR_SLIDE) == "(m_ B^ (m_ N_ g_) B^ e)"


def test_mirror_reverses_the_word_and_keeps_the_tree():
    back = NEAR_SLIDE.mirror()
    assert back.letters == ("B", "g", "N", "m", "B", "m")
    assert signature.tree_string(back) == "(e B^ (g_ N_ m_) B^ m_)"
    assert back.mirror() == NEAR_SLIDE


def test_candidates_stop_at_a_saddle_at_least_as_high():
    # From the left B-saddle toward +b: both inner minima and the lower
    # N-saddle, stopped by the other roof saddle.  Not forced.
    assert signature.candidates(NEAR_SLIDE, 1, +1) == ((2, 4), (3,), None)
    assert signature.forced(NEAR_SLIDE, 1, +1) is None
    # From the N-saddle either way: one minimum, then a higher saddle.
    assert signature.forced(NEAR_SLIDE, 3, -1) == -1
    assert signature.forced(NEAR_SLIDE, 3, +1) == +1


def test_outermost_saddle_escape_is_forced_and_funnel_is_its_own_class():
    # Right B-saddle toward +b: nothing beyond it, u falls to u(infinity).
    assert signature.forced(NEAR_SLIDE, 5, +1) == signature.END_PLUS
    # Left B-saddle toward -b: one minimum, then an open end.  The level
    # sets do not decide it; reading it as a capture is opt-in.
    assert signature.is_funnel(NEAR_SLIDE, 1, -1)
    assert signature.forced(NEAR_SLIDE, 1, -1) is None
    assert signature.forced(NEAR_SLIDE, 1, -1, funnel=True) == -1
    assert signature.forced_attachments(NEAR_SLIDE, funnel=True) is None


def test_mirror_image_portraits_share_a_signature():
    att = ((-1, +3), (-1, +1), (-1, signature.END_PLUS))
    one = signature._assemble(NEAR_SLIDE, att, traced=True)
    two = signature._assemble(NEAR_SLIDE.mirror(),
                              signature._mirror_attachments(att), traced=True)
    assert one.key2 == two.key2 and one.text == two.text
    assert {one.chirality, two.chirality} == {"as_given", "mirrored"}
    # A different attachment on the same backbone is a different class.
    slid = signature._assemble(
        NEAR_SLIDE, ((-1, +1), (-1, +1), (-1, signature.END_PLUS)), True)
    assert slid.key1 == one.key1 and slid.key2 != one.key2


def test_backbone_of_near_slide_needs_no_tracing():
    sig = signature.of_model(_zoo_model("near-slide-d2"))
    assert sig.backbone_text == "(e B^ (m_ N_ g_) B^ m_)"
    assert sig.attachment_text is None and not sig.traced
    assert sig.grades["open_branches"] == 2
    assert sig.grades["funnel_branches"] == 1
    assert sig.grades["eligible_connections"] == 2


def test_near_slide_portrait_records_the_nonadjacent_attachment():
    p = portrait.certified_compute(_zoo_model("near-slide-d2"))
    sig = signature.of_portrait(p)
    # The B-saddle's inward branch skips the adjacent minimum.
    assert sig.attachment_text == "[L,+1] [-1,+1] [-3,+1]"
    assert sig.grades["nonadjacent"] == 1 and sig.grades["unknown"] == 0
    assert signature.forced_disagreements(p, funnel=True) == []


def test_inspector_replay_matches_the_page_generator():
    census = pytest.importorskip("portrait_census")
    # Values produced by the page's own JavaScript (mulberry32 and
    # randomCoeffs from demos/explorer/index.html, run under node) for
    # seed 1, degree 13, f = g, uniform coefficients.
    f, g = census.inspector_case(1, 0, 13, "uniform")
    assert f == g
    assert g[:4] == [0.598479, 0.025242, 0.02728, -0.978578]
    assert g[-1] == 0.225885
