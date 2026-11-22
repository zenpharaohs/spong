"""Combinatorial signature of a portrait: which ones have we seen before?

WHAT THIS IS FOR

Accurate portraits are now cheap enough to make in bulk, so the question
moves from "is this one right" to "is this one NEW".  A signature is a
discrete invariant with two layers, each refining the last:

  1. BACKBONE.  The critical points in b order with their types, and the
     merge tree of the backbone loss u -- the sublevel topology of L, by
     the tube theorem in merge_tree.  Needs the exact enumeration and the
     exact tree only: no stubs, no tracing.
  2. ATTACHMENT.  Where each unstable branch ends: a minimum, recorded as a
     signed offset along the backbone, or a backbone end.  This is the
     Morse-Smale skeleton as a plane tree with escapes.

Two portraits with the same layer-2 signature have the same skeleton up to
the mirror b -> -b (which is the same family seen through x -> -x).  Layer 1
alone CANNOT separate portraits that differ by a slide: the rheostat fixes
every critical b and every ratio of critical values, hence the whole of
layer 1, and still flips an attachment.

WHAT IS EXACT AND WHAT IS NOT

Everything in layer 1 is read off merge_tree.build, whose levels are
verified by exact sign tests; nothing here compares floats.  Layer 2 is
read off a portrait's topology ledger, so it is exactly as certified as
that portrait, and an uncertified end is recorded as '?', never guessed.

THE FORCED PART OF LAYER 2

A branch launched from saddle s on one side enters the component of
{L < u(s)} on that side, and can only end at something that component
holds.  Scanning outward from s along the backbone, the component runs to
the first saddle at least as high as s, or to the backbone end.  If that
run holds exactly one candidate the attachment is FORCED and needs no
trace.  When every branch is forced, layer 2 follows from layer 1 alone
(``forced_attachments``), which is how a census passes over the ordinary
portraits without computing them.  Branches with more than one candidate
are where a portrait can differ from another with the same backbone, and
their number is the first measure of how much there is to look at.

The far-end rule: a run that reaches the backbone end is unbounded there
exactly when u(infinity) < u(s).  With minima in it, that is the funnel's
gap -- the end is one more candidate, not a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib

from . import _poly as P
from . import merge_tree, sturm

END_MINUS = "L"          # b -> -infinity
END_PLUS = "R"           # b -> +infinity
UNKNOWN = "?"


# --------------------------------------------------------------------- #
# layer 1: the backbone                                                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Backbone:
    """The critical points in b order with the level data the tree needs.

    ``letters``  'g' global minimum, 'm' other minimum, 'B'/'N'/'H' saddle
                 by the factor of u' it is a root of.
    ``height``   number of verified separating levels strictly below the
                 point's critical value.  Equal heights are one value class
                 (every B-saddle sits at the roof C), never guessed apart.
    ``height_inf``  the same count for u(infinity).
    """
    letters: tuple[str, ...]
    height: tuple[int, ...]
    height_inf: int

    def mirror(self) -> "Backbone":
        return Backbone(self.letters[::-1], self.height[::-1],
                        self.height_inf)

    def flag(self, i: int) -> str:
        h = self.height[i]
        return "^" if h > self.height_inf else "_" if h < self.height_inf \
            else "="

    def is_saddle(self, i: int) -> bool:
        return self.letters[i] not in ("g", "m")


def backbone(m, e=None, tree=None) -> Backbone:
    """Layer-1 data from the exact enumeration and merge tree."""
    e = sturm.enumerate_critical_points(m) if e is None else e
    if not e.psi_positive:
        raise ValueError("signature requires A(b) > 0 on the real line")
    if not e.morse:
        raise ValueError("signature requires a Morse backbone")
    tree = merge_tree.build(m, e) if tree is None else tree
    n = len(e.points)
    # A point lies in a component of {u < c} exactly when u(point) < c, so
    # the levels NOT enclosing it are the levels at or below its value.
    height = [0] * n
    for comps in tree.components:
        inside = set()
        for comp in comps:
            inside.update(comp.minima)
            inside.update(comp.saddles)
        for i in range(n):
            if i not in inside:
                height[i] += 1
    u_inf = merge_tree.backbone_level_at_infinity(m)
    height_inf = (-1 if u_inf is None
                  else sum(1 for c in tree.levels if c < u_inf))
    minima = [i for i, p in enumerate(e.points) if p.kind == "min"]
    floor = min((height[i] for i in minima), default=0)
    letters = tuple(
        ("g" if height[i] == floor else "m") if p.kind == "min" else p.source
        for i, p in enumerate(e.points))
    return Backbone(letters, tuple(height), height_inf)


def tree_string(bb: Backbone) -> str:
    """The merge tree as a nested word, in b order.

    A node splits its run at the highest saddles in it (several at once
    when they share a value class); a run with no saddle is a minimum, or
    'e' for the empty run beyond an outermost saddle -- the backbone end,
    where an unstable branch can escape.  Each point carries its side of
    u(infinity): '^' above, '_' below, '=' not separated from it.  Heights
    are otherwise forgotten, so the order of critical values in unrelated
    subtrees does not enter.
    """
    def node(lo: int, hi: int) -> str:
        saddles = [i for i in range(lo, hi) if bb.is_saddle(i)]
        if not saddles:
            return "".join(bb.letters[i] + bb.flag(i)
                           for i in range(lo, hi)) or "e"
        top = max(bb.height[i] for i in saddles)
        parts, start = [], lo
        for i in saddles:
            if bb.height[i] == top:
                parts.append(node(start, i))
                parts.append(bb.letters[i] + bb.flag(i))
                start = i + 1
        parts.append(node(start, hi))
        return "(" + " ".join(parts) + ")"
    return node(0, len(bb.letters))


def candidates(bb: Backbone, s: int, direction: int) -> tuple:
    """What a branch leaving saddle ``s`` toward ``direction`` can end at.

    Indices of minima, of lower saddles (possible saddle connections), and
    possibly a backbone end, all inside the launch component.  Returns
    (minima, saddles, end) with end in (None, 'L', 'R').
    """
    n = len(bb.letters)
    minima, saddles, end = [], [], None
    i = s + direction
    while 0 <= i < n:
        if bb.is_saddle(i):
            if bb.height[i] >= bb.height[s]:
                break
            saddles.append(i)
        else:
            minima.append(i)
        i += direction
    else:
        # Reached the backbone end.  The component is unbounded there iff
        # u(infinity) < u(s); '=' is not a decision, so it stays a candidate.
        if bb.height_inf <= bb.height[s]:
            end = END_PLUS if direction > 0 else END_MINUS
    return tuple(minima), tuple(saddles), end


def is_funnel(bb: Backbone, s: int, direction: int) -> bool:
    """One minimum, no lower saddle, and an open backbone end beyond it.

    The level sets do not decide this branch: its launch component is
    unbounded and still holds a minimum.  Every such branch seen so far is
    a capture -- u RISES toward u(infinity) beyond an outermost minimum, so
    the backbone end is uphill -- but that is the far-field funnel's
    argument, not this module's, so it is kept as its own class.
    """
    minima, saddles, end = candidates(bb, s, direction)
    return len(minima) == 1 and not saddles and end is not None


def forced(bb: Backbone, s: int, direction: int, funnel: bool = False):
    """The forced attachment code of one branch, or None if not forced.

    With ``funnel=True`` a funnel branch (see ``is_funnel``) is read as a
    capture.  That is an ASSUMPTION for passing over ordinary portraits in
    a census; ``forced_disagreements`` is the check on it.
    """
    minima, saddles, end = candidates(bb, s, direction)
    if saddles:
        return None
    if len(minima) == 1 and (end is None or funnel):
        return minima[0] - s
    if not minima and end is not None:
        return end
    return None


def forced_attachments(bb: Backbone, funnel: bool = False):
    """Layer 2 from layer 1 alone, or None unless EVERY branch is forced."""
    out = []
    for s in range(len(bb.letters)):
        if not bb.is_saddle(s):
            continue
        pair = (forced(bb, s, -1, funnel), forced(bb, s, +1, funnel))
        if None in pair:
            return None
        out.append(pair)
    return tuple(out)


# --------------------------------------------------------------------- #
# layer 2: attachments                                                   #
# --------------------------------------------------------------------- #

def _nearest(values, x) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - x))


def attachments(p) -> tuple:
    """Per saddle in b order: (code toward -b, code toward +b).

    A code is the signed index offset from the saddle to the minimum the
    branch is certified to reach (+-1 adjacent, +-3 skipping one minimum,
    ...), 'L'/'R' for a certified escape along a backbone end, or '?'.
    """
    e = p.enumeration
    bs = [q.b for q in e.points]
    codes: dict[tuple[int, int], object] = {}
    for end in p.ledger.get("topology", {}).get("unstable_ends", ()):
        br = p.branches[end["branch"]]
        s = _nearest(bs, br.diag["saddle_b"])
        d = 1 if br.diag.get("unstable_direction", 1) > 0 else -1
        if not end.get("certified"):
            code = UNKNOWN
        elif end["kind"] == "finite_capture":
            target = _nearest(bs, end["minimum"][1])
            code = target - s if e.points[target].kind == "min" else UNKNOWN
        elif end["kind"] == "infinity_escape":
            code = (END_PLUS if end.get("end") == "b_plus_infinity"
                    else END_MINUS if end.get("end") == "b_minus_infinity"
                    else UNKNOWN)
        else:
            code = UNKNOWN
        if (s, d) in codes:
            # Two branches of one saddle read the same b direction: the
            # unstable eigenvector is nearly vertical.  Keep both, in the
            # two slots, rather than drop one.
            d = -d
        codes[(s, d)] = code
    return tuple((codes.get((s, -1), UNKNOWN), codes.get((s, +1), UNKNOWN))
                 for s, q in enumerate(e.points) if q.kind == "saddle")


def _mirror_code(code):
    if isinstance(code, int):
        return -code
    return {END_MINUS: END_PLUS, END_PLUS: END_MINUS}.get(code, code)


def _mirror_attachments(att):
    return tuple((_mirror_code(plus), _mirror_code(minus))
                 for minus, plus in att[::-1])


def _attachment_string(att) -> str:
    def show(code):
        return f"{code:+d}" if isinstance(code, int) else str(code)
    return " ".join(f"[{show(a)},{show(b)}]" for a, b in att)


# --------------------------------------------------------------------- #
# the signature                                                          #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Signature:
    backbone_text: str            # canonical layer 1
    attachment_text: str | None   # canonical layer 2, None if not computed
    chirality: str                # 'symmetric' | 'as_given' | 'mirrored'
    traced: bool                  # layer 2 came from a portrait, not forcing
    grades: dict = field(default_factory=dict, compare=False)

    @property
    def key1(self) -> str:
        return hashlib.sha1(self.backbone_text.encode()).hexdigest()[:12]

    @property
    def key2(self) -> str | None:
        if self.attachment_text is None:
            return None
        text = self.backbone_text + " | " + self.attachment_text
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    @property
    def text(self) -> str:
        return (self.backbone_text if self.attachment_text is None
                else self.backbone_text + " | " + self.attachment_text)

    def as_dict(self) -> dict:
        return {"key1": self.key1, "key2": self.key2,
                "backbone": self.backbone_text,
                "attachments": self.attachment_text,
                "chirality": self.chirality, "traced": self.traced,
                "grades": dict(self.grades)}


def _grades(bb: Backbone, att) -> dict:
    saddles = [i for i in range(len(bb.letters)) if bb.is_saddle(i)]
    open_ = funnels = eligible = 0
    for s in saddles:
        for d in (-1, +1):
            _minima, lower, _end = candidates(bb, s, d)
            funnels += is_funnel(bb, s, d)
            open_ += forced(bb, s, d, funnel=True) is None
            eligible += len(lower)
    g = {
        "n_critical": len(bb.letters),
        "n_saddle": len(saddles),
        "n_roof_saddle": sum(bb.letters[i] == "B" for i in saddles),
        "value_classes": len(set(bb.height)),
        # Branches the backbone does not decide (two or more real
        # candidates), funnel branches, and the lower saddles in launch
        # components: the candidate saddle connections.
        "open_branches": open_,
        "funnel_branches": funnels,
        "eligible_connections": eligible,
    }
    if att is not None:
        flat = [c for pair in att for c in pair]
        skips = [(abs(c) - 1) // 2 for c in flat if isinstance(c, int)]
        g.update(
            nonadjacent=sum(k > 0 for k in skips),
            max_skip=max(skips, default=0),
            escapes=sum(c in (END_MINUS, END_PLUS) for c in flat),
            unknown=sum(c == UNKNOWN for c in flat))
    return g


def _assemble(bb: Backbone, att, traced: bool) -> Signature:
    forward = (tree_string(bb),
               None if att is None else _attachment_string(att))
    back_bb = bb.mirror()
    backward = (tree_string(back_bb),
                None if att is None
                else _attachment_string(_mirror_attachments(att)))
    key = lambda t: (t[0], t[1] or "")                      # noqa: E731
    if key(forward) == key(backward):
        chosen, chirality = forward, "symmetric"
    elif key(forward) < key(backward):
        chosen, chirality = forward, "as_given"
    else:
        chosen, chirality = backward, "mirrored"
    return Signature(chosen[0], chosen[1], chirality, traced,
                     _grades(bb, att))


def of_model(m, e=None, tree=None, funnel: bool = False) -> Signature:
    """Signature without tracing: layer 1, and layer 2 when it is forced."""
    bb = backbone(m, e, tree)
    return _assemble(bb, forced_attachments(bb, funnel), traced=False)


def of_portrait(p, tree=None) -> Signature:
    """Signature of a computed portrait: both layers, from its ledger."""
    bb = backbone(p.model, p.enumeration, tree)
    return _assemble(bb, attachments(p), traced=True)


def forced_disagreements(p, tree=None, funnel: bool = False) -> list:
    """Forced predictions the portrait's certified ends contradict.

    Empty on a sound portrait with ``funnel=False``: a forced attachment is
    a theorem about the launch component, so anything listed is a defect in
    the tree, the audit, or this module.  With ``funnel=True`` the funnel
    reading is tested too, and a listing there is a certified escape past
    an outermost minimum -- a finding, not a defect.
    """
    bb = backbone(p.model, p.enumeration, tree)
    saddles = [i for i in range(len(bb.letters)) if bb.is_saddle(i)]
    bad = []
    for s, (minus, plus) in zip(saddles, attachments(p)):
        for d, got in ((-1, minus), (+1, plus)):
            want = forced(bb, s, d, funnel)
            if want is not None and got != UNKNOWN and got != want:
                bad.append({"saddle": s, "direction": d,
                            "forced": want, "certified": got})
    return bad
