"""spong.sturm: exact counting, isolation, positivity, squarefree."""

from fractions import Fraction as F

from spong import _poly as P
from spong import sturm


def test_count_simple():
    # (b-1)(b+2)(b-3) = b^3 - 2b^2 - 5b + 6
    p = P.poly([6, -5, -2, 1])
    assert sturm.count_roots(p) == 3
    assert sturm.count_roots(p, F(0), F(4)) == 2      # roots 1 and 3
    assert sturm.count_roots(p, F(-3), F(0)) == 1     # root -2


def test_multiple_roots_count_once():
    # (b-1)^2 (b+1)
    p = P.mul(P.mul(P.poly([-1, 1]), P.poly([-1, 1])), P.poly([1, 1]))
    assert sturm.count_roots(p) == 2
    assert not sturm.is_squarefree(p)
    assert sturm.is_squarefree(sturm.squarefree_part(p))


def test_isolation_close_pair():
    # roots at 0.1 and 0.1000001, plus 5: exact isolation must separate
    r1, r2, r3 = F(1, 10), F(1, 10) + F(1, 10**7), F(5)
    p = P.mul(P.mul(P.poly([-r1, 1]), P.poly([-r2, 1])), P.poly([-r3, 1]))
    ivs = sturm.isolate_roots(p)
    assert len(ivs) == 3
    for iv, r in zip(ivs, [r1, r2, r3]):
        ref = sturm.refine(p, iv)
        assert ref.lo <= r <= ref.hi
    # disjointness
    assert ivs[0].hi <= ivs[1].lo and ivs[1].hi <= ivs[2].lo


def test_exact_rational_root_detected():
    # root exactly at 1/2 can be hit by bisection midpoints
    p = P.mul(P.poly([F(-1, 2), 1]), P.poly([3, 1]))
    ivs = sturm.isolate_roots(p)
    assert len(ivs) == 2
    hits = [iv for iv in ivs if iv.exact]
    # not guaranteed to land exactly, but if it does the value is right
    for iv in hits:
        assert iv.lo in (F(-3), F(1, 2))


def test_far_root_no_overflow():
    # coefficient spread ~1e12 puts a root at ~1e12: exact arithmetic
    # handles it with no Inf/NaN path by construction.
    p = P.mul(P.poly([-10**12, 1]), P.poly([-1, 1]))     # roots 1 and 1e12
    ivs = sturm.isolate_roots(p)
    assert len(ivs) == 2
    refined = [sturm.refine(p, iv) for iv in ivs]
    assert refined[0].lo <= 1 <= refined[0].hi
    assert refined[1].lo <= 10**12 <= refined[1].hi


def test_positivity():
    assert sturm.is_positive(P.poly([1, 0, 1]))           # 1 + b^2
    assert not sturm.is_positive(P.poly([-1, 0, 1]))      # b^2 - 1
    assert not sturm.is_positive(P.poly([0, 0, 1]))       # b^2 (touches 0)


def test_interval_sign():
    p = P.poly([-1, 1])                                   # b - 1
    iv = sturm.RootInterval(F(2), F(3), False)
    assert sturm.interval_sign(p, iv) == 1
    iv0 = sturm.RootInterval(F(0), F(2), False)           # root inside
    assert sturm.interval_sign(p, iv0) is None
