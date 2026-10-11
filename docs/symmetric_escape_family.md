# Symmetric normal-moment escape sequence

The original case has `f=(0,-3,0,-3)` and `g=(-3,-1,-3,4)`.

## A flatter degree-5 normal example

The named `normal-symmetric-flat-d5` case directly uses
`f(x)=x+x^3+x^5` and `g=(-3,-1,-3,4,-3,-2)` under standard-normal
moments. Its five saddles lie on `a=0` at loss `C=1207`; four finite
minima alternate across that axis. Their losses, ordered by increasing
`b`, are approximately `1206.191615, 1206.763131, 1206.763131,
1206.191615`. Thus the largest depression below the saddle loss is
`0.066975%`. The loss tends to the much lower exact infimum `142/21` as
`b` tends to either infinity. The full portrait's topology certifies.

This case differs from the earlier degree-5 row only in `g_5`: `-2`
instead of `-1`. Rescaling that earlier row's `f=-3(x+x^3+x^5)` to the
unit-coefficient `f` divides its `a` coordinates by `-3` and every loss by
`9`; its largest finite depression is `1.155565%` of `C`. The new row is
therefore substantially flatter without increasing the largest absolute
coefficient in `g` beyond `4`.

The bounded search is small enough to reproduce directly. Hold
`g_0=g_2=g_4=-3` and enumerate nonzero integers
`g_1,g_3,g_5` in `[-4,4]`. For this `f`, the exact cross term is
`B(b)=19g_1 b+123g_3 b^3+1065g_5 b^5`. Five real simple B roots require
the quadratic `19g_1+123g_3 t+1065g_5 t^2` to have two distinct positive
roots. Among the resulting rows, exact Sturm enumeration leaves four
coefficient triples with five saddles and four minima, up to reversing all
odd coefficients. Ranked by the largest relative saddle-to-minimum loss
gap, the rows are:

| `(g_1,g_3,g_5)` | largest relative gap |
|---|---:|
| `(-1,4,-2)` | `0.066975%` |
| `(-1,3,-1)` | `0.142945%` |
| `(-2,4,-1)` | `0.286946%` |
| `(-1,4,-1)` | `1.155565%` |

The gap values are numerical evaluations of the Sturm-isolated minima;
the topology and tail value above have separate certificates. This is an
exhaustive search only within the stated coefficient box and fixed even
coefficients, not a claim about all rational degree-5 instances. Reproduce
the ranking with `python scripts/search_normal_flat_d5.py --height 4`.

## Curated odd-degree sequence

The small-integer continuation in
`spong.zoo.small_integer_symmetric_escape_case` uses degree `d=2m+1` and

\[
 f_d(x)=-3\sum_{r=0}^{m}x^{2r+1},\qquad
 g_d(x)=-3\sum_{r=0}^{m}x^{2r}
       -\sum_{k=0}^{m}(-1)^k h_{m,k}x^{2k+1}.
\]

For standard-normal moments `mu[2n]=(2n-1)!!`, put

\[
 W_{m,k}=\sum_{r=0}^{m}\mu_{2(r+k+1)},\qquad
 B_d(b)=3b\sum_{k=0}^{m}(-1)^k h_{m,k}W_{m,k}(b^2)^k.
\]

The moment weights have the exact degree recurrence
`W[m+1,k] = W[m,k] + (2m+2k+3)!!`.  This is a recurrence for the
**weights**, not yet for the small integer rows `h[m,*]`.  The rows below
were found independently by integer search; five rows cannot establish an
all-degree rule.

| d | `h[m,*]` | critical points | finite minima | portrait status |
|---:|---|---:|---:|---|
| 3 | `(1,4)` | 5 | 2 | certified |
| 5 | `(1,4,1)` | 9 | 4 | certified |
| 7 | `(1,4,4,1)` | 13 | 6 | certified |
| 9 | `(1,7,13,7,1)` | 17 | 8 | certified |
| 11 | `(1,9,24,24,9,1)` | 21 | 10 | unresolved |

All counts and Morse classifications in this table come from exact Sturm
enumeration.  In each row `B_d` has `d` simple real roots; they give saddles
at `a=0`.  Between consecutive roots, `B_d^2/A_d` has a positive interior
maximum, giving an off-axis minimum.  The listed rows have exactly those
`d-1` minima and no other critical points.  Their `a` coordinates alternate
in sign along increasing `b`.  Even normal moments make `A_d` even and
`B_d` odd, so the portrait is symmetric under `(a,b) -> (-a,-b)`.

The degree-9 portrait initially refused `stable_escape_unresolved`.  Stable
branches had been retraced to the exact level-bar box, but the topology audit
checked their terminal crossing against the smaller display-derived box.
Checking the box actually used by each branch certifies degree 9, with no
trajectory or contact-policy change.  Degree 11 has 1,024,536 segments,
above the one-million-segment audit budget.  A diagnostic run with a
1.1-million allowance finished endpoint checks but found two forbidden
contacts (`topology_contact`); the production budget remains unchanged and
degree 11 remains outside the ordinary zoo registry.

A conventional orthogonal-polynomial three-term recurrence for these exact
rows would require successive positive roots of `B_d(b)/b`, as a polynomial
in `b^2`, to interlace.  They do not: the degree-5 roots are approximately
`0.0425, 0.419`, while the degree-7 roots are approximately
`0.0637, 0.0774, 0.1717`.  This rules out that particular recurrence, not
other degree-dependent or higher-order recurrences.

There is a useful **experimental one-parameter recurrence for a different
sequence**.  Write `H_m(t)=sum_k h[m,k] t^k`; seed
`H_1(t)=1+4t` (the original degree-3 case) and
`H_2(t)=1+4t+t^2` (the curated degree-5 case), then set

\[
 H_m(t)=(1+t)H_{m-1}(t)+c_m t H_{m-2}(t),
\]

where `c_m` is the smallest nonnegative integer for which the resulting
`B_(2m+1)` has all `2m+1` roots real, as verified by exact Sturm counting.
The first `c_m` values, for `m=3,...,11`, are
`0, 7, 0, 17, 0, 34, 0, 62, 0`.  Thus it generates a small-integer
degree-7 variant `H_3=(1,5,5,1)` and a degree-9 variant
`H_4=(1,13,38,13,1)`; these **differ from the curated rows above**.
The exact root test succeeds through degree 23.  The integer selection is
still an adaptive search, not a closed formula for `c_m`, and no theorem yet
guarantees it succeeds at every odd degree.  Reproduce or extend the search
with `python scripts/probe_symmetric_escape_recurrence.py --max-degree 23`.
