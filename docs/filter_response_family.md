# Exact equiripple backbone losses

There is an exact filter-response construction for SPONG, using the positive
two-point measure `Pr(X=-1)=Pr(X=+1)=1/2`.  Let `f(x)=x`, let `P` be an odd
polynomial, let `Q` be an even polynomial, and put

\[
g(z)=P(z)+\kappa Q(z),\qquad \kappa\ne0.
\]

On this support, `g(bX)=X P(b)+κQ(b)`.  Consequently the model's exact
moment polynomials are

\[
A(b)=P(b)^2+\kappa^2Q(b)^2,\quad B(b)=P(b),\quad C=1,
\]

and its backbone loss is

\[
u(b)=1-\frac{B(b)^2}{A(b)}
    =\frac{\kappa^2Q(b)^2}{P(b)^2+\kappa^2Q(b)^2}
    =\frac{1}{1+[P(b)/(\kappa Q(b))]^2}.
\]

The last expression is used only where `Q` is nonzero; the preceding
polynomial quotient defines `u` everywhere.  Coprime `P,Q` ensure `A>0`
on the real line.  This is the standard *squared magnitude* shape of a
rational response, realized here as a genuine least-squares loss.  The
construction is implemented by `spong.filter_family.rational_response_model`.

For the polynomial Chebyshev response, take `Q=1`, `κ>0`, and
`P(b)=T_d(b)` for any positive **odd** degree `d`.  The coefficients are
generated exactly by `T_0=1`, `T_1=b`, and
`T_(d+1)=2b T_d-T_(d-1)`; no coefficient search is required.  Then

\[
u_d(b)=\frac{\kappa^2}{\kappa^2+T_d(b)^2}.
\]

On `[-1,1]`, the `d` zeros of `T_d` are saddles at loss `1`, and its
`d-1` interior extrema are finite minima at the **same** loss
`κ²/(κ²+1)`.  The endpoints `b=±1` have that same lower ripple value
but are not stationary points.  The minima alternate between
`a=+1/(κ²+1)` and `a=-1/(κ²+1)`; the saddles all have `a=0`.
At either infinity, `u_d` tends to zero, below every finite minimum.
The full critical-point count is `2d-1` and every point is Morse.
For example, `d=5`, `κ=1` gives

```text
f = [0, 1]
g = [1, 5, 0, -20, 0, 16]
moments = rademacher
```

The interactive explorer now accepts `X = ±1 (equal)` as a moment law;
the response pane shows its two actual support points.  The same example
can be built in Python with
`filter_family.chebyshev_response_model(5)`.  Exact Sturm enumeration
confirms the degree-3, 5, 7, 9, and 11 counts and loss levels; the degree-5
ordinary portrait certifies.

The rational construction is a route to **Cauer/elliptic-inspired** SPONG
instances: supply a real odd numerator `P` and even denominator `Q` from
an appropriate rational ripple function.  It does not by itself synthesize
an optimal elliptic filter, nor does it imply that any arbitrary rational
function is a valid filter response.  Real zeros of `Q` give finite
zero-loss minima if `P` is nonzero there, a new feature compared with
`Q=1`.  Coefficient conditioning and the resulting manifold attachments
must be checked independently for each proposed high-degree member.

This family uses a rank-two moment measure, whereas the previously
studied `normal-symmetric-small-integer-d3/5/7/9` sequence uses a
full-support Gaussian measure.  The exact Chebyshev recurrence explains
the new family; it does **not** prove that the old small-integer rows obey
that recurrence.  Their different moments change the denominator `A`
and their finite minima are not exactly equiripple.
