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

## Smooth densities near the two-point measure

The exact Rademacher moments **cannot** be represented by a continuous
probability density, even though SPONG uses only finitely many moments.
Already `mu_0=mu_2=mu_4=1` gives

\[
\mathbb E[(X^2-1)^2]=\mu_4-2\mu_2+\mu_0=0.
\]

Since the integrand is nonnegative, every representing probability measure
must be supported on `{-1,+1}`.  `mu_1=0` then fixes the two weights at
one half each.  The moment vector is on the boundary of the positive
moment cone; a polynomial–Gaussian density cannot satisfy these equalities
exactly.  A numerical fit with small moment error may nevertheless be
useful, provided that error is reported rather than called exact matching.
Matching only orders `0..3` is different: `N(0,1)` shares those four raw
moments.  The degree-5 SPONG model uses moments through order 10, so it
necessarily includes the obstructing order-4 equality.

There are smooth **nearby** moment vectors that keep the portrait.  Let
`S` be Rademacher, `Z` standard normal and independent, and take
`X=S+sqrt(v) Z`.  For every rational `v>0`, the law is an equal mixture of
two Gaussians and has a strictly positive smooth density.  Its moments are
computed exactly by `filter_family.gaussian_smoothed_moments`; build the
corresponding model with `filter_family.smoothed_chebyshev_model(d,v)`.
For example, `v=1/10000` gives `mu_2=1.0001` and
`mu_4=1.00060003`.  The degree-5 model still has five saddles and four
minima, with a certified portrait and the same ordered unstable-branch
destinations as the discrete model.  The same was observed at `v=1/100`.
These perturbed moment vectors are natural inputs for a continuous-density
moment fitter, including a polynomial–Gaussian method; the explicit
two-Gaussian mixture already proves that a continuous representative exists.
The finite minima are no longer exactly equiripple: at `v=1/100`, their
losses are approximately `0.63766, 0.51612, 0.51612, 0.63766`.
These are checked examples, not a proven maximal interval of `v` for which
the topology persists.

The persistence has a structural explanation.  The `v=0` degree-5 portrait
is certified Morse–Smale, and the moment perturbation changes the polynomial
loss and its derivatives smoothly on compact sets.  Thus its hyperbolic
critical points and transverse manifold attachments have a local stability
margin.  Escape makes the plane noncompact, so this argument alone does not
exclude new critical points arriving from infinity or a change in escape
fates.  Exact Sturm enumeration and portrait audits check those global
obligations for the displayed positive variances.  The loss limit at either
infinity moves continuously from `0` to about `0.001596` at `v=1/10000`
and remains below all finite minima in the checked examples.

The degree-5 connectivity also persists at `v=1/10`.  On that moment
vector, the independent [moment-density implementation](https://github.com/zenpharaohs/moment-density)
accepts moments through order 10 at bandwidth `0.5`: a degree-10
polynomial–Gaussian fit has maximum scaled moment residual about `5e-16`
and passes its numerical positivity audit at tolerance `1e-14`.  At
bandwidth `0.8`, its higher-order correction returns degree 22 with
residual about `1.6e-15`, but the audit permits a tiny negative density
minimum (about `-8e-15`) under a `1e-12` tolerance.  Neither numerical
audit is a formal nonnegativity certificate.  The exact `v=0` moment
vector is rejected by that implementation's singular-Hankel check, as
the fourth-moment identity requires.

## Compact continuous approximations with decaying moments

The tempting sequence `mu_(2k)=h(k)`, `mu_(2k+1)=0`, where
`h(k)=exp(-lambda*k) sum_(j=0)^d (lambda*k)^j/j!`, is **not** a
probability moment sequence for any `lambda>0` and finite `d>=1`.
Indeed, `h(k)=Pr(Poisson(lambda*k)<=d)`. For independent
`N_1,N_2 ~ Poisson(lambda)`, the event `N_1+N_2<=d` is strictly
contained in the event `N_1<=d, N_2<=d`. Thus
`h(2)<h(1)^2`, whereas any measure requires
`mu_4-mu_2^2=Var(X^2)>=0`. The obstruction uses only moments through
order four, even if a finite prefix of this sequence is extremely close
to the two-point moments. For `d=0`, the sequence is realizable but gives
another two-point measure at `X=+/-exp(-lambda/2)`.

A valid compactly supported alternative is `X=S*R`, where `S` is an
independent fair sign and `R=r0+(r1-r0)*U` with
`U~Beta(shape,shape)`, `0<r0<r1<1`, and integer `shape>=2`.
This law has a continuous density on two short intervals. Its odd
moments vanish and `0<mu_(2k)=E[R^(2k)]<=r1^(2k)`; all its moments
converge to the two-point moments as both radii approach one. The
density is globally `C^(shape-2)` at the interval endpoints. Rational
radii give exact rational moments via `beta_shell_moments` and a model
via `beta_shell_chebyshev_model`.

For `r0=99/100`, `r1=199/200`, and `shape=4`, the degree-5 model has
`mu_2=0.9850569444444445`, `mu_4=0.9703399200662879`, and a strictly
positive order-four Hankel gap of about `2.73627e-6`. Exact critical
point enumeration and the portrait audit find five saddles, four
minima, and the same ordered unstable-branch destinations as the
two-point degree-5 model. This is a certified nearby example, not a
proof that all choices of radii preserve the global portrait.

This family uses a rank-two moment measure, whereas the previously
studied `normal-symmetric-small-integer-d3/5/7/9` sequence uses a
full-support Gaussian measure.  The exact Chebyshev recurrence explains
the new family; it does **not** prove that the old small-integer rows obey
that recurrence.  Their different moments change the denominator `A`
and their finite minima are not exactly equiripple.
