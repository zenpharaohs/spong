# Two independent connection candidates

In the inspector's zoo preset menu select **two-independent-connections**.
It displays all invariant manifolds using the usual portrait machinery and
stable-tail extensions. At the stored connection value, four participating
half-branches are replaced by two single blue curves assembled from independent
two-sided shots, including their source and target saddles. This is an
idealized numerical wall-limit display; the tiny meeting gaps remain reported.
Away from that exact stored decimal value, all sixteen ordinary branches
are shown separately so the slide and basin boundaries are visible. These
preview portraits carry no topology certificate.

The fixture lives in `spong.zoo.NUMERICAL_CONNECTION_CASES`, separately from
ordinary `CASES` and the one-connection wall families. Access it with
`zoo.get_numerical_connection("two-independent-connections")`; `build()` returns
the exact rational physical model and `trace_pairs(model)` produces native GL8
shots. Its probability measure is |x| dx on [-1,1], with even moments 2/(k+2)
and odd moments zero. This measure is not uniform or Gaussian.

The stored base F has coefficients (ascending)
`[c,1,1,0,1,0,1,0,1,0,1]`, where
`c=-223518530295946063188153523711403413281097319/285478393996286056500989623222688818581858816`.
G has coefficients `[.460239,0,-1.774853,0,-.852506,0,-.117097,0,-1.048502]`
(the decimal coefficients are exact rationals).
Physical coefficients are F/s and sG, with s the rational representation of
sqrt(Lambda) rounded to 80 decimal significant digits. Model construction
retains these rationals; only display coefficients and tracing use binary64.

Lambda is approximately 1.307179719334000599261222676631751034570168532337215797867979.
This is a numerical representative, not a certified parameter. Native GMP GL8
loss continuation at 192 bits and 1024 steps produced it. At 256 and 512 steps
the values were 1.30717971933400061800459 and 1.30717971933400059933476.
The inspector now recomputes order-10 launch germs in native GMP at each
selected Lambda, at radius .001 and 192-bit requested precision. GL8 arclength shots with
step .001 meet with gaps about 5.65e-15 on each side.

The source saddles have b=±.4729142401695701 and the targets
b=±1.1142465931034908. Reflection explains coincidence at eta=0. Independence
refers to the two-parameter unfolding G_eta(z)=G(z)+eta*z, together with Lambda.
The measured section-gap Jacobian in (Lambda, eta) was approximately
`[[.4903551502,-4.6756383772],[-.4903551502,-4.6756383772]]`,
with determinant -4.585446717. This is numerical evidence of independent
splitting in the full family, not a transversality certificate.

Provenance: directed search seed 628285345, case 139, lifted with
F_even(x)=f_base(x²), G_even(z)=g_base(-z²), followed by adding x to F.


## Inspecting the rheostat

Both double-connection zoo entries expose a Lambda slider, an editable value,
and a **connection value** reset directly under the preset selector. The slider
covers ±0.1 about the stored value. The text entry accepts [0.1,10], although
continuation can refuse if a selected half-branch no longer reaches its loss
section. Each change reconstructs F/sqrt(Lambda), sqrt(Lambda)G exactly as
rational coefficients and recomputes both native launch germs for each pair.
The readout shows Euclidean meeting distance and signed b separation; these
are numerical diagnostics, not certified distances to a bifurcation.

## Breaking reflection symmetry

The new **two-asymmetric-connections** preset uses
G_new(z) = G_even(z) + eta*z + epsilon*z³, with epsilon=0.05 and
eta=-0.019544449395600534. Lambda is approximately 1.3071646494604742256.
The target F and positive measure are unchanged.

In base coordinates, B_odd(b)=eta*b/2 + epsilon*b³/3, which is not zero.
A remains even, so L(alpha,b)-L(alpha,-b)=-4*alpha*B_odd(b).
Thus the loss no longer has b-reflection symmetry. For example the outer
saddles have b approximately -1.115440 and +1.113019, with physical losses
0.535655 and 0.558283. The two sections consequently have different losses.

The independent connection locators gave:

| loss steps | right Lambda | left Lambda |
|---|---|---|
| 512 | 1.30716464946047422096912 | 1.30716464946047423025397 |
| 1024 | 1.30716464946047422089445 | 1.30716464946047423017951 |

The approximately 9.3e-18 mismatch remains because eta is stored with 18 decimal
places; these are numerical near-connections, not a claim that the literal
rational model is exactly non-Smale. Native display shots meet within 8e-15.
The section-gap Jacobian in (Lambda,eta), estimated at 256 loss steps with
native Lambda sensitivities and centered eta differences, is approximately
`[[.4910019654,-4.4009568720],[-.4897141533,-4.9841656498]]`,
with determinant -4.6024459983. Halving the eta-difference step from 5e-5 to
2.5e-5 changes each eta derivative by less than 1.4e-7.

The symmetric subfamily indeed makes the coincidence codimension one.
A nonsingular two-gap Jacobian instead gives the local codimension-two
condition in the larger family. Subject to an exact double connection and
that nonsingularity, the implicit-function theorem allows a small fixed
cubic perturbation to be compensated by changing Lambda and eta. The
calculations above provide numerical evidence for this continuation, not
an interval proof of existence or transversality.

`scripts/asymmetric_connection_probe.py` reproduces the numerical search.
It now uses the bounded native two-parameter solver for both Lambda and eta;
no output is fed into certification verdicts. See `wall_native.md` for the
conditioning guards and refusal semantics.


The general inspector distribution menu includes **|x| dx on [-1,1]**.
`model.moments_absolute_x(n)` supplies its exact moments. The allocator samples
the same law using X=sign(2U-1)*sqrt(abs(2U-1)) for U uniform on [0,1].
This applies to custom models as well as both connection presets.


## Strong asymmetry

The **two-strongly-asymmetric-connections** preset uses epsilon=3/4 and
eta=-0.283652549200640338, with Lambda approximately
1.3136390836357023026894204981250735694446259302946882157339205.
This is fifteen times the original asymmetric fixture's cubic perturbation.
These stored values were originally retuned using native GMP section gaps and
Lambda sensitivities with a Python outer Newton prototype. The reproduction
script now refines both parameters entirely in the native GMP driver. It does
not overwrite the zoo fixture.

Independent scalar connection roots at 1024 loss steps were
1.313639083635702316002139660709743938052501256630282861225323 (right) and
1.313639083635702289376701335540403200836750603959093570242518 (left).
Their difference is 2.66e-17. At 512 steps the average was
1.313639083635702302765985212230484949584029798651370566742427,
about 7.7e-20 from the 1024-step average. Printed digits are provenance,
not certified accuracy; the literal stored rational model is a near-connection.

The two central outer minima are approximately (a,b)=(-1.22452,-.907723)
and (-2.62111,.746567), compared with nearly equal a-coordinates in the weak
fixture. The central minimum has b=-.0990845. The target saddle losses are
approximately .396490 (left) and .738819 (right).

There are now FOUR minima and FIVE saddles: an additional negative-b pair
appeared along this deformation. The extra minimum has b=-1.19324 and the
far saddle has b=-23.19055. The default frame emphasizes both connections;
**fit to skeleton** includes the distant saddle. The model remains exactly
Morse with A>0. Because its number of equilibria differs from the symmetric
fixture, these two complete flows cannot be topologically conjugate.
Their local two-slide pattern may nevertheless remain equivalent; the same
motion under Lambda is not itself a test of reflection symmetry or conjugacy.

Saddle tracking uses the stored b locations rather than inventory indices,
so the new far-field points do not accidentally change the connection pair.

The displayed two-sided shot gaps at the stored strong parameter are
approximately 5.99e-15 and 6.07e-15. The two-gap Jacobian in (Lambda,eta),
evaluated with 128 loss steps and centered eta differences of size 5e-6,
is approximately `[[.4976231,-2.365394],[-.4804402,-32.87012]]`, with
determinant -17.49336. Halving the difference step from 1e-5 changed the
larger eta derivative by 6.55e-5. This supports independent conditions,
while also showing a much stronger eta response on the negative-b side.
The scalar connection roots above were refined separately to 1024 steps.
`scripts/strong_asymmetric_connection_probe.py` reproduces the coarse
continuation through epsilon=.15,.25,.5,.75 (numerical research only).
