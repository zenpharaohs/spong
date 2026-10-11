# Interactive phase-portrait explorer

    python demos/explorer/serve.py        # then http://127.0.0.1:8710

A browser front end for looking at certified portraits.  The browser is a
better instrument than the CLI for validating a portrait by eye, and the hard
cases are exactly the ones worth looking at, so the design goal is that a
degree-17 refusing case is usable rather than merely possible.

## What is certified and what is not

The portrait -- critical points, their kinds, the separatrices, the topology
verdict -- comes from `portrait.certified_compute`.  Nothing here recomputes
it.  The page evaluates only closed-form things: `L(a,b) = C - 2aB + a²A` by
Horner for the heatmap, the level curves `a±(b;c) = a*(b) ± √((c-u(b))/A(b))`,
the backbone, and `a·g(bx)` for the response panel.

The **descent layer is not certified** and says so.  SGD and Adam run in the
page because they are discrete-time algorithms and are the demo's subject.
The continuous method does not: it calls `/trace`, which integrates in the C
core (see below).

## Two stages: the picture, then the verdict

A model request is served twice.  `stage=preview` runs `portrait.compute` at
geometry level 0 with `_skip_audit=True` -- the geometry, no topology
certificate.  `stage=final` runs `certified_compute` to a verdict.  The
enumeration and materialized stubs are cached between them.

This is what makes the hard cases usable.  On `linear-target-d17-thrash` the
branch tracing is about 95s while a full level-0 portrait is 808s, almost all
of it the audit, and the escalation ladder above it refines the *certificate*
rather than the curves.  Blocking the display on the verdict meant twenty
minutes of blank page on the cases the viewer exists for.

## The certificate mark

A glyph beside the portrait caption, because two of the three outcomes are
determinations and one is not:

| | meaning |
|---|---|
| **✓** blue | certified: Morse skeleton and topology both determined |
| **◆** violet | certified NON-Morse: ψ fails or the skeleton is degenerate.  Exact, Sturm-decided -- a different fact about the model, not a lesser one.  Reported straight from the preview, since ψ and Morse are settled upstream of the geometry |
| **?** orange | Morse, topology unresolved in binary64.  The only genuinely open state, and the only one a better method could move |
| **…** grey | work outstanding |

The status line carries the resolution reason alongside the status:
`unstable_endpoint_unresolved` (a far-field escape that could not be certified)
and `topology_contact` (two branches too close to separate) are different
findings.

## Traces

The continuous method integrates through `Kernel.normalized_step` at
`GEOMETRIC_IRK_PRIMARY` order -- the C core's 2-D normalized-gradient
integrator, the same one `charts` falls back to when both graph
parameterizations go singular.

Unit speed is the point.  An explicit method has no business on this field,
and gradient *time* never arrives on a stiff valley: that stiffness is what
forced the whole certified machinery.  Arclength travels the same curve at
constant speed, so the trace-length control is distance along the trajectory,
not elapsed time -- which is the only version of "where does this initial
condition end up" that has an answer.

Lengths run from `quiver` (a direction sample at the click) to `to the end`
(twenty view-widths of curve).  `learning rate` applies to SGD and Adam only.

## Equal versus Thompson allocation

The allocator panel runs the same experiment as
`demos/thompson_moustaches.py` on the active portrait.  It is not restricted
to a named zoo fixture: custom coefficients, wall-family members, and
empirical portraits use the model already cached for `/portrait`.  The
currently visible `(a,b)` window supplies the common low-discrepancy or
blue-noise initialization design, so pan or zoom before running when the
question concerns a local part of a large portrait.

Both policies receive the same starts, per-arm minibatch streams, optimizer,
and requested optimizer-step budget.  The independent equal-allocation arm
prefixes are computed first, in parallel for larger runs, and their chunk
losses and iterates are recorded.  Thompson then plays back those exact same
prefixes; only an arm selected beyond its recorded uniform prefix is extended.
Adam and the AdamW-style auxiliary variant use a fused native minibatch-gradient
and optimizer chunk kernel when a local C compiler is available.  NumPy still
draws each arm's samples in the same order, and the Python implementation is
used automatically when the kernel cannot be built.  Set
`SPONG_DEMO_ADAM_NATIVE=0` to force that reference path for comparison.
Equal allocation is round-robin; Thompson gives every arm one forced
continuation and thereafter selects the smallest exact continuous-Bernoulli
posterior draw after observing `Z = L/(a+L)` for positive finite loss.
The divisor starts at the median initial loss and is recalibrated when it
leaves the recent loss interquartile range.  The posterior state is
sample-and-hold `(N Z,N)` at the arm's
current loss, not a historical average of its trajectory. Capture balls and
portrait topology do not enter selection. The optional population Fisher/Gauss–Newton
retirement check uses enumerated minima with exact gradient and Hessian
derivatives to stop an arm that has arrived at a local minimum; both policies
consume the same cached stopping event. Equal is a broad neutral-grey, half-transparent
underlay; Thompson is a narrower saturated-violet overlay with a haloed violet
cross at every start and a white-centred violet ring at every endpoint.  Within
each policy line weight and opacity record the number of pulls.  Either policy
can be hidden without discarding the result.
For display, ordinary trajectories longer than 350 iterates are reduced with
largest-triangle-three-buckets (LTTB) in the `(a,b)` plane.  When Thompson runs
an arm beyond its uniform prefix, the prefix keeps the same 350-point LTTB
selection as its gray counterpart, and the extension receives up to 250 more
points.  Log-spaced step buckets emphasize early continuation; the late noise
ball gets fewer segments.  Triangle area still favors bends and excursions
within each bucket.  Each drawn segment may span several optimizer steps;
line length is not the optimizer's step size.

The learning-curve panel plots log loss against log optimizer steps for each
arm. Short runs retain every step; long runs sample steps on a logarithmic
grid, separately from the portrait's LTTB thinning. The endpoint-audit button
plots each arm's last observed loss against the loss of the minimum reached by
continued gradient flow. It first asks the exact sublevel merge tree whether
the endpoint is already in a bounded, saddle-free one-minimum component. For
undecided endpoints it tries IRK8-GL descent and asks the tree again along the
way. The same audit probes initializations so basin changes can be counted.
An unresolved continuation is omitted from the scatter rather than assigned
to a nearest minimum.

A non-finite minibatch gradient, optimizer update, or loss terminates that arm at its last finite
iterate and removes it from later posterior draws.  The viewer reports
terminated-arm counts and actual executed steps, so numerical divergence is
visible without producing a stream of NumPy overflow warnings, repeatedly
sampling a failed arm, or drawing non-finite points.
Posterior observations use the sampler's closed family.  Exact zero loss maps
to the point mass at zero and is a resolved success that ends that policy run.
Upper-endpoint failure sentinels are recorded for the result but are not
passed to the sampler.  For finite positive losses, the default observation
uses `L/(a+L)`, with `a` periodically recalibrated to the median recent
allocated loss; all active held-loss posteriors are reset under the new `a`.
Interactive allocation also has a configurable wall-clock limit (60 seconds
by default, at most 300 seconds).  The status shows elapsed time while the
request runs, and Cancel releases the viewer immediately.  The HTTP endpoint
runs allocation beyond a process boundary: ordinary code cooperatively checks
the deadline, while the server can still kill a native sampler call that fails
to return.

Minibatches follow the portrait law: fresh `U(0,1)` or `N(0,1)` samples, or
resampling from the exact finite support of an empirical portrait.  The exact
posterior remains supplied by the separate `continuous-bernoulli` package.
The explorer first honors `CB_CORE_LIBRARY`; otherwise it discovers a sibling
`continuous-bernoulli` checkout and builds `src/c/cb_core.c` into a
content-addressed cache under the system temporary directory.  Thus the
ordinary sibling-checkout launch needs no sampler setup:

    python demos/explorer/serve.py

An installation with the posterior source elsewhere can set
`CB_CORE_SOURCE`, or can still provide a prebuilt library explicitly:

    CB_CORE_LIBRARY=/tmp/libcb_core.dylib python demos/explorer/serve.py

If neither the source nor a prebuilt library can be found, portraits and
ordinary traces remain usable and the allocator panel reports how to supply
one.

The allocator also offers **Fisher/Gauss–Newton + line search** for this
two-parameter model.  Each update draws the configured minibatch, forms its
2×2 Jacobian Gram matrix, solves a relatively damped system, and backtracks
on that same batch.  The schedule and learning-rate controls are disabled for
this method.  The summary reports batch passes as well as optimizer steps:
one derivative pass plus each trial loss pass counts once.  The exact portrait
loss remains the allocator observation, not an input to the optimizer step.
Consequently its population loss can rise even when the sampled batch loss
falls; the selector labels that scope explicitly.
The **exact Fisher/Gauss–Newton** variant forms the population 2×2 metric
and gradient from the model's moments, and backtracks on the exact population
loss. It draws no minibatches; the summary counts exact loss evaluations.
Its accepted steps strictly decrease the rational polynomial loss at the
actual binary64 coordinates. A step that cannot make representable progress
terminates with a precision-limit reason instead of being counted as descent.
Its line search starts near the previous accepted step and compares adjacent
dyadic trial scales. This retains useful basin-crossing steps on a nonconvex
ray while avoiding a nearly loss-neutral full step when a half step is much
better.
This is a full-information comparator with different objective access from
the stochastic methods.
This is demo-layer evidence, not part of portrait certification.

## Cases

**Zoo presets** come from `zoo.names()`; selecting one sends the name alone,
so the server uses the entry's own `f`, `g`, moment distribution and tuned
`default_view` -- the `cli.zoo_phase_portrait` path exactly.

**Random cases** are seeded in the page, so a case is identified by (seed,
degrees, distribution) and can be written down.  Two guards: a vanishing
leading coefficient would silently lower the degree, and `g(0) = 0` makes
`A(0) = E[g(0)²] = 0`, so ψ-positivity fails at the origin for *every*
distribution and the case can never be certified -- at `b = 0` the network
outputs `a·g(0) = 0` whatever `a` is, so it is not a two-parameter family
there.  The integer distribution alone produced that about one draw in nine.

**Separated maximal-critical candidates** fix `f(x) = 1+x` and uniform
moments, prescribe

    B(b) = product_{k=0}^{d-1} (b - (-1)^k Lambda^k),

and recover the coefficients of `g` exactly from
`B_j = g_j <f,x^j>`.  Degree and `Lambda` are live inspector controls.  At
the default `Lambda = 30`, exact Sturm counts attain the algebraic ceiling
`4d-2` for every advertised degree `1 <= d <= 8`; changing `Lambda` is an
experiment, not a promise that the ceiling remains attained.  These are
generated inverse cases rather than eight hard-coded zoo entries.

**Wall families** are the one zoo object that is a path rather than a point,
which is why a slider suits them and the CLI cannot really offer one.  Sliding
Λ from `below` through the wall to `above` shows the saddle connection form
and break; the chambers certify and the wall itself cannot, because a saddle
connection is codimension one and binary64 can bracket it but never confirm
standing on it.  The panel shows `wall_bracket` and says explicitly when Λ is
inside it, where the landing fate is launch-protocol-dependent and the
coordinate is not citable.  The slider holds Λ itself (`step="any"`), so the
stored wall coordinate is representable exactly and "back to the wall" lands
on it; the exact geometric limit is requested only when the slider sits there,
and is drawn in its own colour (`--connection`), since the orbit is a stable
and an unstable manifold at once and should read as neither.

The current `f` and `g` are echoed as selectable text under the Model panel:
coefficients live in `<input>` values, which do not survive copying the page,
so a case that misbehaves could not otherwise be reported.

## PNG export

The **export PNG 2x** button beside the backbone caption writes one shareable
image containing the live phase portrait and backbone panels, with captions
and a compact view/contour record but without the surrounding controls.  It
uses the current parameter, pan/zoom window, contours, crosshair choice, and
trace overlays.  Both panels are redrawn against temporary 2x backing stores,
so the file contains native high-resolution geometry rather than an enlarged
browser screenshot.  Named zoo cases and wall-family members receive readable
filenames; custom models receive a stable short hash of their coefficients and
moment specification.

## Notes

Contour density is a view control (toolbar), not a property of the model: at
high degree there is a nearly flat band around `|b| < 1` where the default
spacing leaves the picture bare.  Levels are closed-form, so changing it costs
a redraw and never a recompute.

The phase-portrait canvas does not zoom in response to wheel or trackpad
gestures.  Ordinary two-finger motion remains available for scrolling the
page, while pinch magnification over the canvas is suppressed; use the zoom
box or the `+`/`-` controls to make a reproducible change of view.  Disabling
the crosshairs removes the entire selection annotation rather than leaving a
centre ring behind.

Hand-entered coefficients always send the current view.  Letting the library
choose is unbounded -- `_trace_box` widens whatever box it derives -- so the
view you are looking at is the honest bound.

The server defaults to `SPONG_ENGINE=native` and `SPONG_WORKERS=8`; both are
overridable from the environment.  Its caches (models, enumerations, portraits)
persist for the process, so a second look at a model is much cheaper than the
first.
