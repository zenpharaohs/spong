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
and requested optimizer-step budget.  Equal allocation is round-robin; Thompson
allocation gives every arm one forced continuation and thereafter selects the
smallest exact continuous-Bernoulli posterior draw after observing
`L/(1+L)`.  The posterior state is sample-and-hold `(N Z,N)` at the arm's
current loss, not a historical average of its trajectory.  Capture balls,
critical-point labels, and portrait topology do not
enter either allocation.  Equal is a broad neutral-grey, half-transparent
underlay; Thompson is a narrower saturated-violet overlay with a haloed violet
cross at every start and a white-centred violet ring at every endpoint.  Within
each policy line weight and opacity record the number of pulls.  Either policy
can be hidden without discarding the result.

A non-finite minibatch gradient, optimizer update, loss, or numerically
saturated upper-endpoint transform terminates that arm at its last finite
iterate and removes it from later posterior draws.  The viewer reports
terminated-arm counts and actual executed steps, so numerical divergence is
visible without producing a stream of NumPy overflow warnings, repeatedly
sampling a failed arm, or drawing non-finite points.
Posterior observations use the sampler's closed family.  Exact zero loss maps
to the point mass at zero and is a resolved success that ends that policy run.
Upper-endpoint failure sentinels are recorded for the result but are not
passed to the sampler, so neither endpoint enters rejection sampling.
Interactive allocation also has a configurable wall-clock limit (20 seconds
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

**Wall families** are the one zoo object that is a path rather than a point,
which is why a slider suits them and the CLI cannot really offer one.  Sliding
Λ from `below` through the wall to `above` shows the saddle connection form
and break; the chambers certify and the wall itself cannot, because a saddle
connection is codimension one and binary64 can bracket it but never confirm
standing on it.  The panel shows `wall_bracket` and says explicitly when Λ is
inside it, where the landing fate is launch-protocol-dependent and the
coordinate is not citable.

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
