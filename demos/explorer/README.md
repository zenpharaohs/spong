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

## Notes

Contour density is a view control (toolbar), not a property of the model: at
high degree there is a nearly flat band around `|b| < 1` where the default
spacing leaves the picture bare.  Levels are closed-form, so changing it costs
a redraw and never a recompute.

Hand-entered coefficients always send the current view.  Letting the library
choose is unbounded -- `_trace_box` widens whatever box it derives -- so the
view you are looking at is the honest bound.

The server defaults to `SPONG_ENGINE=native` and `SPONG_WORKERS=8`; both are
overridable from the environment.  Its caches (models, enumerations, portraits)
persist for the process, so a second look at a model is much cheaper than the
first.
