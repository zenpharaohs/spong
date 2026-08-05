# Interactive explorer

Three linked panels over one certified portrait.

    python demos/explorer/serve.py        # http://127.0.0.1:8710

## What it shows

**Phase portrait** — `L(a,b)` as a log-scaled field with contour banding, the
backbone `a*(b)`, the certified critical points, and the certified
separatrices. Drag to move `(a,b)`; wheel to zoom; shift-drag to launch a
descent trace.

**Backbone panel** — `a*(b)` together with `u(b) = L` restricted to the
backbone, drawn through the model's own reduced `backbone_num / backbone_den`.
Vertical stops mark the certified critical points: minima are local minima of
`u`, saddles are local maxima. Click anywhere to snap `a` to `a*(b)` and slide
along the backbone, which is where every critical point lives.

**Response panel** — `f(x)` against `a·g(bx)` at the selected `(a,b)`, with the
residual shaded. The readout normalizes by `Var(f)` rather than `E[f²]`:
`L/Var = 1` means no better than predicting the mean, which is the honest
reference for a target with a large mean.

## Division of labour

Everything analytic comes from the library. `serve.py` calls

    m = model.build(f, g, mu)
    p = portrait.certified_compute(m, view=view)

and serializes it: `m.alpha`, `m.beta`, `m.C` are the exact `A`, `B`, `C`;
`m.backbone_num` / `m.backbone_den` are the reduced `B²/A`; each
`CriticalPoint` supplies `b`, `a`, `kind`, `source` and the exact sign of `u''`;
the `Enumeration` supplies `psi_positive`, `morse`, `alternates`.

The page evaluates `L = C - 2aB(b) + a²A(b)` by Horner for the heatmap and
draws the two polynomials for the response panel. It does no root finding, no
Morse classification, no finite differences. Changing `f`, `g`, or the moment
selection posts to `/portrait` and recomputes; while a request is in flight the
certified layer greys out and the header reads *stale*, so certified marks are
never shown over a surface they do not belong to.

Empirical moments travel as **sample points**, not moments: the browser draws
the batch, the server forms exact rational empirical moments, and the batch
portrait is a genuine SPONG model rather than a floating-point echo of one.

## The descent layer is not certified

SGD and Adam draw a fresh batch each step and are integrated in the page. They
are labelled as a demo layer and drawn in a separate colour from the certified
separatrices. The moment-filter selector applies a `d`-fold repeated-pole
kernel

    h[t] = C(t+d-1, d-1) (1-λ)^d λ^t

to the moment vector rather than to the gradient — nonnegative, so the filtered
moments stay inside the moment cone. `d = 1` is EWMA.

## Known limits

No compactification: critical points that escape the window simply leave it.
High-degree presets pay a full certified enumeration on every edit, so the
450 ms debounce in `queue()` may want raising.
