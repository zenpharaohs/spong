# spong

**S**ingle **PO**lynomial **N**euron **G**radient — certified phase portraits
of the steepest-descent flow on the MSE loss of a single scalar polynomial
neuron "network" ("network" because one neuron is a trivial network, and the
point is that even the trivial network defeats descent).

spong is the reference instrument of the *beautiful untrainable nets* corpus:
it draws the graph paper on which descent methods go to die, to the limit of
achievable precision, with a machine-checkable certificate attached to every
curve.  Descent methods themselves (SGD, Adam, L-BFGS, …) appear only in
`demos/`, as consumers judged against certified portraits — never as library
code.

**Start here: [SPONG_FOUNDING.md](SPONG_FOUNDING.md)** — the founding
document: why an off-the-shelf phase portraitist cannot do this job, the
complete mathematical recital (§1–§11), the certificate semantics
(EXACT / VALIDATED / RESIDUAL / EMPIRICAL), the chart dispatcher and box
contracts, the architecture, and the migration plan from the MATLAB
predecessor (`mse-bundle`).

Status: **Phase 0** (scaffold).  Trusted core: Python stdlib + NumPy; exact
certificates via stdlib rational arithmetic over the model's (dyadic float)
coefficients; integrators are the two Gauss collocation methods (implicit
midpoint and IRK4-GL) and nothing else.

## Layout

    src/spong/model.py      exact model coefficients; the (b, w) chart      [§1–3]
    src/spong/sturm.py      certified enumeration (Sturm-only)              [§4]
    src/spong/gauss.py      IMM, IRK4-GL, richardson3, dense output, events [§10]
    src/spong/charts.py     graph transforms, jet charts, dispatcher        [§6–7]
    src/spong/atlas.py      Poincaré disk, rim charts, index bookkeeping    [§8]
    src/spong/portrait.py   assembly + certificate ledger                   [§11]
    src/spong/render.py     zoom-proof polylines, contours, disk/plane views
    docs/theorems.md        named theorems and their proof obligations
    demos/                  descent-method overlays; NOT part of the library

## Development

Tests run with the shared environment (no fresh venvs, no editable installs):

    PYTHONPATH=src python -m pytest tests
