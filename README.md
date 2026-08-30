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

Status: **Phases 1–5 landed — the instrument draws** — `spong.model` (exact coefficients, the (b, w)
chart, closed-form level curves, Hessian identities) and `spong.sturm`
(EXACT enumeration: primitive-PRS Sturm chains over dyadic-rational inputs,
isolation, interval-sign classification, alternation invariant; the critical
set is roots(N) ∪ roots(B)).  Trusted core: Python stdlib + NumPy; exact
certificates via stdlib rational arithmetic over the model's (dyadic float)
coefficients; integrators are the two Gauss collocation methods — `spong.gauss` (Phase 2):
IMM and IRK4-GL with block-Newton stages, collocation-polynomial dense
output and event location, `richardson3` step control, and the anadromic
reversal gap exported as a per-span certificate.  `spong.charts` (Phase 3):
the Hadamard slow-graph fixed point, the fast-graph ODE, the chart-switching
continuation engine with the shallow-water handoff (self-certifying zone
rejection at false sounding spikes), seam residuals, and angle-energy
certificates — the tricky d=11 branch traces at E ≤ 1e-12 and its
separatrices at ≤ 1e-11 with the √d asymptote confirmed.  `spong.atlas`
(Phase 4): rim structure (√d_eff diagonals, EXACT C_inf backbone-end
constant), the asymptote certificate, EXACT Poincaré–Hopf index balance
(rational-rectangle winding via Sturm-certified axis crossings — additive
over box subdivisions), genericity/degree-drop handling, and the §8b box
contract.  `spong.portrait` + `spong.render` (Phase 5): full assembly with
the certificate ledger (every branch ships its angle-energy, seam, turn,
termination/attachment, and asymptote residuals; the enumeration ships its
EXACT certificates; the portrait ships its index balance), and pure-stdlib SVG
rendering — spong's own closed-form contour layer (no fcontour, no
marching squares), house palette, plane and Poincaré-disk views.  First
portraits: [docs/gallery/](docs/gallery/).

The complex/Smale certificate layer lives in `spong.complex_structure` and
`spong.hyperelliptic`: complete complex divisors are proved by exact
Lehmer--Schur/Schur--Cohn disk counts; exact rational fibres expose the
hyperelliptic pencil; lifted gradient holonomy is enclosed by rational
trapping tubes; and same-sheet Abel gaps can exclude stable/unstable
connections.  Genus-zero conic differentials have a separate certified
residue--logarithm fast path.  `spong.local_certificate` now proves exact
rational invariant-cone graph launches and hands their `(b,y)` section boxes
to the holonomy tube in either loss direction.  This Python/Fraction oracle is
opt-in until its GMP C counterpart is ready for production timing.  The
portrait ledger keeps the remaining proof boundary explicit: positive-genus
comparisons spanning a branch point still need unwrapped period/Gauss--Manin
transport and portrait-wide orchestration.  See
[`docs/local_graph_certificate.md`](docs/local_graph_certificate.md) and
[`docs/hyperelliptic_smale.md`](docs/hyperelliptic_smale.md).

Run the fast suite: `PYTHONPATH=src python -m pytest tests` (slow founding
gates: add `-m slow`).

Validate the installed native exact-arithmetic core:

    spong-validate-native --output spong-validation.json

This differentially compares the shipped C implementation with the independent
Python `Fraction` oracle on reproducible mundane random polynomials, targeted
close/far/repeated-root families, and every named zoo model.  It validates
analysis, bounded counts, isolation, refinement, and original-polynomial
signs.  The command runs cases in parallel when the host permits it, prints a
compact summary, writes a reproducible JSON report when `--output` is supplied,
and exits nonzero on any disagreement.  The zoo tier currently takes about a
minute on a laptop; use `--no-zoo` for a seconds-scale installation smoke test,
or increase `--mundane`, `--targeted-per-exponent`, and `--exponents` for a
larger stress qualification.

Random portrait inspection:

    PYTHONPATH=src python -m spong.cli --same --f-degree 5 --count 3 --pause

This writes SVG portraits and JSON certificate summaries under
`out/random_portraits/`.  With `--pause`, each portrait is opened in the
default read-only viewer before the command waits for Enter; use
`--viewer inkscape` explicitly when you want to inspect or edit SVG geometry
in Inkscape.  On Windows, `--viewer auto` opens SVGs in a browser so strict
XML/SVG parsing is part of the inspection loop.  Use
`--view A_LO A_HI B_LO B_HI` to specify the plane view box up front, or type
`v` at the pause prompt to render/open a new view box for the current
portrait.  Default rendering keeps a readable display view but traces against
a larger compute box so invariant manifolds that leave and re-enter the view
are still available to the clipper.  Add
`--zoom-close 2` to also write tight metrological zooms around close
approaches of distinct unstable branches that share a captured minimum.
Every portrait traces both stable and unstable branches: the stable
separatrices are required to certify the basin topology.  Individual Morse
objects remain available in the numerical return values for callers that want
to inspect or render them separately.
Installed environments also get the `spong-random-portrait` console command.

For teaching and numerical comparison, installed environments also provide:

    spong-compare-portrait --zoo quadratic-stiff

This generates a side-by-side HTML gallery comparing the certified portrait
with Forward Euler, Backward Euler, explicit/implicit midpoint, and adaptive
RKF45, Rosenbrock-2, and STORK-2/4.  The methods receive a common physical
time horizon,
the gallery includes a bottom-canyon zoom, and its JSON report compares
common-resolution tangent/gradient angle defects.  Open the result on macOS
with `open out/comparisons/quadratic-stiff_comparison.html`.
`--critical-method grid-newton` also replaces exact Morse enumeration with an
explicitly uncertified multistart finite-difference Newton scan.
The report additionally runs the production loss-level order diagnostic on
each proposed manifold set, separating resolved pair roots, unresolved or
critical-level contacts, and self-crossings.  Comparison methods receive no
terminal certificates, so this diagnostic can find topology faults but cannot
certify an otherwise casual portrait.
See [docs/comparisons.md](docs/comparisons.md).

Public resolution calls have a total three-outcome contract:

```python
from spong import ResolutionStatus, model, resolve

m = model.build(f, g, mu)
answer = resolve(m)

if answer.status is ResolutionStatus.CERTIFIED_NON_MORSE:
    # Exact algebraic certificate; geometry was not attempted.
    ...
elif answer.status is ResolutionStatus.MORSE_NUMERICALLY_UNRESOLVED:
    # Morse exactly, with structured arithmetic/geometry refusal diagnostics.
    ...
else:
    portrait = answer.portrait       # certified portrait and topology ledger
```

Thus a valid call never silently drops a difficult model or returns a guessed
portrait.  Frontends may supply a `ResolutionPolicy` with frozen calibrated
binary64 margins; the default records the margins and lets the a-posteriori
geometry certificate decide.

Many-start optimizer overlays are demo consumers of the certified portrait:

    PYTHONPATH=src:. python demos/optimizer_moustaches.py \
        --zoo quadratic-stiff --starts 100

This produces separate SGD, momentum-SGD, and Adam panels for both
low-discrepancy and blue-noise starts, plus a JSON trajectory ledger under
`out/optimizer_moustaches/`.  Muon is recorded as inapplicable: the trainable
state `(a,b)` is a vector, whereas Muon acts on matrix-valued hidden weights
and routes such parameters to auxiliary AdamW in practical implementations.

Named zoo portraits:

    PYTHONPATH=src python -m spong.cli --zoo quadratic-stiff --pause
    PYTHONPATH=src python -m spong.cli --zoo tricky-d11 --pause
    PYTHONPATH=src python -m spong.cli --zoo linear-target-d17-thrash --pause
    PYTHONPATH=src python -m spong.cli --zoo nonnearest-attachment --pause

`quadratic-stiff` is the degree-2 `f=g` case discovered at random seed
`2735729614`: three minima, three saddles, and a bounded unstable branch from
the lowest saddle down to a far finite minimum.  It is the canonical "you
thought quadratic would be simple?" portrait and a regression for the
Hadamard/engine handoff at the shallow-water threshold.

`tricky-d11` is the canonical degree-11 `f=g` case with saddle stiffness
ratio about `8.5e8`.  Ordinary adaptive integrators exhaust their budgets or
collapse onto the nearby backbone; it is the founding regression that
motivated the Hadamard graph-transform chart.

`linear-target-d17-thrash` is the seed `1158725111` case with `deg f = 1` and
`deg g = 17`.  It is a compact overparameterization example: many finite
descent routes exist, but the attractor reached can be a local minimum.  It
also guards the nearly horizontal finite-branch tracer regression.

`nonnearest-attachment` is seed `1802198452`, the counterexample to assigning
an unstable branch to the nearest minimum in backbone order.  The branch from
the saddle at `b=-0.477068...` reaches the nonadjacent minimum at
`b=0.966807...`; a stable separatrix crosses the backbone between them.  Its
zoo gate checks that exact connection and the certified topology.

## Layout

    src/spong/model.py      exact model coefficients; the (b, w) chart      [§1–3]
    src/spong/sturm.py      certified enumeration (Sturm-only)              [§4]
    src/spong/gauss.py      IMM, IRK4-GL, richardson3, dense output, events [§10]
    src/spong/charts.py     graph transforms, jet charts, dispatcher        [§6–7]
    src/spong/local_certificate.py exact invariant-cone launch oracle
    src/spong/complex_structure.py complete certified complex divisors
    src/spong/hyperelliptic.py lifted holonomy, Abel gaps, conic integrals
    src/spong/atlas.py      Poincaré disk, rim charts, index bookkeeping    [§8]
    src/spong/portrait.py   assembly + certificate ledger                   [§11]
    src/spong/resolution.py total public three-outcome resolution contract
    include/spong/         stable frontend-independent C99 ABI
    src/c/                 reusable native core (Python/MATLAB/mobile)
    src/spong/render.py     zoom-proof polylines, contours, disk/plane views
    src/spong/cli.py        random portrait inspection command
    docs/theorems.md        named theorems and their proof obligations
    docs/effective_bounds.md near-poles, box contracts, ineffective bounds
    demos/                  descent-method overlays; NOT part of the library

The native library and frontend migration contract are documented in
[docs/c_api.md](docs/c_api.md).

## Development

Tests run with any Python ≥ 3.10 that has numpy + pytest on path (no fresh
venvs, no editable installs needed — point at your scientific environment):

    PYTHONPATH=src /path/to/sci-env/bin/python -m pytest tests
