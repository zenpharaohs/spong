# Design note: the inverse construction as a regression-suite generator

Prescribe the critical points; solve for polynomials `f` and `g` that have
them.  Used to generate spong's regression suite — in particular to place
critical points at large `|b|`, which is where the descent equation is stiff.
Sampling reaches stiff cases only by luck; prescription chooses them.

Recorded 2026-07-26 from a working discussion.  The method predates this note
and is not currently in the repository; the derivation below is a
reconstruction from `model.py` that the original author confirms matches what
was used.  Treat the mechanism as sound and the details as unverified until it
is rebuilt with tests.

## Mechanism

From §1–3 (`model.py` header):

    L(a,b) = C − 2a·B(b) + a²·A(b) = u(b) + A(b)·w²,   w = a − a*(b)
    u' = B·N / A²   with   N = A'B − 2B'A
    ⇒ critical b-values = real roots of N  ∪  real roots of B

So "prescribe the critical points" means "prescribe real roots of `N` and of
`B`", and the split is what makes it tractable:

- `B` is **linear** in the model coefficients, so its prescribed roots are
  placed directly.
- `N = A'B − 2B'A` is **bilinear** in `(A, B)`.  With `B` already fixed by the
  first stage, prescribing roots of `N` is linear in `A`.

Two linear stages rather than a general nonlinear inverse problem.  The
unknowns are `f`, `g` coefficients (with the moment sequence `μ` fixed), so the
map from those to `(A, B)` must be inverted as well; that is the part most
likely to need care on rebuild.

**Constraint.** Not every configuration is realisable: the alternation
invariant requires minima and saddles to interleave, so the design space is
constrained rather than free.  A prescription that violates alternation has no
model.

## The extras, and why they are mild

The construction yields polynomials that **contain** the prescribed critical
points and generally some additional ones.  This is degree surplus, not a
defect: `deg N = 3d − 2` and `deg B = d`, so `4d − 2` roots are in play while
only the prescribed few are pinned.  The rest are determined by the structure
but not controlled — the same situation as a partial inverse eigenvalue
problem.

Extras only matter if they are **real**; complex roots of `N` or `B` are not
critical points.  Measured over 12 random models per degree (noisy — the real
counts do not support fitting a trend):

| d | total roots | real of N | real of B | critical pts | % real |
|--:|--:|--:|--:|--:|--:|
| 3 | 10 | 3.8 | 1.5 | 5.3 | 53% |
| 5 | 18 | 4.2 | 1.3 | 5.5 | 31% |
| 7 | 26 | 5.0 | 2.0 | 7.0 | 27% |
| 9 | 34 | 4.8 | 2.0 | 6.8 | 20% |
| 11 | 42 | 6.3 | 2.2 | 8.5 | 20% |
| 13 | 50 | 5.0 | 1.8 | 6.8 | 14% |

The surplus goes complex on its own, increasingly so with degree: 50 roots at
d=13 produce about 7 critical points.  So the extras to contend with are a
handful, not dozens, and the parameter budget (2d + 2 coefficients) grows
linearly against a roughly flat real-root count.  That ratio suggests a refined
construction could *steer* surplus roots complex rather than accepting where
they land — untested, and the obvious next experiment.

## Stiffness: the scalar, and why it is a position effect

There are **two** stiffness scalars, and only one of them steers the engine.

| name in `diag` | expression | role |
|---|---|---|
| `kappa_saddle` | `depth_gauge_floor` = `2\|a*'\| / \|w₁'\|` | **the dispatcher's gauge** — `kap[0] < KAPPA_HI` is the shallow-water test |
| `kappa_spectral_saddle` | `sounding` = `2A / \|u''\|` | diagnostic only |

`κ ≥ KAPPA_HI = 1e4` is *shallow water*: the Hadamard slow-graph fixed point
takes over from the ODE, with hysteresis back out at `KAPPA_EXIT = 1e3`.

`sounding`'s own docstring warns it is "DIAGNOSTIC only (lies at
u-inflections)" — where `u'' → 0` the spectral ratio blows up spuriously.  The
depth gauge is taken on the slaved floor, which is what the slow-RHS
denominator actually is.  Note it is 0/0 *at* a critical point; read it just
off, as `trace_unstable` does (`kap[0] ← kap[1]`).

**Measured agreement** over the same 785 saddles:

- vs `log₁₀\|b\|`: spectral **r = +0.748**, depth gauge **r = +0.740** — the
  position conclusion below is the same under either scalar
- spectral vs depth gauge: r = +0.920; typical disagreement 0.30 decades (a
  systematic factor ≈2), but **max 52 decades** at the inflection cases the
  docstring warns about
- shallow-water classification agrees on 777/785 = **99.0%**, and all 8
  disagreements run one way: the gauge says shallow while the spectral says
  mild.  **The spectral scalar misses stiff cases; it never raises a false
  alarm.**

For a stiffness sweep that asymmetry is the wrong direction of error — you
would omit cases the engine treats as shallow water.  **Gate the suite on
`depth_gauge_floor`.**  The spectral scalar is fine for reporting and for the
correlations below, which is how it is used here.

Measured at 785 saddles of random models, degrees 3–13 (spectral scalar):

| predictor | log₁₀ κ |
|---|--:|
| log₁₀ `\|b\|` | **r = +0.748** |
| degree | r = +0.109 |

| quartile | median `\|b\|` | median κ | max κ |
|---|--:|--:|--:|
| innermost | 0.41 | 0.0915 | 2.05e3 |
| outermost | 5.12 | **3.46e10** | **1.86e68** |

Eleven orders of magnitude in κ between the quartile medians, and κ spans 76
orders overall (5.2e−08 … 1.86e68).  **Stiffness is a position effect on the
backbone; degree barely enters.**  Large `|b|` is stiff — which is why the
inverse construction places critical points there.

### Step count and chart switches are the WRONG difficulty metrics

An earlier draft of this note claimed step count was the honest measure of
difficulty.  That is backwards, and the reason is architectural.  The stiff
neighbourhoods are handled by the slow-graph fixed point and by **exact jets**
at critical points whose locations are known exactly from the Sturm
enumeration.  Neither consumes integration steps.  The adaptive IMM/IRK4-GL
stepping runs in the *mild* regions.  So step count measures mild arc, and is
anti-correlated with stiffness:

- steps vs `\|saddle_b\|`: r = −0.470, chart switches vs `\|saddle_b\|`: r = −0.407
- steps vs switches: r = +0.884 (they measure the same thing, and it is not
  stiffness)
- over 168 traced branches, switches ranged only 0…2 with **median 0 in both
  the inner and outer quartiles** — nearly nothing in that sample entered the
  stiff regime at all

Use κ.  Do not use step count, switches, or residual — `richardson3` absorbs
what stiffness reaches it into step count and holds accuracy roughly flat, so
the residual reports what is left *after* adaptation.

### The magic simplification

κ reaching 1e68 would defeat any general-purpose ODE method, and a general
Hadamard graph transform with it.  spong survives it because it does not
integrate through the stiff region: the critical points are known **exactly**
(Sturm, rational arithmetic) and the local **jets are exact expressions**, so
the worst country is evaluated analytically rather than traversed numerically.
That, not the integrator, is why these portraits can be drawn at all — and it
is the reason to doubt that any other phase portrait code would draw them
correctly.

### What sampling gives you, and what prescription gives you

Random draws do reach stiff saddles — 24.6% of the 785 sampled were in shallow
water, and `|b|` ran out to 1.9e3.  So the argument for the inverse
construction is **control, not reachability**: sampling lands where it lands,
while prescription puts a critical point at a chosen `|b|` and therefore at a
chosen κ.  That is what makes a stiffness sweep possible — pick the radii, get
the κ ladder, trace each one, and check the certificates hold across it.

A suite indexed by degree tests the wrong axis (r = +0.109); a suite indexed by
backbone position tests the right one (r = +0.748).

## Why it is also an oracle

Every certificate spong ships today — alternation, Poincaré–Hopf index balance,
seam and angle-energy residuals — is an *internal consistency* condition.  They
establish that a portrait is coherent, not that it is the portrait of the model
that was asked for.  The inverse construction supplies external ground truth,
and the extras change which side it checks rather than weakening it:

- **containment**, `prescribed ⊆ enumerated`, catches **missed** critical
  points — the dangerous failure, since a missing point silently breaks
  alternation and the index sum;
- **index balance and alternation** catch **spurious** critical points, which
  cannot balance the index.

Neither is two-sided alone; together they are.  That is strictly stronger than
the present self-certifying ledger.

## Rebuild checklist

1. Two-stage linear solve as above, including the `(f, g, μ) → (A, B)` inverse.
2. Reject prescriptions that violate alternation, with a clear diagnostic.
3. Report the full critical set (prescribed + extras) — it is exactly
   computable by the Sturm enumeration, so ground truth remains available.
4. Suite indexed by **geometry**: prescribed radii, close approaches at chosen
   separations, clusters — not by degree.
5. Large-`|b|` sweep to map the stiffness law, gating on
   **`depth_gauge_floor`** (`kappa_saddle`) — not the spectral scalar, and not
   step count, switches, or residual.
6. Experiment: can the free parameters push surplus roots complex, giving
   exactly the prescribed real critical set?
