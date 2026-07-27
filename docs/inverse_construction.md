# Design note: the inverse construction as a regression-suite generator

Prescribe the critical points; solve for polynomials `f` and `g` that have
them.  Used to generate spong's regression suite — in particular to place
critical points at large `|b|`, which is where the descent equation is stiff.
Sampling reaches stiff cases only by luck; prescription chooses them.

**Rebuilt 2026-07-26 as `spong.inverse`** (23 tests).  The derivation below is
a reconstruction from `model.py` that the original author confirms matches the
method originally used; the implementation is new.

## Mechanism

From §1–3 (`model.py` header):

    L(a,b) = C − 2a·B(b) + a²·A(b) = u(b) + A(b)·w²,   w = a − a*(b)
    u' = B·N / A²   with   N = A'B − 2B'A
    ⇒ critical b-values = real roots of N  ∪  real roots of B

So "prescribe the critical points" means "prescribe real roots of `N` and of
`B`", and the split is what makes it tractable:

The implementation is simpler than the two-stage sketch this note first gave.
Fix `g` and `μ` — they determine `A` alone.  Then

- `B_j = g_j · Σ_i f_i μ_{i+j}` is **linear in `f`**, and
- `N = A'B − 2B'A` is **linear in `B`**, hence linear in `f`.

So *both* root families are linear conditions on `f`, and a prescription is a
**single homogeneous linear system `M f = 0`**, solved exactly over ℚ.  One
stage, not two.

**Exactness.**  `model.Model` accepts `Fraction` coefficients, so a prescribed
`b` is an *exact* root — `P.eval_at(m.N, b) == 0` on the nose, and the Sturm
enumeration finds it exactly rather than nearby.

### Two preconditions found by building it

**`g(0) ≠ 0`.**  `A(b) = E[g(bX)²]` is positive for every `b ≠ 0` whenever `g`
is nonzero, but `A(0) = g₀²μ₀`.  A vanishing constant term puts a pole of
`a* = B/A` at the origin and the model is not well posed there.

**At most `deg g` points may be prescribed.**  Every condition is a linear
functional on `B`, which has only `deg g + 1` coefficients; `deg g + 1`
independent conditions force `B ≡ 0` — a model with *no critical points at
all*, in which the prescription is satisfied **vacuously**.  This was a live
bug in the first implementation: the exact-root assertions passed while `B` and
`N` were both identically zero.  Raising `deg f` does **not** help; the
bottleneck is `deg g`.  `design()` now rejects the over-prescription up front
and additionally refuses any solution whose `B` vanishes identically.

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

## The ladder, measured

`inverse.stiffness_ladder` designing one model per radius, `g = 1 + b + b²/2`,
uniform-[0,1] moments, tracing every unstable branch:

| `b*` | depth gauge | zone | term | worst seam residual |
|--:|--:|---|---|--:|
| 1/2 | 11.7 | mild | capture | 5.8e−21 |
| 2 | 2.4e3 | mild | capture | 3.6e−22 |
| 4 | 1.6e5 | shallow | capture | 1.4e−12 |
| 16 | 4.2e9 | shallow | capture | 7.4e−12 |
| 64 | 2.1e14 | shallow | capture | 1.1e−09 |
| 256 | 2.0e16 | shallow | capture | 1.8e−07 |

Fifteen orders of gauge, monotone, with the mild→shallow handoff at `b* = 4`,
and every branch terminating in `capture`.  Only measurable because
prescription can build the ladder: the random sample in the section above never
produced a single branch that entered shallow water at all.

### Turning the knob until it breaks — it doesn't

**First: the gauge saturates.**  `depth_gauge_floor` divides by
`max(|w₁'|, 1e-16·|a*'|)`, so it cannot report more than `2/1e-16 = 2e16`.
Every reading at 2e16 is that ceiling, not a measurement.  The raw ratio
`2|a*'|/|w₁'|` is needed to see past it.

Pushing `b*` from 2⁸ to 2²⁶ (≈ 6.7e7), raw gauge 2.3e18 → **5.2e61**:

| `b*` | raw gauge | zones | switches | term | seam |
|--:|--:|--:|--:|---|--:|
| 2⁸ | 2.3e18 | 2 | 0 | capture | 3.2e−07 |
| 2¹² | 1.0e28 | 2 | 0 | capture | 8.7e−07 |
| **2¹⁴** | **6.5e32** | **4** | **1** | capture | **8.6e−05** |
| 2¹⁵ | 1.7e35 | 1 | 0 | capture | 2.6e−16 |
| 2¹⁸ | 2.8e42 | 1 | 0 | capture | 7.5e−16 |
| 2²⁶ | 5.2e61 | 1 | 0 | capture | 6.5e−16 |

Capture never fails, over 43 further orders of raw gauge.  All of these are
genuine traces — Morse, two critical points, ~4000 polyline points, spans out
to 6.7e7 — not degenerate collapses.

**Difficulty is not monotone in stiffness.**  The worst seam residual on the
whole ladder, 8.6e−05, is at `b* = 2¹⁴`, precisely where the branch has *four*
zones and *one* chart switch: it straddles the mild/shallow boundary and
alternates.  Past the transition the branch is entirely in shallow water — one
zone, no handoffs, the Hadamard fixed point owning all of it — and the seam
falls back to roundoff (~6e−16), nine orders better than at the transition.

**Uniform stiffness is easy; mixed stiffness is hard.**  The instrument is
stressed by the *handoff*, not by the magnitude of κ.  A suite that only pushes
the extreme will therefore find nothing; the cases worth generating are the
ones that sit on the mild↔shallow boundary and cross it repeatedly.  Note also
that the prescription controls *where* a critical point is, not *which kind* —
in this family the prescribed point consistently became the far minimum, with
the saddle staying near b = −1.066.

## The transition-straddling suite

`straddling_suite` generates the cases that actually stress the instrument.
Three things had to be got right, two of them the hard way.

**Screening must be hysteretic.**  The engine enters shallow water at
`KAPPA_HI` and leaves only below `KAPPA_EXIT`, so a gauge that pokes above 1e4
and falls back to 5e3 never leaves the fixed point.  Counting raw crossings of
`KAPPA_HI` predicts nothing: over 120 traced candidates the median switch count
was **0 in every raw-crossing group**, and crossings correlated with the seam
at r = −0.32 — the wrong sign.  `hysteretic_zones` reproduces the engine's
state machine instead.

**Even hysteretic screening on a b-grid misses the worst case.**  The gauge is
sampled along the straight span, but the engine follows the *trajectory* and
can run **backward** in b inside a shallow zone.  The worst case found —
`g4`, `b* = 20480` — screens as **zero** transitions and actually produces four
zones, one switch, a seam residual of **3.42** and an angle energy of **32.2**,
while still reporting `term = capture`.  A suite built on prediction alone
omits precisely the case it most needs, so `straddling_suite(verify_all=True)`
traces every candidate and ranks on what the engine did.  `StraddleCase.
mispredicted` flags the disagreement.

**Anatomy of that failure.**  Zones:

    engine(−1.72 → 4.67) → shallow(4.67 → 3.38) → engine(3.38 → 8.50) → shallow(8.50 → 20480)

The second zone runs **backward**: entered at 4.67, the fixed point converged
to 3.38, behind where it started — the ping-pong the `depth_gauge_floor`
docstring warns about.  The engine then re-runs the same territory and hands
off again at 8.50, and that seam is the O(1) one.  The cause is visible in the
gauge profile: it goes **813 → 2e16 with nothing in the hysteresis band**
(`fraction inside [1e3, 1e4) = 0.000`), so the handoff is necessarily made at a
point where neither representation has been validated.

## Improving the handoff: what the suite says

Lowering the thresholds fixes that case spectacularly — at `KAPPA_HI = 1e2` its
seam goes 3.42 → 5.09e−17, sixteen orders, with one zone and no switches, and
raising them makes it monotonically worse (at 1e6: three switches, angle energy
100).  **But that is one case.**  Across the 35-case suite:

| HI / EXIT | median seam | median \|E\| | max \|E\| |
|---|--:|--:|--:|
| **1e4 / 1e3** (current) | 1.6e−11 | **8.8e−12** | 5.4e−07 |
| 1e3 / 1e2 | 1.1e−09 | 1.8e−11 | 1.1e−06 |
| 1e2 / 1e1 | 0.0 | 6.0e−08 | 6.4e−04 |
| 1e5 / 1e4 | 4.2e−12 | 8.5e−12 | 1.2e−05 |

Dropping to 1e2 makes the suite median angle energy **four orders worse**.
**The current 1e4 is well chosen and should not be moved globally** — the
single-case experiment was overfitting, which is exactly what the suite exists
to prevent.

So the improvement directions are local, not global.  Failing loudly is the
backstop, to be reached only after everything that could actually help has been
tried.

### FIXED: the backward shallow zone

Root cause, confirmed by replaying the zone loop's arithmetic.  `grid_index`
rounds to **nearest**, so `bg[i_cur]` can lie *behind* `b_cur`; and when the
`KAPPA_EXIT` walk does not advance (that node's gauge is already below the exit
threshold) `j` stays at `i_cur` and `np.linspace(b_cur, bg[j], ...)` runs
backward.  Measured at `g4`, `b* = 20480`:

    b_cur = 4.673945   i_cur = 1   bg[1] = 3.380928   (behind b_cur)
    kap[1] = 813 < KAPPA_EXIT = 1000  ->  loop never advances  ->  j = 1

The zone was *entered* because the exact scalar gauge at `b_cur` was ≥ 1e4
while the grid node read 813 — four orders apart, because `hstep = 20481/4000
= 5.12` and the whole 813 → 2e16 transition lives inside **one grid cell**.

Fix: the zone endpoint must lie strictly ahead of its start, `(bg[j] − b_cur)·
sgn > 0`.  Measured over 147 straddling cases, baseline versus guarded:

| | baseline | guarded |
|---|--:|--:|
| median seam | 4.61e−12 | 4.61e−12 |
| median \|E\| | 1.40e−09 | 1.40e−09 |
| **max seam** | **3.42e+00** | **2.85e−06** |
| **max \|E\|** | **3.31e+01** | **9.50e−01** |
| cases with \|E\| > 1 | 2 | **0** |
| backward zones | 2 | **0** |

Medians identical, tail repaired — the signature of a correct guard rather than
a retuning.  On the case itself: seam 3.42 → 8.67e−16, angle energy 32.2 →
0.0033, four zones → three, and the backward zone gone.

### TRIED AND REJECTED: the adaptive grid

The obvious next fix was to resolve the gauge table where it moves, since a
uniform `bg` over a span of 2e4 cannot see structure occupying ~5 units near
the saddle.  Implemented as gauge-driven refinement (bisect any interval across
which log₁₀ κ moves by more than `dlog`, nodes built in a normalized coordinate
so the index still increases toward the target) with `grid_index` becoming a
`searchsorted`.

**It is a regression, and the reason inverts the premise.**  On `g2`,
`b* = 2¹⁸` — previously one zone with seam 7.5e−16 — refinement gives two
zones, seam 1.2e−01 and angle energy **2.2e+03**.  Isolating the two changes
shows the fault is the refinement alone, and that as few as **four** extra
nodes are enough:

| `dlog` | nodes | zones | seam | \|E\| |
|--:|--:|--:|--:|--:|
| 0.75 | 4034 | 2 | 1.24e−01 | 2.22e+03 |
| 3.0 | 4005 | 2 | 1.24e−01 | 2.24e+03 |
| 50 (no refinement) | 4001 | 1 | 7.48e−16 | 6.03e−01 |

`searchsorted` alone is harmless — the last row has it active.

**The coarse grid's mislabelling was load-bearing.**  Near the saddle the true
gauge is low, but the nearest coarse node sits far out where the gauge is huge,
so the zone loop read "shallow" and handed the stretch to the Hadamard fixed
point, which traversed it cleanly.  An accurate table labels that stretch deep,
hands it to the continuation engine, and the engine is what fails.

The lesson is about where the weakness actually is: **the fixed point
outperforms the engine even in territory the gauge calls deep.**  `KAPPA_HI`
does not mark where the fixed point becomes *necessary*; the fixed point is
already the better instrument well before that line.  Improving the handoff is
therefore not a matter of locating it more precisely — it is a question of how
much territory the fixed point should own, or of why the engine degrades on
these near-saddle stretches.

Note this cuts against the naive reading of the threshold sweep above, which
found lowering `KAPPA_HI` globally made the suite median worse.  Both
observations are on a uniform grid, where mislabelling already routes some
near-saddle stretches to the fixed point regardless of the threshold; the two
effects are entangled and neither measurement isolates the question.

### ANSWERED: the fixed point's territory should NOT be extended

The experiment the adaptive-grid failure pointed at: run the Hadamard fixed
point on stretches the gauge calls DEEP and score it against the continuation
engine on the same stretch, from the same starting state, with the same
certificate (`angle_energy`, normalized per vertex; the engine's shallow
handoff gated off so it stays in engine mode).  357 stretches, degrees 2–4.

**The fixed point is valid far below `KAPPA_HI`.**  Its lowest converged κ is
**664** (`rel < 1e-10`); below that it genuinely diverges, `rel = inf` and the
certificate runs to 1e224 and worse.  And where both work it is the better
representation by a wide margin — in `[1e3, 1e4)`, entirely below `KAPPA_HI`,
E/vertex is **1.82e−19 against the engine's 1.30e−17**, about 70×.

**But extending its territory makes things worse.**  Holding `KAPPA_EXIT` at
1e3 (safely above the 664 floor) and lowering only `KAPPA_HI`, over 147 cases:

| HI (EXIT = 1e3) | median seam | median \|E\| |
|--:|--:|--:|
| **1e4** (current) | **4.61e−12** | **1.40e−09** |
| 6e3 | 1.14e−11 | 2.21e−09 |
| 3e3 | 4.81e−11 | 5.61e−09 |
| 1.2e3 | 4.35e−10 | 6.74e−09 |

Monotone, 150× on the median seam.  Both results are real, and the resolution
is that **`KAPPA_HI` was never marking fixed-point validity** — it marks where
the TRAJECTORY becomes slaved to the floor, which is the condition the handoff
actually needs (`_continue_curve` already tests `|w − w₁| ≤ 0.05|w₁|`
alongside the κ test).  Measured along traced branches:

| κ band | median \|w − w₁\|/\|w₁\| | fraction slaved (≤ 5%) |
|--:|--:|--:|
| [1e3, 3e3) | 8.20e−02 | 43.9% |
| [3e3, 1e4) | 3.24e−02 | 70.5% |
| **[1e4, 3e4)** | **6.97e−03** | **95.1%** |
| [3e4, 1e6) | 9.99e−03 | 97.4% |

Slaving jumps to 95–97% exactly at `KAPPA_HI`.  Handing off earlier snaps a
trajectory that has not yet reached the floor onto it, and the seam residual is
the size of that snap.  So the current threshold is well chosen, and now has a
mechanism rather than a tuning behind it.

**Where this leaves the weak spot.**  The problem is not the location of the
boundary but that the trajectory is not slaved when one would like to hand off.
That reframes the remaining ideas:

1. **Match where the representations agree best**, rather than at a fixed κ plus
   a 5% test — i.e. choose the handoff to minimize `|w − w₁|` over the approach.
   This is the same matched-asymptotics idea as before but now correctly
   motivated: the quantity to match is the slaving error, not the gauge.
2. ~~**Correct the snap instead of avoiding it.**~~  **RULED OUT — see below.**
3. **Accelerate the approach to the floor** so slaving is achieved sooner.
4. Only then, fail loudly.

### RULED OUT: defect correction of the snap

The idea: at a handoff the engine sits at `w_cur` while the graph is at
`w_zone[0]`, and the code discards the difference.  Since the manifold
*attracts*, that offset is a decaying transient rather than an error, so carry
it — `w(b) = w_zone(b) + δ₀·φ(b)` — instead of snapping.

Along the flow the chart obeys `dw/db = F(b,w) = 2Aw/P − a*′`, so linearizing
about the graph gives `dδ/db = λδ` with

    λ = ∂F/∂w = 2A/P − 2A w P_w / P²,     P_w = 2A′w − 2A a*′

and the carried transient is `δ₀ exp(∫ λ)`.  Implemented and measured.

**It is inapplicable, and the reason is quantitative.**  λ is enormous
wherever the fixed point is usable — measured at the actual handoffs, −8.8e13
to −1.5e46, so `φ` underflows to zero inside the first grid step and the
"correction" is either a no-op or a full snap, with no intermediate regime.
The corrected polyline scored no better than the snapped one on 9 of 10 cases.

Sampling λ across the whole gauge range shows the two requirements never
overlap:

| κ band | median \|λ\| | transient decay length 1/\|λ\| |
|--:|--:|--:|
| [1, 10) | 4.15 | 2.4e−01 |
| [1e2, 1e3) | 675 | 1.5e−03 |
| [1e3, 1e4) | 4.5e3 | 2.2e−04 |
| [1e4, 1e6) | 2.5e5 | 4.0e−06 |
| [1e10, ∞) | 2.4e23 | 4.2e−24 |

Wherever the fixed point converges (κ ≥ 664) the transient dies within ≲1.5e−3
— orders below one grid cell.  Where it would be *resolvable* (κ ≲ 10) the
fixed point diverges.  **No κ makes the correction both applicable and
needed**, and the snap is the correct representation of a boundary layer that
is genuinely thinner than anything the polyline can carry.

**What this rules out about the seam.**  The seam residual is therefore *not*
an uncarried transient.  Measured δ₀ at real handoffs is ~3e−6 while the local
attraction is instantaneous, which means the trajectory is already glued to the
floor to ~3e−6 and the seam is measuring the ENGINE's own accuracy in tracking
it, not a discarded piece of dynamics.  That is consistent with lowering
`KAPPA_HI` making seams worse — an earlier handoff gives the engine less
distance to converge onto the floor — but it is an inference from two
measurements rather than a direct one, and has not been tested.

### The overlap: real, but not a "both fraying" overlap

Andrew's expectation was that near the handoff both methods might be fraying
while their *consensus* stays accurate — the standard matched-asymptotics
picture — and that it would be surprising if the two had no overlap at all.
The overlap is real; the consensus idea is not supported.

**Correction to an earlier claim in this note.**  "No overlap" was said about
the *defect correction's* applicability, not about the two methods.  By the
`angle_energy` certificate both are healthy across κ ∈ [1e3, 1e6] — three
decades — with the engine fraying above 1e6 and the fixed point below 1e3.

**Consensus does not help: 0 of 273 stretches.**  Averaging the two
representations scored between them every time, with `E_consensus ≈ E_eng/4` in
every band — exactly what averaging does when one input is far better than the
other (half the deviation, and E is quadratic in it).  There is no band where
the two are comparably degraded, which is what a useful consensus would need.

**But `angle_energy` cannot settle this**, and that is worth recording: it is
satisfied by ANY integral curve.  The slow manifold is one and the branch from
the saddle is another; E cannot tell them apart, so the seven-order margin it
reports for the fixed point is not a statement about the branch.

Against a trusted trajectory instead (engine at ds/50 from the same start):

| κ | \|engine − ref\| | \|fixed point − ref\| |
|--:|--:|--:|
| 2.5e5 | **4.96e−09** | 1.17e−07 |
| 9.0e7 | 6.23e−06 | 1.09e−06 |
| 2.6e10 | 2.53e−02 | **5.72e−09** |
| 2.0e16 | 3.86e+04 | **8.43e−09** |

so the engine is the better representation of the *branch* at moderate κ and
the fixed point at high κ, with a **crossover somewhere in [1e5, 1e8]** — not
a region of joint degradation but a clean handover of dominance.

**Reference uncertainty bounds all of this.**  Two independent fine runs
(ds and ds/4) agree only to 6e−09 … 2.6e−08, so the "trusted" trajectory is
itself uncertain at ~1e−8.  At κ = 2.6e10 the fixed point's 5.72e−09 is *below*
that floor — indistinguishable from the best trajectory available — while the
coarse engine's 2.53e−02 is far above it and genuinely wrong.

**The open question this raised — now settled: `KAPPA_HI = 1e4` is correct, do
not raise it.**  The per-stretch crossover sits one to four orders ABOVE the
threshold, which looked like an argument for handing off later.  It is not.
Re-running the threshold sweep over the 147-case suite *with* the backward-zone
guard `341c0b6` in place (the earlier sweep predated it):

| `KAPPA_HI` | med seam | max seam | med \|E\| | max \|E\| | \|E\|>1 |
|---|---|---|---|---|---|
| **1e4** | 4.61e−12 | **2.85e−06** | 1.40e−09 | **9.50e−01** | **0** |
| 3e4 | 1.31e−12 | 3.92e−01 | 2.20e−09 | 2.13e+01 | 4 |
| 1e5 | 2.64e−11 | 5.70e−02 | 1.40e−09 | 2.53e+02 | 14 |
| 1e6 | 2.11e−12 | 7.81e−02 | 8.44e−09 | 1.67e+02 | 24 |
| 1e7 | 2.09e−23 | 8.69e−02 | 2.83e−07 | 9.00e+05 | 31 |

The medians are flat across five orders of threshold — every row is ~1e−12 seam,
~1e−9 \|E\| — while the **tail explodes monotonically**.  1e4 is the only
setting with no \|E\|>1 case at all.  So *per-stretch accuracy and the right
place to hand off are different questions*: the fixed point being the better
representation at κ=1e6 does not mean the trace should stay with the engine
until κ=1e6, because a late handoff lands the zone somewhere geometrically
worse.  The threshold is not an accuracy knob; it is a placement knob.

**The reference-free measure, done properly.**  The earlier `inf` everywhere was
a harness bug, not a numerical finding: `charts.slow_rhs(m)` returns a **tuple
`(F, J)`**, the tuple was passed as `F`, every call raised, and a bare
`except Exception` turned that into `inf`.  Nothing was ever measured.

Run correctly (`F, J = charts.slow_rhs(m)`, `jac=J`, short sub-spans away from
both critical points where `P → 0`, 225 stretches, step count swept):

| κ band | n=50 | n=200 | n=800 | n=3200 |
|---|---|---|---|---|
| [1e4, 1e6) | 9.69e−19 | 1.75e−08 | 8.17e−01 | 1.01e+00 |
| [1e6, 1e10) | 4.85e−21 | 5.57e−20 | 9.08e−19 | 1.52e−10 |
| **[1e10, ∞)** | **6.02e+01** | **4.87e+01** | **1.97e+01** | **2.53e+00** |

Read this carefully, because it is **not** an accuracy measure and the gap
mostly *grows* with n.  The docstring says so outright: "for a symmetric method
this is Newton-tolerance + roundoff, NOT a method-order quantity."  It certifies
that the *scheme* is anadromic, and a symmetric scheme passes at any step size.
Worse, a small gap can be small for the wrong reason — once \|hλ\| is enormous
the Gauss stability function saturates at R(−∞) = ±1, so forward-then-backward
cancels *exactly* whether or not anything was resolved.  Saturation and accuracy
are indistinguishable in this statistic.

What it does certify, reference-free, is where the **nonlinear solve** fails:
above κ ≈ 1e10 the engine cannot reverse its own trajectory at any step count.
That independently confirms the catastrophic failure the trusted-trajectory
comparison showed (3.86e+04 at κ = 2e16) without leaning on a reference that is
itself an engine run — and bounds the usable engine region from above.

### Still open
2. **Match in an overlap region.**  When the profile has *no* samples in
   `[KAPPA_EXIT, KAPPA_HI)` there is no overlap in which to match, and the fixed
   thresholds are meaningless for that branch.  Matching where the two
   representations agree best is the matched-asymptotics answer.
3. **Extend the exact jets.**  The handoff happens at b ≈ 4.7 with the saddle at
   −1.74; a certified validity radius for the jet could cover it analytically
   and skip the engine phase.
4. **Only then, fail loudly.**  A branch whose seam survives all of the above
   at O(1) should not report `capture`.

## Status and what is left

Implemented in `spong.inverse`: the exact single-stage solve, the `g(0) ≠ 0`
and `deg g` preconditions, rejection of the vacuous `B ≡ 0` solution,
`depth_gauge_at` / `is_shallow` (reading the gauge just off the critical point,
as `trace_unstable` does), `report()` with containment plus extras plus the
gauge, and `stiffness_ladder`.  Tests: `tests/test_inverse.py`.

Left to do:

1. Reject prescriptions that violate alternation up front, with a clear
   diagnostic — currently a bad prescription is only caught downstream by the
   enumeration.
2. Suite indexed by **geometry**: prescribed radii, close approaches at chosen
   separations, clusters — not by degree.
3. Experiment: can the free parameters push surplus roots complex, giving
   exactly the prescribed real critical set?

### The commutator criterion: right idea, closes off a whole family

`kappa = 2|a*'|/|w1'|` sees only the **spectral gap**.  Slaving can also fail
when the eigenframe *rotates* fast compared to the gap, which no gap scalar can
see.  The commutator is exactly that missing part: along the descent flow
`Hdot = -grad^3 L[grad L]`, and

$$[H,\dot H]_{ij} = (\lambda_i-\lambda_j)\dot H_{ij}$$

annihilates whatever merely rescales the timescales, keeping only what mixes
fast into slow.  `[H,Hdot] = 0` iff the fast/slow splitting is invariant — i.e.
iff the fixed-point construction is *exact* rather than approximate.  Two things
make it practical here: `L = C - 2aB(b) + a^2 A(b)` is quadratic in `a` and
polynomial in `b`, so `grad^3 L` is **closed form** (only `A'''`, `B'''` beyond
what `Model` stores — no finite differences); and for symmetric 2x2 the
commutator is antisymmetric, hence a single *signed* scalar with no scalarizing
choice to make:

$$\omega = (p-r)\dot q - q(\dot p - \dot r),\quad
  \text{gap} = \sqrt{(p-r)^2+4q^2},\quad
  \mathfrak{A} = |\omega|/\text{gap}^3$$

**Measured, 42355 samples along traced branches — it does not detect the
changeover.**  `A` peaks at kappa in [1e1,1e2) and decays monotonically above,
reaching 1e-37; it is `<< 1` *everywhere*.  So the eigenframe is effectively
frozen throughout, and frame rotation is **not** the mechanism of handoff
difficulty.  A per-branch check killed the one suggestive signal too: pooled
across models `sign(omega)<0` swings 98% -> 25% across kappa=1e4, but within a
single branch there are 50-123 sign flips, clustered at the gauge's 2e16
saturation cap.  Band-membership confounding, not structure.

**Why it fails is the real result.**  The *unnormalized* rotation rate is nearly
constant — median |omega| = 5.85, 5.94, 11.2, 10.8, 19.6 across kappa bands
1e3 ... 1e8, i.e. O(10) over eight orders — while the gap moves 6.1e1 -> 1.9e5.
All the variation is in the gap.  Dividing by gap^3 therefore does not build a
new instrument; it rebuilds kappa with a worse exponent.  **Any criterion
carrying the gap in its denominator inherits kappa's monotonicity**, and the
cube amplifies it rather than curing it.

That closes off the whole spectral family: no function of `H`'s spectrum along
the flow can mark the changeover, because the spectrum is monotone in position
while the difficulty is not.  Consistent with the two independent findings
above — log kappa vs log|b| r=+0.748 against degree r=+0.109 (difficulty is
positional), and the threshold sweep leaving every median flat while only the
tail degrades (`KAPPA_HI` is a *placement* knob, not an accuracy knob).

### The jet-based handoff detector: order reduction is the whole story

Andrew's construction, by analogy with the "explicit deviation from implicit"
stiffness scalar (difference of two methods in the `h -> 0` limit): measure the
difference between the ODE integrator and the Hadamard graph transform.  Unlike
the commutator this is a *method* difference, not a gap-normalized quantity, so
nothing forces it to inherit kappa's monotonicity.

**What the two methods actually are.**  The transform is the invariance equation
`w' = 2Aw/P - a*'` solved for `w` (the contracting direction):

```python
w <- P*(a*' + w')/(2A)          # w' from np.gradient
```

so its fixed point IS the exact invariant manifold, and its only error is the
finite-difference `w'`.  Measured self-convergence: **interior order 2.00, edge
order 1.00** -- `np.gradient` defaults to `edge_order=1`, and *the zone edge is
exactly where the seam is measured*.  Fixed (`edge_order=2`); over the 147-case
suite the median seam improves 4.61e-12 -> 1.81e-12 with max seam, max |E| and
the |E|>1 count bit-identical.

**The prediction, and why it was wrong.**  With the engine at order 4 and the
transform at order 2 damped by the contraction factor `1/kappa`, the errors would
balance at `kappa h^2 = C2 J2/(C1 J1)` -- making `kappa*h^2` the dispatch
variable.  Measurement says otherwise: the ratio `err_ODE/err_GT` is **constant
in h**,

| n | 26 | 51 | 101 | 201 | 401 |
|---|--:|--:|--:|--:|--:|
| ratio (kappa=8.9e7) | 1.65e7 | 1.83e7 | 1.83e7 | 1.83e7 | 1.82e7 |

because **both** methods are second order here.  IRK4-GL is 2-stage Gauss:
classical order 4, **stage order 2**, and for stiff problems the observed order
drops to the stage order (B-convergence, Hairer-Wanner II).  This is the
guaranteed rate, not a defect -- verified by control experiment: on non-stiff
problems the same code gives 4.00, 4.00, 4.00 at `|h*lambda|` = 6e-3 and 6e-2,
and 3.98/4.00/4.00 on `y' = -y + sin x`.  In spong the observed order is a clean
2.00 because the integration runs *along the manifold*, where the transient is
absent and the collocation defect is exactly the stage order.

So the `h` dependence CANCELS:

$$\frac{\mathrm{err_{ODE}}}{\mathrm{err_{GT}}} = \frac{C_1J_1}{C_2J_2}\,\kappa
  \qquad\text{independent of }h$$

and the measured ratios track it: kappa=8.9e7 -> 1.8e7, kappa=5.1e9 -> 3.0e9,
kappa=6.2e12 -> 1.0e12, i.e. **ratio ~ 0.2 kappa**.  Refining the step helps both
methods equally; there is no step size at which the engine catches up.

**Consequence: the engine retires far too late.**  The transform wins once
kappa > ~5, tempered only by its own convergence requirement.  Sweeping down:

| `KAPPA_HI` | med seam | max seam | med \|E\| | max \|E\| | \|E\|>1 |
|---|--:|--:|--:|--:|--:|
| 1e4 (current) | 4.61e-12 | 2.85e-06 | 1.40e-09 | 9.50e-01 | 0 |
| **3e2** | **3.58e-16** | 3.74e-06 | 3.17e-08 | 9.50e-01 | 0 |
| 3e1 | 2.09e-23 | 2.47e-03 | 1.18e-04 | 9.50e-01 | 0 |
| 1e1 | 5.72e-24 | 2.08e+00 | 3.08e-04 | 8.61e+00 | 2 |

Four orders of median seam at `HI = 3e2`, max seam essentially unchanged (1.3x),
max |E| and the failure count identical.  The counterweight is median |E|, which
worsens 23x -- so this is a real trade, not a free win, and the threshold change
is NOT applied here pending a decision on which metric governs.  Note this is the
opposite direction from the earlier "raise it" hypothesis, and both are now
measured: raising degrades the tail monotonically, lowering improves the median
seam and degrades median |E|.

### Would IRK6-GL help?  Measured: yes in the mild region, not as a handoff fix

B-convergence says the stiff regime observes the STAGE order: 2-stage Gauss
(GL4) has classical 4 / stage 2; 3-stage (GL6) has classical 6 / stage 3.  So
GL6 should buy +1 where it matters and, since `err_ODE ~ C1 h^3` against
`err_GT ~ C2 h^2/kappa`, un-cancel the `h` dependence -- restoring a crossover
the engine can reach by refining.

**Measured (3-stage Gauss dropped into `_newton_stages`, b = 5/18, 4/9, 5/18).**
Control on non-stiff problems: order 6.00, 6.00, 6.07 -- implementation correct.
Along the manifold the gain is LARGER than the +1 predicted (observed order ~4,
not 3):

| case | kappa | GL4 | GL6 | GT (order 2) | GT (order 4) |
|---|--:|--:|--:|--:|--:|
| g2 2^3 | 7.3e+05 | 8.83e-18 | **6.25e-21** | 9.33e-20 | 4.05e-21 |
| g2 2^5 | 8.9e+07 | 2.16e-16 | 4.46e-22 | 1.19e-23 | **2.58e-26** |
| g4 2^5 | 5.1e+09 | 2.29e-18 | 1.26e-23 | 6.06e-28 | **2.02e-28** |

GL6 **does** move the crossover: at kappa = 7.3e5 it beats the current transform
15x, which GL4 never does at any step size.  But it cannot win the asymptotic
argument -- GL6's gain is a fixed factor while the transform keeps an unbounded
`1/kappa` damping, so above kappa ~ 1e7 the transform wins again.  **The handoff
moves; it does not go away.**

**The cheap alternative is exhausted.**  A 4th-order stencil for `w'` in the
transform costs no Newton stages and is 3-460x better on isolated stretches, but
over the 147-case suite it gives 1.81e-12 -> 1.79e-12 -- nothing.  Reason: the
seam is measured AT THE ZONE EDGE, and once `edge_order=2` fixed the edge the
interior stencil order stopped mattering.  The seam is no longer
stencil-limited; what remains is the placement effect.

**Recommendation.**  Adopt GL6 for the MILD region on classical order alone
(order 6 vs 4 is fewer steps for the same accuracy, and that is where the engine
does its production work), not as a way to push back the handoff.  Cost to weigh
against it: `gl4_scalar` is a Tier-0 pure-float fast path built on a CLOSED-FORM
2x2 stage Newton; three stages needs a 3x3 stage solve and that closed form does
not carry over.

### What the wider overlap actually buys: certification, not consensus

The point of GL6 is not to beat the transform (it cannot -- fixed factor vs
unbounded `1/kappa`).  It is that a wider band in which BOTH methods are
genuinely accurate is what makes *matching* possible.  With GL4 the overlap is
where both are fraying, which is why averaging failed there.

**The band widens by five decades.**  Relative errors at matched `n`, and the
count of stretches where BOTH methods are usable (rel err < 1e-6):

| kappa band | GL4/GT | GL6/GT | both usable (GL4) | (GL6) |
|---|--:|--:|--:|--:|
| 1e4-1e6 | 5.2e+02 | **8.4e-03** | 23/26 | **26/26** |
| 1e6-1e8 | 6.6e+04 | **2.2e-01** | 8/11 | **11/11** |
| 1e8-1e10 | 1.7e+08 | 8.4e+02 | 1/4 | **4/4** |
| 1e10-1e13 | 1.2e+10 | 1.5e+05 | 1/6 | **6/6** |
| 1e13+ | 3.1e+295 | 1.1e+292 | 1/36 | 23/36 |

With GL4 the overlap dies by kappa ~ 1e8; with GL6 it survives past 1e13.  Note
the GL6/GT ratios BELOW 1 in 1e4-1e8: GL6 is better than the transform in the
very band the dispatcher currently gives away at `KAPPA_HI = 1e4`.

**In that band the disagreement is a perfect error estimator** (41 stretches,
kappa in [1e4,1e10)):

```
median |w6 - wGT| / max(err) = 1.000
within 10x of max(err)        : 41/41
never UNDER-estimates by >10x : 41/41
```

This is the prize.  Every accuracy measurement in this note has been limited by
needing a reference trajectory that is itself an engine run, self-consistent only
to ~1e-8.  Two independent representations of comparable accuracy give a
**reference-free certified error bound** at every point of the overlap.

**Consensus (averaging) still fails, and now the reason is visible.**  Better
than both in 1/41, worse than both in 0/41, but median consensus 3.30e-19 against
GL6's 1.24e-20 -- 27x worse than simply taking the better method.  The two errors
differ by ~24x, so the average drags the good estimate toward the bad one.
Blending needs errors that are comparable AND independent; these are neither.

**So the overlap is for certification and placement, not blending.**  Hand off to
whichever method is better, and use the disagreement to (a) certify the result
without a reference and (b) choose WHERE to hand off -- minimizing disagreement
within the overlap, which is the matched-asymptotics answer.  That is directly
the tool the residual problem needs: every other line of attack (raising
`KAPPA_HI`, lowering it, spectral criteria, stencil order) ended by pointing at
placement, and placement is what a reference-free error estimate can finally
resolve.
