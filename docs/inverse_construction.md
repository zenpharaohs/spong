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

### Still open

1. **Resolve the grid adaptively.**  A uniform `bg` over a span of 2e4 cannot
   see structure living in ~5 units near the saddle, which is what put the
   handoff where neither representation was validated.  The residual seam of
   2.8e−06 on that case is the surviving engine→shallow junction at the coarse
   node.  A grid clustered near the saddle (or refined where `kap` jumps
   between adjacent nodes) is the real fix; note `grid_index` would need a
   `searchsorted` rather than its O(1) inverse.
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
