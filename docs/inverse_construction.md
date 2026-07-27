# Design note: the inverse construction as a regression-suite generator

Prescribe the critical points; solve for polynomials `f` and `g` that have
them.  Used to generate spong's regression suite — in particular to place
critical points at large `|b|`, which is where the descent equation is stiff
and which random draws essentially never reach.

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

## Why this is the right suite generator: reachability

Difficulty is geometric, not degree-indexed.  Measured over 166 traced
branches (degrees 3–13, 4 seeds each), predicting integration effort:

| predictor | log₁₀(steps) | log₁₀(residual) |
|---|--:|--:|
| `\|saddle_b\|` | **r = −0.470** | r = +0.165 |
| degree | r = +0.142 | r = +0.005 |

Position on the backbone predicts effort about three times better than degree
does.  A regression suite indexed by degree therefore tests the wrong axis: it
will miss a stiff low-degree case and waste time on a benign high-degree one.

**But the random-draw measurement above does not reach the stiff regime, and
this is the point of the inverse construction.**  Over those 166 branches
`|saddle_b|` spanned only 0.12 … 20.9, with an outermost-quartile median of
5.5 — and within that narrow range the correlation runs *inward* (inner
branches took a median 6374 steps against 4001 outer, while achieving better
residuals, 2.5e−16 against 1.0e−13).  The stiffness at large `|b|` that the
construction was built to exercise lies outside what random `f`, `g` produce at
all.  Sampling cannot get there; prescription can.  The apparent tension
between the measured inward correlation and the design fact is a
regime-coverage artifact, not a contradiction — and resolving it is a good
first use of a rebuilt constructor.

Note also that residual is the wrong difficulty metric: `richardson3` absorbs
stiffness into step count and holds accuracy roughly flat, so the residual
reports what is left *after* adaptation.  **Step count is the honest measure.**

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
5. Large-`|b|` sweep to map the stiffness law outside the reachable-by-sampling
   regime, using step count rather than residual as the difficulty measure.
6. Experiment: can the free parameters push surplus roots complex, giving
   exactly the prescribed real critical set?
