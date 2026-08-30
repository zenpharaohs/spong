# spong — founding document

**spong** (Single POlynomial Neuron Gradient) computes the dynamics of the
steepest-descent flow on the MSE loss of a single scalar polynomial neuron
"network" — "network" because one neuron is a trivial network, and the point
of the exercise is that even the trivial network defeats descent.  spong is
the reference instrument of the *beautiful untrainable nets* corpus: it draws
the graph paper on which descent methods go to die, to the limit of achievable
precision, with a certificate attached to every curve.

Successor to `mse-bundle` (MATLAB).  Trusted core: Python stdlib + NumPy — no
ODE-suite or plotting-suite black boxes.  Exact arithmetic where the ledger
says EXACT is stdlib `fractions.Fraction` over the model's coefficients:
IEEE floats are dyadic rationals, so "exact over the model as given" is a
rigorous notion with no external dependency (coefficient growth in chains is
controlled by subresultant PRS).  Floating kernels (NumPy) carry everything
labeled VALIDATED or RESIDUAL.

**Certificate semantics** (used as labels throughout; the metrology claim is
only as strong as these definitions):

- **EXACT** — computed in rational arithmetic over the dyadic inputs; the
  statement is a theorem about the model as given (e.g. Sturm root count).
- **VALIDATED** — floating computation with a rigorous a-priori error bound
  or interval enclosure (e.g. compensated sums with running error bounds).
- **RESIDUAL** — a-posteriori: a residual the reader can recompute
  independently of the code that produced the object (e.g. angle-energy,
  invariance residual, reversal gap, extrapolant agreement).
- **EMPIRICAL** — regression evidence only (e.g. zoo sweeps); never load-
  bearing for a correctness claim.

---

## Part I — Why this exists (and why not an off-the-shelf phase portraitist)

A reader will reasonably ask why we did not use pplane, MATLAB's streamlines,
matplotlib's `streamplot`, or any of the web's phase-portrait tools.  Three
answers, in increasing order of importance.

**1. Generic tools cannot draw these portraits at all.**  The flow is a
polynomial gradient system whose saddle stiffness ratio κ = λ_fast/λ_slow
reaches 10⁹ at modest degree.  Measured on the canonical hard case (degree 11,
κ = 8.5×10⁸): a normalized-Euler tracer takes 20,000 steps to move 10⁻⁶ and
oscillates at 90° to the true curve; a Hessian-eigenframe integrator reaches
the target but with 69° worst-case angle error through the bend; generic
adaptive time-steppers exhaust their budgets in the stiff zone or silently
substitute a nearby easier curve (the backbone), which is *wrong* and looks
plausible.  Grid-based contouring cannot resolve level-set structure 10⁻⁹
units wide.  The methods in Part II draw the same branch with angle-energy
residual 5×10⁻¹³ in a fraction of a second.

**2. An uncertified reference cannot adjudicate.**  The purpose of these
portraits is adversarial: they are the ground truth against which descent
methods (SGD, Adam, L-BFGS, …) are judged.  A referee whose own output is
uncertified settles nothing — the skeptic will (correctly) suspect the
portrait before suspecting the optimizer.  Every spong output therefore ships
with machine-checkable residuals (Part II §11): the reader need not trust the
tracer, only re-evaluate the certificates.

**3. The portrait is the mean-field object.**  The descent ODE is Ljung's
mean-field limit of SGD: we deliberately hand the reader the noiseless flow so
that "it's just SGD noise" is foreclosed as an explanation.  The one-neuron
net sees the data distribution only through its moments μ₀…μ₂d, so a batch is
a point in moment space and the portrait is one fiber of a family over moment
space.  The walls where the critical-point inventory ceases to be Morse are
algebraic and can be certified (Part II §4).  The full separatrix
configuration has additional global handle-slide walls, defined by a shooting
condition and invisible to the algebraic discriminant.  None of this structure
is visible to, or exploitable by, a generic streamline plotter.

Descent methods themselves are **not part of spong**.  They appear only in
`demos/`, as consumers, overlaid on certified portraits.

---

## Part II — The mathematics (complete recital)

Everything spong draws follows from the identities below.  Coefficient-level
constructions (A, B, N, discriminants, Sturm chains) are EXACT — rational
arithmetic over the dyadic inputs.  Point evaluations and traces are floating,
certified RESIDUAL (with VALIDATED enclosures where a-priori bounds are
cheap), using only the Gauss collocation integrators of §10.

### 1. Model

Data x ~ X with finite moments μ_k = E[X^k], k = 0…2d.  Neuron: input scalar
x, output a·g(bx); target f(x); f, g polynomials of degree ≤ d.  Loss:

    L(a,b) = E[(f(X) − a·g(bX))²] = C − 2a·B(b) + a²·A(b)

    A(b) = E[g(bX)²]  = Σ_m α_m b^m,   α_m = μ_m Σ_{i+j=m} g_i g_j
    B(b) = E[f(X)g(bX)] = Σ_j β_j b^j,   β_j = g_j Σ_i f_i μ_{i+j}
    C    = E[f(X)²]

A ≡ ψ and B ≡ φ in the older notes.  Standing hypothesis (**ψ-nice**):
A(b) > 0 for all real b — true whenever μ is a genuine moment sequence and
g ≢ 0, and *certified* by a Sturm positivity test on A (§4), with a Hamburger
moment check on μ.

### 2. The chart (completing the square)

    L(a,b) = u(b) + A(b)·w²,    w = a − a*(b),
    a*(b) = B/A  (the backbone),   u(b) = C − B²/A  (the reduced loss).

This identity is exact and is the coordinate system of the whole program:

- vertical slices are upward parabolas — **L has no maxima**;
- level curves are closed form:  a_±(b;c) = a*(b) ± √((c − u(b))/A(b));
- the gradient is  ∇L = (2Aw, P),  P(b,w) = u′ + A′w² − 2Aw·a*′;
- the descent flow is  ȧ = −2Aw,  ḃ = −P.

The deviation w is carried as its own variable everywhere: recomputing it as
a − B/A near the backbone is a catastrophic cancellation and is forbidden in
the implementation.

### 3. Critical points and Hessian identities

∇L = 0 ⟺ w = 0 and u′(b) = 0, and u′ = B·N/A² with

    N(b) = α′(b)β(b) − 2β′(b)α(b)         (deg ≤ 3d−2: leading term cancels)

so the critical b-values are the real roots of B UNION the real roots of N.
At any backbone point (identities universal on the backbone, no critical-
point substitution needed):

    H₁₁ = 2A,   H₁₂ = −2A·a*′,   H₂₂ = 2A·a*′² + u″
    H = 2A·nnᵀ + u″·e₂e₂ᵀ,   n = (1, −a*′)ᵀ (backbone normal)
    det H = 2A·u″

Consequences (all load-bearing):

- **Classification is one-dimensional**: u″(b*) > 0 ⇒ minimum, < 0 ⇒ saddle;
  no other types exist (A > 0 ⇒ no maxima).  At a simple B-root,
  N = −2B′A there, so u″ = −2B′²/A < 0 automatically: **every B-root is a
  saddle** (det H = 2A·u″ = −4B′², the same universal identity).  Reduce
  \(B^2/A=P/D\) and write \(u'=H/D^2\), where
  \(H=PD'-P'D\).  Then \(L\) is Morse iff \(H\) and \(H'\) have no
  common **real** root.  Squarefree, coprime \(B,N\) are the fast generic
  factorized case, not a necessary condition: common complex factors can
  cancel without affecting the real phase portrait.
- **Alternation**: the signs of u″ alternate along the complete ordered
  critical set (ordinary one-dimensional Morse alternation), and L-types
  follow u″ everywhere.  Since every simple B-root is a saddle, its finite
  critical neighbors, when present, are minima.  Earlier text incorrectly
  mixed the N-root subset with the complete critical set.
- **Eigenvectors**: exactly backbone-normal/-tangent only in the stiff limit
  u″/2A → 0; the first-order tilt is a*′u″/(2A(1+a*′²)).  Launch directions
  therefore always come from the exact eigenvector, never the limit formula.
- **True stiffness ratio**: κ = λ₁/|λ₂| ≈ 2A(1+a*′²)²/|u″|, unbounded in
  practice (10⁹ observed at d = 11).

### 4. Certified enumeration

Candidates and counts come from Sturm theory on the exact N — **the companion
matrix is banned**, even as a candidate generator (measured failure: at d=13
with N(0,1) moments it silently lost two close critical-point pairs that the
Sturm count found; an incorrect critical-point set makes the portrait
worthless).  The count-and-isolate layer runs in EXACT rational arithmetic
(stdlib `Fraction`, subresultant PRS — a one-time cost, trivial at deg ≤ 40);
a floating replica may run first for speed, but the exact pass is what the
certificate cites.  Pipeline:

1. EXACT coefficients of N (rational arithmetic; the floating replica uses
   compensated summation where sums cancel);
2. Sturm chain and root count: EXACT (the floating chain uses sign-preserving
   normalization and relative leading-zero stripping, and must agree);
3. bisection isolation using the same chain — sign-based, hence Inf-safe even
   where polyval overflows (sign(±Inf) is correct);
4. Halley polish (guarded: any non-finite intermediate falls back to the
   bisection value);
5. 2-D Newton on ∇L = 0 with a **gradient noise-floor** convergence test
   (|∇L| cannot fall below ~eps·(term scale); e.g. at the perfect-fit minimum
   (1,1) when f = g, g_a = 2(A−B) is an exact cancellation of large sums);
6. far roots (|b| at the coefficient-ratio scale, e.g. 10¹²) live in a
   **reciprocal-polynomial zone**: count and classify in t = 1/b where the
   arithmetic is O(1)-conditioned.  A single linear rescaling is *known bad*:
   it crushes the O(1) root cluster below isolation resolution.

Certificates: exact Sturm count; alternation of types (a violated alternation
proves the enumeration wrong); square-freeness of N (Bezoutian / displacement
LDL); ψ-positivity (Sturm).  Newton residuals reported against the noise floor.

### 5. The flow

ż = −∇L is a polynomial gradient flow: L is a global Lyapunov function, so
there are no periodic orbits and no recurrence; every forward and backward
limit is a single critical point, finite or at infinity (Łojasiewicz for the
finite case).  Trajectories cross level curves orthogonally.  The portrait's
content is the **separatrix skeleton**: for each saddle, two unstable branches
(descent) and two stable branches (ascent = the separatrices).

### 6. Invariant manifolds as graphs (the core method)

Time is eliminated *from the equations*, not just the narrative.  On a graph
w = w(b) the manifold satisfies the parametrization-free **invariance
equation**

    dw/db = 2Aw/P − a*′            (division of ẇ by ḃ; the speed cancels)

and on the transposed graph b = h(w):

    db/dw = P / (2Aw − a*′P).

**Unstable manifolds (slow graphs).**  Where the sounding κ(b) = 2A/|u″|
exceeds a threshold (default 10⁴), the Hadamard graph transform

    w ← P(b,w)·(a*′ + w′) / (2A)

is a contraction with rate ~1/κ: *the stiffness that kills time integration
is the convergence rate of the manifold computation*.  First iterate in
closed form: w₁ = a*′u′/(2A) (this is also the exact statement of how far the
manifold sits off the backbone, i.e. why backbone substitution is wrong).
The multiplicative form above is REQUIRED in the stiff zone: the divided form
2Aw/P is a difference of huge cancelling terms there and produces noise that
looks like the method failing.  Where κ is moderate, the scalar invariance
ODE is integrated directly — it is non-stiff there by construction of the
split.  In the overlap both representations must agree to tolerance (a seam
certificate, matched-asymptotics style).

**Stable manifolds (fast graphs).**  Near the saddle the dominant balance in
P is −2Awa*′ (not u′): both numerator and denominator of db/dw vanish
linearly and the ratio is regular — a removable 0/0 resolved by the exact
eigenvector jet.  No contraction exists there and none is needed: the κ that
defeated time integration enters the ascent flow only as *speed*, which the
graph parametrization deletes.  The fast graph is therefore the **primary**
chart for a separatrix (measured on the hard case: 76 non-stiff scalar steps
and E ≈ 10⁻¹² where arclength-normalized ascent stalled at 2×10⁵ steps) —
primary, not sole.  A separatrix trace is dispatcher-owned like every other
branch: it may fold out of the fast chart, hug the backbone at large |b|
(re-entering genuine shallow water, where the *same* slow-graph fixed point
applies — the invariance equation does not know the direction of time), and
it terminates in the far-field/rim chart.  There is no "one magic tracer";
there is one invariance equation and a dispatcher.

**Chart dispatcher contract.**  Every branch is owned at all times by exactly
one chart; handoffs happen only at events; every seam carries a residual.

| regime | owning chart | entry / exit events | seam certificate |
|---|---|---|---|
| critical-point neighborhood | jet chart (§7) | exit: jet radius (10% rule) | jet invariance residual over chart |
| shallow water (κ ≥ κ_hi, small w) | slow graph w(b), fixed point | exit: κ drops below κ_hi | overlap agreement with deep-water trace in [κ_lo, κ_hi] |
| deep water | graph ODE (w(b) or b(w) by dominant eigvec component), Gauss IVP | exit: fold event (slope explosion before P or 2Aw−a*′P crosses 0) → switch to the other chart or arclength; capture event; box/rim exit | reversal gap + extrapolant agreement per span; chart-switch point re-verified in the new chart |
| far field / rim | asymptote chart (§8) | entry: sounding says leading-form dominance | traced exit tangent vs asymptote (RESIDUAL) |

Chart selection at a saddle uses the exact eigenvector components (|v₁ −
a*′v₂| vs |v₂|): the two eigenvectors are orthogonal, so at least one chart is
well-posed for each manifold at every saddle, including tilted ones.

**Former Theorem 1 (adjacency and algebraic rigidity — refuted).**
Backbone order does not determine planar attachment: a stable separatrix may
cross the backbone away from a critical point, and an unstable branch may
therefore terminate at a nonadjacent minimum.  More strongly, ψ-nice Morse
families admit saddle--saddle handle slides; a simple slide is generically
codimension one.  Along the
constructed Λ-path, ψ-positivity, the exact Sturm counts, the discriminant
zero set (and scale-free root-collision margin), and every critical
b-coordinate remain fixed while the separatrix attaching map changes.  The
intervening wall
is the zero of a global level-section shooting map, not an algebraic
critical-point discriminant.

What remains rigid inside an algebraic Morse chamber is the critical-point
inventory: root count, order, and local Morse indices.  The attaching maps are
not rigid.  Runtime continuation therefore captures against every feasible
minimum, records the observed attachment, scans all invariant manifolds for
forbidden contacts, and returns `fp64_unresolved` when the separatrix skeleton
cannot be certified.  See `docs/theorems.md`, Theorems 4–6.

### 7. Jet charts at critical points

Every critical point owns a neighborhood in which the manifold is represented
by the Taylor jet of its graph, obtained by differentiating the invariance
equation at the point (first coefficient = exact eigenvector slope; each
further order is polynomial algebra in A, B derivatives).  Unlike the full
Poincaré normal form, the jet has no homological divisors and **no μ_u/μ_s
stiffness cap** — it works exactly where the normal-form launch degenerates.
Chart radius by self-selection: largest r with (next-order term) ≤ 10% of
(leading term).  Minima get entry jets too (tangency w′ = a*′u″/2A), so
traces terminate at chart boundaries and the last segment is analytic.
Manifolds are traced only *between* chart boundaries.  In the stiff zone the
saddle jet chart grows into the slow-manifold fixed point — the same object
at higher truncation.  Certificate: the jet's invariance residual over its
chart, a polynomial evaluation.

### 8. Infinity (Poincaré compactification) and globality

The polynomial field extends to the Poincaré disk.  **Genericity conditions**
(stated because a reference instrument must degrade explicitly, not
silently): the analysis below uses the *effective* degree d_eff = deg g as
realized (largest m with g_m ≠ 0 — "degree ≤ d" inputs with degree drops are
recomputed at d_eff, never at the nominal d), and requires the leading
coefficient α_{2d_eff} = g_{d_eff}²·μ_{2d_eff} > 0, i.e. μ_{2d_eff} > 0 —
automatic for a nondegenerate Hamburger moment sequence and certified with
ψ-niceness.  If coefficient cancellations reduce the leading form further
(e.g. deg B < d_eff shifts which terms dominate on the axes), the equatorial
classification is recomputed from the actual leading form; when that form is
degenerate the portrait declares the rim analysis at reduced certification
rather than asserting the generic picture.  Under the generic conditions,
the leading form of −∇L gives the equatorial tangential field
∝ a·b^{2d−1}·(d·a² − b²) (d = d_eff throughout) and radial component
−2α(1+d)·a²·b^{2d} ≤ 0.  Hence the equilibria at infinity are:

- four **diagonal points b = ±√d·a** — hyperbolic-type; these are the
  asymptotic directions of all separatrices (verified: measured slope of the
  d = 11 separatrix converges to −√11 like 1/r over five decades of radius);
- two **compactified backbone ends** (the ±b-axis points at infinity) —
  degenerate, local model
  ḃ ~ −C_inf/b², b(t) ~ t^{1/3}; the unbounded unstable branches end here;
  these require quasi-homogeneous blow-up and carry asymptotic-series-grade
  (weaker) certificates;
- two a-axis points, degenerate at leading order, resolved at next order.

**Completed separatrix configuration.**  The commonly quoted
Markus–Neumann theorem requires the corrected formulation: separatrices plus
the canonical-region data (equivalently, suitable representative orbits),
under its stated surface-flow hypotheses; see
[Espín Buendía–Jiménez López (2018)](https://doi.org/10.1016/j.jde.2018.07.021).
The present certified deliverable
is the compactified separatrix skeleton with its endpoint and contact audit.
Calling that skeleton a complete topological classification additionally
requires the corrected theorem's canonical-region/representative-orbit data,
or a proof that SPONG's gradient setting makes those data redundant.  That is
an explicit obligation, not an assumed reduction to a bare connection graph.
Global cross-check: **Poincaré–Hopf index balance** on the disk — if the
certified finite equilibria and equatorial equilibria do not balance,
something was missed and the instrument says so.

### 8b. The box contract

The compactification makes the mathematics global; the *deliverable* is a
certified finite portrait inside a requested view.  The two must not be
conflated, so the API contract is explicit:

- **view box** — user-requested; what gets rendered.
- **compute box** — where branches are computed; always contains the view
  box in its interior, and by default contains all finite nondegenerate
  critical points (policy flag for the rare user who wants a strict crop;
  the portrait then declares which critical points were excluded).
- **legal maximum compute box** — set by the rim analysis: outside it the
  far-field/asymptote chart owns the dynamics and box-style tracing is
  neither needed nor permitted.  User view boxes are clipped to it.
- **branch policy** — unstable branches are continued until a certified
  finite capture, a certified far-field exit, or the resolution budget is
  exhausted.  A branch may terminate at a nonadjacent minimum or, on a wall,
  at a lower saddle.  Stable branches are continued to a certified
  superlevel end.  Curves are clipped to the view box only after their
  computed terminal status is recorded; unresolved branches remain
  explicitly unresolved.

### 9. Level curves

Two representations, used both for rendering and as numerical objects:
(i) closed form a_±(b;c) from §2 — exact, grid-free;
(ii) orbits of the skew-gradient flow ż = J∇L, which is **Hamiltonian with
H = L**.  The Gauss integrators of §10 are symplectic, so L has no secular
drift along a traced contour (contours close and never cross); each step may
additionally be projected exactly onto {L = c} using (i).  Unit-ish speed
reparametrizations must use *even* Sundman time transformations to preserve
reversibility.  Closed-contour return gap is reported as a certificate.

### 10. Numerics doctrine

**Integrators: the Gauss collocation family, and nothing else.**
IMM (implicit midpoint = 1-stage Gauss, order 2), IRK4-GL (2-stage, order 4)
and IRK6-GL (3-stage, order 6, **the default**).  Selected by three
requirements that all point at the same family:

- *portable* — a few dozen lines each, no black boxes;
- *anadromic* (time-symmetric, Φ₋ₕ = Φₕ⁻¹) — REQUIRED, twice over:
  (a) **manifold duality**: one trajectory is both W^u(descent) and
  W^s(ascent) with limits at both time infinities; a non-symmetric scheme has
  no guaranteed/unique discrete backward orbit and draws two different curves
  for the same manifold; a symmetric scheme draws one, and its even-order
  modified equation adds no directional bias; chart handoffs become
  direction-independent;
  (b) **level-set conservation**: symplecticity of the Gauss family is what
  makes §9 true;
- *implicit / A-stable* — nearly free here, because every ODE spong
  integrates is **scalar** (graph ODEs), so the stage equations are s×s
  Newton solves with analytic Jacobians.

Which member is the default is a **tolerance** question, not a stiffness one.
Stiff problems observe the *stage* order (s), not the classical order (2s), so
GL4 runs at order 2 and GL6 at order 3–4 in stiff country; but where the
integrator actually earns its keep — the mild arc, resolved by the Richardson
ladder — GL6 climbs 3–4 fewer rungs than GL4 and runs 4.8–10.8× faster at
tol = 1e-12, while being ~1.5× *slower* on easy spans at 1e-8 where GL4's error
was never the binding constraint.  spong's gates are stated at 1e-12, so GL6 is
the default.  IRK8-GL was measured and declined: it captures only the last ~22%
(net 0.93–1.85× in the scalar path) at a worse conditioned 4×4 stage solve.

**The stage solve is closed-form and provably safe**, which is what keeps
"no black boxes" honest at 3 stages.  The stage matrix is M = I − h·diag(J_i)·A
with A the constant tableau, and

    det(I − zA) = Q_s(z), the (s,s) Padé denominator of exp

so **A-stability *is* the statement that M cannot be singular for dissipative
h·λ** — every root of Q_s lies in Re z > 0 — and |det M| *grows* like |z|^s.
Measured cond₂(M) saturates (10.44 frozen-D, 24.6 varying-D) *independently of
stiffness*.  Because the entries are essentially exact (exact tableau, analytic
Jacobians), small backward error IS small forward error, so any backward-stable
solve serves and only an ill-conditioning **guard** is needed — a Hadamard
ratio, free from the factorization.  spong uses row-equilibrated LU with partial
pivoting: unpivoted LU is already stable below the diagonal-dominance threshold
|h·J| < 1.64, pivoting covers above it, and bounding the multipliers by 1 removes
overflow structurally.  Iterative refinement is *not* used — measured against an
exact rational solve, the raw factorization is already at machine precision.

A failed stage solve is a **step-size signal, not a fatal error**: M is
diagonally dominant for small enough h, so the engine halves and retries.

**Step control: coarse, halve, halve, extrapolate.**  Solve each span at h,
h/2, h/4 (compared at shared nodes); apply `richardson3` (Aitken Δ², rate-free,
with the cancellation guard); refine until successive extrapolants agree
within

    tol_plot = span / (Z_max × pixels),   Z_max = 1000,

so the step policy and the rendering guarantee ("exact at every zoom the user
is promised") are the same inequality.  Because the base methods are
symmetric, error expansions are even in h and one Aitken sweep on IMM gains
two orders (the Gragg–Bulirsch–Stoer mechanism).  All trajectories are finite
(chart-to-chart on the compact disk), so refinement terminates on a bounded
budget.  The same `richardson3` primitive serves: fixed-point acceleration and
its convergence certificate (linear rate 1/κ), far-field asymptote
extrapolation (linear in 1/r), and contour-closure assessment.

**Arithmetic rules.**  Sign-based operations (Sturm, bisection) are Inf-safe
and may run to the Cauchy bound; value-based operations (Halley, Newton,
integrator stages) carry non-finite guards; convergence tests are noise-floor
aware; cancelling sums use compensated (Ogita–Rump–Oishi) accumulation; far
roots use the reciprocal chart; w is a state variable, never a difference.

### 11. The certificate ledger

Every portrait ships with, per object:

| object | certificates (semantics label per §Certificate semantics) |
|---|---|
| critical points | Sturm root count [EXACT]; min/saddle alternation via interval sign of u″ on the isolating intervals [EXACT]; N square-free (Bezoutian) [EXACT]; ψ > 0 (Sturm) [EXACT]; Newton residual vs noise floor [RESIDUAL] |
| complex divisors and exact fibres | reduced pole/critical polynomials [EXACT]; complete one-root complex disks by Lehmer--Schur/Schur--Cohn or linear Rouché witnesses [VALIDATED]; real branch-point count by Sturm [EXACT] |
| local Frobenius/Poincare launch | centered polynomial gradient over the certified algebraic saddle interval [EXACT enclosure]; invariant-cone transverse and lateral-face inequalities [VALIDATED]; exact rational loss-section crossing and `(b,y)` launch rectangle [VALIDATED].  The current `Fraction` oracle is opt-in pending a differentially checked GMP C kernel. |
| Smale holonomy candidate | inward lateral-face inequalities for a lifted `(ell,b,y)` trapping tube in increasing or decreasing loss [VALIDATED]; validated local-launch handoff [VALIDATED]; same-sheet Abel-gap zero exclusion [VALIDATED]; conic residue--log root-sum form [EXACT] and rational definite-integral enclosure [VALIDATED].  Cross-sheet positive-genus comparisons require unwrapped period transport. |
| each manifold branch | jet invariance residual over its chart [RESIDUAL]; angle-energy E = Σ ½‖d_⊥‖² over the RESOLVED vertices, with the resolved/unresolved counts (E = 0 ⟺ discrete integral curve) [RESIDUAL]; backbone residual max\|w\|/\|a*\| over the UNRESOLVED vertices [RESIDUAL]; anadromic reversal gap [RESIDUAL]; Richardson extrapolant agreement vs tol_plot [RESIDUAL]; seam agreement at every chart handoff [RESIDUAL]; level-tube inventory or shrinking backbone-funnel signs [EXACT at the measured dyadic point/ray]; local strong-convexity capture ball [RESIDUAL]; observed saddle-connection and asymptote agreement [RESIDUAL] |
| each level curve | closure gap [RESIDUAL]; L-drift (zero secular by construction; measured residual reported) [RESIDUAL] |
| the portrait | Poincaré–Hopf index balance on the disk [EXACT under §8 genericity, else VALIDATED with the declared reduction]; Morse certificate [EXACT]; moment-space algebraic-discriminant distance (distance to loss of the certified critical-point inventory, **not** to a global topology change) [EXACT]; separatrix contact/intersection audit and observed attaching map [RESIDUAL] |
| rendering | max vertex turn ≤ 0.2°; chord sag below pixel at 1000× zoom [RESIDUAL] |

**Two certificates per branch, because one cannot span it.**  `angle_energy` is
GEOMETRIC: it needs the *direction* of ∇L, whose significant digits fall as
‖∇L‖ ~ C_inf/b² approaches its own evaluation floor — so it decays OUTWARD, and
past a computable radius it measures its own noise rather than the curve.  The
backbone residual is ALGEBRAIC: far out the branch *is* a* = B/A, an exact
rational function, so it IMPROVES outward.  The two cross where both are strong,
and a branch is certified in two pieces.  Each is scoped to where it is relied
upon — measuring either one where the other governs makes it report failure on a
claim it was never asked to support — and the resolved/unresolved counts are
published so a certificate can never pass by measuring nothing.

The claim "as correct as it can be" is precisely: every drawn object carries
residuals a skeptic can recompute without trusting the code that drew it.

---

## Part III — Architecture

    spong.model      f, g, μ → exact coefficients; A, B, C, a*, u, N and
                     derivatives; the (b, w) chart; closed-form level curves;
                     Hessian identities; moment-space utilities (algebraic
                     Morse discriminant of N, ψ-positivity boundary; neither
                     claims global structural stability)
    spong.sturm      chains, counts, isolation, Halley, Bezoutian,
                     positivity, reciprocal far zone   [§4]
    spong.gauss      IMM, IRK4-GL, scalar Newton stages, richardson3,
                     collocation dense output, event location on the
                     collocation polynomial, reversal-gap reporting  [§10]
    spong.charts     slow/fast graph transforms, invariance ODEs, jet charts,
                     sounding dispatcher, seam verification  [§6–7]
    spong.atlas      Poincaré disk, equatorial charts, blow-ups at the
                     degenerate poles, index bookkeeping  [§8]
    spong.portrait   assembly: enumeration → jets → branches → skeleton;
                     certificate ledger  [§11]
    spong.render     zoom-proof polylines (turn-angle densification against
                     exact re-evaluators), contour layer, plane and disk
                     views, view-box cropping
    demos/           NOT part of the library: SGD / Adam / L-BFGS overlays on
                     certified portraits; batch-morphing across moment space
                     with exact algebraic Morse walls and numerically
                     bracketed global handle-slide walls; the d = 11 "tricky"
                     showcase; gallery/merch renders

Dependencies: NumPy only in the core.  Plotting via matplotlib in
`spong.render`/`demos` (rendering consumes spong's polylines; no numerical
work happens in the plotting layer).

## Part IV — Migration from mse-bundle

Oracles carried over (parity targets, from the MATLAB working tree):
`mse_trace_unstable_slowmanifold` + `audit_tricky_branch` (branch numbers:
d_target 0, ang_max ≤ 0.02°, E_angle ≤ 10⁻¹²), the MTC certification stack
(`certify_morse_loss`, `certify_positive_sturm`, `build_sturm_chain`,
two-zone `sturm_count`), the d = 13 / N(0,1) enumeration zoo (close pairs +
far root), the 30-portrait d = 2..6 regression sweep, and the separatrix
asymptote check (slope → −√d as 1/r).

Phases (each gated by its acceptance tests):

0. scaffold; CI.  (PyPI name registration is worthwhile but deliberately
   decoupled: repo founding must not block on, or be gated by, packaging.)
1. `spong.model` + `spong.sturm` — gates: N(0,1) d = 13 zoo exact counts;
   close-pair resolution; alternation on 1000 random models; far-root
   classification via reciprocal zone (no overflow-NaN path anywhere).
2. `spong.gauss` — gates: order/convergence tests; reversal gap ~ roundoff on
   reversible spans; zero secular L-drift on skew-flow circuits; richardson3
   stop matches tol_plot.
3. `spong.charts` — gates: tricky branch at E ≤ 10⁻¹²; both separatrices ≤
   10⁻¹¹; seam agreement; fold handoff on the d = 17 zoo without warnings.
4. `spong.atlas` — gates: √d asymptote convergence; index balance across the
   random zoo; backbone-end model matches C_inf analysis.
5. `spong.portrait` + `spong.render` — gates: full ledger emitted; MATLAB
   parity on the tricky portrait; 1000× zoom inspection windows clean.
6. `demos/` — SGD/Adam/L-BFGS vs the graph paper; moment-space morphing with
   certified walls.

Deliberately left behind: the five saddle-connection refinement variants, the
Poincaré launch machinery (superseded by jets; its α₀ self-selection and
third-derivative extraction survive inside `spong.charts`), backbone routing
(superseded by w₁-corrected graphs), companion-matrix candidates, all
MATLAB-suite dependencies (`ode15s`, `ode45`, `deval`, `fcontour`), and
time-parametrized manifold tracing in every form.
