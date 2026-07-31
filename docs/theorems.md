# spong theorems — statements and proof obligations

Named theorems the instrument relies on.  Each entry records: statement,
proof status, what the code assumes, and how the claim is enforced at
runtime regardless of status.  A theorem may be cited as EXACT in the
certificate ledger only when its status here is PROVED.

---

## Former Theorem 1 (backbone adjacency — refuted)

**Former statement.**  An unstable branch converges to the nearest minimum on
the side selected by the b-component of its unstable eigenvector.

**Status: REFUTED (2026-07-28).**  Seed 1802198452 is an exact ψ-nice Morse
counterexample.  The positive-b branch of the saddle at b=-0.4770682828
converges to the minimum at b=0.9668071440, not the nearer minimum at
b=-0.0157631516.  Independent DOP853 and Radau integrations at successively
tighter tolerances agree.  A stable separatrix crosses the backbone away from
a critical point, so backbone order does not determine the planar basins.

**Why the former proof sketch fails.**
(i) A > 0 ⇒ L strictly convex on vertical slices ⇒ no local maxima ⇒ a
backward-ascending stable branch has no finite critical point available as a
backward limit other than a saddle — which is exactly what must be excluded.
(ii) Each stable manifold (both branches plus the saddle) is a properly
embedded line from infinity to infinity and separates the plane; unstable
branches of other saddles cannot cross it, by uniqueness of solutions.
(iii) Transverse contraction ẇw ≈ −2Aw² slaves each unstable branch to
w* = a*′P/(2A), while ḃ = −P moves b monotonically across the interval
between the saddle and the adjacent minimum, where u′ is single-signed.
The counterexample invalidates the confinement step between (ii) and (iii):
the stable line may cross the backbone away from its saddle, so the local
departure can begin in a nonadjacent minimum's basin without crossing it.

**Runtime consequence.**  Unstable continuation tests capture against every
minimum and records the observed connection.  Stub departure direction never
preassigns a destination.

**Superseded by Theorem 4:** not only can the landing minimum be nonadjacent —
the weaker belief that every unstable branch terminates in *some* minimum is
also false family-wide.  Saddle–saddle connections exist in the ψ-nice Morse
family, on codimension-one walls (Theorem 4); Theorem 5 records what survives.

---

## Theorem 2 (Backbone reduction — proved)

**Statement (corrected 2026-07-02, simplified same day).**  All critical
points of L lie on the backbone a = B/A, at roots of B·N (u′ = B·N/A²;
N = α′β − 2β′α).  The identities

    H₁₂ = −2A·a*′,   H₂₂ = 2A·a*′² + u″,   det H = 2A·u″

hold at EVERY backbone point — no critical-point substitution required
(verified symbolically and numerically at arbitrary non-critical b).
Hence classification is uniformly one-dimensional: u″ > 0 ⇔ minimum,
u″ < 0 ⇔ saddle.  At a simple B-root, N(b₀) = −2B′A, so
u″ = B′N/A² = −2B′²/A < 0 automatically: **every simple B-root is a
saddle**, with det H = 2A·u″ = −4B′² — the SAME identity, not an
exception to it.  The SIGNS of u″ alternate along b (1D Morse on u);
consecutive L-saddles occur wherever a B-root falls.  In reduced form
\(B^2/A=P/D\), write \(u'=H/D^2\) with \(H=PD'-P'D\).  Then \(L\) is
Morse iff \(H\) and \(H'\) have no common **real** root.  Squarefree,
coprime \(B,N\) give a cheaper sufficient factorization, but are not
necessary: a common complex factor does not make a real critical point
degenerate.

**Status: PROVED.**  History: the original min/saddle-alternation
phrasing was FALSE (caught by the Poincaré–Hopf index certificate); the
first correction wrongly claimed det H = 2A·u″ fails at B-roots (caught
in external review — Codex — by direct algebra: the identity is
universal).  Strip-localized exact winding confirms index −1 at every
B-root.

---

## Theorem 3 (Equatorial structure — proved under genericity)

**Statement.**  Under the §8 genericity conditions (effective degree d_eff,
μ_{2·d_eff} > 0), the Poincaré compactification of −∇L has equatorial
equilibria exactly at the axes directions and the four diagonals
b = ±√d_eff·a; the equatorial radial component is ≤ 0; separatrices approach
the diagonal equilibria asymptotic to those lines.

**Status: PROVED at leading order** (direct computation of the leading form;
the degenerate backbone poles' local model ḃ ~ −C_inf/b² is established in
mse_morse_info's compactification analysis).  The quasi-homogeneous blow-up
at the degenerate poles carries asymptotic-series-grade certificates only —
recorded as the atlas's weakest link in SPONG_FOUNDING §8.

---

## Theorem 4 (Saddle–saddle connections exist — computer-assisted)

**Statement.**  The ψ-nice Morse family contains instances whose descent flow
has a heteroclinic saddle–saddle connection.  For a planar gradient flow with
hyperbolic finite critical points and the compactified ends separately
controlled, a saddle–saddle connection is the remaining finite-plane
Morse–Smale obstruction (the connection is automatically non-transverse).
The family realizes this gap: connections occur on codimension-one walls in
instance space (Kupka–Smale: generic instances have none, which is why random
sampling never exhibits one).

**The Λ-rheostat.**  (f, g, μ) → (f/√Λ, √Λ·g, μ) realizes (A, B) → (ΛA, B)
inside the full SPONG instance family (β_j picks up √Λ from g and 1/√Λ from
f; C → C/Λ is invisible to ∇L).  Since N → ΛN, every critical b-value is
Λ-independent; Morse-ness, ψ-niceness, Sturm counts, squarefreeness, and the
discriminant zero/nonzero status are frozen while κ ∝ Λ² sweeps from the wild
regime to the slaved regime.  A scale-free root-collision margin is likewise
unchanged.  (The raw discriminant value is not: for \(n=\deg N\),
\(\operatorname{disc}(\Lambda N)=\Lambda^{2n-2}\operatorname{disc}(N)\).)
Thus the path moves the global separatrix data without moving the
zero-dimensional Morse skeleton.

**Construction (on zoo `nonnearest-attachment`, uniform01 moments).**  Track
the +b unstable branch of the B-root saddle S at b = −0.4770682827686173
(level C).  Computed landings: far minimum (b = 0.9668071440) for Λ ∈ {1, 2}
(Λ = 1 is the recorded adjacency refutation); near minimum (b = −0.0157631516)
for Λ ∈ {4, 8, 16, 64, 256}.  The connection is forced at the flip:

1. *Confinement (exact; Theorem 5.2).*  On any B-root vertical,
   L(a, b₀) = C_Λ + ΛA(b₀)a² ≥ C_Λ, while the branch has L < C_Λ strictly for
   t > 0: it can never cross b = −1.5180886643 or b = +1.4162716731.  A > 0 on
   the compact b-interval bounds a as well; the forward orbit is bounded.
2. *Single-point ω-limit (exact).*  Polynomial gradient flow + bounded orbit
   ⇒ the ω-limit is one critical point (Łojasiewicz).  Candidates in the
   strip: three minima, S, and the N-saddle S′ at b = 0.6402740884.  Not S:
   L has strictly decreased below C_Λ = L(S).
3. *IVT on Λ (exact given the endpoint landings).*  Each set
   {Λ : branch lands at minimum m} is open (hyperbolic sinks, smooth
   dependence of the unstable branch on Λ).  [2, 4] is connected with
   endpoints in different landing sets, so some Λ* lands at no minimum —
   leaving only S′.  At Λ* the branch IS a heteroclinic S → S′
   (B-saddle, level C_Λ → N-saddle, level below C_Λ).

**Measured wall.**  Λ* = 2.177709563954844; Radau (rtol 1e-12) and DOP853
(rtol 1e-13) agree on the far/near classification down to |ΔΛ/Λ*| = 1e-12,
and disagree only in the last two ulps, consistent with integrator truncation
error dominating there.  The two fp64 neighbours of Λ* pass within ~1e-12 of
S′ before forking to opposite minima.  A wall instance requires a distinct
zoo representation: the existing `expected_connections` field describes
saddle-to-minimum capture and does not encode an expected saddle connection.

**Status: COMPUTER-ASSISTED.**  Steps 1–3 are exact; the two hypotheses
consumed are the finite endpoint traces (Λ = 2 → far, Λ = 4 → near), each a
robust capture deep inside a sink basin, cross-checked by two integrators —
the same evidence grade as the Former-Theorem-1 refutation.  Upgrade path to
PROVED: validated (interval) enclosures of those two finite orbits; both are
non-stiff at their Λ and land far from basin boundaries.

**Independent confirmation (degree 3, different mechanism).**  The SPSA +
signed level-section shooting search (demos/search_saddle_connection.py and
companions) produced an exact rational affine coefficient segment with
shooting residuals +8.687e-4 at parameter 473/512 and −8.088e-4 at 947/1024,
secant-refined to mismatch 1.4e-7 at 10299890/11143041
(out/saddle_connection_degree3_*.json): an N→N pair, source level 0.13824 →
target level 0.12822, in a 2-saddle/2-minima portrait.  Outstanding
obligation for that bracket (which the Λ-path gets for free from its frozen
critical skeleton): certify that the tracked pair's identity — both branches,
the section, the Morse data — persists across the whole segment, so the
signed residual is a continuous function of the parameter.

**Consequences.**
- The rigidity claim that the ψ-nice Morse family has only algebraic
  bifurcations is FALSE: along the Λ-path every algebraic certificate is
  constant, yet the Markus–Neumann–Peixoto invariant changes at Λ*.
  Connection walls are global shooting walls; algebraic Morse-discriminant
  distance does NOT certify structural stability of the current vector
  field, and the demos/README "walls are algebraic" remark applies only to
  critical-point bifurcations.
- The Λ-path varies \(f\) and \(g\), not μ alone.  Therefore this construction
  does not by itself prove that a fixed-\((f,g)\), moment-only path crosses a
  handle-slide wall.  That narrower batch-moment claim requires its own
  transverse shooting bracket.
- The runtime posture above (capture against every minimum, record observed
  connections, escalate to fp64_unresolved) is the correct behavior; capture
  times diverge logarithmically at a wall, so `fp64_unresolved` is the honest
  verdict exactly there.
- The Λ-rheostat is a zoo instrument in its own right: an isospectral-in-b
  stiffness dial for regression cases.

---

## Theorem 5 (Level-C structure and downward confinement — proved)

Elementary consequences of L = C − 2aB + a²A with A > 0; proofs inline.

**5.1  {L = C} = {a = 0} ∪ {a = 2B/A}.**  L − C = a(aA − 2B).  In particular
the a-axis is a level curve, every B-root saddle sits on it at level exactly
C, and — since a connection strictly decreases L — **no B-saddle can connect
to a B-saddle.**

**5.2  Vertical block.**  L(a, b₀) = u(b₀) + A(b₀)w² ≥ u(b₀) for every a: an
orbit whose current level is below u(b₀) can never cross the vertical line
b = b₀.  Applied at B-roots (u = C): the open regions between consecutive
B-roots are bounded, forward-invariant "bubbles"; nothing below level C
crosses a pinch.  Applied at any saddle: a branch cannot pass over a saddle
at or above its current level.

**5.3  Downward confinement.**  An unstable branch of a saddle at level c
stays in its connected component of {L < c} (level strictly decreases; 5.2
bounds the component's b-extent when u exceeds c outside a compact interval,
and A > 0 then bounds a), so its ω-limit is a critical point of THAT
component: **a minimum, or a strictly lower saddle in the same component.**

**5.4  Safe components.**  If the component contains no lower saddle — in
particular the single-interior-saddle bubble, and any N-saddle whose
component of {L < c} contains only minima — the branch terminates at a
minimum of that component.  This is the correct salvage of Former Theorem 1.

**Status: PROVED** (the statements are three-line algebra/topology over
Theorem 2's identities; 5.3's compactness hypothesis — u > c outside a
compact b-interval — is u > c ⟺ B² − (C−c)·A < 0).  For a B-root source,
\(c=C\) is rational over dyadic inputs and the condition is an ordinary exact
Sturm check.  For an N-root source, \(c=u(b_s)\) is generally an algebraic
number, not a rational one; an exact implementation must evaluate polynomial
signs in the isolated real-algebraic extension generated by \(b_s\).

**Runtime use.**  5.2–5.4 justify capture-target pruning: the capture set for
a branch from level c is the critical points of its {L < c} component.  This
is immediately exact with the current rational Sturm machinery for B-root
sources.  Claiming the same exact pre-trace pruning for a general N-root
source is conditional on adding the algebraic-number sign evaluation just
described.  A certified capture outside the resulting set is a numerical
defect; a component with no admissible minimum-only conclusion flags a
potential connection instance for escalation.

---

## Theorem 6 (Branch-fate trichotomy; walls, unfoldings, shadowing)

**Setup.**  {L < c} fibers over {u < c} ⊂ ℝ with vertical intervals of
half-width √((c−u)/A), so its connected components ("tubes") correspond to
the interval components of {u < c}.  An unstable branch of a saddle at level
c enters, and remains forever in, the tube on its departure side.  All entry
data of a tube is coefficient-level: its b-extent, its critical inventory
(roots of B·N inside), whether it is unbounded (compare c with
u_∞ = C − β_d²/α_{2d} when deg B = d_eff; u_∞ = C on a degree drop), and the
tail sign of u′, which for b → +∞ is the sign of the leading coefficient of
B·N.  For N-root sources these comparisons live in the real-algebraic layer
of Theorem 5's status note; for B-root sources (c = C) they are rational.

**Statement.**  Each unstable branch has exactly one of three fates, decided
by its tube:

(i) *Bounded tube.*  The branch terminates at a critical point of the tube —
a minimum, or a strictly lower saddle (Theorem 5.3).

(ii) *Unbounded tube, inward tail* (u′ > 0 eventually on the unbounded end;
u climbs to u_∞ from below).  The branch still terminates at a finite
critical point of the tube (which necessarily contains a minimum: u leaves
level c, dips, and returns toward u_∞ ≤ c from below).  Far out the tube
pinches onto the backbone like b^(−d_eff) while the transverse rate 2A grows
like b^(2·d_eff), so beyond a computable b† every orbit in the tube is
slaved to the scalar dynamics ḃ = −u′ < 0 and is pushed back inward;
boundedness follows, then Łojasiewicz.

(iii) *Unbounded tube, outward tail* (u′ < 0 eventually; u decreases to u_∞
from above).  Two sub-cases:
  - *Empty side*: if the tube contains no critical points, escape to the
    backbone pole is FORCED — the ω-limit would need a critical point in the
    tube's closure at level < c; there is none, and an equal-level boundary
    saddle is unreachable because L strictly decreases from c.
  - *Occupied side*: if the tube contains critical points, its outermost
    critical point on the unbounded end is necessarily a saddle S″ (the last
    interior extremum of u before a decreasing tail is a maximum).  BOTH
    capture and escape occur in the family — large-Λ slaving captures at the
    adjacent minimum; nothing blocks an overflight that passes the b_{S″}
    vertical at level above u(S″), misses W^s(S″), and rides the empty
    sub-tube out.  The capture/escape wall is a saddle connection to S″.
    The fate is decided by no algebraic function of the entry data
    (Theorem 4's frozen-skeleton argument applies verbatim) — only
    a-posteriori, by the termination tests below.

**Corollary (degree drop).**  If deg B < d_eff then u_∞ = C, every N-saddle
tube is bounded, every B-saddle outer tube has an inward tail, and every
unstable branch of every saddle terminates at a finite critical point: no
unbounded unstable branches exist at all.  Conversely "extreme saddle" does
NOT imply escape: an extreme saddle with an inward tail (case ii) is
captured.  Backbone position was never the controlling invariant; the tube
and its tail sign are.

**Status.**  (i) PROVED (Theorem 5.3).  (ii) and (iii-empty) PROVED modulo
the **far-field funnel lemma** — existence of a computable b† beyond which
the tube is a certified contraction funnel onto the slow graph with
sign(ḃ) = −sign(u′): the same two-zone machinery as the Hadamard slow graph
plus the §8 leading form; explicit constants are a proof obligation.
(iii-occupied) the dichotomy is proved modulo the same lemma; an interior
saddle with an escaping branch has not yet been exhibited — expected via the
Λ-rheostat overflight protocol (run Λ down from the slaved regime on a
case-(iii) geometry and watch the landing flip from adjacent minimum to
funnel exit).

**Exact termination tests (the engine contract).**  Proximity to a minimum
is a scheduling heuristic, never a certificate.  The certificates are:
- *Capture test*: with ℓ a validated enclosure of L at the current point, if
  the current point's component of {u < ℓ} contains no saddle, it contains
  exactly one minimum and capture there is forced (Łojasiewicz leaves no
  alternative).  One rigorous evaluation plus root data — no basin geometry,
  and no trust in the trace history.
- *Escape test*: certified entry into the far-field funnel (b beyond b†,
  outward tail sign verified) — the unstable-side analogue of the stable
  exits' superlevel-end certificate.
- *Neither fires*: the diagnostic is the level gap ℓ − u(S″) to the
  outermost (or nearest blocking) saddle of the tube; gap → 0 is the
  connection signature, and `fp64_unresolved` is the honest verdict.
Portrait computation is therefore harder than assumed — branch fates are not
algebraically pre-assignable — but remains tractable: off the walls both
tests fire in finite time with cost growing like log(1/gap), and the wall
set itself is measure zero.

**Walls, structural stability, and shadowing.**  A wall instance is Morse
but not finite-plane Morse–Smale, hence not structurally stable.  By the
Lipschitz-shadowing/structural-stability equivalences
(Pilyugin–Tikhomirov type), quantitative shadowing fails there: near a wall,
a pseudo-orbit with small per-step residual may be shadowed by NO true orbit
— in particular a traced polyline can drift across W^s(S″), which no true
orbit does.  Consequence for the ledger: single-trace residual certificates
(angle-energy, reversal gap) certify pseudo-orbit quality but lose their
orbit-existence force exactly near walls.  Admissible evidence there is
two-sided: shooting brackets with opposite signed residuals, and the
threshold tests above, which certify the true orbit's fate from an enclosure
rather than from the trace.  This is why the capture test is stated as a
sublevel-component test and not as "the trace got close".

**Local unfolding at a simple wall (enumeration of adjacent skeletons).**
At an instance with exactly one saddle connection S → S″, every other branch
robustly resolved: the connection breaks to one side of W^s(S″) or the
other, after which the S-branch shadows one of S″'s two unstable branches to
that branch's (locally constant) destination.  The adjacent Morse–Smale
classes are therefore exactly TWO, computable from the wall data alone:
terminal(S-branch) ∈ {terminal(S″ branch⁺), terminal(S″ branch⁻)}.  The wall
configuration is itself a legitimate Markus–Neumann–Peixoto class
(separatrix configurations encompass connections), so the certified
deliverable near a wall is the ordered triple (left class, wall class, right
class) plus the bracket.  No combinatorial explosion occurs at simple walls.
The explosion lives elsewhere: codimension-k strata (k simultaneous
connections, or cascades S → S′ → S″) have up to 2^k adjacent classes, and
the chamber count of the shooting stratification over a single Morse cell
can grow with the number of saddles.  But a generic one-parameter path meets
only simple walls, every chamber transition is a certified-detectable wall
crossing, and enumeration is on-demand per wall — the burden is global
bookkeeping, not local ambiguity.

---

## A-posteriori topology certificate

The portraitist audits every polyline self-contact and every pairwise contact
with robust FP64 orientation predicates and a bounding-volume hierarchy.
Contacts involving a stable invariant manifold remain forbidden except at
certified critical endpoints: stable manifolds are basin boundaries, so such a
contact can change the Morse decomposition.

Two unstable branches captured at the same minimum are different.  They are
basin-interior representatives of the same terminal component, not basin
boundaries; their sampled suffixes may be replaced by disjoint arcs without
changing the Morse complex.  Their mutual contacts are therefore trimmed only
after both independent continuations certify the same critical endpoint.
Sublevel suffixes below the applicable saddle level and superlevel stable
suffixes in the same compactified end provide stronger local trimming
certificates where available.

The trace box is escalated when a stable exit has not yet entered either a
certified superlevel end or a valid algebraic asymptotic regime.  Failure after
the escalation budget remains `fp64_unresolved`; it is never promoted merely
because every finite branch happened to terminate.
