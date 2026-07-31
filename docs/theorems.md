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
family; simple connections are generically codimension-one walls (Theorem 4).
Theorem 5 records what survives.

---

## Theorem 2 (Backbone reduction — proved)

**Statement (corrected 2026-07-31).**  All critical points of L lie on the
backbone a = B/A, at roots of B·N (u′ = B·N/A²;
N = α′β − 2β′α).  The identities

    H₁₂ = −2A·a*′,   H₂₂ = 2A·a*′² + u″,   det H = 2A·u″

hold at EVERY backbone point — no critical-point substitution required
(verified symbolically and numerically at arbitrary non-critical b).
Hence classification is uniformly one-dimensional: u″ > 0 ⇔ minimum,
u″ < 0 ⇔ saddle.  At a simple B-root, N(b₀) = −2B′A, so
u″ = B′N/A² = −2B′²/A < 0 automatically: **every simple B-root is a
saddle**, with det H = 2A·u″ = −4B′² — the SAME identity, not an
exception to it.  Because \(u\) is a one-dimensional Morse function, the
signs of \(u''\), and hence the planar types, alternate along the complete
ordered critical set.  In particular a simple B-root is a saddle whose
neighboring finite critical points, when present, are minima.  In reduced form
\(B^2/A=P/D\), write \(u'=H/D^2\) with \(H=PD'-P'D\).  Then \(L\) is
Morse iff \(H\) and \(H'\) have no common **real** root.  Squarefree,
coprime \(B,N\) give a cheaper sufficient factorization, but are not
necessary: a common complex factor does not make a real critical point
degenerate.

**Status: PROVED.**  The universal Hessian identity gives index −1 at
every simple B-root, while ordinary one-dimensional alternation applies to
the full reduced numerator of \(u'\).  Earlier text confused alternation of
the complete critical set with classification of the N-root subset alone.

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
The family realizes this gap.  A simple connection is generically a
codimension-one shooting wall; the construction below proves existence but
does not, by itself, prove simplicity or uniqueness of the wall it crosses.

**The Λ-rheostat.**  (f, g, μ) → (f/√Λ, √Λ·g, μ) realizes (A, B) → (ΛA, B)
inside the full SPONG instance family (β_j picks up √Λ from g and 1/√Λ from
f; C → C/Λ is invisible to ∇L).  Since N → ΛN, every critical b-value is
Λ-independent; Morse-ness, ψ-niceness, Sturm counts, squarefreeness, and the
discriminant zero/nonzero status are frozen while \(\kappa\) grows
asymptotically like Λ² from the wild regime to the slaved regime.  A
scale-free root-collision margin is likewise unchanged.  (The raw
discriminant value is not: for \(n=\deg N\),
\(\operatorname{disc}(\Lambda N)=\Lambda^{2n-2}\operatorname{disc}(N)\).)
Thus the path moves the global separatrix data while preserving the ordered
critical b-inventory and its local indices.  The critical a-coordinates and
critical values do scale with Λ.

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

**Measured wall.**  Λ* ≈ 2.177709563954844; Radau (rtol 1e-12) and DOP853
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
  bifurcations is FALSE: along the Λ-path the listed algebraic Morse data and
  the scale-free collision margin are constant, yet the separatrix attaching
  map changes at Λ*.
  Connection walls are global shooting walls; algebraic Morse-discriminant
  distance does NOT certify structural stability of the current vector
  field, and the demos/README "walls are algebraic" remark applies only to
  critical-point bifurcations.
- The Λ-path varies \(f\) and \(g\), not μ alone.  Therefore this construction
  does not by itself prove that a fixed-\((f,g)\), moment-only path crosses a
  handle-slide wall.  That narrower batch-moment claim requires its own
  transverse shooting bracket.
- The runtime posture above (capture against every feasible minimum, record
  observed connections, escalate to `fp64_unresolved`) is the correct
  behavior.  Near a simple wall the residence time near the blocking saddle
  grows logarithmically in inverse shooting distance; at the wall no finite
  capture test should guess a destination.
- The Λ-rheostat is a zoo instrument in its own right: an isospectral-in-b
  stiffness dial for regression cases.

---

## Theorem 5 (Sublevel tubes and bounded branch fates — proved)

Elementary consequences of

    L(a,b) = u(b) + A(b)(a-a*(b))²,       A(b) > 0.

**5.1  Level C.**  Since \(L-C=a(aA-2B)\),

    {L=C} = {a=0} ∪ {a=2B/A}.

Every simple B-root saddle has level C.  Strict loss decrease therefore
rules out a connection from one B-saddle to another B-saddle.

**5.2  Vertical barrier.**  For every \(b_0\),

    L(a,b_0) ≥ u(b_0).

An orbit at level below \(u(b_0)\) cannot cross the vertical line \(b=b_0\).
In particular, level-C branches cannot cross another B-root vertical after
leaving their source.

**5.3  Tube confinement.**  The components of \(\{L<c\}\) correspond exactly
to the interval components of \(\{u<c\}\); over such an interval the fiber is

    |a-a*(b)| < sqrt((c-u(b))/A(b)).

An unstable branch from a level-c saddle enters one of its two departure
tubes and remains there for all forward time.

**5.4  Bounded fate.**  If that branch is bounded, the Łojasiewicz theorem
for polynomial gradient flows gives a single critical-point omega limit.
It is either a minimum or a saddle of strictly lower loss in the same tube.
If the tube contains exactly one minimum and no lower saddle, the bounded
branch terminates at that minimum.

**Status: PROVED.**  The theorem deliberately makes no finite-termination
claim for an unbounded tube.  For a B-root source \(c=C\), tube endpoints and
critical inventory are accessible to the current rational Sturm machinery.
For a general N-root source, the level \(c=u(b_s)\) is algebraic; exact
pre-trace tube classification requires sign evaluation in the
real-algebraic extension
generated by the isolated root \(b_s\).

**5.5  Degree-drop corollary.**  If \(\deg B<d_{\rm eff}\), then
\(u(b)\to C\) at both ends.  A B-root branch has loss below \(C\) immediately
after departure; an N-root saddle has source level below \(C\).  At any later
point choose a strict upper level \(\ell<C\).  The relevant component of
\(\{u<\ell\}\) is bounded, so every unstable branch is bounded and has a
finite critical-point limit.  Thus degree-drop models have no unbounded
unstable branches; this conclusion needs no far-field funnel lemma.

---

## Theorem 6 (Certified branch decisions)

The tube is exact prior information, but it does not in general decide the
branch fate.  The intended total engine contract has two positive
certificates and otherwise returns unresolved.

**6.1  Capture certificate.**  Let \(\ell_{\rm hi}\) be a validated strict
upper bound for the loss at a point on the branch, and let \(I\) be the
component of \(\{u<\ell_{\rm hi}\}\) containing its b-coordinate.  Capture
at a minimum \(m\) is certified when:

1. \(I\) contains \(m\), no other minimum, and no saddle;
2. the corresponding sublevel tube is bounded, or validated inward
   far-field funnels exclude every unbounded end.

Future loss is below \(\ell_{\rm hi}\), so the true orbit cannot leave that
tube; Theorem 5.4 then forces convergence to \(m\).  Nearness to \(m\) is
useful for scheduling but is not itself a certificate.

**6.2  Escape certificate.**  Escape is certified only after the branch
enters a validated forward-invariant outward far-field funnel with a named
compactified terminal.  The required funnel lemma must provide an explicit
threshold \(b^\dagger\), transverse contraction, the sign of the longitudinal
drift, and preservation of the selected end.  Asymptotic sign alone is not a
finite-arithmetic certificate.

For reference, when \(\deg B=d_{\rm eff}\),

    u_∞ = C - β_d²/α_{2d},

and on a degree drop \(u_\infty=C\).  These values and the eventual sign of
\(u'\) nominate inward and outward ends; they do not replace the funnel
proof.

**6.3  Unresolved case.**  If neither certificate fires, the engine continues
within its budget and then returns `fp64_unresolved`.  Possible causes include
insufficient arithmetic, an uncertified far field, or approach to the stable
manifold of a lower saddle.  Theorem 4 shows that the last possibility is
real: tube inventory and the algebraic Morse skeleton do not determine the
attaching map.

**6.4  Simple handle slide (conditional local model).**  Suppose a
one-parameter family crosses transversely a wall with exactly one connection
\(S\to S'\), and every other branch and compactified end is robustly
resolved.  Standard invariant-manifold dependence then gives two local
chambers.  On the two sides, the affected branch from S follows the two sides
of \(W^s(S')\) and subsequently has the respective fates of the two unstable
branches of \(S'\).  This is the local handle-slide model.  The hypotheses
— especially transversality and uniqueness of the connection — must be
validated for any claimed wall certificate.

**Numerical consequence.**  A small residual proves that a polyline is a
good pseudo-orbit; it does not prove on which side of a stable separatrix the
true branch lies.  Endpoint fate therefore comes only from 6.1 or 6.2;
otherwise it remains unresolved.  A two-sided signed shooting bracket is
separate evidence for a wall between parameter instances.  No global
shadowing theorem is invoked.  The usual Lipschitz-shadowing equivalences
concern the full flow of a \(C^1\) vector field on a closed manifold; they do
not directly certify the designated finite-plane separatrix, and SPONG's
compactification also has degenerate equilibria at infinity.

**Status.**  6.1 follows from Theorem 5 once its level enclosure, component
inventory, and boundedness hypotheses are validated.  The current code uses
an exact-upper-level sublevel component as a safe candidate filter and has
local arrival checks; packaging the complete 6.1 implication as one ledger
certificate remains work.  Section 6.2 is conditional on an explicit
far-field funnel certificate, which remains a proof and implementation
obligation.  Section 6.3 is the intended total contract.  Section 6.4 is a
conditional classical local model, not a claim that every observed landing
flip is already a certified simple wall.

---

## Continuation program (not a theorem)

The Λ-rheostat is useful for experiments because it freezes all critical
b-values while changing the global separatrices.  A future continuation
mode may start in a validated strongly slaved regime and record signed
handle-slide crossings while returning to Λ=1.  Three obligations precede
such a mode:

1. certify a finite slaving threshold Λ₀ and its base attaching map;
2. cover each tracked branch/section by charts on which its shooting
   residual is continuous (analytic where the section crossing is
   transverse);
3. use interval subdivision or another exclusion argument to rule out
   flip-and-return wall pairs between mesh points.  Endpoint agreement alone
   certifies the endpoint graph, not the complete wall word.

The slow-graph balance must be written in transformed quantities.  If

    A_Λ=ΛA₀,       a*_Λ=a*₀/Λ,       u_Λ=u₀/Λ,

then to leading order

    w*_Λ = a*_Λ′ u_Λ′ /
            (2 A_Λ (1+a*_Λ′²))
          = a*₀′ u₀′ /
            (2 Λ³ A₀ (1+a*₀′²/Λ²)),

and

    P_Λ|w* = u_Λ′/(1+a*_Λ′²) + higher-order terms.

These formulas motivate a base chamber but do not certify one.

**Computed evidence.**  In `nonnearest-attachment`, the graph at Λ=1 and
the graph at Λ=256 differ in the right branch of the saddle near
b=−0.477.  Conditional on the endpoint landing evidence, Theorem 4's
computer-assisted argument forces at least one saddle connection between
the two landing regimes.  The measured crossing near
Λ≈2.177709563954844 is strong evidence for a single simple slide.  A claim
that it is the only slide on the entire dial segment requires the exhaustive
continuation certificate above.

---

## Theorem 8 (Minimal wall portraits)

How small can a Morse instance sitting on a saddle-connection wall be?
"Small" = number of finite critical points.

**8.1  Floor (PROVED; classification corrected after review).**  A
connection needs two saddles at strictly different levels; alternation
(Theorem 2) puts a minimum between them, so three critical points is the
floor.  A Morse 3-critical portrait has THREE types, not two: (i) both
saddles simple B-roots — both at level C, no connection possible; (ii) both
saddles N-roots, B real-root free; (iii) MIXED — one simple B-root saddle
(level C) and one N-saddle, exemplified by the exact uniform01 instance
f = (−3/16, 3/16, 27/32, 1/2), g = (7/16, 9/32, 13/16, −15/16) (ψ-positive,
Morse, S_N m S_B at b = −0.1340, 0.7035, 1.5679).  Types (ii) AND (iii) are
connection candidates.  Parity: at full degree, deg(B·N) = 4d_eff − 2 is
even, so an odd critical count forces a degree-drop stratum — EVERY
3-critical portrait lies on one (the earlier claim of open 3-critical sets
at d_eff ≥ 4 was false; caught in review — Codex).

**8.2  Overflight action bound (PROVED; scaling corrected after review).**
Along any descent orbit, |dL/db| = |∇L|²/|ḃ| ≥ |∇L| ≥ 2√(A(L−u)) wherever
the motion has a b-component (all quantities of the instance itself).
Hence a branch of level c₁ that passes the longitude of a saddle at level
c₂ < c₁ (keeping L > c₂, as the vertical block requires) must dissipate at
least the action integral, giving the necessary condition

    c₁ − c₂  ≥  2 ∫ √( A(b) · (c₂ − u(b)) ) db

over the well span {u < c₂} between the saddles.  Under the Λ-rheostat, in
baseline quantities A₀, u₀, c₀ this reads

    (c₁,₀ − c₂,₀)/Λ  ≥  2 ∫ √( A₀(b) · (c₂,₀ − u₀(b)) ) db,

so the overflight-forbidding threshold is LINEAR in Λ (an earlier √Λ form
mixed baseline and rheostat quantities; caught in review — Codex).
Corollaries: a per-instance quantitative adjacency threshold — the capture
half of the slaved-regime claim with no funnel machinery — and a pre-filter
for wall hunts.  Certification grade: the integrand is algebraic but the
integral is a VALIDATED interval-quadrature object, not an exact Sturm
computation.  The bound is necessary, not sufficient: instances satisfying
it comfortably still capture (slaving is a separate, dynamical
obstruction).

**8.3  Three-critical rigidity (partially proved; conjecture).**  Two cases
are PROVED, both degree-free:
- *Mirror-symmetric instances always capture.*  On the d_eff = 2 drop
  stratum, if A is even about b_c = −α₃/(4α₄) then B is automatically even
  about the same center; the reflection is flow-equivariant, an escaping
  inner branch must cross the center longitude, and its mirror image — the
  other saddle's inner branch — would pass through the same point: two
  distinct orbits through one point.  So both inner branches capture.
- *Equal-level saddles.*  A branch of level c can neither terminate on nor
  pass an equal-level saddle (vertical block + strict decrease).
EMPIRICAL: 88/88 scale-sane instances of the d_eff = 2 type-(ii) stratum,
both branches, Λ swept over [1e-3, 1e2]: the inner branch captures every
time; the passing fate never occurs.  The type-(iii) mixed stratum is
essentially untested: the single 8.1 exemplar, dialed over
Λ ∈ [1e-2, 1e2], also captures throughout, but one instance is not a
sweep.  **Conjecture: every 3-critical ψ-nice Morse instance is
finite-plane Morse–Smale.**  The proved cases do not cover type (iii)
asymmetric instances; a systematic mixed-stratum hunt is the falsification
target, and the conjectured minimality of four in 8.4 is conditional on
exactly this.

**8.4  Four-critical landing-flip witness (COMPUTER-ASSISTED).**  Subject
to 8.3, the minimum is four, and a landing-flip wall is realized there —
"wall" meaning a parameter at which the branch terminates at no minimum;
whether its type is the finite connection S → S′ or a rim termination is
settled only at the 8.5 evidence grade: zoo `minimal-quartet`
(d_eff = 2, uniform01, B positive definite — no B-saddles; quartic N with
four simple real roots, S m S m).  Criticals b = −0.6247727737 (S, high,
Q = 1.914), 0.3185624009 (m₁), 0.6395949203 (S′, low, Q = 14.285),
1.8472900359 (m₂); Q_∞ = 8.053.  Under the Λ-rheostat the +b branch of S
lands at m₂ for Λ below and m₁ for Λ above the wall

    Λ* ≈ 7.651823524762  (fp64-width bracket).

Wall coordinates at this precision are launch-protocol-sensitive
(re-polishing the saddle coordinates moved an earlier instance's bracket by
~2e-13 relative): a wall should be cited together with its protocol, or
certified by a signed shooting residual instead.  At Λ = 1 the stored
instance lands at m₂ outright — skipping m₁ and S′ — so the quartet is also
the minimal nonadjacent-attachment example (4 criticals against
`nonnearest-attachment`'s 9).

**8.5  Tube parity; the wall type at four criticals (PROVED / EMPIRICAL).**
At d_eff = 2 with B root-free, the outermost critical point on each side is
approached monotonically from a tail at height Q_∞, so Q_∞ exceeds the
outer saddles' Q values; since the high saddle's Q is below everything in
its target tube, **the high saddle's target tube is unbounded in every
4-critical arrangement**.  Consequently the exact confinement argument
cannot exclude a rim termination at the wall: at Λ* the ω-limit is S′ or
the rim, and only the observed hug-scaling — closest approach to S′
falling monotonically (1.3e-1, 5.1e-2, 2.2e-2, 9.5e-3 at bracket offsets
1e-4 … 1e-13) — identifies the connection S → S′ (EMPIRICAL grade).  The
smallest configuration whose wall type is FORCED exactly is five criticals,
S_B m S_N m S_B: the B-root verticals bound the bubble and Theorem 4's
confinement argument applies verbatim.  Theorem 4's `nonnearest-attachment`
wall is the wild-caught representative of that type (9 criticals); the
designed 5-critical version lives on the same deg-N-drop stratum and is an
open construction target.

**Summary.**  3 = the combinatorial floor, conjecturally never on a wall
(open in particular for the mixed type); 4 = the minimal landing-flip wall
(witnessed; saddle-connection type EMPIRICAL); 5 = the minimal wall whose
type is exactly forced.  Runtime consequence: the conjecture authorizes
nothing — 3-critical portraits certify Morse–Smale exactly as any portrait
does, by every branch passing a capture or escape test; the conjecture is a
research flag, and a certified 3-critical wall would refute it (a
discovery, surfaced by the same audit).

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
