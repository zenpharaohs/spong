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
