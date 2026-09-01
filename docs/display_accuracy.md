# Display accuracy and the level-step policy

STATUS: design note for markup, 2026-09-01.  Nothing here is implemented;
the policy below changes vertex densities and therefore the RESIDUAL
golden tier, so it wants agreement before code.

## The principle (Andrew, 2026-09-01)

Everything depicted in a view is accurate to the pixel.  Magnification is
limited: when the numerical fuzz of what would be drawn exceeds a couple
of pixels, that magnification is refused rather than served.  Display
accuracy is therefore a few ulps of the trace's working precision --
binary64 today; relative to binary32 if an FP32 tracer exists.

## Two consumers, two requirements

The stored polyline serves two masters that the current fixed resolution
(n_levels = 12000 potential-rate steps per prefix) conflates.

CERTIFICATES consume vertices through: (a) the turn test -- per-step
turning is bounded, kappa*h <= theta_max, where kappa is the orbit's
curvature and h the arclength step; (b) the contact scan -- segment
tests carry a sagitta bound kappa*h^2/8 for the gap between chord and
curve; (c) the LEVEL BAR -- a prefix step must never cross more than one
bar constant, or a crossing event escapes the record.  (c) is a LOSS
step bound: delta_L <= the gap between adjacent bar levels ahead.  None
of these asks for 12000 steps on a straight stretch; all of them are
per-step inequalities the stepper can enforce adaptively.

DISPLAY consumes vertices through interpolation, and its requirement is
geometric: the drawn curve must sit within one pixel of the true curve
at the current magnification.

## The sagitta arithmetic

Let p be the pixel size in model units at some magnification (view width
/ canvas width), kappa the orbit curvature at a vertex, h the arclength
to the next vertex.

  Chords (what the canvas draws today):    error ~ kappa h^2 / 8
      pixel-accurate iff   h <= sqrt(8 p / kappa)

  Cubic Hermite from vertex + tangent (tangents are free: the unit
  gradient at each vertex):                error ~ |kappa'| h^3 / c3
      pixel-accurate at h larger by roughly a decade for the same p in
      practice.

Curvature is computable per step from quantities the C phases already
have: with v = grad L and H the Hessian, kappa = |v x Hv| / |v|^3.
The conversion between the potential-rate parameter and arclength is
h = delta_L / |v|.

## The policy

1.  The STORED polyline is certificate-dense plus display-adequate at a
    DEFAULT resolution p0 (the zoo view on a ~1200px canvas).  The
    stepper grows the level step freely subject to, per step:
        kappa h   <= theta_max          (turn certificate, exists today)
        kappa h^2 <= 8 p0               (chord sagitta at default view)
        delta_L   <= next bar gap       (level-bar completeness)
        h         <= box_span / 256     (structure floor)
    On straight stretches all four are loose and the 12000 collapses to
    the structure floor: ~256 vertices.  Near critical points kappa and
    the bar spacing bind, which is where the density belongs.

2.  ZOOM within the stored polyline's competence draws chords (or
    Hermite segments) as today.  When the view's pixel p_view drops
    below what the stored sagitta supports, the inspector RE-TRACES the
    visible portion between the two certified vertices bracketing the
    view, with p_view as the sagitta target.  The C phases run at
    microseconds per step and the visible arc is short at high zoom:
    milliseconds, on demand, cacheable by (branch, view).  Certified
    vertices are anchors; a re-trace never replaces them, it interpolates
    between them, so certificates are untouched.

3.  The MAGNIFICATION LIMIT.  A traced vertex carries absolute position
    error on the order of a few ulps of the coordinate scale:
    e ~ c_eps * eps * max(|a|, |b|, scale), eps = 2^-52 (or 2^-23 for an
    FP32 tracer).  When p_view < ~2 e -- the fuzz would exceed a couple
    of pixels -- the inspector refuses further zoom and says why.  For
    binary64 that is a linear magnification of roughly p0 / e ~ 10^12
    from the default view; for FP32, ~10^5.  The refusal message should
    state the limit in the same terms ("display precision exhausted at
    this scale: traced in binary64").

4.  RE-TRACING between anchors at high zoom uses the same certified
    stepper, so its per-step guarantees are those of the certificate; no
    separate "display integrator" exists.  The only display-only object
    remains the beyond-the-box extension (serve._stable_extensions),
    already drawn distinguishably.

## What changes, what does not

Vertex density changes everywhere the four inequalities are looser than
n_levels = 12000: the RESIDUAL golden tier re-freezes once, with the
angle-resolved and n_points fields moving.  Decisions (EXACT tier),
certificates' verdicts, and the parity corpora's recorded segments are
untouched -- the corpora record calls that were made, and new-policy
calls are new recordings.  The population profile says this is worth
~60% of a typical random case; thrash gains little (its cost is stubs
and pullbacks).

## Open constants, for markup

  theta_max        exists (turn threshold); unchanged?
  p0               default-view pixel: zoo view / 1200?
  c_eps            ulp multiplier in the zoom limit: 4?
  structure floor  256 vertices per branch?
  Hermite or chords for the client?  (Hermite: ship tangents, fewer
                   vertices, bezier draw calls; chords: simplest.)
