# Why not just use a standard phase-portrait recipe?

SPONG ships deliberately ordinary, explicitly **uncertified** comparison
backends.  They are teaching instruments, not fallbacks.  Each comparison
holds the model and display box fixed while changing one numerical decision.

Generate a comparison gallery with:

```sh
spong-compare-portrait --zoo quadratic-stiff
```

The output directory contains a certified reference SVG, one SVG for each
requested alternative, a side-by-side HTML page, and a JSON diagnostic report.
The HTML file is a document, not a shell command.  On macOS, open it with:

```sh
open out/comparisons/quadratic-stiff_comparison.html
```

For a particularly visible Forward Euler example:

```sh
spong-compare-portrait \
  --zoo quadratic-stiff \
  --methods forward-euler \
  --step-size 0.05
```

The canonical graph-transform challenge is:

```sh
spong-compare-portrait \
  --zoo tricky-d11 \
  --step-size 0.05 \
  --time-horizon 100
```

## Geometry alternatives

The available methods integrate the actual, unnormalized gradient flow from
fixed eigenvector offsets at each saddle:

- `forward-euler`: one explicit slope evaluation;
- `backward-euler`: an implicit endpoint equation solved by Newton;
- `explicit-midpoint`: the second-order explicit Runge--Kutta method;
- `implicit-midpoint`: the second-order symmetric implicit method;
- `rkf45`: the classical adaptive Fehlberg embedded 4(5) pair;
- `ros2`: an adaptive, L-stable, two-stage Rosenbrock method of order two,
  included as an open analogue of MATLAB's `ode23s`; and
- `stork2`: the 20-stage Stabilized Taylor Orthogonal Runge--Kutta order-two
  recurrence, using its published Forward Euler startup and first-order
  time-Taylor virtual function evaluations; and
- `stork4`: the corresponding ROCK4-based order-four recurrence and
  composition.  The comparison includes the authors' tabulated coefficients
  for 9 or 20 recurrence stages.  Change the stabilized degree for either
  STORK method with `--stork-stages`.

The fixed-step methods use exactly the requested `--step-size`.  The adaptive
methods begin at that step and apply conventional local-error control.  Every
method receives the same physical `--time-horizon`, so an adaptive method
cannot appear to progress farther merely by taking larger steps.  All use a
finite step/vertex budget and simple distance capture.  They do not use
Poincaré conditioning, graph-transform stubs, Gauss collocation, topology
auditing, or the exact Morse structure.  Their SVG footer is therefore marked
`UNCERTIFIED COMPARISON`.

STORK's virtual stages were designed to reduce expensive neural-network
evaluations in diffusion and flow-matching samplers.  A SPONG vector field is
an inexpensive exact polynomial evaluation, so that economy is irrelevant to
the production engine.  The comparison retains the virtual-stage Taylor
approximation because the point is to test STORK itself, rather than the
underlying non-Taylor RKG2/ROCK4 methods.  Its enlarged linear stability
interval does not condition the eigenvector launch and does not certify
invariant manifold incidence.  With first-order Taylor virtual evaluations,
STORK-4 reduces on an autonomous linear mode to the second-order Taylor
stability polynomial, independent of the tested recurrence degree; the
non-Taylor ROCK4 stability interval therefore does not carry over unchanged.

The adaptive defaults, `rtol=1e-3` and `atol=1e-6`, match the familiar
MATLAB/SciPy casual defaults; both can be changed on the command line.

The report also measures tangent/gradient misalignment after resampling every
curve at a common geometric spacing.  It gives an angle energy, chord-weighted
RMS angle, and maximum local angle, along with a separate record for every
branch.  Points where the gradient is below one-thousandth of that branch's
maximum are marked unresolved rather than allowing a near-critical tangent
angle to dominate.  This avoids making a method look smoother merely because
it emitted fewer vertices, while acknowledging that direction is ill
conditioned as the vector field vanishes.  Each gallery includes an automatic
zoom around the deepest certified unstable branch.

The gallery also applies SPONG's loss-level order diagnostic to every
uncertified manifold proposal.  Chord-intersection candidates are classified
by resolved transverse order on regular loss levels.  The JSON and HTML
reports distinguish:

- isolated resolved pair roots, which are numerical violations of ODE
  uniqueness;
- same-order chord contacts, which do not reverse the represented curves;
- contacts at a critical loss or without resolved witnesses, which remain
  unresolved; and
- transverse self-crossings and ambiguous self-contacts.

The textbook methods inherit no production terminal, local-graph, or endpoint
certificates.  Their apparent asymptotic contacts therefore cannot be silently
discharged, and an `accepted` contact diagnostic still does **not** certify the
portrait.  It says only that this scan found no contact fault.  Use
`--contact-threshold` to change the resolved-order margin and
`--contact-limit` to bound pathological scribble output.

The handle-slide wall has a dedicated geometry comparison:

```sh
PYTHONPATH=src:. python3 demos/saddle_connection_comparison.py
open out/saddle_connection_comparison/nonnearest-saddle-connection-casual-comparison.html
```

All panels in that gallery share the exact Sturm critical points.  The only
change is the manifold tracer.  The exact wall branch terminates at the target
saddle.  A casual finite-radius saddle stop can make an inaccurate trace look
accidentally correct, so the comparison continues each numerical unstable
branch until a minimum.  The casual panels contain only the green/red phase
portrait produced by the named discretization; optimizer and highlighted-
trajectory overlays belong to separate demos.  The selected outgoing branch records the
macroscopic topology error generated by a small launch or integration error;
it is not asserted to be the wall's branch incidence.

The gallery also reports the signed separation of the independently traced
`W^u(B)` and `W^s(N)` branches on the midpoint regular loss level.  This exposes
subpixel disagreements which an ordinary phase-portrait view conceals.  It is
a two-sided shooting measurement, not a pure anadromy test, because the two
branches also have independent launches.  The sequential experiment that
isolates launch, Poincare conditioning, Hadamard graph construction, GL8
continuation, and topology auditing is specified in
`docs/geometry_ablation.md`.

The fixed-step wall comparison defaults to `h=0.05`; `h=0.01` takes roughly
five times as many steps and makes even Euler look artificially polished at
portrait scale.  The report therefore includes centerline-subtracted
transverse panels.  At a family of regular loss sections these remove the
common longitudinal position and plot the two independently computed
manifolds at `-d/2` and `+d/2`, where `d` is their signed mismatch.  Every panel
prints its own physical transverse scale.  These panels are intentionally
anisotropic metrology views, not pictures of the ambient plane.

No single step or tolerance is a qualification.  Fixed-step methods must be
run as a convergence sequence, and adaptive methods as a tolerance sequence.
In particular, RKF45's wall mismatch is not monotone in its local error
tolerance: local IVP error control is not a topology certificate.

Backward Euler is included because stability alone is not sufficient:
its nonlinear solve can fail or converge to the wrong implicit solution.
Implicit midpoint avoids backward Euler's artificial damping but still needs
a reliable nonlinear stage solve.  RKF45 is a very good nonstiff IVP method;
the comparison demonstrates that adaptive local IVP error control does not by
itself certify separatrix topology in a stiff near-critical problem.
MATLAB's
[`ode23s`](https://www.mathworks.com/help/matlab/ref/ode23s.html) is a
single-step, modified Rosenbrock method of order two intended for stiff
problems.  It evaluates a Jacobian at every step and can be effective at crude
tolerances.  The `ros2` comparison has the same broad pedagogical role, but is
not claimed to reproduce MATLAB's private implementation coefficient for
coefficient.

## Critical-point alternative

Use `--critical-method grid-newton` to replace exact Sturm enumeration with a
common casual recipe:

1. place a rectangular grid of initial guesses;
2. run damped Newton using a finite-difference Jacobian;
3. discard unconverged or out-of-box iterates;
4. merge nearby answers; and
5. classify the retained points from the numerical Hessian.

This can find a useful picture, but it cannot prove completeness.  A coarser
grid can miss a Newton basin; the deduplication tolerance can merge distinct
critical points; and near-singular Hessians defeat both convergence and
classification.  On the `quadratic-stiff` zoo case, for example, a 9-by-9
scan finds five critical points while a 13-by-13 scan finds all six.

```sh
spong-compare-portrait \
  --zoo quadratic-stiff \
  --critical-method grid-newton \
  --critical-grid 9
```

## What common plotting tools do

There is no generally complete built-in phase-portrait oracle.

- MATLAB's [`ode45`](https://www.mathworks.com/help/matlab/ref/ode45.html)
  traces selected initial-value problems with a Dormand--Prince explicit
  Runge--Kutta 4(5) pair; `odephas2` plots two solution components.
- MATLAB's [`stream2`](https://www.mathworks.com/help/matlab/ref/stream2.html)
  integrates streamlines through a vector field supplied on a grid.  Its
  documented defaults are a grid-data step of 0.1 and at most 10,000
  vertices.
- Matplotlib's
  [`streamplot`](https://matplotlib.org/stable/gallery/images_contours_and_fields/plot_streamplot.html)
  similarly seeds and integrates streamlines of a sampled 2-D vector field.
- SciPy's
  [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
  defaults to explicit RK45 and recommends Radau or BDF when stiffness is
  present.
- Equilibria are normally found separately with a local nonlinear solver,
  such as MATLAB `fsolve` or SciPy
  [`root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html),
  from user-chosen starting guesses.

Those are appropriate exploratory tools.  What they do not supply is an exact
count of all critical points, certified invariant-manifold launches, or an
a-posteriori proof that the drawn curves have the correct planar topology.
