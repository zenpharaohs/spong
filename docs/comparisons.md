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
- `implicit-midpoint`: the second-order symmetric implicit method; and
- `rkf45`: the classical adaptive Fehlberg embedded 4(5) pair; and
- `ros2`: an adaptive, L-stable, two-stage Rosenbrock method of order two,
  included as an open analogue of MATLAB's `ode23s`.

The fixed-step methods use exactly the requested `--step-size`.  The adaptive
methods begin at that step and apply conventional local-error control.  Every
method receives the same physical `--time-horizon`, so an adaptive method
cannot appear to progress farther merely by taking larger steps.  All use a
finite step/vertex budget and simple distance capture.  They do not use
Poincaré conditioning, graph-transform stubs, Gauss collocation, topology
auditing, or the exact Morse structure.  Their SVG footer is therefore marked
`UNCERTIFIED COMPARISON`.

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
