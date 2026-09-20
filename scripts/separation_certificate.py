#!/usr/bin/env python3
"""Certify that a candidate pair carries NO saddle connection, and say how far.

TWO STATEMENTS, ONE COMPUTATION.

  separation   The source's unstable half-branch and the target's stable
               half-branch are shot to a common regular loss level, where
               their oriented separation D is a signed scalar.  Propagating
               each materialized stub's certified width to that level by a
               Gronwall bound and requiring the total to fall short of |D|
               proves the two manifolds do not meet: no connection on this
               pair, at THIS model.

  ball radius  The useful statement is stronger -- no model NEAR this one
               has one either.  D is smooth in the coefficients, so

                   dist(theta, non-Smale)  >=  |D| / sup ||dD/dtheta|| ,

               and dD/dtheta solves the variational equation along the same
               shot, amplified by the same exp(INT ||J|| dL).

Both use one amplification per shot, which is why they are computed together
and saved together: the elaboration layer (IVT bracketing of an actual
connection) needs the identical tube, amplification and stub widths, plus
dD/dLambda.  Nothing here is recomputed there.

MEASURED ON near-slide-d2 AT Lambda = 1 (branch 6, target b=0.984296,
stable_sign=+1), which is an ordinary Morse-Smale member about eight percent
from a reported rheostat wall:

    separation |D|            1.60e-01
    stub widths               1.36e-09 (source), 1.74e-10 (target)
    INT ||J|| dL bounds       4.43 (down), 7.99 (up)   [true 3.64, 2.87]
    amplifications            84, 2965
    propagated stub widths    1.15e-07, 5.15e-07   -> margin about 2.5e5
    sup ||dD/dtheta|| bound   1.73e5               [true 0.6 to 9.8]
    certified ball radius     9.3e-07              -- ten orders above the
                                                      binary64 representation
                                                      error of the inputs

WHERE THE PESSIMISM IS, recorded so nobody re-derives it: the Gronwall form
bounds ||z_theta|| by the worst case at every step -- forcing always aligned
with the most expanding direction of J, never cancelling -- while the true
variational solution does neither, since the forcing largely pushes ALONG
the trajectory (changing arrival time, not position) and successive
contributions partly cancel.  Sharpening that means propagating the
fundamental matrix rather than its norm, which belongs with the root
enclosure, not here.  For a NEGATIVE statement a loose bound costs only the
occasional case that could have been certified with more work.

    python scripts/separation_certificate.py
    python scripts/separation_certificate.py --case near-slide-d2 --branch 6 \
        --target-b 0.984296 --stable-sign 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import lipschitz, zoo                                  # noqa: E402
from two_ended_connection import measure                          # noqa: E402


@dataclass
class ShotBound:
    """Everything the certificate knows about one shot."""
    label: str
    stub_width: float
    radius: float
    integral: float
    amplification: float
    propagated: float
    local_error: float
    segments: int
    bootstrap_rounds: int


@dataclass
class Certificate:
    separation: float
    shots: list
    total_enclosure: float
    separated: bool
    sensitivity: float | None = None
    ball_radius: float | None = None
    per_coefficient: dict = field(default_factory=dict)


def _sagitta(points):
    """Max distance from a vertex to the chord through its neighbours.

    An upper bound on how far the TRUE arc departs from the polyline it is
    represented by, which the tube radius must cover: Gronwall's hypothesis
    is a Lipschitz bound on a region containing both the computed path and
    the true solution, not on the computed path alone.
    """
    worst = 0.0
    for i in range(1, len(points) - 1):
        p0 = np.asarray(points[i-1], float)
        p1 = np.asarray(points[i], float)
        p2 = np.asarray(points[i+1], float)
        span = p2 - p0
        length = float(np.hypot(*span))
        if length == 0.0:
            continue
        # 2-D cross product written out: np.cross no longer accepts
        # 2-vectors in NumPy 2.
        offset = p1 - p0
        area = float(span[0]*offset[1] - span[1]*offset[0])
        worst = max(worst, abs(area)/length)
    return worst


def bound_shot(m, label, shot, stub_width, space_atol, space_rtol,
               *, max_rounds=6, verbose=False):
    """Amplify a stub width along one shot, bootstrapping the tube radius.

    THE RADIUS IS NOT A FREE PARAMETER.  The bound is valid on a tube of
    half-width r about the computed chords; the enclosure it produces must
    then FIT inside that tube, or it was computed on a region that does not
    contain the true solution.  So: guess r, bound, check, enlarge, repeat.
    The loop converges in one or two rounds when the margin is large and
    reports honestly when it does not.
    """
    points, levels = shot.path, shot.path_levels
    span = abs(float(levels[-1]) - float(levels[0]))
    sagitta = _sagitta(points)
    # Local truncation, amplified: each accepted step contributes at most its
    # own tolerance, and the controller ran far inside it (max_space_ratio was
    # about 1e-6 here), but the bound uses the tolerance, not the observed.
    chord = max(float(np.hypot(*(np.asarray(points[i+1], float)
                                 - np.asarray(points[i], float))))
                for i in range(len(points)-1))
    per_step = space_atol + space_rtol*chord
    local_error = per_step*(len(points) - 1)

    radius = max(sagitta*2.0, stub_width*16.0, local_error*16.0, 1e-12)
    for rounds in range(1, max_rounds + 1):
        integral, contributions = lipschitz.path_integral_upper(
            m, points, levels, radius)
        if integral is None:
            failed = len(contributions) - 1
            raise RuntimeError(
                f"{label}: ||J|| not certified at segment {failed} of "
                f"{len(points)-1} with radius {radius:.3e}")
        total = float(integral)
        if total > 700.0:
            raise RuntimeError(
                f"{label}: amplification overflows (INT = {total:.4g})")
        amplification = math.exp(total)
        propagated = (stub_width + local_error)*amplification
        if verbose:
            print(f"     {label} round {rounds}: r={radius:.3e} "
                  f"INT={total:.4g} amp={amplification:.4g} "
                  f"enclosure={propagated:.3e}")
        if propagated + sagitta <= radius:
            return ShotBound(label, stub_width, radius, total, amplification,
                             propagated, local_error, len(points)-1, rounds)
        radius = 4.0*(propagated + sagitta)
    raise RuntimeError(f"{label}: tube radius did not settle in {max_rounds}")


def certify(base, lam, branch, target_b, stable_sign, *, order=6,
            launch_level=4, n_steps=400, fraction=0.5,
            space_rtol=1e-12, space_atol=1e-15, verbose=False):
    """Separation verdict and ball radius for one candidate pair.

    THE TOLERANCES ARE THE BINDING KNOB, and the relative one is what binds:
    with a max chord near 0.03, space_rtol = 1e-8 contributes 2.8e-10 per
    step against space_atol's 1e-13, so the absolute term was never in
    play.  Tightening the pair is nearly free and compounds through the
    radius bootstrap -- smaller local error gives a thinner tube, hence a
    smaller ||J|| bound, hence a smaller amplification, hence a smaller
    enclosure.  Measured on near-slide-d2 at Lambda = 1, n_steps = 400:

        rtol 1e-8,  atol 1e-13  ->  enclosure 6.98e-4, ball 1.68e-7
        rtol 1e-10, atol 1e-13  ->  enclosure 1.50e-6, ball 1.76e-6
        rtol 1e-12, atol 1e-15  ->  enclosure 1.97e-7, ball 5.34e-6

    a factor of 3500 in the enclosure for no extra segments (199 -> 203)
    and no extra time.  Returns diminish past 1e-12, where the stub widths
    (1.4e-9, 1.7e-10) and the tube geometry take over again, so that is the
    knee and the default.
    """
    r = measure(base, lam, branch, target_b, stable_sign, [fraction], order,
                launch_level, n_steps, 0.01, 0.01, space_rtol, space_atol,
                1e-8, 1e-12)
    row = r["rows"][0]
    separation = abs(float(row["D_b"]))
    m = _model_of(base, lam)
    shots = []
    for label, key, stub in (("down", "down_shot", r["source_stub_error"]),
                             ("up", "up_shot", r["target_stub_error"])):
        shots.append(bound_shot(m, label, r[key], float(stub),
                                space_atol, space_rtol, verbose=verbose))
    total = sum(s.propagated for s in shots)
    cert = Certificate(separation, shots, total, total < separation)

    if cert.separated:
        worst = 0.0
        for name, dA, dB in lipschitz.coefficient_derivatives(m):
            contribution = 0.0
            for label, key in (("down", "down_shot"), ("up", "up_shot")):
                shot = r[key]
                bound = next(s for s in shots if s.label == label)
                acc = 0.0
                for i in range(len(shot.path)-1):
                    value = lipschitz.field_sensitivity(
                        m, dA, dB, shot.path[i], shot.path[i+1], bound.radius)
                    if value is None:
                        acc = None
                        break
                    acc += float(value)*abs(float(shot.path_levels[i+1])
                                            - float(shot.path_levels[i]))
                if acc is None:
                    contribution = None
                    break
                contribution += acc*bound.amplification
            cert.per_coefficient[name] = contribution
            if contribution is not None:
                worst = max(worst, contribution)
        # Directions combine: the worst unit direction is bounded by the
        # euclidean norm of the per-coefficient sensitivities, not by their
        # largest component.
        values = [v for v in cert.per_coefficient.values() if v is not None]
        if values and len(values) == len(cert.per_coefficient):
            cert.sensitivity = math.sqrt(sum(v*v for v in values))
            cert.ball_radius = separation/cert.sensitivity
    return cert


def _model_of(base, lam):
    from level_crossing import rheostat
    return rheostat(base, lam)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="near-slide-d2")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--branch", type=int, default=6)
    ap.add_argument("--target-b", type=float, default=0.984296)
    ap.add_argument("--stable-sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--launch-level", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--space-rtol", type=float, default=1e-12)
    ap.add_argument("--space-atol", type=float, default=1e-15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    base = zoo.get(args.case)
    cert = certify(base, args.lam, args.branch, args.target_b,
                   args.stable_sign, order=args.order,
                   launch_level=args.launch_level, n_steps=args.n_steps,
                   fraction=args.fraction, space_rtol=args.space_rtol,
                   space_atol=args.space_atol, verbose=args.verbose)

    print(f"\n{args.case} at Lambda={args.lam:g}: branch {args.branch} "
          f"vs saddle near b={args.target_b:g} (stable sign "
          f"{args.stable_sign:+d})")
    print(f"   separation |D|            {cert.separation:.6e}")
    for s in cert.shots:
        print(f"   {s.label:>4}: stub {s.stub_width:.3e}  local "
              f"{s.local_error:.3e}  r {s.radius:.3e}  INT {s.integral:.4g}"
              f"  amp {s.amplification:.4g}  -> {s.propagated:.3e}"
              f"  [{s.segments} segments, {s.bootstrap_rounds} rounds]")
    print(f"   total enclosure           {cert.total_enclosure:.6e}")
    print(f"   SEPARATED                 {cert.separated}"
          + (f"   margin {cert.separation/cert.total_enclosure:.4g}x"
             if cert.separated else ""))
    if cert.ball_radius is not None:
        print(f"   sup ||dD/dtheta||         {cert.sensitivity:.4e}")
        print(f"   CERTIFIED BALL RADIUS     {cert.ball_radius:.4e}"
              f"   (relative coefficient perturbation)")
        worst = max(cert.per_coefficient.items(),
                    key=lambda kv: (kv[1] is not None, kv[1]))
        print(f"   worst single direction    {worst[0]} at "
              f"{worst[1]:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
