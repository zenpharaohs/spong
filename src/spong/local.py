"""Conditioned local data attached to the zero-dimensional Morse skeleton.

The global model has exact rational coefficients, but evaluating it near a
critical point can lose nearly all significant digits through cancellation.
This module translates the finite polynomial gradient with Fraction arithmetic
about a very tightly refined rational representative of the isolated root.
Only the resulting centered coefficients are rounded to binary64.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb

import numpy as np

from . import _poly as P
from ._poly import Poly
from .model import Model


def _shift(p: Poly, c: Fraction) -> tuple[Fraction, ...]:
    """Coefficients of p(c+x), exactly over Q."""
    return tuple(
        sum((p[k] * comb(k, j) * c ** (k - j)
             for k in range(j, len(p))), Fraction(0))
        for j in range(len(p))
    )


def _lincomb(*terms: tuple[Fraction, tuple[Fraction, ...]]
             ) -> tuple[Fraction, ...]:
    n = max((len(p) for _, p in terms), default=1)
    return tuple(sum((s * p[k] for s, p in terms if k < len(p)),
                     Fraction(0)) for k in range(n))


@dataclass(frozen=True)
class SpectralData:
    """High-precision eigendata derived from the exact centered Hessian."""

    eigenvalues: tuple[float, float]       # ascending
    frame: tuple[tuple[float, float], tuple[float, float]]
    trace: Fraction
    determinant: Fraction
    decimal_precision: int
    decimal_eigenvalues: tuple[Decimal, Decimal]
    decimal_frame: tuple[tuple[Decimal, Decimal],
                         tuple[Decimal, Decimal]]


def _decimal(x: Fraction) -> Decimal:
    return Decimal(x.numerator) / Decimal(x.denominator)


def _scaled_norm(x) -> float:
    """Overflow/underflow-safe Euclidean norm for diagnostic vectors."""
    x = np.asarray(x, dtype=float).ravel()
    scale = float(np.max(np.abs(x), initial=0.0))
    if scale == 0.0:
        return 0.0
    return scale*float(np.sqrt(np.sum((x/scale)**2)))


def _solve2(J, rhs):
    """Guarded closed-form solve for the near-identity Poincare Jacobian."""
    a, b = float(J[0][0]), float(J[0][1])
    c, d = float(J[1][0]), float(J[1][1])
    r0, r1 = float(rhs[0]), float(rhs[1])
    s0, s1 = max(abs(a), abs(b)), max(abs(c), abs(d))
    if s0 == 0.0 or s1 == 0.0:
        raise FloatingPointError("singular Poincare coordinate Jacobian")
    aa, bb, rr0 = a/s0, b/s0, r0/s0
    cc, dd, rr1 = c/s1, d/s1, r1/s1
    determinant = aa*dd-bb*cc
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise FloatingPointError(
            "ill-conditioned Poincare coordinate Jacobian")
    return ((dd*rr0-bb*rr1)/determinant,
            (aa*rr1-cc*rr0)/determinant)


def _exact_symmetric_spectral(
        H: tuple[tuple[Fraction, Fraction],
                 tuple[Fraction, Fraction]],
        precision: int = 100) -> SpectralData:
    """Diagonalize a rational symmetric 2x2 matrix before FP64 rounding.

    The eigenvalue of smaller magnitude is recovered from det/lambda_large,
    avoiding the subtraction which erased the saddle signature in the
    large-radius qualification cases.
    """
    a, c, d = H[0][0], H[0][1], H[1][1]
    trace = a + d
    determinant = a*d - c*c
    with localcontext() as context:
        context.prec = precision
        A, C, D = _decimal(a), _decimal(c), _decimal(d)
        T, Det = _decimal(trace), _decimal(determinant)
        root = ((A-D)*(A-D) + Decimal(4)*C*C).sqrt()
        lm_direct = (T-root)/Decimal(2)
        lp_direct = (T+root)/Decimal(2)
        if abs(lp_direct) >= abs(lm_direct):
            lp = lp_direct
            lm = Det/lp if lp else lm_direct
            large_is_plus = True
            large = lp
        else:
            lm = lm_direct
            lp = Det/lm if lm else lp_direct
            large_is_plus = False
            large = lm

        # Use whichever null-equation row gives the larger unnormalized
        # vector.  The other eigenvector is its exact orthogonal complement.
        candidates = ((C, large-A), (large-D, C))
        x, y = max(candidates, key=lambda v: v[0]*v[0]+v[1]*v[1])
        norm = (x*x+y*y).sqrt()
        if not norm:
            # Diagonal matrix: select the coordinate belonging to `large`.
            x, y = ((Decimal(1), Decimal(0))
                    if abs(large-A) <= abs(large-D)
                    else (Decimal(0), Decimal(1)))
            norm = Decimal(1)
        x, y = x/norm, y/norm
        if large_is_plus:
            vminus, vplus = (-y, x), (x, y)
        else:
            vminus, vplus = (x, y), (-y, x)
        frame = ((float(vminus[0]), float(vplus[0])),
                 (float(vminus[1]), float(vplus[1])))
        eigenvalues = (float(lm), float(lp))
    return SpectralData(
        eigenvalues, frame, trace, determinant, precision,
        (lm, lp),
        ((vminus[0], vplus[0]), (vminus[1], vplus[1])))


@dataclass(frozen=True)
class PoincareData:
    """Selective quadratic normal-form candidate for one saddle manifold."""

    manifold: str
    frame: tuple[tuple[float, float], tuple[float, float]]
    eigenvalues: tuple[float, float]       # graph, transverse
    full_map: tuple[tuple[float, float, float],
                    tuple[float, float, float]]
    selected_map: tuple[tuple[float, float, float],
                        tuple[float, float, float]]
    transformed_quadratic: tuple[tuple[float, float, float],
                                 tuple[float, float, float]]
    divisors: tuple[tuple[float, float, float],
                    tuple[float, float, float]]
    retained: tuple[tuple[bool, bool, bool],
                    tuple[bool, bool, bool]]
    desired_reach: float
    coordinate_budget: float
    normal_native: object | None = None
    conditioned_native: object | None = None

    def centered(self, u: float, s: float) -> tuple[float, float]:
        p, q = self.selected_map
        U = u + p[0]*u*u + p[1]*u*s + p[2]*s*s
        S = s + q[0]*u*u + q[1]*u*s + q[2]*s*s
        R = np.asarray(self.frame)
        z = R @ np.array([U, S])
        return float(z[0]), float(z[1])

    def physical(self, local, u: float, s: float) -> tuple[float, float]:
        da, db = self.centered(u, s)
        return local.a + da, local.b + db

    def velocity(self, local, u: float, s: float) -> tuple[float, float]:
        """Conditioned vector field in normal-form coordinates."""
        if self.conditioned_native is not None:
            return self.conditioned_native.gradient(u, s)
        p, q = self.selected_map
        U = u + p[0]*u*u + p[1]*u*s + p[2]*s*s
        S = s + q[0]*u*u + q[1]*u*s + q[2]*s*s
        if self.normal_native is not None:
            g = np.asarray(self.normal_native.gradient(U, S))
        else:
            point = self.physical(local, u, s)
            physical_g = np.asarray(
                local.gradient(point[0]-local.a, point[1]-local.b))
            g = np.asarray(self.frame).T @ physical_g
        J = np.array([
            [1+2*p[0]*u+p[1]*s, p[1]*u+2*p[2]*s],
            [2*q[0]*u+q[1]*s, 1+q[1]*u+2*q[2]*s],
        ])
        return _solve2(J, g)

    def graph(self, local, sign: int, n: int = 513,
              tol: float = 1e-12, max_iter: int = 80,
              reach: float | None = None):
        reach = self.desired_reach if reach is None else reach
        # This is a centered coordinate computation.  Do not impose the ulp
        # of the translated physical critical point on its start radius:
        # mapping back may coalesce a few initial display samples, while the
        # local jet and invariant graph remain fully resolved.
        start = reach*1e-10
        if not start < reach:
            raise FloatingPointError("no resolved Poincare graph interval")
        if self.conditioned_native is not None:
            flat_map = (0.0,)*6
            kernel = self.conditioned_native
            frame = ((1.0, 0.0), (0.0, 1.0))
        else:
            flat_map = tuple(x for row in self.selected_map for x in row)
            kernel = self.normal_native or local.native
            frame = ((1.0, 0.0), (0.0, 1.0)) \
                if self.normal_native is not None else self.frame
        x, h, iterations, rel = kernel.poincare_graph(
            tuple(x for row in frame for x in row), flat_map,
            *self.eigenvalues, start, reach, int(sign), n, tol, max_iter)
        x, h = np.asarray(x), np.asarray(h)
        centered = np.asarray([
            self.centered(sign*float(u), float(s)) for u, s in zip(x, h)])
        physical = centered + (local.a, local.b)
        physical = np.vstack(([local.a, local.b], physical))
        return physical, {"iterations": int(iterations),
                          "relative_change": float(rel),
                          "start": float(start), "reach": float(reach),
                          "grid_points": n,
                          "normal_x": x, "normal_h": h,
                          "centered": centered}


@dataclass(frozen=True)
class InvariantStub:
    manifold: str
    orientation: int
    b_direction: int
    destination_b: float | None
    destination_kind: str
    curve: tuple[tuple[float, float], ...]
    preferred_chart: str
    certificates: tuple[tuple[str, float], ...]

    @property
    def handoff(self):
        return self.curve[-1]


@dataclass(frozen=True)
class LocalJet:
    """Centered finite gradient jet, computed exactly then rounded once.

    ``grad[component][a_power][b_power]`` is the coefficient of
    ``da**a_power * db**b_power``.  The constant vector is imposed as exactly
    zero: the center represents the certified algebraic critical point, while
    ``center_interval`` records its remaining rational enclosure.
    """

    a: float
    b: float
    center_interval: object
    grad: tuple[tuple[tuple[float, ...], ...], ...]
    exact_grad: tuple[tuple[tuple[Fraction, ...], ...], ...]
    hessian: tuple[tuple[float, float], tuple[float, float]]
    exact_hessian: tuple[tuple[Fraction, Fraction],
                         tuple[Fraction, Fraction]]
    spectral: SpectralData
    native: object | None = None
    poincare: tuple[PoincareData, ...] = ()

    def gradient(self, da: float, db: float) -> tuple[float, float]:
        if self.native is not None:
            return self.native.gradient(da, db)
        out = []
        for component in self.grad:
            value = 0.0
            apow = 1.0
            for row in component:
                acc = 0.0
                for coefficient in reversed(row):
                    acc = acc * db + coefficient
                value += apow * acc
                apow *= da
            out.append(value)
        return out[0], out[1]

    def potential(self, da: float, db: float) -> float:
        """Centered potential difference whose gradient is ``gradient``."""
        # Integrate the a-component from (0, db) to (da, db), then the
        # b-component along a=0 from (0, 0) to (0, db).
        value = 0.0
        apow = da
        for i, row in enumerate(self.grad[0]):
            acc = 0.0
            for coefficient in reversed(row):
                acc = acc*db + coefficient
            value += apow*acc/(i+1)
            apow *= da
        bpow = db
        row = self.grad[1][0]
        for j, coefficient in enumerate(row):
            value += bpow*coefficient/(j+1)
            bpow *= db
        return value

    def normalized_step(self, da: float, db: float, h: float,
                        order: int = 6) -> tuple[float, float]:
        """Native IRK continuation step entirely in centered coordinates."""
        if self.native is None:
            raise RuntimeError("native centered continuation is unavailable")
        return self.native.normalized_step(da, db, h, order)

    def raw_step(self, da: float, db: float, h: float,
                 order: int = 6) -> tuple[float, float]:
        """Native IRK step for the regular centered gradient field."""
        if self.native is None:
            raise RuntimeError("native centered continuation is unavailable")
        return self.native.raw_step(da, db, h, order)

    def unstable_graph(self, reach: float, n: int = 513,
                       sign: int = 1, tol: float = 1e-13,
                       max_iter: int = 80, manifold: str = "unstable"):
        """Native Hadamard fixed point for one unstable branch.

        The graph is represented in the orthonormal Hessian eigenframe and
        returned in centered physical coordinates.
        """
        if self.native is None:
            raise RuntimeError("native centered graph transform is unavailable")
        lam = np.asarray(self.spectral.eigenvalues)
        V = np.asarray(self.spectral.frame)
        if not (lam[0] < 0.0 < lam[1]):
            raise ValueError("unstable graph requires a Morse saddle")
        depart = 0 if manifold == "unstable" else 1
        transverse = 1 - depart
        R = np.column_stack((V[:, depart], V[:, transverse]))
        u = np.linspace(0.0, float(sign) * abs(reach), n)
        f, iterations, rel = self.native.graph_fixed_point(
            u, tuple(R.ravel()), float(lam[depart]), float(lam[transverse]),
            tol, max_iter)
        f = np.asarray(f, dtype=float)
        centered = u[:, None] * R[:, 0] + f[:, None] * R[:, 1]
        return centered, {
            "iterations": int(iterations),
            "relative_change": float(rel),
            "lambda_graph": float(lam[depart]),
            "lambda_transverse": float(lam[transverse]),
            "manifold": manifold,
            "grid_points": int(n),
            "reach": float(reach),
        }

    def compare_graph_acceleration(self, reach: float, n: int = 513,
                                   sign: int = 1, tol: float = 1e-12,
                                   max_iter: int = 80,
                                   manifold: str = "unstable"):
        """Compare plain Hadamard iteration with safeguarded depth-3 RRE."""
        lam = np.asarray(self.spectral.eigenvalues)
        V = np.asarray(self.spectral.frame)
        if not (lam[0] < 0.0 < lam[1]):
            raise ValueError("graph acceleration requires a Morse saddle")
        depart = 0 if manifold == "unstable" else 1
        transverse = 1-depart
        V = np.column_stack((V[:, depart], V[:, transverse]))
        u = np.linspace(0.0, float(sign) * abs(reach), n)
        frame = tuple(V.ravel())

        def G(x):
            return np.asarray(self.native.graph_transform(
                u, frame, float(lam[depart]), float(lam[transverse]), x),
                dtype=float)

        def norm(x):
            return _scaled_norm(x) / np.sqrt(x.size)

        plain = np.zeros(n)
        accelerated = np.zeros(n)
        history = [accelerated.copy()]
        plain_done = rre_done = None
        attempted = accepted = rejected_condition = rejected_residual = 0
        plain_residual = rre_residual = np.inf
        for iteration in range(1, max_iter + 1):
            pn = G(plain)
            plain_residual = norm(pn - plain)
            plain = pn
            if plain_done is None and plain_residual <= tol*(1+norm(plain)):
                plain_done = iteration

            an = G(accelerated)
            rre_residual = norm(an - accelerated)
            accelerated = an
            history.append(accelerated.copy())
            if len(history) >= 4:
                attempted += 1
                X = np.asarray(history[-4:])
                U = (X[1:] - X[:-1]).T
                try:
                    # Intentional sole production use of np.linalg: U is an
                    # empirical FP64 iterate-difference matrix, not exact
                    # model data, and rank revelation is exactly the task
                    # SVD is backward-stable for.  The condition guard below
                    # rejects lost rank, and no extrapolate is accepted unless
                    # an independent graph-transform residual improves.
                    _left, singular, right = np.linalg.svd(
                        U, full_matrices=False)
                    if singular[-1] <= 1e-13*singular[0]:
                        raise np.linalg.LinAlgError
                    ones = np.ones(U.shape[1])
                    # Constrained least-residual coefficients from the SVD
                    # itself.  Do not square the condition number by forming
                    # and solving U.T@U.
                    relative = singular/singular[0]
                    y = right.T @ ((right @ ones)/(relative*relative))
                    gamma = y / np.sum(y)
                    if np.sum(np.abs(gamma)) > 20:
                        raise np.linalg.LinAlgError
                    candidate = gamma @ X[:-1]
                except np.linalg.LinAlgError:
                    rejected_condition += 1
                else:
                    cn = G(candidate)
                    cres = norm(cn-candidate)
                    if np.all(np.isfinite(candidate)) and cres < rre_residual:
                        accelerated = candidate
                        rre_residual = cres
                        history[-1] = candidate.copy()
                        accepted += 1
                    else:
                        rejected_residual += 1
            if rre_done is None and rre_residual <= tol*(1+norm(accelerated)):
                rre_done = iteration
            if plain_done is not None and rre_done is not None:
                break
        return {
            "plain": plain, "rre": accelerated,
            "plain_iterations": plain_done, "rre_iterations": rre_done,
            "plain_residual": plain_residual, "rre_residual": rre_residual,
            "rre_attempted": attempted, "rre_accepted": accepted,
            "rre_rejected_condition": rejected_condition,
            "rre_rejected_residual": rejected_residual,
            "u": u, "frame": V,
        }

    def normal_form(self, manifold: str) -> PoincareData:
        return next(x for x in self.poincare if x.manifold == manifold)


def _poincare_candidates(jet: LocalJet, nearby_b: tuple[float, ...],
                         coordinate_budget: float = 0.25
                         ) -> tuple[PoincareData, ...]:
    lam = np.asarray(jet.spectral.eigenvalues)
    eig = np.asarray(jet.spectral.frame)
    def transformed_native(Rd, chart_eigenvalues):
        """Round only after the exact jet has been rotated in high precision."""
        def power(value, exponent):
            return Decimal(1) if exponent == 0 else value**exponent

        degree = max(
            i+j
            for component in jet.exact_grad
            for i, row in enumerate(component)
            for j, value in enumerate(row) if value)
        normal = [
            [[Decimal(0) for _ in range(degree+1)]
             for _ in range(degree+1)]
            for _ in range(2)
        ]
        with localcontext() as context:
            context.prec = jet.spectral.decimal_precision
            for physical_component, component in enumerate(jet.exact_grad):
                for apower, row in enumerate(component):
                    for bpower, rational in enumerate(row):
                        if not rational:
                            continue
                        value = _decimal(rational)
                        for ka in range(apower+1):
                            ca = (Decimal(comb(apower, ka))
                                  * power(Rd[0][0], ka)
                                  * power(Rd[0][1], apower-ka))
                            for kb in range(bpower+1):
                                cb = (Decimal(comb(bpower, kb))
                                      * power(Rd[1][0], kb)
                                      * power(Rd[1][1], bpower-kb))
                                upower = ka+kb
                                spower = apower+bpower-upower
                                term = value*ca*cb
                                for normal_component in range(2):
                                    normal[normal_component][upower][spower] += (
                                        Rd[physical_component][normal_component]
                                        * term)
            # The exact critical identity and exact diagonal linearization
            # are structural data, not residuals to be re-estimated after
            # cancellation in the coordinate transform.
            for component in range(2):
                normal[component][0][0] = Decimal(0)
            normal[0][1][0] = chart_eigenvalues[0]
            normal[0][0][1] = Decimal(0)
            normal[1][1][0] = Decimal(0)
            normal[1][0][1] = chart_eigenvalues[1]
        rounded = tuple(
            tuple(tuple(float(x) for x in row) for row in component)
            for component in normal)
        try:
            from . import _native
        except (ImportError, AttributeError):
            return rounded, None, normal
        # A rotated polynomial generally has more than three powers of its
        # first coordinate.  The native kernel supports the full rectangular
        # coefficient array; a ValueError here means malformed chart data and
        # must not silently select a kernel expressed in different coordinates.
        return rounded, _native.LocalKernel(rounded), normal

    def pulled_back_python(normal, selected_decimal, chart_eigenvalues):
        """Build adj(DT) F(T(z)) before the one binary64 rounding.

        The determinant omitted from the inverse Jacobian is a positive
        scalar inside a certified chart, so this polynomial has exactly the
        same oriented integral curves as the Poincare-coordinate vector
        field.  Forming it symbolically prevents the removable normal-form
        terms from being cancelled at runtime in FP64.
        """
        zero = Decimal(0)
        one = Decimal(1)

        def add(a, b, scale=one):
            out = dict(a)
            for key, value in b.items():
                out[key] = out.get(key, zero) + scale*value
                if not out[key]:
                    del out[key]
            return out

        def mul(a, b):
            out = {}
            for (i, j), x in a.items():
                for (k, l), y in b.items():
                    key = (i+k, j+l)
                    out[key] = out.get(key, zero) + x*y
            return {key: value for key, value in out.items() if value}

        def power_poly(a, exponent):
            out = {(0, 0): one}
            for _ in range(exponent):
                out = mul(out, a)
            return out

        p, q = selected_decimal
        U = {(1, 0): one, (2, 0): p[0],
             (1, 1): p[1], (0, 2): p[2]}
        S = {(0, 1): one, (2, 0): q[0],
             (1, 1): q[1], (0, 2): q[2]}
        fields = []
        for component in normal:
            field = {}
            for i, row in enumerate(component):
                for j, coefficient in enumerate(row):
                    if coefficient:
                        term = mul(power_poly(U, i), power_poly(S, j))
                        field = add(field, term, coefficient)
            fields.append(field)
        j00 = {(0, 0): one, (1, 0): 2*p[0], (0, 1): p[1]}
        j01 = {(1, 0): p[1], (0, 1): 2*p[2]}
        j10 = {(1, 0): 2*q[0], (0, 1): q[1]}
        j11 = {(0, 0): one, (1, 0): q[1], (0, 1): 2*q[2]}
        pulled = (
            add(mul(j11, fields[0]), mul(j01, fields[1]), -one),
            add(mul(j00, fields[1]), mul(j10, fields[0]), -one),
        )
        # Preserve the structural critical point and diagonal linear part.
        for component in pulled:
            component.pop((0, 0), None)
        pulled[0][(1, 0)] = chart_eigenvalues[0]
        pulled[0].pop((0, 1), None)
        pulled[1].pop((1, 0), None)
        pulled[1][(0, 1)] = chart_eigenvalues[1]
        degree_u = max(i for component in pulled for i, _ in component)
        degree_s = max(j for component in pulled for _, j in component)
        rounded = tuple(
            tuple(
                tuple(float(component.get((i, j), zero))
                      for j in range(degree_s+1))
                for i in range(degree_u+1))
            for component in pulled)
        return rounded

    def pulled_back_native(normal, selected_decimal, chart_eigenvalues):
        """Compose the chart in the frontend-independent precision core."""
        try:
            from . import _native
        except (ImportError, AttributeError):
            return None
        nu = len(normal[0])
        ns = len(normal[0][0])
        flat_normal = tuple(
            str(value)
            for component in normal
            for row in component
            for value in row)
        flat_selected = tuple(
            float(value) for row in selected_decimal for value in row)
        rounded = _native.poincare_pullback(
            flat_normal, nu, ns, flat_selected,
            str(chart_eigenvalues[0]), str(chart_eigenvalues[1]),
            max(192, 4*jet.spectral.decimal_precision))
        return _native.LocalKernel(rounded)
    out = []
    for manifold, depart in (("unstable", 0), ("stable", 1)):
        transverse = 1-depart
        R = np.column_stack((eig[:, depart], eig[:, transverse]))
        decimal_columns = jet.spectral.decimal_frame
        Rd = (
            (decimal_columns[0][depart], decimal_columns[0][transverse]),
            (decimal_columns[1][depart], decimal_columns[1][transverse]),
        )
        ld, lt = float(lam[depart]), float(lam[transverse])
        normal_grad, normal_native, normal_decimal = transformed_native(
            Rd, (jet.spectral.decimal_eigenvalues[depart],
                 jet.spectral.decimal_eigenvalues[transverse]))
        D = np.array([
            [normal_grad[0][2][0], normal_grad[0][1][1],
             normal_grad[0][0][2]],
            [normal_grad[1][2][0], normal_grad[1][1][1],
             normal_grad[1][0][2]],
        ])
        divisors = np.array([
            [ld, lt, 2*lt-ld],
            [2*ld-lt, ld, lt],
        ])
        full = np.zeros((2, 3))
        safe = np.abs(divisors) > 64*np.finfo(float).eps*max(
            abs(ld), abs(lt), 1.0)
        # `divisors` stores m·lambda-lambda_i.  For the physical map
        # x = y + H(y), the pulled-back quadratic coefficient is
        # D -(m·lambda-lambda_i)H, hence H=D/divisor.
        full[safe] = D[safe]/divisors[safe]

        cap_neighbor = np.inf
        if nearby_b and abs(R[1, 0]) > 1e-15:
            cap_neighbor = 0.5*min(abs(x-jet.b) for x in nearby_b) \
                / abs(R[1, 0])
        reach = min(0.1, cap_neighbor)
        selected = full.copy()
        retained = ~safe
        # Along the outgoing invariant graph s=O(u^3), these four entries
        # dominate DT-I.  Retain a formally removable term when eliminating
        # it would consume too much of the requested chart's injectivity.
        for component, monomial, multiplier in (
                (0, 0, 2.0), (0, 1, 1.0),
                (1, 0, 2.0), (1, 1, 1.0)):
            if multiplier*abs(full[component, monomial])*reach \
                    > coordinate_budget:
                selected[component, monomial] = 0.0
                retained[component, monomial] = True
        # Pull back by the exact values of the binary64 map which will be
        # used to map the finished curve.  Any residual caused by rounding a
        # theoretically removable coefficient is then retained explicitly
        # in the conditioned polynomial instead of being recreated through
        # catastrophic runtime subtraction.
        selected_decimal = tuple(
            tuple(Decimal.from_float(float(x)) for x in row)
            for row in selected)
        conditioned_native = pulled_back_native(
            normal_decimal, selected_decimal,
            (jet.spectral.decimal_eigenvalues[depart],
             jet.spectral.decimal_eigenvalues[transverse]))
        out.append(PoincareData(
            manifold, tuple(map(tuple, R)),
            (ld, lt), tuple(map(tuple, full)), tuple(map(tuple, selected)),
            tuple(map(tuple, D - divisors*selected)),
            tuple(map(tuple, divisors)), tuple(map(tuple, retained)),
            float(reach), coordinate_budget, normal_native,
            conditioned_native))
    return tuple(out)


def build_local_jet(m: Model, interval, source: str,
                    nearby_b: tuple[float, ...] = (),
                    root_poly: Poly | None = None) -> LocalJet:
    """Build conditioned critical-point data after exact enumeration."""
    root_poly = (m.N if source == "N" else m.beta) \
        if root_poly is None else root_poly
    # About 48 decimal digits: comfortably beyond the one final binary64
    # rounding, while all translation below remains exact rational arithmetic.
    from .sturm import refine
    tight = refine(root_poly, interval, Fraction(1, 2**160))
    b0 = tight.mid
    A0 = P.eval_at(m.alpha, b0)
    B0 = P.eval_at(m.beta, b0)
    a0 = B0 / A0

    As = _shift(m.alpha, b0)
    Bs = _shift(m.beta, b0)
    Aps = _shift(P.deriv(m.alpha), b0)
    Bps = _shift(P.deriv(m.beta), b0)

    # grad_a = 2[(a0+da) A(b0+db) - B(b0+db)]
    ga0 = list(_lincomb((2 * a0, As), (Fraction(-2), Bs)))
    ga1 = _lincomb((Fraction(2), As))
    # grad_b = -2(a0+da)B' + (a0+da)^2 A'
    gb0 = list(_lincomb((-2 * a0, Bps), (a0 * a0, Aps)))
    gb1 = _lincomb((Fraction(-2), Bps), (2 * a0, Aps))
    gb2 = _lincomb((Fraction(1), Aps))

    # The representative is within 2^-160 of the algebraic root.  Preserve
    # the exact critical identity rather than a meaningless tiny midpoint
    # residual.
    ga0[0] = Fraction(0)
    gb0[0] = Fraction(0)
    exact = ((tuple(ga0), ga1), (tuple(gb0), gb1, gb2))
    grad = tuple(tuple(tuple(float(c) for c in row) for row in component)
                 for component in exact)
    h01 = (exact[0][0][1] + exact[1][1][0]) / 2
    exact_H = ((exact[0][1][0], h01),
               (h01, exact[1][0][1]))
    H = tuple(tuple(float(x) for x in row) for row in exact_H)
    spectral = _exact_symmetric_spectral(exact_H)
    try:
        from . import _native
        native = _native.LocalKernel(grad)
    except (ImportError, AttributeError, ValueError):
        native = None
    base = LocalJet(
        float(a0), float(b0), tight, grad, exact, H, exact_H, spectral, native)
    if spectral.determinant < 0:
        base = replace(base, poincare=_poincare_candidates(base, nearby_b))
    return base


def build_stubs(m: Model, point, minima) -> tuple[InvariantStub, ...]:
    """Materialize and certify the four local invariant-manifold stubs."""
    local = point.local
    if local is None or not local.poincare:
        return ()
    spectral_abs = np.abs(np.asarray(local.spectral.eigenvalues, dtype=float))
    spectral_resolution_margin = float(
        np.min(spectral_abs)
        / max(np.finfo(float).eps*np.max(spectral_abs), 1e-300))
    stubs = []
    for chart in local.poincare:
        for orientation in (-1, 1):
            reach = chart.desired_reach
            reason = "no attempt"
            use_poincare = True

            def centered_graph(n, current_reach):
                lam = np.asarray(local.spectral.eigenvalues)
                if not (lam[0] < 0.0 < lam[1]):
                    raise FloatingPointError(
                        "FP64 cannot resolve the centered saddle "
                        f"linearization at b={point.b:.17g}; "
                        f"eigenvalues={tuple(map(float, lam))}")
                centered, diag = local.unstable_graph(
                    current_reach, n=n, sign=orientation,
                    manifold=chart.manifold)
                centered = np.asarray(centered)
                physical = np.vstack((
                    [local.a, local.b],
                    centered + (local.a, local.b)))
                R = np.asarray(chart.frame)
                diag.update({
                    "normal_x": centered @ R[:, 0],
                    "normal_h": centered @ R[:, 1],
                    "centered": centered,
                })
                return physical, diag

            def evaluate(current_reach, conditioned, coarse_n=257):
                fine_n = 2*coarse_n-1
                if conditioned:
                    coarse, dc = chart.graph(
                        local, orientation, n=coarse_n,
                        reach=current_reach)
                    fine, df = chart.graph(
                        local, orientation, n=fine_n,
                        reach=current_reach)
                else:
                    coarse, dc = centered_graph(coarse_n, current_reach)
                    fine, df = centered_graph(fine_n, current_reach)
                if not (np.all(np.isfinite(coarse))
                        and np.all(np.isfinite(fine))
                        and np.isfinite(dc["relative_change"])
                        and np.isfinite(df["relative_change"])):
                    raise FloatingPointError("nonfinite invariant graph")
                # Certify the graph before mapping it back into possibly
                # poorly scaled physical coordinates.  At a far translated
                # critical point, subtraction in binary64 can otherwise make
                # two identical normal-form graphs appear different.
                hc = np.asarray(dc["normal_h"])
                hf = np.asarray(df["normal_h"])
                coarse_change_error = (dc["relative_change"]
                                       * float(np.max(np.abs(hc)))
                                       / max(current_reach, 1e-300))
                fine_change_error = (df["relative_change"]
                                     * float(np.max(np.abs(hf)))
                                     / max(current_reach, 1e-300))
                normal_grid_absolute = float(np.max(np.abs(hc-hf[::2])))
                grid_error = normal_grid_absolute/max(
                    current_reach, 1e-300)
                bending_grid_error = normal_grid_absolute/max(
                    float(np.max(np.abs(hf))),
                    current_reach*np.finfo(float).eps, 1e-300)
                aligned = np.vstack((fine[0], fine[1::2]))
                physical_scale = max(
                    current_reach,
                    64*np.finfo(float).eps*np.hypot(local.a, local.b),
                    1e-300)
                physical_map_grid_error = float(np.max(np.hypot(
                    (coarse-aligned)[:, 0],
                    (coarse-aligned)[:, 1]))/physical_scale)
                endpoint = fine[-1]
                centered_fine = np.vstack((
                    np.zeros(2), np.asarray(df["centered"])))
                delta = centered_fine[-1]
                if conditioned and chart.normal_native is not None:
                    u = orientation*np.asarray(df["normal_x"])
                    s = np.asarray(df["normal_h"])
                    pmap, qmap = chart.selected_map
                    U = u + pmap[0]*u*u + pmap[1]*u*s + pmap[2]*s*s
                    S = s + qmap[0]*u*u + qmap[1]*u*s + qmap[2]*s*s
                    j00 = 1+2*pmap[0]*u+pmap[1]*s
                    j01 = pmap[1]*u+2*pmap[2]*s
                    j10 = 2*qmap[0]*u+qmap[1]*s
                    j11 = 1+qmap[1]*u+2*qmap[2]*s
                    if chart.conditioned_native is not None:
                        pulled_gradient = np.asarray(
                            chart.conditioned_native.gradient(
                                float(u[-1]), float(s[-1])))
                        # pulled = adj(J) F.  J*pulled = det(J)F;
                        # det(J)>0 in the certified chart, so this reconstructs
                        # the physical gradient direction without cancellation.
                        normal_gradient = np.array((
                            j00[-1]*pulled_gradient[0]
                            + j01[-1]*pulled_gradient[1],
                            j10[-1]*pulled_gradient[0]
                            + j11[-1]*pulled_gradient[1]))
                    else:
                        normal_gradient = np.asarray(
                            chart.normal_native.gradient(float(U[-1]),
                                                         float(S[-1])))
                    glocal = np.asarray(chart.frame) @ normal_gradient
                    jacobian_hadamard = np.abs(j00*j11-j01*j10) / np.maximum(
                        np.hypot(j00, j01)*np.hypot(j10, j11), 1e-300)
                    injectivity_margin = float(
                        np.min(jacobian_hadamard, initial=np.inf))
                else:
                    glocal = np.asarray(local.gradient(
                        float(delta[0]), float(delta[1])))
                    injectivity_margin = 1.0
                gglobal = np.asarray(m.gradL(
                    float(endpoint[0]), float(endpoint[1])))
                scale = max(np.hypot(*glocal), np.hypot(*gglobal),
                            1e-300)
                field_error = float(np.hypot(*(glocal-gglobal))/scale)
                field_absolute_error = float(np.hypot(*(glocal-gglobal)))
                norm_product = max(
                    np.hypot(*glocal)*np.hypot(*gglobal), 1e-300)
                cosine = float(np.dot(glocal, gglobal) / norm_product)
                field_direction_error = float(abs(
                    glocal[0]*gglobal[1]-glocal[1]*gglobal[0])
                    / norm_product)
                a_endpoint, b_endpoint = map(float, endpoint)
                eps = np.finfo(float).eps
                scale_a = 2.0*(abs(a_endpoint)*m.A(b_endpoint)
                               + abs(m.B(b_endpoint)))
                scale_b = (2.0*abs(a_endpoint)*abs(m.Bp(b_endpoint))
                           + a_endpoint*a_endpoint*abs(m.Ap(b_endpoint)))
                global_roundoff_floor = float(
                    16.0*eps*np.hypot(scale_a, scale_b))
                global_resolution_margin = float(
                    np.hypot(*gglobal)
                    / max(global_roundoff_floor, np.finfo(float).tiny))
                L = np.asarray([
                    local.potential(float(da), float(db))
                    for da, db in centered_fine])
                dL = np.diff(L)
                monotone_failures = int(np.count_nonzero(
                    dL > 1e-12*(1+np.abs(L[:-1]))
                    if chart.manifold == "unstable"
                    else dL < -1e-12*(1+np.abs(L[:-1]))))
                if conditioned and chart.conditioned_native is not None:
                    graph_u = orientation*np.asarray(df["normal_x"])
                    graph_s = np.asarray(df["normal_h"])
                    graph_slope = np.gradient(
                        graph_s, graph_u, edge_order=2)
                    graph_velocity = np.asarray([
                        chart.conditioned_native.gradient(float(uu), float(ss))
                        for uu, ss in zip(graph_u, graph_s)])
                    invariance_denominator = np.hypot(
                        1.0, graph_slope)*np.hypot(
                            graph_velocity[:, 0], graph_velocity[:, 1])
                    invariance_error = np.divide(
                        np.abs(graph_velocity[:, 1]
                               - graph_slope*graph_velocity[:, 0]),
                        invariance_denominator,
                        out=np.full_like(graph_slope, np.inf),
                        where=invariance_denominator > 1e-300)
                else:
                    graph_tangent = np.gradient(
                        np.asarray(df["centered"]), axis=0)
                    graph_velocity = np.asarray([
                        local.gradient(float(da), float(db))
                        for da, db in df["centered"]])
                    invariance_denominator = np.hypot(
                        graph_tangent[:, 0], graph_tangent[:, 1])*np.hypot(
                            graph_velocity[:, 0], graph_velocity[:, 1])
                    invariance_error = np.divide(
                        np.abs(graph_tangent[:, 0]*graph_velocity[:, 1]
                               - graph_tangent[:, 1]*graph_velocity[:, 0]),
                        invariance_denominator,
                        out=np.full(len(graph_tangent), np.inf),
                        where=invariance_denominator > 1e-300)
                invariance_direction_error = float(
                    np.max(invariance_error[8:], initial=0.0))
                accepted = (coarse_change_error < 1e-12
                            and fine_change_error < 1e-12
                            and grid_error < 1e-6
                            and invariance_direction_error < 1e-5
                            and monotone_failures == 0)
                global_ready = float(
                    np.all(np.isfinite(glocal))
                    and np.all(np.isfinite(gglobal))
                    and injectivity_margin > 1e-6
                    and global_resolution_margin >= 1024.0
                    # Continuation uses the geometric (normalized) field, so
                    # the handoff quantity is direction, not magnitude.
                    and field_direction_error <= max(
                        1e-10, 64.0/global_resolution_margin)
                    and cosine > 0.0)
                return {
                    "coarse": coarse, "fine": fine, "dc": dc, "df": df,
                    "centered_fine": centered_fine,
                    "coarse_change_error": coarse_change_error,
                    "fine_change_error": fine_change_error,
                    "grid_error": grid_error,
                    "bending_grid_error": bending_grid_error,
                    "physical_map_grid_error": physical_map_grid_error,
                    "glocal": glocal, "gglobal": gglobal,
                    "field_error": field_error,
                    "field_absolute_error": field_absolute_error,
                    "cosine": cosine,
                    "field_direction_error": field_direction_error,
                    "global_roundoff_floor": global_roundoff_floor,
                    "global_resolution_margin": global_resolution_margin,
                    "injectivity_margin": injectivity_margin,
                    "invariance_direction_error":
                        invariance_direction_error,
                    "monotone_failures": monotone_failures,
                    "accepted": accepted, "global_ready": global_ready,
                    "coarse_n": coarse_n,
                }

            def continuation_ready(candidate):
                """Whether the global scalar charts can safely own the tail."""
                if not candidate["global_ready"]:
                    return False
                if chart.manifold != "unstable":
                    return True
                spectral_stiffness = (
                    abs(chart.eigenvalues[1])
                    / max(abs(chart.eigenvalues[0]), 1e-300))
                if spectral_stiffness < 1e4:
                    # In ordinary deep water the global dispatcher is the
                    # better representation; extending a local graph merely
                    # duplicates a well-conditioned continuation.
                    return True
                a_endpoint, b_endpoint = map(
                    float, candidate["fine"][-1])
                w_endpoint = a_endpoint-m.a_star(b_endpoint)
                A = m.A(b_endpoint)
                asp = m.a_star_p(b_endpoint)
                Pv = (m.u_p(b_endpoint) + m.Ap(b_endpoint)*w_endpoint**2
                      - 2.0*A*w_endpoint*asp)
                numerator = abs(2.0*A*w_endpoint)+abs(asp*Pv)
                denominator = abs(2.0*A*w_endpoint-asp*Pv)
                gauge = numerator/max(
                    denominator, 1e-16*numerator+1e-300)
                candidate["handoff_depth_gauge"] = float(gauge)
                # "The global field is evaluable" is weaker than "an
                # implicit stage system selects the local-flow root."  Probe
                # the exact production GL step at the graph's own terminal
                # chord and require Armijo expected descent.  A failed probe
                # asks the graph transform to carry the branch farther.
                native = getattr(m, "_native_kernel", None)
                if native is None:
                    return gauge < 1e3
                chord = float(np.hypot(*(
                    candidate["fine"][-1]-candidate["fine"][-2])))
                g0 = np.asarray(m.gradL(a_endpoint, b_endpoint))
                L0 = float(m.L(a_endpoint, b_endpoint))
                slack = 64.0*np.finfo(float).eps*(1.0+abs(L0))
                for order in (6, 4):
                    try:
                        a1, b1 = native.normalized_step(
                            a_endpoint, b_endpoint, -chord, order)
                    except (ArithmeticError, ValueError,
                            FloatingPointError, OverflowError):
                        continue
                    if not (np.isfinite(a1) and np.isfinite(b1)):
                        continue
                    delta = np.array([a1-a_endpoint, b1-b_endpoint])
                    expected = float(g0@delta)
                    actual = float(m.L(a1, b1)-L0)
                    if expected < 0.0 and actual <= 1e-4*expected+slack:
                        candidate["handoff_probe_order"] = float(order)
                        return True
                candidate["handoff_probe_order"] = 0.0
                return False

            data = None
            for cuts in range(16):
                try:
                    data = evaluate(reach, use_poincare)
                except (ValueError, FloatingPointError) as exc:
                    if use_poincare:
                        # A near resonance or transformed-coordinate
                        # underflow may reject the selective normal form.
                        # The exact centered jet remains an admissible chart.
                        use_poincare = False
                        try:
                            data = evaluate(reach, False)
                        except (ValueError, FloatingPointError) as centered_exc:
                            reason = str(centered_exc)
                            reach *= 0.5
                            continue
                    else:
                        reason = str(exc)
                        reach *= 0.5
                        continue
                if data["accepted"]:
                    break
                reason = (
                    f"graph=({data['dc']['relative_change']:.3g},"
                    f"{data['df']['relative_change']:.3g}), "
                    f"grid={data['grid_error']:.3g}, "
                    f"field={data['field_error']:.3g}, "
                    f"invariance={data['invariance_direction_error']:.3g}, "
                    f"cosine={data['cosine']:.9g}, "
                    f"monotone={data['monotone_failures']}")
                reach *= 0.5
            else:
                # The critical point remains exact even when binary64 cannot
                # resolve any finite invariant-graph prefix.  Materialize an
                # explicit refusal stub rather than turning one pathological
                # saddle into an exception for the entire Morse skeleton.
                b_depart = orientation*np.asarray(chart.frame)[1, 0]
                b_direction = 1 if b_depart > 0 else -1
                refused = (
                    ("graph_certified", 0.0),
                    ("graph_residual_coarse",
                     float(data["dc"]["relative_change"])),
                    ("graph_residual_fine",
                     float(data["df"]["relative_change"])),
                    ("graph_change_error_coarse",
                     float(data["coarse_change_error"])),
                    ("graph_change_error_fine",
                     float(data["fine_change_error"])),
                    ("graph_iterations_coarse",
                     float(data["dc"]["iterations"])),
                    ("graph_iterations_fine",
                     float(data["df"]["iterations"])),
                    ("grid_error", float(data["grid_error"])),
                    ("bending_grid_error",
                     float(data["bending_grid_error"])),
                    ("physical_map_grid_error",
                     float(data["physical_map_grid_error"])),
                    ("poincare_conditioned", float(use_poincare)),
                    ("field_error", float(data["field_error"])),
                    ("field_absolute_error",
                     float(data["field_absolute_error"])),
                    ("field_cosine", float(data["cosine"])),
                    ("field_direction_error",
                     float(data["field_direction_error"])),
                    ("global_roundoff_floor",
                     float(data["global_roundoff_floor"])),
                    ("global_resolution_margin",
                     float(data["global_resolution_margin"])),
                    ("injectivity_margin",
                     float(data["injectivity_margin"])),
                    ("invariance_direction_error",
                     float(data["invariance_direction_error"])),
                    ("global_field_ready", 0.0),
                    ("spectral_resolution_margin",
                     spectral_resolution_margin),
                    ("fp64_spectral_resolved",
                     float(spectral_resolution_margin >= 64.0)),
                    ("monotone_failures",
                     float(data["monotone_failures"])),
                    ("reach", float(reach)),
                    ("reach_halvings", 16.0),
                    ("extension_steps", 0.0),
                    ("extension_grid_refinements", 0.0),
                    ("flow_extension_steps", 0.0),
                    ("flow_extension_arclength", 0.0),
                )
                stubs.append(InvariantStub(
                    chart.manifold, orientation, b_direction, None,
                    "undetermined" if chart.manifold == "unstable"
                    else "infinity",
                    ((local.a, local.b),), "conditioned_refusal", refused))
                continue

            # If the invariant graph is valid but the downstream divided ODE
            # is still in shallow water, keep advancing a certified graph.
            # The Poincare chart is tried first.  If its larger-domain fixed
            # point loses contraction, the exact centered polynomial is a
            # second representation of the same local flow and may still
            # carry the graph to deep water.
            extension_steps = 0
            grid_refinements = 0
            centered_extension = False
            for _ in range(16):
                if continuation_ready(data):
                    break
                candidate_reach = 2.0*reach
                candidate = None
                selected_conditioning = use_poincare
                modes = (use_poincare, False) if use_poincare else (False,)
                for conditioned in modes:
                    # A larger interval may need a finer representation even
                    # though its invariant graph remains perfectly regular.
                    for coarse_n in (257, 513, 1025, 2049):
                        try:
                            trial = evaluate(
                                candidate_reach, conditioned, coarse_n)
                        except (ValueError, FloatingPointError):
                            break
                        if trial["accepted"]:
                            candidate = trial
                            selected_conditioning = conditioned
                            grid_refinements += int(coarse_n > 257)
                            break
                    if candidate is not None:
                        break
                if candidate is None:
                    break
                reach = candidate_reach
                data = candidate
                if use_poincare and not selected_conditioning:
                    centered_extension = True
                use_poincare = selected_conditioning
                extension_steps += 1
            handoff_ready = continuation_ready(data)

            # If the graph loses its contraction before reaching a globally
            # resolved point, preserve the last certified prefix and refuse
            # the handoff.  A scalar IRK march of ds/du is not an acceptable
            # substitute here: the same extreme eigenvalue ratio which ends
            # the graph's contraction gives the stage equation several roots.
            flow_extension_steps = 0
            flow_extension_arclength = 0.0

            coarse, fine = data["coarse"], data["fine"]
            dc, df = data["dc"], data["df"]
            centered_fine = data["centered_fine"]
            coarse_change_error = data["coarse_change_error"]
            fine_change_error = data["fine_change_error"]
            grid_error = data["grid_error"]
            bending_grid_error = data["bending_grid_error"]
            physical_map_grid_error = data["physical_map_grid_error"]
            glocal, gglobal = data["glocal"], data["gglobal"]
            field_error = data["field_error"]
            field_absolute_error = data["field_absolute_error"]
            cosine = data["cosine"]
            field_direction_error = data["field_direction_error"]
            global_roundoff_floor = data["global_roundoff_floor"]
            global_resolution_margin = data["global_resolution_margin"]
            injectivity_margin = data["injectivity_margin"]
            invariance_direction_error = data[
                "invariance_direction_error"]
            global_ready = data["global_ready"]
            monotone_failures = data["monotone_failures"]

            b_depart = orientation*np.asarray(chart.frame)[1, 0]
            b_direction = 1 if b_depart > 0 else -1
            # A local departure direction does not determine the global
            # basin: a stable separatrix may cross the backbone away from a
            # critical point.  Connections are discovered by continuation.
            destination_b = None
            destination_kind = (
                "undetermined" if chart.manifold == "unstable" else "infinity")
            asp = -local.hessian[0][1]/local.hessian[0][0]
            vw = glocal[0]-asp*glocal[1]
            preferred = "slow" if abs(vw) <= 4*abs(glocal[1]) else "fast"
            certs = (
                ("graph_certified", 1.0),
                ("graph_residual_coarse", dc["relative_change"]),
                ("graph_residual_fine", df["relative_change"]),
                ("graph_change_error_coarse", coarse_change_error),
                ("graph_change_error_fine", fine_change_error),
                ("graph_iterations_coarse", float(dc["iterations"])),
                ("graph_iterations_fine", float(df["iterations"])),
                ("grid_error", grid_error),
                ("bending_grid_error", bending_grid_error),
                ("physical_map_grid_error", physical_map_grid_error),
                ("poincare_conditioned", float(use_poincare)),
                ("field_error", field_error),
                ("field_absolute_error", field_absolute_error),
                ("field_cosine", cosine),
                ("field_direction_error", field_direction_error),
                ("global_roundoff_floor", global_roundoff_floor),
                ("global_resolution_margin", global_resolution_margin),
                ("injectivity_margin", injectivity_margin),
                ("invariance_direction_error",
                 invariance_direction_error),
                ("global_field_ready", global_ready),
                ("spectral_resolution_margin",
                 spectral_resolution_margin),
                ("fp64_spectral_resolved",
                 float(spectral_resolution_margin >= 64.0)),
                ("monotone_failures", float(monotone_failures)),
                ("reach", float(reach)),
                ("reach_halvings", float(cuts)),
                ("extension_steps", float(extension_steps)),
                ("extension_grid_refinements", float(grid_refinements)),
                ("centered_extension", float(centered_extension)),
                ("handoff_depth_gauge",
                 float(data.get("handoff_depth_gauge", np.inf))),
                ("handoff_probe_order",
                 float(data.get("handoff_probe_order", 0.0))),
                ("continuation_ready", float(handoff_ready)),
                ("flow_extension_steps", float(flow_extension_steps)),
                ("flow_extension_arclength",
                 float(flow_extension_arclength)),
            )
            stubs.append(InvariantStub(
                chart.manifold, orientation, b_direction, destination_b,
                destination_kind, tuple(map(tuple, fine)), preferred, certs))
    return tuple(stubs)
