"""Demo-grade optimizers — consumers of the portrait, never library code.

The one-neuron net sees data only through moments, so a BATCH is a point
in moment space and SGD's noise is literally moment-space jitter: the
stochastic gradient below is computed from raw samples x_i drawn from the
same uniform, normal, or finite empirical law as the portrait,

    L_batch(a,b) = (1/n) Σ (f(x_i) − a·g(b·x_i))²
    ∂a = −(2/n) Σ (f(x_i) − a·g(b x_i))·g(b x_i)
    ∂b = −(2/n) Σ (f(x_i) − a·g(b x_i))·a·g′(b x_i)·x_i

which is the honest SGD of the network.  The full-batch gradient is the
model's ∇L (Ljung's mean-field ODE is the portrait itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

DEFAULT_FISHER_DAMPING = 1e-3


def _polyval(coeffs, x):
    acc = np.zeros_like(x, dtype=float)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


class BatchGradient:
    """Stochastic gradient from raw samples of a supported portrait law.

    ``uniform01`` remains the default for compatibility with the original
    moustache galleries.  ``normal01`` draws fresh standard-normal samples;
    ``absolute_x`` draws from density |x| on [-1,1];
    ``empirical`` resamples the exact finite support used to build an
    interactive empirical portrait.
    """

    def __init__(self, f_coeffs, g_coeffs, batch_size, rng,
                 distribution="uniform01", samples=None):
        self.f = [float(c) for c in f_coeffs]
        self.g = [float(c) for c in g_coeffs]
        self.gp = [k * self.g[k] for k in range(1, len(self.g))]
        self.n = int(batch_size)
        if self.n <= 0:
            raise ValueError("batch_size must be positive")
        self.rng = rng
        self.distribution = str(distribution)
        self.samples = (None if samples is None
                        else np.asarray(samples, dtype=float))
        if self.distribution not in {"uniform01", "normal01", "absolute_x", "empirical"}:
            raise ValueError(
                f"unsupported minibatch distribution {self.distribution!r}")
        if self.distribution == "empirical" and (
                self.samples is None or self.samples.size == 0):
            raise ValueError("empirical minibatches need at least one sample")

    def _draw(self):
        if self.distribution == "uniform01":
            return self.rng.random(self.n)
        if self.distribution == "normal01":
            return self.rng.normal(size=self.n)
        if self.distribution == "absolute_x":
            # Inverse CDF of the normalized density |x| on [-1,1].
            y = 2.0 * self.rng.random(self.n) - 1.0
            return np.copysign(np.sqrt(np.abs(y)), y)
        return self.rng.choice(self.samples, size=self.n, replace=True)

    def _draw_many(self, steps):
        """Draw consecutive batches in one array, preserving stream order."""
        shape = (int(steps), self.n)
        if self.distribution == "uniform01":
            return self.rng.random(shape)
        if self.distribution == "normal01":
            return self.rng.normal(size=shape)
        if self.distribution == "absolute_x":
            y = 2.0 * self.rng.random(shape) - 1.0
            return np.copysign(np.sqrt(np.abs(y)), y)
        return self.rng.choice(self.samples, size=shape, replace=True)

    def __call__(self, a, b):
        x = self._draw()
        # A divergent demo optimizer is an ordinary experimental outcome,
        # not a reason to flood the terminal with one warning per minibatch.
        # Non-finite results remain visible to ``checked_step`` below, which
        # terminates the trajectory at its last finite iterate.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bx = b * x
            gv = _polyval(self.g, bx)
            res = _polyval(self.f, x) - a * gv
            da = -2.0 * np.mean(res * gv)
            db = -2.0 * np.mean(
                res * a * _polyval(self.gp, bx) * x)
        return np.array([da, db])


@dataclass
class FisherGNState:
    """Damped two-coordinate empirical Fisher/Gauss--Newton descent.

    Each step draws one batch, forms J=[g(bx), a*x*g'(bx)], and backtracks
    against that same batch.  The relative diagonal damping makes the 2x2
    solve meaningful when the two parameter sensitivities differ greatly.
    A batch evaluation counts one pass over the drawn samples, whether it
    computes derivatives or only the candidate loss.
    """

    z: np.ndarray
    damping: float = DEFAULT_FISHER_DAMPING
    armijo: float = 1e-4
    max_halvings: int = 60
    batch_evaluations: int = 0

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.z.shape != (2,):
            raise ValueError("Fisher-GN needs two parameters")
        if not np.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("Fisher-GN damping must be positive and finite")

    def step_batch(self, oracle):
        before = self.z.copy()
        x = oracle._draw()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bx = before[1]*x
            gv = _polyval(oracle.g, bx)
            residual = _polyval(oracle.f, x)-before[0]*gv
            jac = np.column_stack((gv, before[0]*x*_polyval(oracle.gp, bx)))
            loss = float(np.mean(residual*residual))
            grad = -2.0*np.mean(residual[:, None]*jac, axis=0)
            metric = jac.T@jac/len(x)
        self.batch_evaluations += 1
        if not (np.isfinite(loss) and np.all(np.isfinite(grad))
                and np.all(np.isfinite(metric))):
            return before, "nonfinite_fisher"
        scale = np.sqrt(np.maximum(np.diag(metric), np.finfo(float).tiny))
        correlation = metric/np.outer(scale, scale)
        try:
            direction = np.linalg.solve(
                correlation+self.damping*np.eye(2), -grad/scale)/scale
        except np.linalg.LinAlgError:
            return before, "singular_fisher"
        slope = float(grad@direction)
        if not np.all(np.isfinite(direction)) or not np.isfinite(slope):
            return before, "nonfinite_direction"
        if slope >= 0:
            return before, "non_descent_direction"
        step = 1.0
        for _ in range(self.max_halvings):
            candidate = before+step*direction
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                value = _polyval(oracle.f, x)-candidate[0]*_polyval(
                    oracle.g, candidate[1]*x)
                candidate_loss = float(np.mean(value*value))
            self.batch_evaluations += 1
            if (np.all(np.isfinite(candidate)) and np.isfinite(candidate_loss)
                    and candidate_loss <= loss+self.armijo*step*slope):
                self.z = candidate
                return self.z, None
            step *= 0.5
        return before, "line_search_failure"


@dataclass
class ExactFisherGNState:
    """Population Fisher/Gauss--Newton step from the model's exact moments.

    This is deliberately a full-information comparator: unlike the minibatch
    variant, neither its direction nor its line search consumes sampled data.
    The line search evaluates the rational polynomial at the actual binary64
    candidate coordinates.  This prevents rounded-equal losses or no-op
    coordinates from passing Armijo near a numerical minimum.
    ``objective_evaluations`` counts these exact loss evaluations.
    """

    z: np.ndarray
    model: object
    damping: float = DEFAULT_FISHER_DAMPING
    armijo: float = 1e-4
    max_halvings: int = 60
    objective_evaluations: int = 0
    trial_step: float = 1.0
    known_minima: tuple[float, ...] | None = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.z.shape != (2,):
            raise ValueError("exact Fisher-GN needs two parameters")
        if not np.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("exact Fisher-GN damping must be positive and finite")
        if not np.isfinite(self.trial_step) or not 0 < self.trial_step <= 1:
            raise ValueError("exact Fisher-GN trial step must lie in (0, 1]")
        if not np.isfinite(self.armijo) or not 0 < self.armijo < 1:
            raise ValueError("exact Fisher-GN Armijo factor must lie in (0, 1)")
        # E[(X g'(bX))²] = Σ_kl kl g_k g_l mu_{k+l} b^{k+l-2}.
        g = self.model.g
        coefficients = np.zeros(max(1, 2*len(g)-3), dtype=float)
        for k in range(1, len(g)):
            for l in range(1, len(g)):
                coefficients[k+l-2] += (
                    k*l*float(g[k])*float(g[l])*float(self.model.mu[k+l]))
        self._fisher_bb = coefficients
        self._exact_A = tuple(self.model.alpha)
        self._exact_B = tuple(self.model.beta)
        self._exact_Ap = tuple(k*c for k, c in enumerate(self._exact_A[1:], 1))
        self._exact_Bp = tuple(k*c for k, c in enumerate(self._exact_B[1:], 1))
        self._exact_App = tuple(k*c for k, c in enumerate(self._exact_Ap[1:], 1))
        self._exact_Bpp = tuple(k*c for k, c in enumerate(self._exact_Bp[1:], 1))

    @staticmethod
    def _exact_polyval(coefficients, value):
        result = Fraction(0)
        for coefficient in reversed(coefficients):
            result = result*value+coefficient
        return result

    def _exact_values(self, a, b, *, gradient=False):
        a = Fraction.from_float(float(a))
        b = Fraction.from_float(float(b))
        A = self._exact_polyval(self._exact_A, b)
        B = self._exact_polyval(self._exact_B, b)
        loss = self.model.C-2*a*B+a*a*A
        if not gradient:
            return loss
        Ap = self._exact_polyval(self._exact_Ap, b)
        Bp = self._exact_polyval(self._exact_Bp, b)
        return loss, (2*(a*A-B), a*a*Ap-2*a*Bp)

    def at_local_minimum(self, relative_radius=1e-8):
        """Check positive curvature and a small exact Newton correction.

        The test uses rational derivatives at the stored binary64 point.
        A flat far-field arm has a large Newton correction and is not retired
        merely because its gradient happens to be tiny.
        """
        a, b = map(lambda x: Fraction.from_float(float(x)), self.z)
        _, (ga, gb) = self._exact_values(*self.z, gradient=True)
        A = self._exact_polyval(self._exact_A, b)
        Ap = self._exact_polyval(self._exact_Ap, b)
        Bp = self._exact_polyval(self._exact_Bp, b)
        App = self._exact_polyval(self._exact_App, b)
        Bpp = self._exact_polyval(self._exact_Bpp, b)
        h11 = 2*A
        h12 = 2*a*Ap-2*Bp
        h22 = a*a*App-2*a*Bpp
        determinant = h11*h22-h12*h12
        if h11 <= 0 or determinant <= 0:
            return False
        da = (h22*ga-h12*gb)/determinant
        db = (h11*gb-h12*ga)/determinant
        radius = Fraction.from_float(float(relative_radius))
        close = (abs(da) <= radius*max(1, abs(a))
                 and abs(db) <= radius*max(1, abs(b)))
        if not close or self.known_minima is None:
            return close
        return any(abs(float(b)-root_b) <= 1e-6*max(1, abs(float(b)))
                   for root_b in self.known_minima)

    def step_batch(self, _oracle=None):
        before = self.z.copy()
        a, b = before
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            metric = np.array([
                [self.model.A(b), 0.5*a*self.model.Ap(b)],
                [0.5*a*self.model.Ap(b),
                 a*a*np.polynomial.polynomial.polyval(b, self._fisher_bb)]])
        self.objective_evaluations += 1
        loss, exact_grad = self._exact_values(a, b, gradient=True)
        try:
            grad = np.asarray([float(value) for value in exact_grad])
        except OverflowError:
            return before, "nonfinite_fisher"
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(metric))):
            return before, "nonfinite_fisher"
        scale = np.sqrt(np.maximum(np.diag(metric), np.finfo(float).tiny))
        correlation = metric/np.outer(scale, scale)
        try:
            direction = np.linalg.solve(
                correlation+self.damping*np.eye(2), -grad/scale)/scale
        except np.linalg.LinAlgError:
            return before, "singular_fisher"
        if not np.all(np.isfinite(direction)):
            return before, "nonfinite_direction"
        exact_directional_slope = sum(
            g*Fraction.from_float(float(d))
            for g, d in zip(exact_grad, direction))
        if exact_directional_slope >= 0:
            return before, "non_descent_direction"
        best = None
        armijo_factor = Fraction.from_float(self.armijo)

        def evaluate(step):
            candidate = before+step*direction
            if not np.all(np.isfinite(candidate)) or np.array_equal(candidate, before):
                return None
            candidate_loss = self._exact_values(*candidate)
            self.objective_evaluations += 1
            displacement = tuple(
                Fraction.from_float(float(new))-Fraction.from_float(float(old))
                for old, new in zip(before, candidate))
            actual_slope = sum(g*d for g, d in zip(exact_grad, displacement))
            armijo = (candidate_loss < loss and actual_slope < 0
                      and candidate_loss <= loss+armijo_factor*actual_slope)
            return candidate_loss, candidate, armijo

        def accept(record, step):
            self.z = record[1]
            self.trial_step = min(1.0, 2.0*step)
            return self.z, None

        # A nonconvex ray can have several stationary points.  First find a
        # decreasing dyadic step, then compare its immediate neighbours.  In
        # particular, do not interpolate between zero and an overshooting
        # unit step: that can select a shallow local minimum while the half
        # step crosses into a much better basin.
        checked = {}
        step = self.trial_step
        for _ in range(self.max_halvings):
            record = evaluate(step)
            checked[step] = record
            if record is not None and record[2]:
                break
            step *= 0.5
        else:
            return before, "line_search_precision_limit"
        for neighbour in (step*0.5, min(1.0, step*2.0)):
            if neighbour not in checked:
                checked[neighbour] = evaluate(neighbour)
        for trial, record in checked.items():
            if (record is not None and record[2]
                    and (best is None or record[0] < best[0][0])):
                best = (record, trial)

        if best is not None:
            return accept(*best)
        return before, "line_search_precision_limit"


def cosine_schedule(base_lr, total_steps, warmup_fraction=0.05,
                    final_fraction=0.01):
    """Linear warmup followed by cosine decay."""
    return CosineSchedule(float(base_lr), int(total_steps),
                          float(warmup_fraction), float(final_fraction))


def inverse_sqrt_schedule(base_lr, warmup_steps=100):
    """Linear warmup followed by the conventional inverse-square-root tail."""
    return InverseSqrtSchedule(float(base_lr), max(1, int(warmup_steps)))


@dataclass(frozen=True)
class CosineSchedule:
    base_lr: float
    total_steps: int
    warmup_fraction: float
    final_fraction: float

    def __call__(self, t):
        warmup = max(1, int(round(self.total_steps*self.warmup_fraction)))
        if t <= warmup:
            return self.base_lr*t/warmup
        u = min(1.0, (t-warmup)/max(1, self.total_steps-warmup))
        multiplier = self.final_fraction+(1-self.final_fraction)*0.5*(
            1+np.cos(np.pi*u))
        return self.base_lr*float(multiplier)


@dataclass(frozen=True)
class InverseSqrtSchedule:
    base_lr: float
    warmup_steps: int

    def __call__(self, t):
        if t <= self.warmup_steps:
            return self.base_lr*t/self.warmup_steps
        return self.base_lr*np.sqrt(self.warmup_steps/t)


def _lr_at(lr, t):
    return float(lr(t) if callable(lr) else lr)


@dataclass
class SGDState:
    z: np.ndarray
    lr: object
    momentum: float = 0.0
    nesterov: bool = False
    velocity: np.ndarray | None = None
    t: int = 0

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.velocity is None:
            self.velocity = np.zeros_like(self.z)

    def step(self, gradient):
        self.t += 1
        self.velocity = self.momentum*self.velocity+gradient
        direction = (gradient+self.momentum*self.velocity
                     if self.nesterov else self.velocity)
        self.z = self.z-_lr_at(self.lr, self.t)*direction
        return self.z


@dataclass
class AdamState:
    z: np.ndarray
    lr: object
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    first: np.ndarray | None = None
    second: np.ndarray | None = None
    t: int = 0

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.first is None:
            self.first = np.zeros_like(self.z)
        if self.second is None:
            self.second = np.zeros_like(self.z)

    def step(self, gradient):
        self.t += 1
        self.first = self.b1*self.first+(1-self.b1)*gradient
        self.second = self.b2*self.second+(1-self.b2)*gradient*gradient
        mh = self.first/(1-self.b1**self.t)
        vh = self.second/(1-self.b2**self.t)
        lr = _lr_at(self.lr, self.t)
        self.z = ((1-lr*self.weight_decay)*self.z
                  - lr*mh/(np.sqrt(vh)+self.eps))
        return self.z


@dataclass
class VectorMuonState:
    """The 2x1 polar-factor surrogate, not matrix-parameter Muon.

    For the scalar-neuron state (a,b), the polar factor of the 2x1 momentum
    matrix is just its Euclidean normalization.  Real Muon normally routes
    vector/scalar parameter groups to AdamW; the explicit name prevents this
    pedagogical surrogate from being mistaken for transformer-style Muon.
    """

    z: np.ndarray
    lr: object
    momentum: float = 0.95
    nesterov: bool = True
    buffer: np.ndarray | None = None
    eps: float = 1e-12
    t: int = 0

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.buffer is None:
            self.buffer = np.zeros_like(self.z)

    def step(self, gradient):
        self.t += 1
        self.buffer = self.momentum*self.buffer+(1-self.momentum)*gradient
        update = (gradient+self.momentum*self.buffer
                  if self.nesterov else self.buffer)
        norm = float(np.hypot(*update))
        if norm > self.eps:
            update = update/norm
        self.z = self.z-_lr_at(self.lr, self.t)*update
        return self.z


def checked_step(state, grad):
    """Advance once, or preserve the last finite iterate on FP64 failure.

    Returns ``(z, None)`` on success and ``(z, reason)`` on failure.  The
    optimizer's auxiliary accumulators may already be non-finite after a
    failed update, so callers must regard any failure as terminal.
    """
    before = np.asarray(state.z, dtype=float).copy()
    if not np.all(np.isfinite(before)):
        return before, "nonfinite_state"
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gradient = np.asarray(grad(*before), dtype=float)
    except ArithmeticError:
        return before, "gradient_failure"
    if gradient.shape != before.shape:
        raise ValueError(
            f"gradient shape {gradient.shape} does not match {before.shape}")
    if not np.all(np.isfinite(gradient)):
        return before, "nonfinite_gradient"
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = np.asarray(state.step(gradient), dtype=float)
    except ArithmeticError:
        state.z = before
        return before, "step_failure"
    if z.shape != before.shape:
        raise ValueError(
            f"optimizer state shape {z.shape} does not match {before.shape}")
    if not np.all(np.isfinite(z)):
        state.z = before
        return before, "nonfinite_step"
    return z, None


def run_state(state, grad, n_steps, box=None):
    """Advance a resumable optimizer state; Thompson scheduling uses one step."""
    trajectory = [state.z.copy()]
    for _ in range(n_steps):
        z, failure = checked_step(state, grad)
        if failure is not None:
            break
        trajectory.append(z.copy())
        if box and not (
                box[0] <= z[0] <= box[1] and box[2] <= z[1] <= box[3]):
            break
    return np.asarray(trajectory)


def run_sgd(grad, z0, lr, n_steps, box=None, momentum=0.0,
            nesterov=False):
    state = SGDState(np.asarray(z0), lr, momentum=momentum,
                     nesterov=nesterov)
    return run_state(state, grad, n_steps, box=box)


def run_adam(grad, z0, lr, n_steps, box=None, b1=0.9, b2=0.999, eps=1e-8):
    state = AdamState(np.asarray(z0), lr, b1=b1, b2=b2, eps=eps)
    return run_state(state, grad, n_steps, box=box)


def run_adamw(grad, z0, lr, n_steps, box=None, b1=0.9, b2=0.95,
              eps=1e-10, weight_decay=0.0):
    state = AdamState(
        np.asarray(z0), lr, b1=b1, b2=b2, eps=eps,
        weight_decay=weight_decay)
    return run_state(state, grad, n_steps, box=box)


def run_vector_muon(grad, z0, lr, n_steps, box=None, momentum=0.95,
                    nesterov=True):
    state = VectorMuonState(
        np.asarray(z0), lr, momentum=momentum, nesterov=nesterov)
    return run_state(state, grad, n_steps, box=box)


def run_lbfgs(m, z0, n_steps, box=None, mem=8):
    """Full-batch L-BFGS (two-loop recursion, Armijo backtracking) on the
    model's exact loss — the classical deterministic contender."""
    z = np.array(z0, dtype=float)
    traj = [z.copy()]
    S, Yv = [], []
    g = m.gradL(z[0], z[1])
    for _ in range(n_steps):
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Yv)):
            rho = 1.0 / max(float(y @ s), 1e-300)
            al = rho * float(s @ q)
            q = q - al * y
            alphas.append((al, rho, s, y))
        if Yv:
            y_last, s_last = Yv[-1], S[-1]
            q = q * float(s_last @ y_last) / max(float(y_last @ y_last),
                                                 1e-300)
        for al, rho, s, y in reversed(alphas):
            beta = rho * float(y @ q)
            q = q + (al - beta) * s
        p = -q                                   # descent direction
        if float(p @ g) > 0:
            p = -g                               # safeguard
        # Armijo backtracking
        L0 = m.L(z[0], z[1])
        t = 1.0
        for _bt in range(40):
            z_new = z + t * p
            if np.all(np.isfinite(z_new)) and \
                    m.L(z_new[0], z_new[1]) <= L0 + 1e-4 * t * float(g @ p):
                break
            t *= 0.5
        z_new = z + t * p
        g_new = m.gradL(z_new[0], z_new[1])
        S.append(z_new - z)
        Yv.append(g_new - g)
        if len(S) > mem:
            S.pop(0)
            Yv.pop(0)
        z, g = z_new, g_new
        traj.append(z.copy())
        if box and not (box[0] <= z[0] <= box[1] and box[2] <= z[1] <= box[3]):
            break
        if float(np.hypot(*g)) < 1e-12:
            break
    return np.array(traj)


def nearest_critical(enumeration, z):
    pts = enumeration.points
    d = [float(np.hypot(p.a - z[0], p.b - z[1])) for p in pts]
    k = int(np.argmin(d))
    return pts[k], d[k]
