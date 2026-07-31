"""Demo-grade optimizers — consumers of the portrait, never library code.

The one-neuron net sees data only through moments, so a BATCH is a point
in moment space and SGD's noise is literally moment-space jitter: the
stochastic gradient below is computed from raw samples x_i ~ U(0,1),

    L_batch(a,b) = (1/n) Σ (f(x_i) − a·g(b·x_i))²
    ∂a = −(2/n) Σ (f(x_i) − a·g(b x_i))·g(b x_i)
    ∂b = −(2/n) Σ (f(x_i) − a·g(b x_i))·a·g′(b x_i)·x_i

which is the honest SGD of the network.  The full-batch gradient is the
model's ∇L (Ljung's mean-field ODE is the portrait itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _polyval(coeffs, x):
    acc = np.zeros_like(x, dtype=float)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


class BatchGradient:
    """Stochastic gradient from raw U(0,1) samples (demo sampler)."""

    def __init__(self, f_coeffs, g_coeffs, batch_size, rng):
        self.f = [float(c) for c in f_coeffs]
        self.g = [float(c) for c in g_coeffs]
        self.gp = [k * self.g[k] for k in range(1, len(self.g))]
        self.n = batch_size
        self.rng = rng

    def __call__(self, a, b):
        x = self.rng.random(self.n)
        gv = _polyval(self.g, b * x)
        res = _polyval(self.f, x) - a * gv
        da = -2.0 * np.mean(res * gv)
        db = -2.0 * np.mean(res * a * _polyval(self.gp, b * x) * x)
        return np.array([da, db])


def cosine_schedule(base_lr, total_steps, warmup_fraction=0.05,
                    final_fraction=0.01):
    """Linear warmup followed by cosine decay."""
    warmup = max(1, int(round(total_steps*warmup_fraction)))

    def schedule(t):
        if t <= warmup:
            return float(base_lr)*t/warmup
        u = min(1.0, (t-warmup)/max(1, total_steps-warmup))
        multiplier = final_fraction+(1-final_fraction)*0.5*(
            1+np.cos(np.pi*u))
        return float(base_lr)*float(multiplier)

    return schedule


def inverse_sqrt_schedule(base_lr, warmup_steps=100):
    """Linear warmup followed by the conventional inverse-square-root tail."""
    warmup_steps = max(1, int(warmup_steps))

    def schedule(t):
        if t <= warmup_steps:
            return float(base_lr)*t/warmup_steps
        return float(base_lr)*np.sqrt(warmup_steps/t)

    return schedule


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


def run_state(state, grad, n_steps, box=None):
    """Advance a resumable optimizer state; Thompson scheduling uses one step."""
    trajectory = [state.z.copy()]
    for _ in range(n_steps):
        gradient = np.asarray(grad(*state.z), dtype=float)
        z = state.step(gradient)
        trajectory.append(z.copy())
        if not np.all(np.isfinite(z)):
            break
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
