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


def run_sgd(grad, z0, lr, n_steps, box=None):
    z = np.array(z0, dtype=float)
    traj = [z.copy()]
    for _ in range(n_steps):
        z = z - lr * grad(z[0], z[1])
        traj.append(z.copy())
        if not np.all(np.isfinite(z)):
            break
        if box and not (box[0] <= z[0] <= box[1] and box[2] <= z[1] <= box[3]):
            break
    return np.array(traj)


def run_adam(grad, z0, lr, n_steps, box=None, b1=0.9, b2=0.999, eps=1e-8):
    z = np.array(z0, dtype=float)
    mom = np.zeros(2)
    vel = np.zeros(2)
    traj = [z.copy()]
    for t in range(1, n_steps + 1):
        g = grad(z[0], z[1])
        mom = b1 * mom + (1 - b1) * g
        vel = b2 * vel + (1 - b2) * g * g
        mh = mom / (1 - b1**t)
        vh = vel / (1 - b2**t)
        z = z - lr * mh / (np.sqrt(vh) + eps)
        traj.append(z.copy())
        if not np.all(np.isfinite(z)):
            break
        if box and not (box[0] <= z[0] <= box[1] and box[2] <= z[1] <= box[3]):
            break
    return np.array(traj)


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
