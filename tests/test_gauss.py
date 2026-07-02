"""Phase-2 gates for spong.gauss (SPONG_FOUNDING Part IV, phase 2).

Gates: order/convergence; reversal gap ~ roundoff on reversible spans;
zero secular L-drift on skew-flow circuits; richardson3 stop matches the
requested tolerance; event location on the collocation polynomial.
"""

import numpy as np
import pytest

from spong import gauss, model


# ------------------------- order / convergence ------------------------- #

def _endpoint_error(method, n):
    # generic nonautonomous linear: y' = cos(x) y, exact y = exp(sin x).
    # (The Riccati y' = y^2 is a bad order probe: Gauss stability functions
    # are Pade rationals and its Moebius flow cancels the h^4 constant.)
    F = lambda x, y: np.cos(x) * y
    J = lambda x, y: np.array([[np.cos(x)]])
    t = gauss.solve(F, 0.0, 1.0, [1.0], n, method=method, jac=J)
    return abs(t.y_end[0] - np.exp(np.sin(1.0)))


@pytest.mark.parametrize("method,order,ns", [("imm", 2, (40, 80)),
                                             ("gl4", 4, (5, 10))])
def test_convergence_order(method, order, ns):
    e1, e2 = (_endpoint_error(method, n) for n in ns)
    assert e2 > 1e-12          # off the Newton-tolerance floor
    rate = np.log2(e1 / e2)
    assert rate == pytest.approx(order, abs=0.35)


def test_gl4_beats_imm():
    assert _endpoint_error("gl4", 40) < 1e-4 * _endpoint_error("imm", 40)


# ------------------------------ anadromy ------------------------------ #

@pytest.mark.parametrize("method", ["imm", "gl4"])
def test_reversal_gap_roundoff(method):
    # nonlinear 2D system (Volterra-ish), generic non-reversible field:
    # the METHOD is symmetric, so back-integration returns to start at
    # Newton-tolerance + roundoff — orders below the local error.
    F = lambda x, y: np.array([y[0] * (1 - y[1]), y[1] * (y[0] - 1)])
    J = lambda x, y: np.array([[1 - y[1], -y[0]], [y[1], y[0] - 1]])
    gap = gauss.reversal_gap(F, 0.0, 5.0, [1.2, 0.8], 200,
                             method=method, jac=J)
    assert gap < 1e-10
    # and the forward endpoint is NOT the start (the test is not vacuous)
    t = gauss.solve(F, 0.0, 5.0, [1.2, 0.8], 200, method=method, jac=J)
    assert np.max(np.abs(t.y_end - np.array([1.2, 0.8]))) > 1e-3


# ------------------ symplectic: skew flow conserves L ------------------ #

@pytest.fixture(scope="module")
def m():
    return model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))


def _skew(m):
    # Hamiltonian flow with H = L: (da, db)/dt = (−L_b, L_a)
    def F(x, y):
        g = m.gradL(y[0], y[1])
        return np.array([-g[1], g[0]])

    def J(x, y):
        H = m.hessL(y[0], y[1])
        return np.array([[-H[1, 0], -H[1, 1]], [H[0, 0], H[0, 1]]])

    return F, J


@pytest.mark.parametrize("method,band", [("imm", 5e-4), ("gl4", 5e-8)])
def test_skew_flow_no_secular_drift(m, method, band):
    """Level curves are Hamiltonian orbits (H = L); the Gauss pair is
    symplectic, so L oscillates in a bounded band with NO secular trend."""
    F, J = _skew(m)
    y0 = np.array([m.a_star(1.0) + 0.4, 1.0])       # near the (1,1) minimum
    L0 = m.L(*y0)
    t = gauss.solve(F, 0.0, 200.0, y0, 20000, method=method, jac=J)
    L = np.array([m.L(a, b) for a, b in t.ys])
    dev = L - L0
    assert np.max(np.abs(dev)) < band * max(1.0, abs(L0))
    # no secular trend: drift of the mean between halves << oscillation band
    half = len(dev) // 2
    trend = abs(np.mean(dev[half:]) - np.mean(dev[:half]))
    assert trend < 0.05 * (np.max(np.abs(dev)) + 1e-15)


# ----------------------------- richardson3 ----------------------------- #

def test_richardson3_accelerates_geometric():
    # x_k = L + c r^k  ->  Aitken recovers L exactly (up to roundoff)
    L, c, r = 3.7, 2.0, 0.6
    x2, x1, x0 = (L + c * r**k for k in (2, 1, 0))
    y = gauss.richardson3(x2, x1, x0)
    assert y == pytest.approx(L, abs=1e-12)


def test_richardson3_passthrough_when_converged():
    y = gauss.richardson3(1.0, 1.0, 1.0)
    assert y == 1.0                    # converged mask: no 0/0 division


def test_solve_richardson_meets_tolerance():
    F = lambda x, y: -y
    J = lambda x, y: np.array([[-1.0]])
    for tol in (1e-6, 1e-10):
        r = gauss.solve_richardson(F, 0.0, 1.0, [1.0], tol,
                                   method="imm", jac=J)
        assert r.converged
        assert abs(r.y_end[0] - np.exp(-1.0)) < 10 * tol


# --------------------------- event location --------------------------- #

def test_event_on_collocation_polynomial():
    F = lambda x, y: -y
    J = lambda x, y: np.array([[-1.0]])
    # between-node collocation accuracy is O(h^3) for GL2: n = 200 over
    # [0, 2] gives h = 0.01, h^3 = 1e-6 -> locate to ~1e-6.
    t = gauss.solve(F, 0.0, 2.0, [1.0], 200, method="gl4", jac=J)
    hit = gauss.find_event(t, lambda x, y: y[0] - 0.6)
    assert hit is not None
    x_star, y_star = hit
    assert x_star == pytest.approx(-np.log(0.6), abs=1e-5)
    assert y_star[0] == pytest.approx(0.6, abs=1e-5)


def test_dense_output_accuracy():
    F = lambda x, y: -y
    J = lambda x, y: np.array([[-1.0]])
    t = gauss.solve(F, 0.0, 1.0, [1.0], 20, method="gl4", jac=J)
    st = t.steps[7]
    for th in (0.25, 0.5, 0.75):     # between nodes: O(h^3), h = 0.05
        x = st.x + th * st.h
        assert st.dense(th)[0] == pytest.approx(np.exp(-x), abs=2e-5)
    # at the step endpoint the collocation polynomial IS the step: h^4-tight
    assert st.dense(1.0)[0] == pytest.approx(st.y1[0], abs=1e-14)
