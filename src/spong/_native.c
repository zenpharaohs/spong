#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "spong/spong_resolution.h"
#include "spong/spong_exact.h"
#include "spong/spong_topology.h"

typedef struct {
    PyObject_HEAD
    Py_ssize_t na, nap, napp, nb, nbp, nbpp, nn, nnp;
    double *a, *ap, *app, *b, *bp, *bpp, *n, *np;
} Kernel;

typedef struct {
    PyObject_HEAD
    double **c[2];
    Py_ssize_t *n[2];
    int rows[2];
} LocalKernel;

typedef struct {
    PyObject_HEAD
    spong_sturm_plan *plan;
    spong_sturm_analysis analysis;
} NativeSturmPlan;

typedef struct {
    PyObject_HEAD
    Py_buffer first;
    Py_buffer second;
    spong_contact_scan *scan;
} NativeContactScan;

static const double SQRT3 = 1.73205080756887729352744634150587237;
static const double SQRT15 = 3.87298334620741688517926539978239961;
/* ill-conditioning trip for the closed-form stage solve; see gauss.py */
static const double STAGE_GUARD = 1e-6;
static const double NEWTON_TOL = 1e-13;
static const int NEWTON_MAX = 30;

static double horner(const double *c, Py_ssize_t n, double x) {
    double acc = 0.0;
    for (Py_ssize_t i = n; i-- > 0;) {
        acc = acc * x + c[i];
    }
    return acc;
}

static int copy_seq(PyObject *obj, double **out, Py_ssize_t *n_out) {
    PyObject *seq = PySequence_Fast(obj, "coefficients must be a sequence");
    if (seq == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "coefficient sequence is empty");
        return -1;
    }
    double *v = (double *)PyMem_Calloc((size_t)n, sizeof(double));
    if (v == NULL) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; i++) {
        v[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            PyMem_Free(v);
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    *out = v;
    *n_out = n;
    return 0;
}

static void Kernel_dealloc(Kernel *self) {
    PyMem_Free(self->a);
    PyMem_Free(self->ap);
    PyMem_Free(self->app);
    PyMem_Free(self->b);
    PyMem_Free(self->bp);
    PyMem_Free(self->bpp);
    PyMem_Free(self->n);
    PyMem_Free(self->np);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int Kernel_init(Kernel *self, PyObject *args, PyObject *kwds) {
    PyObject *a, *ap, *app, *b, *bp, *bpp, *n, *np;
    static char *kwlist[] = {
        "a", "ap", "app", "b", "bp", "bpp", "n", "np", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOOOOOO", kwlist,
            &a, &ap, &app, &b, &bp, &bpp, &n, &np)) {
        return -1;
    }
    if (copy_seq(a, &self->a, &self->na) < 0 ||
        copy_seq(ap, &self->ap, &self->nap) < 0 ||
        copy_seq(app, &self->app, &self->napp) < 0 ||
        copy_seq(b, &self->b, &self->nb) < 0 ||
        copy_seq(bp, &self->bp, &self->nbp) < 0 ||
        copy_seq(bpp, &self->bpp, &self->nbpp) < 0 ||
        copy_seq(n, &self->n, &self->nn) < 0 ||
        copy_seq(np, &self->np, &self->nnp) < 0) {
        return -1;
    }
    return 0;
}

static void eval_base(Kernel *k, double x,
                      double *A, double *Ap, double *App,
                      double *B, double *Bp, double *Bpp,
                      double *Nv, double *Np) {
    *A = horner(k->a, k->na, x);
    *Ap = horner(k->ap, k->nap, x);
    *App = horner(k->app, k->napp, x);
    *B = horner(k->b, k->nb, x);
    *Bp = horner(k->bp, k->nbp, x);
    *Bpp = horner(k->bpp, k->nbpp, x);
    *Nv = horner(k->n, k->nn, x);
    *Np = horner(k->np, k->nnp, x);
}

static double slow_fj(void *ctx, double b, double w, double *jac) {
    Kernel *k = (Kernel *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    eval_base(k, b, &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
    double A2 = A * A;
    double asp = Bp / A - B * Ap / A2;
    double Pv = B * Nv / A2 + Ap * w * w - 2.0 * A * w * asp;
    if (jac != NULL) {
        double P_w = 2.0 * Ap * w - 2.0 * A * asp;
        *jac = 2.0 * A / Pv - 2.0 * A * w * P_w / (Pv * Pv);
    }
    return 2.0 * A * w / Pv - asp;
}

static double fast_fj(void *ctx, double w, double b, double *jac) {
    Kernel *k = (Kernel *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    eval_base(k, b, &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
    double A2 = A * A;
    double A3 = A2 * A;
    double asp = Bp / A - B * Ap / A2;
    double aspp = ((Bpp * A - B * App) / A2
                   - 2.0 * Ap * (Bp * A - B * Ap) / A3);
    double up = B * Nv / A2;
    double upp = ((Bp * Nv + B * Np) / A2
                  - 2.0 * B * Nv * Ap / A3);
    double Pv = up + Ap * w * w - 2.0 * A * w * asp;
    double D = 2.0 * A * w - asp * Pv;
    if (jac != NULL) {
        double P_b = upp + App * w * w - 2.0 * w * (Ap * asp + A * aspp);
        double D_b = 2.0 * Ap * w - aspp * Pv - asp * P_b;
        *jac = (P_b * D - Pv * D_b) / (D * D);
    }
    return Pv / D;
}

typedef double (*FJ)(void *, double, double, double *);

static double gl4_step(void *ctx, FJ fj, double x, double y, double h) {
    const double c1 = 0.5 - SQRT3 / 6.0;
    const double c2 = 0.5 + SQRT3 / 6.0;
    const double a11 = 0.25;
    const double a12 = 0.25 - SQRT3 / 6.0;
    const double a21 = 0.25 + SQRT3 / 6.0;
    const double a22 = 0.25;
    double x1 = x + c1 * h;
    double x2 = x + c2 * h;
    double K1 = fj(ctx, x, y, NULL);
    double K2 = K1;
    int converged = 0;
    for (int it = 0; it < NEWTON_MAX; it++) {
        double Y1 = y + h * (a11 * K1 + a12 * K2);
        double Y2 = y + h * (a21 * K1 + a22 * K2);
        double f1 = fj(ctx, x1, Y1, NULL);
        double f2 = fj(ctx, x2, Y2, NULL);
        double r1 = K1 - f1;
        double r2 = K2 - f2;
        double m = fabs(K1) > fabs(K2) ? fabs(K1) : fabs(K2);
        double r = fabs(r1) > fabs(r2) ? fabs(r1) : fabs(r2);
        if (r < NEWTON_TOL * (1.0 + m)) {
            converged = 1;
            break;
        }
        double J1, J2;
        (void)fj(ctx, x1, Y1, &J1);
        (void)fj(ctx, x2, Y2, &J2);
        double m11 = 1.0 - h * a11 * J1;
        double m12 = -h * a12 * J1;
        double m21 = -h * a21 * J2;
        double m22 = 1.0 - h * a22 * J2;
        double det = m11 * m22 - m12 * m21;
        double d1 = (-m22 * r1 + m12 * r2) / det;
        double d2 = (m21 * r1 - m11 * r2) / det;
        K1 += d1;
        K2 += d2;
        if (fmax(fabs(d1), fabs(d2)) < NEWTON_TOL * (1.0 + m)) {
            converged = 1;
            break;
        }
    }
    if (!converged) return NAN;
    return y + h * 0.5 * (K1 + K2);
}

/* 3-stage Gauss (IRK6-GL), closed-form 3x3 stage Newton by adjugate.
 *
 * Safe rather than merely convenient: det(I - zA) is the (3,3) Pade
 * denominator of exp, whose roots all have Re z > 0 (A-stability), so for
 * dissipative h*lambda the stage matrix cannot be singular and |det| GROWS
 * like |z|^3/120.  cond_2 saturates at 10.44 (frozen D) / 24.6 (varying D),
 * flat to |z| = 1e14.  Entries are essentially exact, so small backward error
 * IS small forward error and no refinement is needed; the only guard required
 * is the ill-conditioning trip below.  Must stay bit-comparable with
 * gauss.gl6_scalar -- tests/test_native_parity.py pins that. */
static double gl6_step(void *ctx, FJ fj, double x, double y, double h) {
    const double c1 = 0.5 - SQRT15 / 10.0;
    const double c2 = 0.5;
    const double c3 = 0.5 + SQRT15 / 10.0;
    const double a11 = 5.0 / 36.0;
    const double a12 = 2.0 / 9.0 - SQRT15 / 15.0;
    const double a13 = 5.0 / 36.0 - SQRT15 / 30.0;
    const double a21 = 5.0 / 36.0 + SQRT15 / 24.0;
    const double a22 = 2.0 / 9.0;
    const double a23 = 5.0 / 36.0 - SQRT15 / 24.0;
    const double a31 = 5.0 / 36.0 + SQRT15 / 30.0;
    const double a32 = 2.0 / 9.0 + SQRT15 / 15.0;
    const double a33 = 5.0 / 36.0;
    double x1 = x + c1 * h;
    double x2 = x + c2 * h;
    double x3 = x + c3 * h;
    double K1 = fj(ctx, x, y, NULL);
    double K2 = K1;
    double K3 = K1;
    int converged = 0;
    /* Preserve full Newton as the normal fast path.  If it exhausts its
     * iteration budget, restart from the Euler stages and globalize the same
     * Newton direction with Armijo descent of 1/2 ||stage residual||^2.
     * The Python reference solver has long had this restart; keeping it out
     * of the production C path made a finite arrival fail at the spatial
     * step floor even though the stage equations still had a usable basin. */
    for (int pass = 0; pass < 2 && !converged; pass++) {
      K1 = fj(ctx, x, y, NULL);
      K2 = K1;
      K3 = K1;
      for (int it = 0; it < NEWTON_MAX; it++) {
        double Y1 = y + h * (a11 * K1 + a12 * K2 + a13 * K3);
        double Y2 = y + h * (a21 * K1 + a22 * K2 + a23 * K3);
        double Y3 = y + h * (a31 * K1 + a32 * K2 + a33 * K3);
        double r1 = K1 - fj(ctx, x1, Y1, NULL);
        double r2 = K2 - fj(ctx, x2, Y2, NULL);
        double r3 = K3 - fj(ctx, x3, Y3, NULL);
        double m = fabs(K1);
        if (fabs(K2) > m) m = fabs(K2);
        if (fabs(K3) > m) m = fabs(K3);
        double r = fabs(r1);
        if (fabs(r2) > r) r = fabs(r2);
        if (fabs(r3) > r) r = fabs(r3);
        if (r < NEWTON_TOL * (1.0 + m)) {
            converged = 1;
            break;
        }
        double J1, J2, J3;
        (void)fj(ctx, x1, Y1, &J1);
        (void)fj(ctx, x2, Y2, &J2);
        (void)fj(ctx, x3, Y3, &J3);
        double m11 = 1.0 - h * a11 * J1;
        double m12 = -h * a12 * J1;
        double m13 = -h * a13 * J1;
        double m21 = -h * a21 * J2;
        double m22 = 1.0 - h * a22 * J2;
        double m23 = -h * a23 * J2;
        double m31 = -h * a31 * J3;
        double m32 = -h * a32 * J3;
        double m33 = 1.0 - h * a33 * J3;
        /* ROW-SCALE first: the adjugate forms TRIPLE products, so unscaled it
         * overflows ~1e150 against the 2x2's ~1e300.  Found out of sample as a
         * NaN step on a random portrait GL4 handled.  Scaling row i and r_i by
         * the row inf-norm leaves dK unchanged and bounds every entry by 1, so
         * the Hadamard ratio is just |det| of the scaled matrix. */
        double n1 = fabs(m11);
        if (fabs(m12) > n1) n1 = fabs(m12);
        if (fabs(m13) > n1) n1 = fabs(m13);
        double n2 = fabs(m21);
        if (fabs(m22) > n2) n2 = fabs(m22);
        if (fabs(m23) > n2) n2 = fabs(m23);
        double n3 = fabs(m31);
        if (fabs(m32) > n3) n3 = fabs(m32);
        if (fabs(m33) > n3) n3 = fabs(m33);
        if (n1 == 0.0 || n2 == 0.0 || n3 == 0.0) {
            return NAN;
        }
        /* Row-equilibrate, then LU with PARTIAL PIVOTING: half the flops of
         * an adjugate, determinant just as free (product of pivots), and every
         * multiplier bounded by 1 -- removing the overflow hazard structurally
         * (the unscaled adjugate died at |h*J| ~ 1e150 on triple products).
         * Must stay bit-comparable with gauss.gl6_scalar. */
        double A_[3][3] = {{m11 / n1, m12 / n1, m13 / n1},
                           {m21 / n2, m22 / n2, m23 / n2},
                           {m31 / n3, m32 / n3, m33 / n3}};
        double v_[3] = {r1 / n1, r2 / n2, r3 / n3};
        double det = 1.0;
        for (int col = 0; col < 3; col++) {
            int p = col;
            double big = fabs(A_[col][col]);
            for (int row = col + 1; row < 3; row++) {
                if (fabs(A_[row][col]) > big) {
                    big = fabs(A_[row][col]);
                    p = row;
                }
            }
            if (p != col) {
                for (int k2 = 0; k2 < 3; k2++) {
                    double t = A_[col][k2];
                    A_[col][k2] = A_[p][k2];
                    A_[p][k2] = t;
                }
                double t = v_[col];
                v_[col] = v_[p];
                v_[p] = t;
                det = -det;
            }
            double piv = A_[col][col];
            det *= piv;
            if (piv == 0.0) break;
            for (int row = col + 1; row < 3; row++) {
                double f2 = A_[row][col] / piv;
                for (int k2 = col; k2 < 3; k2++) {
                    A_[row][k2] -= f2 * A_[col][k2];
                }
                v_[row] -= f2 * v_[col];
            }
        }
        /* |det| of the equilibrated matrix IS the Hadamard ratio */
        if (fabs(det) < STAGE_GUARD) {
            return NAN;         /* caller rejects the step; it halves */
        }
        double d3 = v_[2] / A_[2][2];
        double d2 = (v_[1] - A_[1][2] * d3) / A_[1][1];
        double d1 = (v_[0] - A_[0][1] * d2 - A_[0][2] * d3) / A_[0][0];
        double alpha = 1.0;
        if (pass == 1) {
            double phi = 0.5 * (r1*r1 + r2*r2 + r3*r3);
            int accepted = 0;
            for (int ls = 0; ls <= 12; ls++) {
                double C1 = K1 - alpha*d1;
                double C2 = K2 - alpha*d2;
                double C3 = K3 - alpha*d3;
                double Z1 = y + h * (a11*C1 + a12*C2 + a13*C3);
                double Z2 = y + h * (a21*C1 + a22*C2 + a23*C3);
                double Z3 = y + h * (a31*C1 + a32*C2 + a33*C3);
                double q1 = C1 - fj(ctx, x1, Z1, NULL);
                double q2 = C2 - fj(ctx, x2, Z2, NULL);
                double q3 = C3 - fj(ctx, x3, Z3, NULL);
                double phic = 0.5 * (q1*q1 + q2*q2 + q3*q3);
                if (isfinite(phic)
                        && phic <= phi * (1.0 - 1e-4 * alpha)) {
                    accepted = 1;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) break;
        }
        K1 -= alpha*d1;
        K2 -= alpha*d2;
        K3 -= alpha*d3;
        double d = fmax(fabs(d1), fmax(fabs(d2), fabs(d3)));
        /* A tiny line-search alpha is not convergence: the undamped Newton
         * correction measures the stage-equation error.  Using alpha*d here
         * can accept a stalled Armijo iteration as a solution. */
        (void)d;  /* convergence is certified by the residual at loop head */
      }
    }
    if (!converged) return NAN;
    return y + h * (5.0 / 18.0 * K1 + 4.0 / 9.0 * K2 + 5.0 / 18.0 * K3);
}

static PyObject *Kernel_slow_step(Kernel *self, PyObject *args) {
    double x, y, h;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &h)) {
        return NULL;
    }
    return PyFloat_FromDouble(gl6_step(self, slow_fj, x, y, h));
}

static PyObject *Kernel_fast_step(Kernel *self, PyObject *args) {
    double x, y, h;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &h)) {
        return NULL;
    }
    return PyFloat_FromDouble(gl6_step(self, fast_fj, x, y, h));
}

static PyObject *Kernel_slow_step_gl4(Kernel *self, PyObject *args) {
    double x, y, h;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &h)) {
        return NULL;
    }
    return PyFloat_FromDouble(gl4_step(self, slow_fj, x, y, h));
}

static PyObject *Kernel_fast_step_gl4(Kernel *self, PyObject *args) {
    double x, y, h;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &h)) {
        return NULL;
    }
    return PyFloat_FromDouble(gl4_step(self, fast_fj, x, y, h));
}

static PyObject *Kernel_velocities(Kernel *self, PyObject *args) {
    double b, w;
    if (!PyArg_ParseTuple(args, "dd", &b, &w)) {
        return NULL;
    }
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    eval_base(self, b, &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
    double A2 = A * A;
    double asp = Bp / A - B * Ap / A2;
    double Pv = B * Nv / A2 + Ap * w * w - 2.0 * A * w * asp;
    return Py_BuildValue("dd", -Pv, -2.0 * A * w + asp * Pv);
}

static PyObject *Kernel_slow_fixed_point(Kernel *self, PyObject *args) {
    PyObject *grid_obj;
    double tol = 1e-13;
    int max_iter = 40;
    if (!PyArg_ParseTuple(args, "O|di", &grid_obj, &tol, &max_iter)) return NULL;
    PyObject *grid = PySequence_Fast(grid_obj, "b_grid must be a sequence");
    if (grid == NULL) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(grid);
    if (n < 5) {
        Py_DECREF(grid);
        PyErr_SetString(PyExc_ValueError,
                        "native graph transform requires at least 5 grid points");
        return NULL;
    }
    double *b = PyMem_Malloc((size_t)n * sizeof(double));
    double *A = PyMem_Malloc((size_t)n * sizeof(double));
    double *Ap = PyMem_Malloc((size_t)n * sizeof(double));
    double *asp = PyMem_Malloc((size_t)n * sizeof(double));
    double *up = PyMem_Malloc((size_t)n * sizeof(double));
    double *w = PyMem_Malloc((size_t)n * sizeof(double));
    double *wp = PyMem_Malloc((size_t)n * sizeof(double));
    double *wn = PyMem_Malloc((size_t)n * sizeof(double));
    if (!b || !A || !Ap || !asp || !up || !w || !wp || !wn) {
        PyMem_Free(b); PyMem_Free(A); PyMem_Free(Ap); PyMem_Free(asp);
        PyMem_Free(up); PyMem_Free(w); PyMem_Free(wp); PyMem_Free(wn);
        Py_DECREF(grid);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(grid);
    for (Py_ssize_t i = 0; i < n; i++) {
        b[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) goto fail;
    }
    Py_DECREF(grid);
    grid = NULL;
    double h = b[1] - b[0];
    if (!isfinite(h) || h == 0.0) {
        PyErr_SetString(PyExc_ValueError, "b_grid spacing must be finite and nonzero");
        goto fail;
    }
    for (Py_ssize_t i = 1; i < n - 1; i++) {
        double di = b[i + 1] - b[i];
        if (fabs(di - h) > 1e-12 * fabs(h)) {
            PyErr_SetString(PyExc_ValueError,
                            "native graph transform requires a uniform grid");
            goto fail;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        double App, B, Bp, Bpp, Nv, Np;
        eval_base(self, b[i], &A[i], &Ap[i], &App,
                  &B, &Bp, &Bpp, &Nv, &Np);
        double A2 = A[i] * A[i];
        asp[i] = Bp / A[i] - B * Ap[i] / A2;
        up[i] = B * Nv / A2;
        w[i] = asp[i] * up[i] / (2.0 * A[i]);
    }
    double rel = INFINITY;
    int it = 0;
    for (it = 1; it <= max_iter; it++) {
        wp[0] = (-25.0*w[0] + 48.0*w[1] - 36.0*w[2]
                 + 16.0*w[3] - 3.0*w[4]) / (12.0*h);
        wp[1] = (-3.0*w[0] - 10.0*w[1] + 18.0*w[2]
                 - 6.0*w[3] + w[4]) / (12.0*h);
        for (Py_ssize_t i = 2; i < n - 2; i++) {
            wp[i] = (w[i-2] - 8.0*w[i-1] + 8.0*w[i+1] - w[i+2])
                    / (12.0*h);
        }
        wp[n-2] = (3.0*w[n-1] + 10.0*w[n-2] - 18.0*w[n-3]
                   + 6.0*w[n-4] - w[n-5]) / (12.0*h);
        wp[n-1] = (25.0*w[n-1] - 48.0*w[n-2] + 36.0*w[n-3]
                   - 16.0*w[n-4] + 3.0*w[n-5]) / (12.0*h);
        double scale = 0.0, change = 0.0;
        for (Py_ssize_t i = 0; i < n; i++) {
            double Pv = up[i] + Ap[i] * w[i] * w[i]
                        - 2.0 * A[i] * w[i] * asp[i];
            wn[i] = Pv * (asp[i] + wp[i]) / (2.0 * A[i]);
            if (!isfinite(wn[i])) {
                rel = INFINITY;
                goto done;
            }
            if (fabs(wn[i]) > scale) scale = fabs(wn[i]);
            if (fabs(wn[i] - w[i]) > change) change = fabs(wn[i] - w[i]);
        }
        rel = change / fmax(scale, 1e-300);
        double *tmp = w; w = wn; wn = tmp;
        if (rel < tol) break;
    }
    if (it > max_iter) it = max_iter;
done:
    {
        PyObject *out = PyList_New(n);
        if (out == NULL) goto fail;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *v = PyFloat_FromDouble(w[i]);
            if (v == NULL) {
                Py_DECREF(out);
                goto fail;
            }
            PyList_SET_ITEM(out, i, v);
        }
        PyMem_Free(b); PyMem_Free(A); PyMem_Free(Ap); PyMem_Free(asp);
        PyMem_Free(up); PyMem_Free(w); PyMem_Free(wp); PyMem_Free(wn);
        return Py_BuildValue("Nid", out, it, rel);
    }
fail:
    Py_XDECREF(grid);
    PyMem_Free(b); PyMem_Free(A); PyMem_Free(Ap); PyMem_Free(asp);
    PyMem_Free(up); PyMem_Free(w); PyMem_Free(wp); PyMem_Free(wn);
    return NULL;
}

static int solve_small(double M[8][8], double r[8], double x[8], int n) {
    double original[8][8], right[8];
    double max_matrix = 0.0, max_right = 0.0;
    for (int i = 0; i < n; i++) {
        right[i] = -r[i];
        x[i] = right[i];
        if (fabs(right[i]) > max_right) max_right = fabs(right[i]);
        double row_scale = 0.0;
        for (int j = 0; j < n; j++) {
            original[i][j] = M[i][j];
            if (fabs(M[i][j]) > row_scale) row_scale = fabs(M[i][j]);
            if (fabs(M[i][j]) > max_matrix) max_matrix = fabs(M[i][j]);
        }
        if (!(row_scale > 0.0) || !isfinite(row_scale)) return 0;
        for (int j = 0; j < n; j++) M[i][j] /= row_scale;
        x[i] /= row_scale;
    }
    const double pivot_floor = 64.0*DBL_EPSILON;
    for (int col = 0; col < n; col++) {
        int p = col;
        double big = fabs(M[col][col]);
        for (int row = col + 1; row < n; row++) {
            if (fabs(M[row][col]) > big) {
                big = fabs(M[row][col]); p = row;
            }
        }
        if (!(big > pivot_floor) || !isfinite(big)) return 0;
        if (p != col) {
            for (int j = 0; j < n; j++) {
                double t = M[col][j]; M[col][j] = M[p][j]; M[p][j] = t;
            }
            double t = x[col]; x[col] = x[p]; x[p] = t;
        }
        for (int row = col + 1; row < n; row++) {
            double f = M[row][col] / M[col][col];
            M[row][col] = 0.0;
            for (int j = col + 1; j < n; j++) M[row][j] -= f * M[col][j];
            x[row] -= f * x[col];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
        x[i] /= M[i][i];
        if (!isfinite(x[i])) return 0;
    }
    /* Componentwise construction above is certified against the original,
     * unequilibrated Newton system.  This rejects a finite but meaningless
     * correction instead of allowing it to contaminate a high-order step. */
    double residual_max = 0.0, solution_max = 0.0;
    for (int j = 0; j < n; j++)
        if (fabs(x[j]) > solution_max) solution_max = fabs(x[j]);
    for (int i = 0; i < n; i++) {
        double residual = -right[i];
        for (int j = 0; j < n; j++) residual += original[i][j]*x[j];
        if (fabs(residual) > residual_max) residual_max = fabs(residual);
    }
    double denominator = n*max_matrix*solution_max + max_right;
    double backward_error = residual_max/fmax(denominator, 1e-300);
    if (!isfinite(backward_error) || backward_error > 1e-11) return 0;
    return 1;
}

static int normalized_fj(void *ctx, const double z[2],
                         double f[2], double J[2][2]) {
    Kernel *k = (Kernel *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    eval_base(k, z[1], &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
    double a = z[0];
    double g0 = 2.0 * (a * A - B);
    double g1 = -2.0 * a * Bp + a * a * Ap;
    double ng = hypot(g0, g1);
    if (!(ng > 1e-300) || !isfinite(ng)) return 0;
    f[0] = g0 / ng; f[1] = g1 / ng;
    if (J != NULL) {
        double H00 = 2.0 * A;
        double H01 = -2.0 * Bp + 2.0 * a * Ap;
        double H11 = -2.0 * a * Bpp + a * a * App;
        double Hg0 = H00 * g0 + H01 * g1;
        double Hg1 = H01 * g0 + H11 * g1;
        double ng3 = ng * ng * ng;
        J[0][0] = H00/ng - g0*Hg0/ng3;
        J[0][1] = H01/ng - g0*Hg1/ng3;
        J[1][0] = H01/ng - g1*Hg0/ng3;
        J[1][1] = H11/ng - g1*Hg1/ng3;
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

static int potential_rate_fj(void *ctx, const double z[2],
                             double f[2], double J[2][2]) {
    Kernel *k = (Kernel *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    eval_base(k, z[1], &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
    double a = z[0];
    double g0 = 2.0 * (a * A - B);
    double g1 = -2.0 * a * Bp + a * a * Ap;
    double q = g0*g0 + g1*g1;
    if (!(q > 1e-300) || !isfinite(q)) return 0;
    f[0] = g0 / q; f[1] = g1 / q;
    if (J != NULL) {
        double H00 = 2.0 * A;
        double H01 = -2.0 * Bp + 2.0 * a * Ap;
        double H11 = -2.0 * a * Bpp + a * a * App;
        double Hg0 = H00 * g0 + H01 * g1;
        double Hg1 = H01 * g0 + H11 * g1;
        double q2 = q*q;
        J[0][0] = H00/q - 2.0*g0*Hg0/q2;
        J[0][1] = H01/q - 2.0*g0*Hg1/q2;
        J[1][0] = H01/q - 2.0*g1*Hg0/q2;
        J[1][1] = H11/q - 2.0*g1*Hg1/q2;
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

typedef int (*VecFJ)(void *, const double[2], double[2], double[2][2]);

static int irk2_evaluate(void *ctx, VecFJ fj, const double z[2], double h, int s,
                         const double AT[4][4], const double K[4][2],
                         double R[8], double Js[4][2][2],
                         double *scale, double *rmax, double *phi) {
    *scale = 1.0; *rmax = 0.0; *phi = 0.0;
    for (int i = 0; i < s; i++) {
        double Y[2] = {z[0], z[1]}, F[2];
        for (int d = 0; d < 2; d++)
            for (int j = 0; j < s; j++) Y[d] += h*AT[i][j]*K[j][d];
        if (!fj(ctx, Y, F, Js == NULL ? NULL : Js[i])) return 0;
        for (int d = 0; d < 2; d++) {
            double rv = K[i][d] - F[d];
            R[2*i+d] = rv;
            *phi += 0.5 * rv * rv;
            if (fabs(K[i][d]) > *scale) *scale = fabs(K[i][d]);
            if (fabs(rv) > *rmax) *rmax = fabs(rv);
        }
    }
    return isfinite(*phi);
}

static int irk2_step(void *ctx, VecFJ fj, const double z[2], double h, int order,
                     double out[2]) {
    static const double A4[4][4] = {
        {0.25, 0.25 - SQRT3/6.0, 0.0, 0.0},
        {0.25 + SQRT3/6.0, 0.25, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}
    };
    static const double B4[4] = {0.5, 0.5, 0.0, 0.0};
    static const double A6[4][4] = {
        {5.0/36.0, 2.0/9.0-SQRT15/15.0,
         5.0/36.0-SQRT15/30.0, 0.0},
        {5.0/36.0+SQRT15/24.0, 2.0/9.0,
         5.0/36.0-SQRT15/24.0, 0.0},
        {5.0/36.0+SQRT15/30.0, 2.0/9.0+SQRT15/15.0,
         5.0/36.0, 0.0},
        {0.0, 0.0, 0.0, 0.0}
    };
    static const double B6[4] = {
        5.0/18.0, 4.0/9.0, 5.0/18.0, 0.0};
    /* Four-stage Gauss--Legendre (order eight), independently generated
     * from a_ij = integral_0^c_i L_j and recorded at binary64 precision. */
    static const double A8[4][4] = {
        {0.08696371128436346, -0.02660418008499879,
         0.012627462689404725, -0.003555149685795683},
        {0.18811811749986807, 0.16303628871563654,
         -0.027880428602470895, 0.006735500594538155},
        {0.16719192197418877, 0.35395300603374397,
         0.16303628871563654, -0.014190694931141142},
        {0.17748257225452260, 0.31344511474186835,
         0.35267675751627190, 0.08696371128436346}
    };
    static const double B8[4] = {
        0.17392742256872693, 0.32607257743127307,
        0.32607257743127307, 0.17392742256872693};
    const double (*AT)[4] = order == 4 ? A4 : (order == 6 ? A6 : A8);
    const double *BT = order == 4 ? B4 : (order == 6 ? B6 : B8);
    int s = order/2, n = 2*s;
    double f0[2], K[4][2], K0[4][2], Js[4][2][2];
    if (!fj(ctx, z, f0, NULL)) return 0;
    for (int i = 0; i < s; i++) {
        K0[i][0] = f0[0]; K0[i][1] = f0[1];
    }
    int converged = 0;
    for (int pass = 0; pass < 2 && !converged; pass++) {
        memcpy(K, K0, sizeof(K));
        for (int it = 0; it < NEWTON_MAX; it++) {
            double R[8], scale, rmax, phi;
            if (!irk2_evaluate(ctx, fj, z, h, s, AT, K, R, Js,
                               &scale, &rmax, &phi)) break;
            if (rmax < NEWTON_TOL * scale) { converged = 1; break; }
            double M[8][8] = {{0}}, delta[8];
            for (int i = 0; i < s; i++) for (int j = 0; j < s; j++)
                for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++) {
                    M[2*i+r][2*j+c] = -h * AT[i][j] * Js[i][r][c];
                    if (i == j && r == c) M[2*i+r][2*j+c] += 1.0;
                }
            if (!solve_small(M, R, delta, n)) break;
            double dmax = 0.0;
            for (int i = 0; i < n; i++)
                if (fabs(delta[i]) > dmax) dmax = fabs(delta[i]);
            double alpha = 1.0;
            if (pass == 1) {
                int accepted = 0;
                while (alpha >= 1.0/4096.0) {
                    double Kc[4][2], Rc[8], sc, rm, phic;
                    memcpy(Kc, K, sizeof(Kc));
                    for (int i = 0; i < s; i++) for (int d = 0; d < 2; d++)
                        Kc[i][d] += alpha * delta[2*i+d];
                    if (irk2_evaluate(ctx, fj, z, h, s, AT, Kc, Rc, NULL,
                                      &sc, &rm, &phic)
                            && phic <= phi * (1.0 - 1e-4 * alpha)) {
                        memcpy(K, Kc, sizeof(K)); accepted = 1; break;
                    }
                    alpha *= 0.5;
                }
                if (!accepted) break;
            } else {
                for (int i = 0; i < s; i++) for (int d = 0; d < 2; d++)
                    K[i][d] += delta[2*i+d];
            }
            if (alpha*dmax < NEWTON_TOL * scale) {
                converged = 1; break;
            }
        }
    }
    if (!converged) return 0;
    out[0] = z[0]; out[1] = z[1];
    for (int i = 0; i < s; i++) {
        out[0] += h * BT[i] * K[i][0];
        out[1] += h * BT[i] * K[i][1];
    }
    return isfinite(out[0]) && isfinite(out[1]);
}

static PyObject *Kernel_normalized_step(Kernel *self, PyObject *args) {
    double a, b, h;
    int order = 6;
    if (!PyArg_ParseTuple(args, "ddd|i", &a, &b, &h, &order)) return NULL;
    if (order != 4 && order != 6 && order != 8) {
        PyErr_SetString(PyExc_ValueError, "order must be 4, 6, or 8");
        return NULL;
    }
    double z[2] = {a, b}, out[2];
    if (!irk2_step(self, normalized_fj, z, h, order, out)) {
        return Py_BuildValue("dd", NAN, NAN);
    }
    return Py_BuildValue("dd", out[0], out[1]);
}

static PyObject *Kernel_potential_step(Kernel *self, PyObject *args) {
    double a, b, h;
    int order = 6;
    if (!PyArg_ParseTuple(args, "ddd|i", &a, &b, &h, &order)) return NULL;
    if (order != 4 && order != 6 && order != 8) {
        PyErr_SetString(PyExc_ValueError, "order must be 4, 6, or 8");
        return NULL;
    }
    double z[2] = {a, b}, out[2];
    if (!irk2_step(self, potential_rate_fj, z, h, order, out))
        return Py_BuildValue("dd", NAN, NAN);
    return Py_BuildValue("dd", out[0], out[1]);
}

static void LocalKernel_dealloc(LocalKernel *self) {
    for (int d = 0; d < 2; d++) {
        if (self->c[d] != NULL)
            for (int r = 0; r < self->rows[d]; r++)
                PyMem_Free(self->c[d][r]);
        PyMem_Free(self->c[d]);
        PyMem_Free(self->n[d]);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LocalKernel_init(LocalKernel *self, PyObject *args, PyObject *kwds) {
    PyObject *grad;
    static char *kwlist[] = {"grad", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &grad)) return -1;
    PyObject *comps = PySequence_Fast(grad, "grad must have two components");
    if (comps == NULL) return -1;
    if (PySequence_Fast_GET_SIZE(comps) != 2) {
        Py_DECREF(comps);
        PyErr_SetString(PyExc_ValueError, "grad must have two components");
        return -1;
    }
    PyObject **ci = PySequence_Fast_ITEMS(comps);
    for (int d = 0; d < 2; d++) {
        PyObject *rows = PySequence_Fast(ci[d], "component must contain rows");
        if (rows == NULL) { Py_DECREF(comps); return -1; }
        Py_ssize_t nr = PySequence_Fast_GET_SIZE(rows);
        if (nr < 1 || nr > INT_MAX) {
            Py_DECREF(rows); Py_DECREF(comps);
            PyErr_SetString(PyExc_ValueError, "invalid local jet row count");
            return -1;
        }
        self->c[d] = PyMem_Calloc((size_t)nr, sizeof(*self->c[d]));
        self->n[d] = PyMem_Calloc((size_t)nr, sizeof(*self->n[d]));
        if (self->c[d] == NULL || self->n[d] == NULL) {
            Py_DECREF(rows); Py_DECREF(comps);
            PyErr_NoMemory();
            return -1;
        }
        self->rows[d] = (int)nr;
        PyObject **ri = PySequence_Fast_ITEMS(rows);
        for (Py_ssize_t r = 0; r < nr; r++)
            if (copy_seq(ri[r], &self->c[d][r], &self->n[d][r]) < 0) {
                Py_DECREF(rows); Py_DECREF(comps); return -1;
            }
        Py_DECREF(rows);
    }
    Py_DECREF(comps);
    return 0;
}

static void local_poly(LocalKernel *k, const double z[2],
                       double g[2], double H[2][2]) {
    double da = z[0], db = z[1];
    for (int d = 0; d < 2; d++) {
        double value = 0.0, derivative_a = 0.0, derivative_b = 0.0;
        for (int r = k->rows[d]; r-- > 0;) {
            double row = horner(k->c[d][r], k->n[d][r], db);
            double drow = 0.0;
            for (Py_ssize_t j = k->n[d][r]; j-- > 1;)
                drow = drow*db + (double)j*k->c[d][r][j];
            derivative_a = derivative_a*da + value;
            value = value*da + row;
            derivative_b = derivative_b*da + drow;
        }
        g[d] = value;
        if (H != NULL) {
            H[d][0] = derivative_a;
            H[d][1] = derivative_b;
        }
    }
}

typedef struct {
    LocalKernel *kernel;
    int independent;
} LocalCurveContext;

static double local_curve_fj(void *opaque, double x, double y, double *jac) {
    LocalCurveContext *ctx = (LocalCurveContext *)opaque;
    int i = ctx->independent, d = 1-i;
    double z[2], g[2], H[2][2];
    z[i] = x;
    z[d] = y;
    local_poly(ctx->kernel, z, g, jac == NULL ? NULL : H);
    if (!isfinite(g[i]) || !isfinite(g[d]) || fabs(g[i]) < 1e-300)
        return NAN;
    if (jac != NULL)
        *jac = (H[d][d]*g[i] - g[d]*H[i][d])/(g[i]*g[i]);
    return g[d]/g[i];
}

static int local_normalized_fj(void *ctx, const double z[2],
                               double f[2], double J[2][2]) {
    double g[2], H[2][2];
    local_poly((LocalKernel *)ctx, z, g, J == NULL ? NULL : H);
    double ng = hypot(g[0], g[1]);
    if (!(ng > 1e-300) || !isfinite(ng)) return 0;
    f[0] = g[0]/ng; f[1] = g[1]/ng;
    if (J != NULL) {
        /* g^T H columns; do not assume rounded cross-partials are identical. */
        double Hg[2] = {g[0]*H[0][0] + g[1]*H[1][0],
                        g[0]*H[0][1] + g[1]*H[1][1]};
        double ng3 = ng*ng*ng;
        for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++)
            J[r][c] = H[r][c]/ng - g[r]*Hg[c]/ng3;
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

static int local_raw_fj(void *ctx, const double z[2],
                        double f[2], double J[2][2]) {
    double H[2][2];
    local_poly((LocalKernel *)ctx, z, f, J == NULL ? NULL : H);
    if (J != NULL) {
        J[0][0] = H[0][0]; J[0][1] = H[0][1];
        J[1][0] = H[1][0]; J[1][1] = H[1][1];
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

static PyObject *LocalKernel_gradient(LocalKernel *self, PyObject *args) {
    double z[2], g[2];
    if (!PyArg_ParseTuple(args, "dd", &z[0], &z[1])) return NULL;
    local_poly(self, z, g, NULL);
    return Py_BuildValue("dd", g[0], g[1]);
}

static PyObject *LocalKernel_normalized_step(LocalKernel *self, PyObject *args) {
    double z[2], h, out[2]; int order = 6;
    if (!PyArg_ParseTuple(args, "ddd|i", &z[0], &z[1], &h, &order)) return NULL;
    if (order != 4 && order != 6 && order != 8) {
        PyErr_SetString(PyExc_ValueError, "order must be 4, 6, or 8"); return NULL;
    }
    if (!irk2_step(self, local_normalized_fj, z, h, order, out))
        return Py_BuildValue("dd", NAN, NAN);
    return Py_BuildValue("dd", out[0], out[1]);
}

static PyObject *LocalKernel_raw_step(LocalKernel *self, PyObject *args) {
    double z[2], h, out[2]; int order = 6;
    if (!PyArg_ParseTuple(args, "ddd|i", &z[0], &z[1], &h, &order)) return NULL;
    if (order != 4 && order != 6 && order != 8) {
        PyErr_SetString(PyExc_ValueError, "order must be 4, 6, or 8"); return NULL;
    }
    if (!irk2_step(self, local_raw_fj, z, h, order, out))
        return Py_BuildValue("dd", NAN, NAN);
    return Py_BuildValue("dd", out[0], out[1]);
}

static PyObject *LocalKernel_curve_step(LocalKernel *self, PyObject *args) {
    double x, y, h;
    int independent, order = 6;
    if (!PyArg_ParseTuple(
            args, "dddi|i", &x, &y, &h, &independent, &order)) return NULL;
    if ((independent != 0 && independent != 1)
            || (order != 4 && order != 6)) {
        PyErr_SetString(
            PyExc_ValueError,
            "independent must be 0 or 1 and order must be 4 or 6");
        return NULL;
    }
    LocalCurveContext ctx = {self, independent};
    double out = order == 6
        ? gl6_step(&ctx, local_curve_fj, x, y, h)
        : gl4_step(&ctx, local_curve_fj, x, y, h);
    return PyFloat_FromDouble(out);
}

static PyObject *LocalKernel_graph_fixed_point(LocalKernel *self,
                                                PyObject *args) {
    PyObject *grid_obj, *frame_obj;
    double lu, ls, tol = 1e-13;
    int max_iter = 80;
    if (!PyArg_ParseTuple(args, "OOdd|di", &grid_obj, &frame_obj,
                          &lu, &ls, &tol, &max_iter)) return NULL;
    if (!(lu*ls < 0.0) || !(tol > 0.0) || max_iter < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "graph and transverse eigenvalues must have opposite signs");
        return NULL;
    }
    PyObject *grid = PySequence_Fast(grid_obj, "u_grid must be a sequence");
    PyObject *frame = PySequence_Fast(frame_obj, "frame must have four entries");
    if (grid == NULL || frame == NULL) {
        Py_XDECREF(grid); Py_XDECREF(frame); return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(grid);
    if (n < 5 || PySequence_Fast_GET_SIZE(frame) != 4) {
        Py_DECREF(grid); Py_DECREF(frame);
        PyErr_SetString(PyExc_ValueError,
                        "need at least five grid points and a 2x2 frame");
        return NULL;
    }
    double V[4];
    PyObject **fi = PySequence_Fast_ITEMS(frame);
    for (int i = 0; i < 4; i++) {
        V[i] = PyFloat_AsDouble(fi[i]);
        if (PyErr_Occurred()) { Py_DECREF(grid); Py_DECREF(frame); return NULL; }
    }
    Py_DECREF(frame);
    double *u = PyMem_Malloc((size_t)n*sizeof(double));
    double *f = PyMem_Calloc((size_t)n, sizeof(double));
    double *fp = PyMem_Malloc((size_t)n*sizeof(double));
    double *fn = PyMem_Malloc((size_t)n*sizeof(double));
    if (!u || !f || !fp || !fn) {
        PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp); PyMem_Free(fn);
        Py_DECREF(grid); return PyErr_NoMemory();
    }
    PyObject **gi = PySequence_Fast_ITEMS(grid);
    for (Py_ssize_t i = 0; i < n; i++) {
        u[i] = PyFloat_AsDouble(gi[i]);
        if (PyErr_Occurred()) goto graph_fail;
    }
    Py_DECREF(grid); grid = NULL;
    double h = u[1] - u[0];
    if (!isfinite(h) || h == 0.0) {
        PyErr_SetString(PyExc_ValueError, "grid spacing must be nonzero");
        goto graph_fail;
    }
    for (Py_ssize_t i = 1; i < n - 1; i++)
        if (fabs((u[i+1]-u[i])-h) > 1e-12*fabs(h)) {
            PyErr_SetString(PyExc_ValueError, "grid must be uniform");
            goto graph_fail;
        }

    double rel = INFINITY;
    int it;
    for (it = 1; it <= max_iter; it++) {
        fp[0] = (-25*f[0]+48*f[1]-36*f[2]+16*f[3]-3*f[4])/(12*h);
        fp[1] = (-3*f[0]-10*f[1]+18*f[2]-6*f[3]+f[4])/(12*h);
        for (Py_ssize_t i = 2; i < n-2; i++)
            fp[i] = (f[i-2]-8*f[i-1]+8*f[i+1]-f[i+2])/(12*h);
        fp[n-2] = (3*f[n-1]+10*f[n-2]-18*f[n-3]+6*f[n-4]-f[n-5])/(12*h);
        fp[n-1] = (25*f[n-1]-48*f[n-2]+36*f[n-3]-16*f[n-4]+3*f[n-5])/(12*h);
        double scale = 0.0, change = 0.0;
        for (Py_ssize_t i = 0; i < n; i++) {
            /* Columns of V are unstable and stable unit eigenvectors. */
            double z[2] = {V[0]*u[i] + V[1]*f[i],
                           V[2]*u[i] + V[3]*f[i]};
            double g[2];
            local_poly(self, z, g, NULL);
            double vu = V[0]*g[0] + V[2]*g[1];
            double vs = V[1]*g[0] + V[3]*g[1];
            double Ru = vu - lu*u[i];
            double Rs = vs - ls*f[i];
            fn[i] = (fp[i]*(lu*u[i] + Ru) - Rs)/ls;
            if (i == 0) fn[i] = 0.0;
            if (!isfinite(fn[i])) { rel = INFINITY; goto graph_done; }
            scale = fmax(scale, fabs(fn[i]));
            change = fmax(change, fabs(fn[i]-f[i]));
        }
        rel = change/fmax(scale, 1e-300);
        double *tmp = f; f = fn; fn = tmp;
        if (rel < tol) break;
    }
    if (it > max_iter) it = max_iter;
graph_done:
    {
        PyObject *out = PyList_New(n);
        if (out == NULL) goto graph_fail;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *x = PyFloat_FromDouble(f[i]);
            if (x == NULL) { Py_DECREF(out); goto graph_fail; }
            PyList_SET_ITEM(out, i, x);
        }
        PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp); PyMem_Free(fn);
        return Py_BuildValue("Nid", out, it, rel);
    }
graph_fail:
    Py_XDECREF(grid);
    PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp); PyMem_Free(fn);
    return NULL;
}

static PyObject *LocalKernel_graph_transform(LocalKernel *self,
                                              PyObject *args) {
    PyObject *grid_obj, *frame_obj, *f_obj;
    double lu, ls;
    if (!PyArg_ParseTuple(args, "OOddO", &grid_obj, &frame_obj,
                          &lu, &ls, &f_obj)) return NULL;
    PyObject *grid = PySequence_Fast(grid_obj, "u_grid must be a sequence");
    PyObject *frame = PySequence_Fast(frame_obj, "frame must have four entries");
    PyObject *fs = PySequence_Fast(f_obj, "graph must be a sequence");
    if (!grid || !frame || !fs) {
        Py_XDECREF(grid); Py_XDECREF(frame); Py_XDECREF(fs); return NULL;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(grid);
    if (n < 5 || PySequence_Fast_GET_SIZE(fs) != n
            || PySequence_Fast_GET_SIZE(frame) != 4) {
        Py_DECREF(grid); Py_DECREF(frame); Py_DECREF(fs);
        PyErr_SetString(PyExc_ValueError, "incompatible graph transform inputs");
        return NULL;
    }
    double V[4], *u = PyMem_Malloc((size_t)n*sizeof(double));
    double *f = PyMem_Malloc((size_t)n*sizeof(double));
    double *fp = PyMem_Malloc((size_t)n*sizeof(double));
    if (!u || !f || !fp) {
        PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp);
        Py_DECREF(grid); Py_DECREF(frame); Py_DECREF(fs);
        return PyErr_NoMemory();
    }
    PyObject **vi = PySequence_Fast_ITEMS(frame);
    for (int j = 0; j < 4; j++) V[j] = PyFloat_AsDouble(vi[j]);
    PyObject **ui = PySequence_Fast_ITEMS(grid);
    PyObject **xi = PySequence_Fast_ITEMS(fs);
    for (Py_ssize_t i = 0; i < n; i++) {
        u[i] = PyFloat_AsDouble(ui[i]); f[i] = PyFloat_AsDouble(xi[i]);
    }
    Py_DECREF(grid); Py_DECREF(frame); Py_DECREF(fs);
    if (PyErr_Occurred()) goto transform_fail;
    double h = u[1]-u[0];
    fp[0] = (-25*f[0]+48*f[1]-36*f[2]+16*f[3]-3*f[4])/(12*h);
    fp[1] = (-3*f[0]-10*f[1]+18*f[2]-6*f[3]+f[4])/(12*h);
    for (Py_ssize_t i = 2; i < n-2; i++)
        fp[i] = (f[i-2]-8*f[i-1]+8*f[i+1]-f[i+2])/(12*h);
    fp[n-2] = (3*f[n-1]+10*f[n-2]-18*f[n-3]+6*f[n-4]-f[n-5])/(12*h);
    fp[n-1] = (25*f[n-1]-48*f[n-2]+36*f[n-3]-16*f[n-4]+3*f[n-5])/(12*h);
    PyObject *out = PyList_New(n);
    if (out == NULL) goto transform_fail;
    for (Py_ssize_t i = 0; i < n; i++) {
        double z[2] = {V[0]*u[i]+V[1]*f[i], V[2]*u[i]+V[3]*f[i]}, g[2];
        local_poly(self, z, g, NULL);
        double vu = V[0]*g[0]+V[2]*g[1];
        double vs = V[1]*g[0]+V[3]*g[1];
        double Ru = vu-lu*u[i], Rs = vs-ls*f[i];
        double value = i == 0 ? 0.0 : (fp[i]*(lu*u[i]+Ru)-Rs)/ls;
        PyObject *x = PyFloat_FromDouble(value);
        if (x == NULL) { Py_DECREF(out); goto transform_fail; }
        PyList_SET_ITEM(out, i, x);
    }
    PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp);
    return out;
transform_fail:
    PyMem_Free(u); PyMem_Free(f); PyMem_Free(fp);
    return NULL;
}

static PyObject *LocalKernel_poincare_graph(LocalKernel *self, PyObject *args) {
    PyObject *frame_obj, *map_obj;
    double ld, lt, alpha0, alpha1, tol = 1e-12;
    int sign, n, max_iter = 80;
    if (!PyArg_ParseTuple(args, "OOddddii|di", &frame_obj, &map_obj,
                          &ld, &lt, &alpha0, &alpha1, &sign, &n,
                          &tol, &max_iter)) return NULL;
    if (!(alpha0 > 0 && alpha1 > alpha0) || (sign != -1 && sign != 1)
            || n < 5 || !(ld*lt < 0)) {
        PyErr_SetString(PyExc_ValueError, "invalid Poincare graph parameters");
        return NULL;
    }
    double V[4], M[6];
    PyObject *fr = PySequence_Fast(frame_obj, "frame needs four entries");
    PyObject *mp = PySequence_Fast(map_obj, "map needs six entries");
    if (!fr || !mp || PySequence_Fast_GET_SIZE(fr) != 4
            || PySequence_Fast_GET_SIZE(mp) != 6) {
        Py_XDECREF(fr); Py_XDECREF(mp);
        PyErr_SetString(PyExc_ValueError, "frame/map size mismatch"); return NULL;
    }
    for (int i=0;i<4;i++) V[i]=PyFloat_AsDouble(PySequence_Fast_ITEMS(fr)[i]);
    for (int i=0;i<6;i++) M[i]=PyFloat_AsDouble(PySequence_Fast_ITEMS(mp)[i]);
    Py_DECREF(fr); Py_DECREF(mp);
    if (PyErr_Occurred()) return NULL;
    double *x=PyMem_Malloc((size_t)n*sizeof(double));
    double *h=PyMem_Calloc((size_t)n,sizeof(double));
    double *hn=PyMem_Malloc((size_t)n*sizeof(double));
    double *Q=PyMem_Malloc((size_t)n*sizeof(double));
    if (!x||!h||!hn||!Q) {
        PyMem_Free(x);PyMem_Free(h);PyMem_Free(hn);PyMem_Free(Q);
        return PyErr_NoMemory();
    }
    double dt=log(alpha1/alpha0)/(n-1), rho=fabs(ld), nu=fabs(lt);
    double kappa=nu/rho, q=kappa*dt, decay=exp(-q);
    double w0, w1;
    if (fabs(q) < 1e-3) {
        /*
         * Exact exponential weights for linearly interpolated forcing.
         * Their direct expressions lose every useful bit when kappa*dt is
         * small (the stable chart can readily have kappa < 1e-15).
         */
        double q2=q*q, q3=q2*q, q4=q2*q2;
        w1=dt*(0.5-q/6.0+q2/24.0-q3/120.0+q4/720.0);
        w0=dt*(0.5-q/3.0+q2/8.0-q3/30.0+q4/144.0);
    } else {
        w1=dt*(q-1.0+decay)/(q*q);
        w0=dt*(1.0-decay*(1.0+q))/(q*q);
    }
    for(int i=0;i<n;i++) x[i]=alpha0*exp(dt*i);
    double rel=INFINITY; int it;
    double orient=ld>0?1.0:-1.0;
    for(it=1;it<=max_iter;it++) {
        Q[0]=0.0; int valid=1;
        for(int i=1;i<n;i++) {
            double u=sign*x[i], s=h[i];
            double U=u+M[0]*u*u+M[1]*u*s+M[2]*s*s;
            double S=s+M[3]*u*u+M[4]*u*s+M[5]*s*s;
            double z[2]={V[0]*U+V[1]*S,V[2]*U+V[3]*S},g[2];
            local_poly(self,z,g,NULL);
            double ve0=V[0]*g[0]+V[2]*g[1];
            double ve1=V[1]*g[0]+V[3]*g[1];
            double j00=1+2*M[0]*u+M[1]*s, j01=M[1]*u+2*M[2]*s;
            double j10=2*M[3]*u+M[4]*s, j11=1+M[4]*u+2*M[5]*s;
            double det=j00*j11-j01*j10;
            if (!(fabs(det)>1e-14) || !isfinite(det)) {valid=0;break;}
            double vu=(j11*ve0-j01*ve1)/det;
            double vs=(-j10*ve0+j00*ve1)/det;
            double vx=sign*orient*vu, vy=orient*vs;
            if (!(fabs(vx)>1e-300) || !isfinite(vx)) {valid=0;break;}
            /* t=log(x): h_t + (nu/rho)h = x*vy/vx + kappa*h. */
            Q[i]=x[i]*vy/vx+kappa*s;
        }
        if(!valid){rel=INFINITY;break;}
        hn[0]=0.0; double scale=0,change=0;
        for(int i=1;i<n;i++) {
            hn[i]=decay*hn[i-1]+w0*Q[i-1]+w1*Q[i];
            scale=fmax(scale,fabs(hn[i]));
            change=fmax(change,fabs(hn[i]-h[i]));
        }
        rel=change/fmax(scale,1e-300);
        double *tmp=h;h=hn;hn=tmp;
        if(rel<tol)break;
    }
    if(it>max_iter)it=max_iter;
    PyObject *xo=PyList_New(n),*ho=PyList_New(n);
    if(!xo||!ho){Py_XDECREF(xo);Py_XDECREF(ho);goto pg_fail;}
    for(int i=0;i<n;i++){
        PyObject *a=PyFloat_FromDouble(x[i]),*b=PyFloat_FromDouble(h[i]);
        if(!a||!b){Py_XDECREF(a);Py_XDECREF(b);Py_DECREF(xo);Py_DECREF(ho);goto pg_fail;}
        PyList_SET_ITEM(xo,i,a);PyList_SET_ITEM(ho,i,b);
    }
    PyMem_Free(x);PyMem_Free(h);PyMem_Free(hn);PyMem_Free(Q);
    return Py_BuildValue("NNid",xo,ho,it,rel);
pg_fail:
    PyMem_Free(x);PyMem_Free(h);PyMem_Free(hn);PyMem_Free(Q);return NULL;
}

static PyMethodDef LocalKernel_methods[] = {
    {"gradient", (PyCFunction)LocalKernel_gradient, METH_VARARGS,
     "Evaluate the centered finite gradient jet."},
    {"normalized_step", (PyCFunction)LocalKernel_normalized_step, METH_VARARGS,
     "One centered normalized-gradient GL4, GL6, or GL8 step."},
    {"raw_step", (PyCFunction)LocalKernel_raw_step, METH_VARARGS,
     "One centered unnormalized-gradient GL4, GL6, or GL8 step."},
    {"curve_step", (PyCFunction)LocalKernel_curve_step, METH_VARARGS,
     "One centered integral-curve graph GL4 or GL6 step."},
    {"graph_fixed_point", (PyCFunction)LocalKernel_graph_fixed_point,
     METH_VARARGS, "Hadamard invariant graph in the Hessian eigenframe."},
    {"graph_transform", (PyCFunction)LocalKernel_graph_transform,
     METH_VARARGS, "One Hadamard graph-transform iteration."},
    {"poincare_graph", (PyCFunction)LocalKernel_poincare_graph,
     METH_VARARGS, "Poincare-conditioned integrating-factor graph fixed point."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject LocalKernelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spong._native.LocalKernel",
    .tp_basicsize = sizeof(LocalKernel),
    .tp_dealloc = (destructor)LocalKernel_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native centered critical-point jet kernel.",
    .tp_methods = LocalKernel_methods,
    .tp_init = (initproc)LocalKernel_init,
    .tp_new = PyType_GenericNew,
};

static PyMethodDef Kernel_methods[] = {
    {"slow_step", (PyCFunction)Kernel_slow_step, METH_VARARGS,
     "One scalar GL6 slow-chart step (the default)."},
    {"fast_step", (PyCFunction)Kernel_fast_step, METH_VARARGS,
     "One scalar GL6 fast-chart step (the default)."},
    {"slow_step_gl4", (PyCFunction)Kernel_slow_step_gl4, METH_VARARGS,
     "One scalar GL4 slow-chart step (kept for parity tests)."},
    {"fast_step_gl4", (PyCFunction)Kernel_fast_step_gl4, METH_VARARGS,
     "One scalar GL4 fast-chart step (kept for parity tests)."},
    {"velocities", (PyCFunction)Kernel_velocities, METH_VARARGS,
     "Descent velocities in the deviation chart."},
    {"slow_fixed_point", (PyCFunction)Kernel_slow_fixed_point, METH_VARARGS,
     "Hadamard graph transform on a uniform grid."},
    {"normalized_step", (PyCFunction)Kernel_normalized_step, METH_VARARGS,
     "One 2D normalized-gradient ascent step by GL4, GL6, or GL8."},
    {"potential_step", (PyCFunction)Kernel_potential_step, METH_VARARGS,
     "One 2D constant-potential-rate ascent step by GL4, GL6, or GL8."},
    {NULL, NULL, 0, NULL}
};

static PyObject *native_resolution_preflight(
        PyObject *self, PyObject *args) {
    (void)self;
    spong_morse_analysis analysis;
    spong_resolution_policy policy;
    unsigned int enabled;
    if (!PyArg_ParseTuple(
            args, "ppppdpdpdIddd",
            &analysis.exact_morse,
            &analysis.A_positive,
            &analysis.critical_coordinates_binary64_distinct,
            &analysis.has_root_collision_margin,
            &analysis.root_collision_margin_log2_eps,
            &analysis.has_hessian_relative_nonsingularity,
            &analysis.min_hessian_relative_nonsingularity,
            &analysis.has_gamma_target_product,
            &analysis.max_gamma_target_product_log2,
            &enabled,
            &policy.min_root_collision_margin_log2_eps,
            &policy.max_hessian_condition_loss_bits,
            &policy.max_gamma_target_product_log2))
        return NULL;
    policy.enabled = (uint32_t)enabled;
    spong_resolution_result out;
    if (spong_resolution_preflight(&analysis, &policy, &out) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "native resolution preflight failed");
        return NULL;
    }
    return Py_BuildValue(
        "iiI", (int)out.status, (int)out.primary_reason,
        (unsigned int)out.reason_mask);
}

static PyObject *native_resolution_finalize(
        PyObject *self, PyObject *args) {
    (void)self;
    int certified, branch_aborted;
    if (!PyArg_ParseTuple(args, "pp", &certified, &branch_aborted))
        return NULL;
    spong_resolution_result out;
    if (spong_resolution_finalize(certified, branch_aborted, &out) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "native resolution finalize failed");
        return NULL;
    }
    return Py_BuildValue(
        "iiI", (int)out.status, (int)out.primary_reason,
        (unsigned int)out.reason_mask);
}

static PyObject *native_topology_decide(PyObject *self, PyObject *args) {
    (void)self;
    spong_topology_analysis analysis;
    unsigned long long value[12];
    int branch_aborted;
    if (!PyArg_ParseTuple(
            args, "KKKKKKKKKKKKp",
            &value[0], &value[1], &value[2], &value[3],
            &value[4], &value[5], &value[6], &value[7],
            &value[8], &value[9], &value[10], &value[11],
            &branch_aborted))
        return NULL;
    analysis.saddle_count = (uint64_t)value[0];
    analysis.branch_count = (uint64_t)value[1];
    analysis.stable_count = (uint64_t)value[2];
    analysis.unstable_count = (uint64_t)value[3];
    analysis.segment_count = (uint64_t)value[4];
    analysis.segment_budget = (uint64_t)value[5];
    analysis.raw_event_count = (uint64_t)value[6];
    analysis.raw_event_budget = (uint64_t)value[7];
    analysis.forbidden_count = (uint64_t)value[8];
    analysis.ambiguous_count = (uint64_t)value[9];
    analysis.uncertified_unstable_ends = (uint64_t)value[10];
    analysis.uncertified_stable_tails = (uint64_t)value[11];
    analysis.branch_aborted = branch_aborted;
    spong_topology_result result;
    if (spong_topology_decide(&analysis, &result) != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid topology analysis");
        return NULL;
    }
    return Py_BuildValue(
        "iiisKK", (int)result.certified, (int)result.audit_complete,
        (int)result.branch_inventory_certified,
        spong_topology_reason_name(result.primary_reason),
        (unsigned long long)result.expected_stable,
        (unsigned long long)result.expected_unstable);
}

static PyObject *native_sturm_analyze(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *coefficients;
    unsigned long long max_bits, max_coefficients, max_steps;
    if (!PyArg_ParseTuple(args, "OKKK", &coefficients, &max_bits,
                          &max_coefficients, &max_steps))
        return NULL;
    PyObject *seq = PySequence_Fast(
        coefficients, "coefficients must be a nonempty sequence");
    if (seq == NULL)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "coefficient sequence is empty");
        return NULL;
    }
    const char **text = (const char **)PyMem_Calloc(
        (size_t)n, sizeof(char *));
    PyObject **owned = (PyObject **)PyMem_Calloc(
        (size_t)n, sizeof(PyObject *));
    if (text == NULL || owned == NULL) {
        PyMem_Free(text);
        PyMem_Free(owned);
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        owned[i] = PyObject_Str(items[i]);
        if (owned[i] == NULL)
            goto exact_cleanup;
        text[i] = PyUnicode_AsUTF8(owned[i]);
        if (text[i] == NULL)
            goto exact_cleanup;
    }
    spong_exact_policy policy = {
        (uint64_t)max_bits, (uint64_t)max_coefficients, (uint64_t)max_steps
    };
    spong_sturm_analysis out;
    (void)spong_sturm_analyze_decimal(
        text, (size_t)n, &policy, &out);
    for (Py_ssize_t i = 0; i < n; ++i)
        Py_XDECREF(owned[i]);
    PyMem_Free(text);
    PyMem_Free(owned);
    Py_DECREF(seq);
    return Py_BuildValue(
        "{s:i,s:I,s:I,s:i,s:i,s:K,s:K,s:K,s:K}",
        "status", (int)out.status,
        "distinct_real_roots", (unsigned int)out.distinct_real_roots,
        "repeated_real_roots", (unsigned int)out.repeated_real_roots,
        "input_degree", (int)out.input_degree,
        "squarefree_degree", (int)out.squarefree_degree,
        "prs_steps", (unsigned long long)out.work.prs_steps,
        "chain_polynomials",
            (unsigned long long)out.work.chain_polynomials,
        "chain_coefficients",
            (unsigned long long)out.work.chain_coefficients,
        "peak_coefficient_bits",
            (unsigned long long)out.work.peak_coefficient_bits);

exact_cleanup:
    for (Py_ssize_t i = 0; i < n; ++i)
        Py_XDECREF(owned[i]);
    PyMem_Free(text);
    PyMem_Free(owned);
    Py_DECREF(seq);
    return NULL;
}

static PyObject *native_sturm_count_interval(
        PyObject *self, PyObject *args) {
    (void)self;
    PyObject *coefficients, *lo_num, *lo_den, *hi_num, *hi_den;
    unsigned long long max_bits, max_coefficients, max_steps;
    if (!PyArg_ParseTuple(
            args, "OOOOOKKK", &coefficients,
            &lo_num, &lo_den, &hi_num, &hi_den,
            &max_bits, &max_coefficients, &max_steps))
        return NULL;
    PyObject *seq = PySequence_Fast(
        coefficients, "coefficients must be a nonempty sequence");
    if (seq == NULL)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "coefficient sequence is empty");
        return NULL;
    }
    const char **text = (const char **)PyMem_Calloc(
        (size_t)n, sizeof(char *));
    PyObject **owned = (PyObject **)PyMem_Calloc(
        (size_t)n, sizeof(PyObject *));
    PyObject *bounds[4] = {NULL, NULL, NULL, NULL};
    if (text == NULL || owned == NULL) {
        PyMem_Free(text);
        PyMem_Free(owned);
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        owned[i] = PyObject_Str(items[i]);
        if (owned[i] == NULL)
            goto interval_cleanup;
        text[i] = PyUnicode_AsUTF8(owned[i]);
        if (text[i] == NULL)
            goto interval_cleanup;
    }
    PyObject *input_bounds[4] = {lo_num, lo_den, hi_num, hi_den};
    const char *bound_text[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; ++i) {
        if (input_bounds[i] == Py_None)
            continue;
        bounds[i] = PyObject_Str(input_bounds[i]);
        if (bounds[i] == NULL)
            goto interval_cleanup;
        bound_text[i] = PyUnicode_AsUTF8(bounds[i]);
        if (bound_text[i] == NULL)
            goto interval_cleanup;
    }
    spong_exact_policy policy = {
        (uint64_t)max_bits, (uint64_t)max_coefficients, (uint64_t)max_steps
    };
    spong_sturm_analysis analysis;
    spong_sturm_plan *plan = NULL;
    uint32_t count = 0;
    int created = spong_sturm_plan_create_decimal(
        text, (size_t)n, &policy, &plan, &analysis);
    int counted = (created == 0) ? spong_sturm_plan_count(
        plan, bound_text[0], bound_text[1],
        bound_text[2], bound_text[3], &count) : -1;
    spong_sturm_plan_destroy(plan);
    for (int i = 0; i < 4; ++i)
        Py_XDECREF(bounds[i]);
    for (Py_ssize_t i = 0; i < n; ++i)
        Py_XDECREF(owned[i]);
    PyMem_Free(text);
    PyMem_Free(owned);
    Py_DECREF(seq);
    if (created == 0 && counted != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid rational interval");
        return NULL;
    }
    return Py_BuildValue(
        "{s:i,s:I,s:K,s:K,s:K,s:K}",
        "status", (int)analysis.status,
        "count", (unsigned int)count,
        "prs_steps", (unsigned long long)analysis.work.prs_steps,
        "chain_polynomials",
            (unsigned long long)analysis.work.chain_polynomials,
        "chain_coefficients",
            (unsigned long long)analysis.work.chain_coefficients,
        "peak_coefficient_bits",
            (unsigned long long)analysis.work.peak_coefficient_bits);

interval_cleanup:
    for (int i = 0; i < 4; ++i)
        Py_XDECREF(bounds[i]);
    for (Py_ssize_t i = 0; i < n; ++i)
        Py_XDECREF(owned[i]);
    PyMem_Free(text);
    PyMem_Free(owned);
    Py_DECREF(seq);
    return NULL;
}

static void NativeSturmPlan_dealloc(NativeSturmPlan *self) {
    spong_sturm_plan_destroy(self->plan);
    self->plan = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int NativeSturmPlan_init(
        NativeSturmPlan *self, PyObject *args, PyObject *kwds) {
    PyObject *coefficients;
    unsigned long long max_bits = 0, max_coefficients = 0, max_steps = 0;
    static char *kwlist[] = {
        "coefficients", "max_coefficient_bits", "max_chain_coefficients",
        "max_prs_steps", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|KKK", kwlist, &coefficients, &max_bits,
            &max_coefficients, &max_steps))
        return -1;
    PyObject *seq = PySequence_Fast(
        coefficients, "coefficients must be a nonempty sequence");
    if (seq == NULL)
        return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "coefficient sequence is empty");
        return -1;
    }
    const char **text = (const char **)PyMem_Calloc(
        (size_t)n, sizeof(char *));
    PyObject **owned = (PyObject **)PyMem_Calloc(
        (size_t)n, sizeof(PyObject *));
    if (text == NULL || owned == NULL) {
        PyMem_Free(text);
        PyMem_Free(owned);
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    int result = -1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        owned[i] = PyObject_Str(items[i]);
        if (owned[i] == NULL)
            goto cleanup;
        text[i] = PyUnicode_AsUTF8(owned[i]);
        if (text[i] == NULL)
            goto cleanup;
    }
    spong_exact_policy policy = {
        (uint64_t)max_bits, (uint64_t)max_coefficients, (uint64_t)max_steps
    };
    spong_sturm_plan_destroy(self->plan);
    self->plan = NULL;
    if (spong_sturm_plan_create_decimal(
            text, (size_t)n, &policy, &self->plan, &self->analysis) != 0) {
        PyErr_Format(
            PyExc_ArithmeticError,
            "native Sturm plan refused with status %d",
            (int)self->analysis.status);
        goto cleanup;
    }
    result = 0;
cleanup:
    for (Py_ssize_t i = 0; i < n; ++i)
        Py_XDECREF(owned[i]);
    PyMem_Free(text);
    PyMem_Free(owned);
    Py_DECREF(seq);
    return result;
}

static int rational_text(
        PyObject *value, PyObject **numerator, PyObject **denominator,
        PyObject **numerator_text, PyObject **denominator_text,
        const char **num, const char **den) {
    *numerator = *denominator = *numerator_text = *denominator_text = NULL;
    *num = *den = NULL;
    if (value == Py_None)
        return 0;
    *numerator = PyObject_GetAttrString(value, "numerator");
    *denominator = PyObject_GetAttrString(value, "denominator");
    if (*numerator == NULL || *denominator == NULL)
        return -1;
    *numerator_text = PyObject_Str(*numerator);
    *denominator_text = PyObject_Str(*denominator);
    if (*numerator_text == NULL || *denominator_text == NULL)
        return -1;
    *num = PyUnicode_AsUTF8(*numerator_text);
    *den = PyUnicode_AsUTF8(*denominator_text);
    return (*num != NULL && *den != NULL) ? 0 : -1;
}

static PyObject *NativeSturmPlan_count(
        NativeSturmPlan *self, PyObject *args) {
    PyObject *lower, *upper;
    if (!PyArg_ParseTuple(args, "OO", &lower, &upper))
        return NULL;
    PyObject *objects[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const char *lo_num, *lo_den, *hi_num, *hi_den;
    if (rational_text(
            lower, &objects[0], &objects[1], &objects[2], &objects[3],
            &lo_num, &lo_den) != 0
            || rational_text(
                upper, &objects[4], &objects[5], &objects[6], &objects[7],
                &hi_num, &hi_den) != 0)
        goto cleanup;
    uint32_t count;
    if (spong_sturm_plan_count(
            self->plan, lo_num, lo_den, hi_num, hi_den, &count) != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid rational interval");
        goto cleanup;
    }
    for (size_t i = 0; i < 8; ++i)
        Py_XDECREF(objects[i]);
    return PyLong_FromUnsignedLong((unsigned long)count);
cleanup:
    for (size_t i = 0; i < 8; ++i)
        Py_XDECREF(objects[i]);
    return NULL;
}

static PyObject *NativeSturmPlan_sign_at(
        NativeSturmPlan *self, PyObject *args) {
    PyObject *value;
    if (!PyArg_ParseTuple(args, "O", &value))
        return NULL;
    PyObject *objects[4] = {NULL, NULL, NULL, NULL};
    const char *num, *den;
    if (rational_text(
            value, &objects[0], &objects[1], &objects[2], &objects[3],
            &num, &den) != 0)
        goto cleanup;
    int32_t sign;
    if (spong_sturm_plan_sign_at(self->plan, num, den, &sign) != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid rational point");
        goto cleanup;
    }
    for (size_t i = 0; i < 4; ++i)
        Py_XDECREF(objects[i]);
    return PyLong_FromLong((long)sign);
cleanup:
    for (size_t i = 0; i < 4; ++i)
        Py_XDECREF(objects[i]);
    return NULL;
}

static PyObject *NativeSturmPlan_refine(
        NativeSturmPlan *self, PyObject *args) {
    PyObject *lower, *upper, *relative_width;
    unsigned long long max_bisections = 0, max_endpoint_bits = 0;
    if (!PyArg_ParseTuple(
            args, "OOO|KK", &lower, &upper, &relative_width,
            &max_bisections, &max_endpoint_bits))
        return NULL;
    PyObject *objects[12] = {
        NULL, NULL, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, NULL, NULL, NULL
    };
    const char *lo_num, *lo_den, *hi_num, *hi_den, *rel_num, *rel_den;
    if (rational_text(
            lower, &objects[0], &objects[1], &objects[2], &objects[3],
            &lo_num, &lo_den) != 0
            || rational_text(
                upper, &objects[4], &objects[5], &objects[6], &objects[7],
                &hi_num, &hi_den) != 0
            || rational_text(
                relative_width, &objects[8], &objects[9],
                &objects[10], &objects[11], &rel_num, &rel_den) != 0)
        goto cleanup;
    spong_refinement_policy policy = {
        (uint64_t)max_bisections, (uint64_t)max_endpoint_bits
    };
    spong_refinement_work work;
    spong_root_interval *interval = NULL;
    int refined = spong_sturm_plan_refine(
        self->plan, lo_num, lo_den, hi_num, hi_den, rel_num, rel_den,
        &policy, &interval, &work);
    PyObject *item = Py_None;
    Py_INCREF(item);
    if (refined == 0) {
        Py_DECREF(item);
        item = Py_BuildValue(
            "ssssI", interval->lower_numerator, interval->lower_denominator,
            interval->upper_numerator, interval->upper_denominator,
            (unsigned int)interval->exact);
        if (item == NULL) {
            spong_root_intervals_destroy(interval, 1);
            goto cleanup;
        }
    }
    spong_root_intervals_destroy(interval, refined == 0 ? 1 : 0);
    for (size_t i = 0; i < 12; ++i)
        Py_XDECREF(objects[i]);
    return Py_BuildValue(
        "{s:i,s:N,s:K,s:K}",
        "status", (int)work.status,
        "interval", item,
        "bisections", (unsigned long long)work.bisections,
        "max_endpoint_bits",
            (unsigned long long)work.max_endpoint_bits);
cleanup:
    for (size_t i = 0; i < 12; ++i)
        Py_XDECREF(objects[i]);
    return NULL;
}

static PyObject *NativeSturmPlan_stats(
        NativeSturmPlan *self, PyObject *Py_UNUSED(ignored)) {
    return Py_BuildValue(
        "{s:i,s:I,s:I,s:i,s:i,s:n,s:K,s:K,s:K,s:K,s:K}",
        "status", (int)self->analysis.status,
        "distinct_real_roots",
            (unsigned int)self->analysis.distinct_real_roots,
        "repeated_real_roots",
            (unsigned int)self->analysis.repeated_real_roots,
        "input_degree", (int)self->analysis.input_degree,
        "squarefree_degree", (int)self->analysis.squarefree_degree,
        "sturm_chain_length",
            (Py_ssize_t)spong_sturm_plan_chain_length(self->plan),
        "sturm_chain_coefficients",
            (unsigned long long)spong_sturm_plan_chain_coefficients(self->plan),
        "prs_steps", (unsigned long long)self->analysis.work.prs_steps,
        "chain_polynomials",
            (unsigned long long)self->analysis.work.chain_polynomials,
        "chain_coefficients",
            (unsigned long long)self->analysis.work.chain_coefficients,
        "peak_coefficient_bits",
            (unsigned long long)self->analysis.work.peak_coefficient_bits);
}

static PyObject *NativeSturmPlan_isolate(
        NativeSturmPlan *self, PyObject *args) {
    unsigned long long max_nodes = 0, max_punctures = 0;
    unsigned long long max_endpoint_bits = 0, max_intervals = 0;
    if (!PyArg_ParseTuple(
            args, "|KKKK", &max_nodes, &max_punctures,
            &max_endpoint_bits, &max_intervals))
        return NULL;
    spong_isolation_policy policy = {
        (uint64_t)max_nodes, (uint64_t)max_punctures,
        (uint64_t)max_endpoint_bits, (uint64_t)max_intervals
    };
    spong_isolation_work work;
    spong_root_interval *intervals = NULL;
    size_t count = 0;
    int isolated = spong_sturm_plan_isolate(
        self->plan, &policy, &intervals, &count, &work);
    PyObject *items = PyList_New(isolated == 0 ? (Py_ssize_t)count : 0);
    if (items == NULL) {
        spong_root_intervals_destroy(intervals, count);
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        PyObject *item = Py_BuildValue(
            "ssssI", intervals[i].lower_numerator,
            intervals[i].lower_denominator, intervals[i].upper_numerator,
            intervals[i].upper_denominator, (unsigned int)intervals[i].exact);
        if (item == NULL) {
            Py_DECREF(items);
            spong_root_intervals_destroy(intervals, count);
            return NULL;
        }
        PyList_SET_ITEM(items, (Py_ssize_t)i, item);
    }
    spong_root_intervals_destroy(intervals, count);
    return Py_BuildValue(
        "{s:i,s:N,s:K,s:K,s:K,s:K,s:K,s:K}",
        "status", (int)work.status,
        "intervals", items,
        "subdivision_nodes",
            (unsigned long long)work.subdivision_nodes,
        "variation_evaluations",
            (unsigned long long)work.variation_evaluations,
        "polynomial_evaluations",
            (unsigned long long)work.polynomial_evaluations,
        "puncture_halvings",
            (unsigned long long)work.puncture_halvings,
        "max_subdivision_depth",
            (unsigned long long)work.max_subdivision_depth,
        "max_endpoint_bits",
            (unsigned long long)work.max_endpoint_bits);
}

static int contact_buffer(PyObject *object, Py_buffer *view) {
    if (PyObject_GetBuffer(
            object, view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0)
        return -1;
    if (view->ndim != 2 || view->shape == NULL || view->shape[0] < 2
            || view->shape[1] != 2 || view->itemsize != (Py_ssize_t)sizeof(double)
            || view->format == NULL || strcmp(view->format, "d") != 0
            || !PyBuffer_IsContiguous(view, 'C')) {
        PyBuffer_Release(view);
        memset(view, 0, sizeof(*view));
        PyErr_SetString(
            PyExc_ValueError,
            "contact curves must be C-contiguous float64 arrays of shape (n,2)");
        return -1;
    }
    return 0;
}

static void NativeContactScan_dealloc(NativeContactScan *self) {
    spong_contact_scan_destroy(self->scan);
    if (self->second.obj != NULL) PyBuffer_Release(&self->second);
    if (self->first.obj != NULL) PyBuffer_Release(&self->first);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int NativeContactScan_init(
        NativeContactScan *self, PyObject *args, PyObject *kwds) {
    PyObject *first, *second;
    double tolerance;
    int self_scan = 0;
    static char *kwlist[] = {
        "first", "second", "tolerance", "self_scan", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOd|p", kwlist,
            &first, &second, &tolerance, &self_scan))
        return -1;
    if (contact_buffer(first, &self->first) < 0) return -1;
    if (!self_scan && contact_buffer(second, &self->second) < 0) {
        PyBuffer_Release(&self->first);
        memset(&self->first, 0, sizeof(self->first));
        return -1;
    }
    self->scan = spong_contact_scan_create(
        (const double *)self->first.buf, (size_t)self->first.shape[0],
        self_scan ? NULL : (const double *)self->second.buf,
        self_scan ? 0 : (size_t)self->second.shape[0],
        tolerance, self_scan);
    if (self->scan == NULL) {
        if (self->second.obj != NULL) PyBuffer_Release(&self->second);
        PyBuffer_Release(&self->first);
        memset(&self->first, 0, sizeof(self->first));
        memset(&self->second, 0, sizeof(self->second));
        PyErr_SetString(PyExc_ValueError, "could not initialize contact scan");
        return -1;
    }
    return 0;
}

static PyObject *NativeContactScan_iternext(NativeContactScan *self) {
    spong_contact_event event;
    int status = spong_contact_scan_next(self->scan, &event);
    if (status < 0) {
        PyErr_SetString(PyExc_RuntimeError, "native contact scan failed");
        return NULL;
    }
    if (status == 0) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    const char *kind = event.kind == SPONG_CONTACT_CROSS
        ? "cross" : "ambiguous";
    return Py_BuildValue(
        "KKs(dd)",
        (unsigned long long)event.first_segment,
        (unsigned long long)event.second_segment,
        kind, event.x, event.y);
}

static PyTypeObject NativeContactScanType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spong._native.ContactScan",
    .tp_basicsize = sizeof(NativeContactScan),
    .tp_dealloc = (destructor)NativeContactScan_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Streaming native BVH contact scan over float64 polylines.",
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)NativeContactScan_iternext,
    .tp_init = (initproc)NativeContactScan_init,
    .tp_new = PyType_GenericNew,
};

static PyMethodDef NativeSturmPlan_methods[] = {
    {"count", (PyCFunction)NativeSturmPlan_count, METH_VARARGS,
     "Count distinct roots exactly on (lower, upper]."},
    {"sign_at", (PyCFunction)NativeSturmPlan_sign_at, METH_VARARGS,
     "Evaluate the exact sign of the original polynomial."},
    {"refine", (PyCFunction)NativeSturmPlan_refine, METH_VARARGS,
     "Refine a certified one-root interval with bounded exact work."},
    {"isolate", (PyCFunction)NativeSturmPlan_isolate, METH_VARARGS,
     "Isolate all distinct real roots exactly with bounded work."},
    {"stats", (PyCFunction)NativeSturmPlan_stats, METH_NOARGS,
     "Return exact-plan construction statistics."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject NativeSturmPlanType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spong._native.SturmPlan",
    .tp_basicsize = sizeof(NativeSturmPlan),
    .tp_dealloc = (destructor)NativeSturmPlan_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Persistent exact GMP squarefree Sturm chain.",
    .tp_methods = NativeSturmPlan_methods,
    .tp_init = (initproc)NativeSturmPlan_init,
    .tp_new = PyType_GenericNew,
};

static PyMethodDef module_methods[] = {
    {"resolution_preflight", native_resolution_preflight, METH_VARARGS,
     "Apply the shared C resolution policy to exact-Morse measurements."},
    {"resolution_finalize", native_resolution_finalize, METH_VARARGS,
     "Map a geometry certificate to the shared terminal resolution state."},
    {"topology_decide", native_topology_decide, METH_VARARGS,
     "Reduce topology evidence to the shared certificate outcome."},
    {"sturm_analyze", native_sturm_analyze, METH_VARARGS,
     "Exact GMP Sturm analysis of an integer polynomial."},
    {"sturm_count_interval", native_sturm_count_interval, METH_VARARGS,
     "Exact GMP distinct-root count on a rational interval."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject KernelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spong._native.Kernel",
    .tp_basicsize = sizeof(Kernel),
    .tp_dealloc = (destructor)Kernel_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native scalar portrait kernel.",
    .tp_methods = Kernel_methods,
    .tp_init = (initproc)Kernel_init,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "spong._native",
    .m_doc = "Optional native scalar kernels for spong.",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject *m;
    if (PyType_Ready(&KernelType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&LocalKernelType) < 0) return NULL;
    if (PyType_Ready(&NativeSturmPlanType) < 0) return NULL;
    if (PyType_Ready(&NativeContactScanType) < 0) return NULL;
    m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }
    Py_INCREF(&KernelType);
    if (PyModule_AddObject(m, "Kernel", (PyObject *)&KernelType) < 0) {
        Py_DECREF(&KernelType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&LocalKernelType);
    if (PyModule_AddObject(m, "LocalKernel",
                           (PyObject *)&LocalKernelType) < 0) {
        Py_DECREF(&LocalKernelType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&NativeSturmPlanType);
    if (PyModule_AddObject(m, "SturmPlan",
                           (PyObject *)&NativeSturmPlanType) < 0) {
        Py_DECREF(&NativeSturmPlanType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&NativeContactScanType);
    if (PyModule_AddObject(m, "ContactScan",
                           (PyObject *)&NativeContactScanType) < 0) {
        Py_DECREF(&NativeContactScanType);
        Py_DECREF(m);
        return NULL;
    }
#define ADD_NATIVE_CONSTANT(name)                                           \
    do {                                                                    \
        if (PyModule_AddIntConstant(m, #name, (long)(name)) < 0) {          \
            Py_DECREF(m);                                                   \
            return NULL;                                                    \
        }                                                                   \
    } while (0)
    ADD_NATIVE_CONSTANT(SPONG_RESOLUTION_PROCEED);
    ADD_NATIVE_CONSTANT(SPONG_CERTIFIED_NON_MORSE);
    ADD_NATIVE_CONSTANT(SPONG_MORSE_NUMERICALLY_UNRESOLVED);
    ADD_NATIVE_CONSTANT(SPONG_CERTIFIED_PORTRAIT);
    ADD_NATIVE_CONSTANT(SPONG_REASON_NONE);
    ADD_NATIVE_CONSTANT(SPONG_REASON_EXACT_NON_MORSE);
    ADD_NATIVE_CONSTANT(SPONG_REASON_MODEL_HYPOTHESIS);
    ADD_NATIVE_CONSTANT(SPONG_REASON_ROOT_COLLISION_MARGIN);
    ADD_NATIVE_CONSTANT(SPONG_REASON_HESSIAN_RESOLUTION);
    ADD_NATIVE_CONSTANT(SPONG_REASON_LOCAL_NONLINEARITY);
    ADD_NATIVE_CONSTANT(SPONG_REASON_BINARY64_COORDINATE_COLLISION);
    ADD_NATIVE_CONSTANT(SPONG_REASON_ARITHMETIC_FAILURE);
    ADD_NATIVE_CONSTANT(SPONG_REASON_BRANCH_ABORT);
    ADD_NATIVE_CONSTANT(SPONG_REASON_TOPOLOGY_UNRESOLVED);
    ADD_NATIVE_CONSTANT(SPONG_POLICY_ROOT_COLLISION);
    ADD_NATIVE_CONSTANT(SPONG_POLICY_HESSIAN_CONDITION);
    ADD_NATIVE_CONSTANT(SPONG_POLICY_LOCAL_GAMMA);
    ADD_NATIVE_CONSTANT(SPONG_POLICY_DISTINCT_BINARY64);
    ADD_NATIVE_CONSTANT(SPONG_ABI_VERSION);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_OK);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_INVALID_ARGUMENT);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_ALLOCATION_FAILURE);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_PARSE_FAILURE);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_WORK_LIMIT);
    ADD_NATIVE_CONSTANT(SPONG_EXACT_INTERNAL_FAILURE);
#undef ADD_NATIVE_CONSTANT
    return m;
}
