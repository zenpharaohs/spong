#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    PyObject_HEAD
    Py_ssize_t na, nap, napp, nb, nbp, nbpp, nn, nnp;
    double *a, *ap, *app, *b, *bp, *bpp, *n, *np;
} Kernel;

static const double SQRT3 = 1.73205080756887729352744634150587237;
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

static double slow_fj(Kernel *k, double b, double w, double *jac) {
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

static double fast_fj(Kernel *k, double w, double b, double *jac) {
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

typedef double (*FJ)(Kernel *, double, double, double *);

static double gl4_step(Kernel *k, FJ fj, double x, double y, double h) {
    const double c1 = 0.5 - SQRT3 / 6.0;
    const double c2 = 0.5 + SQRT3 / 6.0;
    const double a11 = 0.25;
    const double a12 = 0.25 - SQRT3 / 6.0;
    const double a21 = 0.25 + SQRT3 / 6.0;
    const double a22 = 0.25;
    double x1 = x + c1 * h;
    double x2 = x + c2 * h;
    double K1 = fj(k, x, y, NULL);
    double K2 = K1;
    for (int it = 0; it < NEWTON_MAX; it++) {
        double Y1 = y + h * (a11 * K1 + a12 * K2);
        double Y2 = y + h * (a21 * K1 + a22 * K2);
        double f1 = fj(k, x1, Y1, NULL);
        double f2 = fj(k, x2, Y2, NULL);
        double r1 = K1 - f1;
        double r2 = K2 - f2;
        double m = fabs(K1) > fabs(K2) ? fabs(K1) : fabs(K2);
        double r = fabs(r1) > fabs(r2) ? fabs(r1) : fabs(r2);
        if (r < NEWTON_TOL * (1.0 + m)) {
            break;
        }
        double J1, J2;
        (void)fj(k, x1, Y1, &J1);
        (void)fj(k, x2, Y2, &J2);
        double m11 = 1.0 - h * a11 * J1;
        double m12 = -h * a12 * J1;
        double m21 = -h * a21 * J2;
        double m22 = 1.0 - h * a22 * J2;
        double det = m11 * m22 - m12 * m21;
        K1 += (-m22 * r1 + m12 * r2) / det;
        K2 += (m21 * r1 - m11 * r2) / det;
    }
    return y + h * 0.5 * (K1 + K2);
}

static PyObject *Kernel_slow_step(Kernel *self, PyObject *args) {
    double x, y, h;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &h)) {
        return NULL;
    }
    return PyFloat_FromDouble(gl4_step(self, slow_fj, x, y, h));
}

static PyObject *Kernel_fast_step(Kernel *self, PyObject *args) {
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

static PyMethodDef Kernel_methods[] = {
    {"slow_step", (PyCFunction)Kernel_slow_step, METH_VARARGS,
     "One scalar GL4 slow-chart step."},
    {"fast_step", (PyCFunction)Kernel_fast_step, METH_VARARGS,
     "One scalar GL4 fast-chart step."},
    {"velocities", (PyCFunction)Kernel_velocities, METH_VARARGS,
     "Descent velocities in the deviation chart."},
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
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject *m;
    if (PyType_Ready(&KernelType) < 0) {
        return NULL;
    }
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
    return m;
}
