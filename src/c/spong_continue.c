/*
 * spong_continue.c -- native engine segment.  Milestone 2, ordinary path,
 * plus the floor-fallback ladder, the normalized-arclength rescue and the
 * stall trim (2026-09-02; see the header for why).
 *
 * A line-by-line port of charts._continue_curve.  The reference
 * implementation stays authoritative; this is the fast path, and
 * tests/corpus/continue_curve.json judges it.
 *
 * STATUS
 *   covered here : chart switching, the shallow handoff (healthy case and
 *                  the stall trim), step halving, the descent-realization
 *                  test, the turn budget, the 1.06 chord ramp, capture, box
 *                  exit, stationary / switch-limit / nonfinite / max-steps /
 *                  step-failure exits, the floor-fallback ladder (GL4 on the
 *                  active chart, GL6 and GL4 on the other), the
 *                  normalized-arclength rescue at orders 8, 6, 4
 *   delegated    : the centered-chart rescue, and only when the caller has
 *                  a centered local jet (centered_available)
 *
 * DUPLICATED CODE, DELIBERATELY AND TEMPORARILY
 *   slow_fj, fast_fj and gl6_step below are verbatim copies of the statics in
 *   _native.c.  Duplicating a delicate stage solver is exactly how two
 *   implementations drift apart, so this is not the resting state: once this
 *   file is in the build, _native.c should include it and delete its copies,
 *   and tests/test_native_parity.py should pin Kernel.slow_step against this
 *   translation unit.  Copying verbatim first keeps the initial diff readable
 *   and bit-identical; merging is a separate, checkable commit.
 */

#include "spong/spong_continue.h"
#include "spong/spong_gauss2.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/*
 * FLOATING-POINT CONTRACTION IS MIXED, DELIBERATELY, AND PER FUNCTION.
 *
 * The reference implementation evaluates emitted points in Python --
 * model.horner, s_a_star, _s_P, _s_depth_gauge_floor -- where no FMA is
 * possible, and takes its Gauss steps inside _native.c, which setup.py builds
 * at -O3 with clang's default contraction.  So the SAME polynomial is
 * evaluated two ways in one trace: uncontracted when a vertex is written,
 * contracted when a stage is solved.  Reproducing that needs two Horners, not
 * one, and per-function control rather than a compile flag.
 *
 * Measured.  Contracted throughout: 14 of 45 segments match, and the failures
 * start at vertex 0, which is a bare s_a_star ratio.  Uncontracted
 * throughout: 2 match, because the stage solve parts company with _native.c.
 * File-scope `#pragma STDC FP_CONTRACT` is not honoured selectively by clang
 * -- it gave 3 -- so the block-scope clang pragma is used instead.
 */
#if defined(__clang__)
#  define SPONG_FP_EXACT   _Pragma("clang fp contract(off)")
#  define SPONG_FP_FUSED   _Pragma("clang fp contract(on)")
#else
#  define SPONG_FP_EXACT
#  define SPONG_FP_FUSED
#endif

/* charts.py module constants.  Kept as literals rather than derived so a
 * change on either side shows up as a corpus failure rather than silently
 * tracking. */
#define R_SWITCH        20.0
#define MAX_SWITCHES    12
#define KAPPA_HI        1e4
#define RETRY_ATTEMPTS  8
#define STALL_WINDOW    12
#define NOISE_C         4.0     /* gauss._NOISE_C: evaluation-floor convergence */

static const double SQRT15 = 3.87298334620741688517926539978239961;
static const double SQRT3  = 1.73205080756887729352744634150587237;
static const double STAGE_GUARD = 1e-6;
static const double NEWTON_TOL = 1e-13;
static const int NEWTON_MAX = 30;

/* np.cos(np.radians(0.75)); computed the same way so the comparison against
 * the reference is exact. */
static double turn_max(void) { return cos(0.75 * 3.14159265358979323846 / 180.0); }

/* model.horner: the Python scalar fast path, which cannot fuse. */
static double horner_p(const double *c, size_t n, double x) {
    SPONG_FP_EXACT
    double acc = 0.0;
    for (size_t i = n; i-- > 0;) acc = acc * x + c[i];
    return acc;
}

/* _native.c's horner, compiled with clang's default fusion. */
static double horner_c(const double *c, size_t n, double x) {
    SPONG_FP_FUSED
    double acc = 0.0;
    for (size_t i = n; i-- > 0;) acc = acc * x + c[i];
    return acc;
}

/* Horner on |c_i| at |x| -- the running-error bound's building block;
 * fused, as _native.c's horner_abs is. */
static double horner_abs_c(const double *c, size_t n, double x) {
    SPONG_FP_FUSED
    double acc = 0.0, ax = fabs(x);
    for (size_t i = n; i-- > 0;) acc = acc * ax + fabs(c[i]);
    return acc;
}

/* ------------------------------------------------------------------ *
 * derived field quantities -- Python side, uncontracted                *
 * ------------------------------------------------------------------ */

typedef struct {
    double A, Ap, App, B, Bp, Bpp, N, Np;
} base;

static void eval_base_p(const spong_continue_field *f, double b, base *o) {
    o->A   = horner_p(f->A,   f->nA,   b);
    o->Ap  = horner_p(f->Ap,  f->nAp,  b);
    o->App = horner_p(f->App, f->nApp, b);
    o->B   = horner_p(f->B,   f->nB,   b);
    o->Bp  = horner_p(f->Bp,  f->nBp,  b);
    o->Bpp = horner_p(f->Bpp, f->nBpp, b);
    o->N   = horner_p(f->N,   f->nN,   b);
    o->Np  = horner_p(f->Np,  f->nNp,  b);
}

static void eval_base_c(const spong_continue_field *f, double b, base *o) {
    o->A   = horner_c(f->A,   f->nA,   b);
    o->Ap  = horner_c(f->Ap,  f->nAp,  b);
    o->App = horner_c(f->App, f->nApp, b);
    o->B   = horner_c(f->B,   f->nB,   b);
    o->Bp  = horner_c(f->Bp,  f->nBp,  b);
    o->Bpp = horner_c(f->Bpp, f->nBpp, b);
    o->N   = horner_c(f->N,   f->nN,   b);
    o->Np  = horner_c(f->Np,  f->nNp,  b);
}

static double a_star(const spong_continue_field *f, double b) {
    SPONG_FP_EXACT
    return horner_p(f->B, f->nB, b) / horner_p(f->A, f->nA, b);
}

static double loss(const spong_continue_field *f, double a, double b) {
    SPONG_FP_EXACT
    double A = horner_p(f->A, f->nA, b), B = horner_p(f->B, f->nB, b);
    return f->C - 2.0 * a * B + a * a * A;
}

static void grad_loss(const spong_continue_field *f, double a, double b,
                      double *ga, double *gb) {
    SPONG_FP_EXACT
    double A  = horner_p(f->A,  f->nA,  b);
    double B  = horner_p(f->B,  f->nB,  b);
    double Ap = horner_p(f->Ap, f->nAp, b);
    double Bp = horner_p(f->Bp, f->nBp, b);
    *ga = 2.0 * (a * A - B);
    *gb = -2.0 * a * Bp + a * a * Ap;
}

/* charts._s_velocities via model.P_of */
static void velocities(const spong_continue_field *f, double b, double w,
                       double *vb, double *vw) {
    SPONG_FP_EXACT
    base k; eval_base_p(f, b, &k);
    double asp = k.Bp / k.A - k.B * k.Ap / (k.A * k.A);
    double up  = k.B * k.N / (k.A * k.A);
    double P   = up + k.Ap * w * w - 2.0 * k.A * w * asp;
    *vb = -P;
    *vw = -2.0 * k.A * w + asp * P;
}

/* charts._s_depth_gauge_floor, term for term. */
static double depth_gauge_floor(const spong_continue_field *f, double b) {
    SPONG_FP_EXACT
    base k; eval_base_p(f, b, &k);
    double asp  = k.Bp / k.A - k.B * k.Ap / (k.A * k.A);
    double aspp = ((k.Bpp * k.A - k.B * k.App) / (k.A * k.A)
                   - 2.0 * k.Ap * (k.Bp * k.A - k.B * k.Ap)
                     / (k.A * k.A * k.A));
    double up   = k.B * k.N / (k.A * k.A);
    double upp  = ((k.Bp * k.N + k.B * k.Np) / (k.A * k.A)
                   - 2.0 * k.B * k.N * k.Ap / (k.A * k.A * k.A));
    double w1p  = (aspp * up + asp * upp) / (2.0 * k.A)
                  - asp * up * k.Ap / (2.0 * k.A * k.A);
    double lo = 1e-16 * fabs(asp) + 1e-300;
    double d = fabs(w1p);
    return 2.0 * fabs(asp) / (d > lo ? d : lo);
}

static double slaved_w1(const spong_continue_field *f, double b) {
    SPONG_FP_EXACT
    base k; eval_base_p(f, b, &k);
    double asp = k.Bp / k.A - k.B * k.Ap / (k.A * k.A);
    double up  = k.B * k.N / (k.A * k.A);
    return asp * up / (2.0 * k.A);
}

/* ------------------------------------------------------------------ *
 * chart right-hand sides and the GL6 stage solve                       *
 * (verbatim from _native.c -- see the duplication note at the top)     *
 *                                                                      *
 * CONTRACTION FUSED from here: these must match _native.c, which is built  *
 * at -O3 with clang's default fusion.                                  *
 * ------------------------------------------------------------------ */

static double slow_fj(const spong_continue_field *f, double b, double w,
                      double *jac) {
    SPONG_FP_FUSED
    base k; eval_base_c(f, b, &k);
    double A2 = k.A * k.A;
    double asp = k.Bp / k.A - k.B * k.Ap / A2;
    double Pv = k.B * k.N / A2 + k.Ap * w * w - 2.0 * k.A * w * asp;
    if (jac != NULL) {
        double P_w = 2.0 * k.Ap * w - 2.0 * k.A * asp;
        *jac = 2.0 * k.A / Pv - 2.0 * k.A * w * P_w / (Pv * Pv);
    }
    return 2.0 * k.A * w / Pv - asp;
}

static double fast_fj(const spong_continue_field *f, double w, double b,
                      double *jac) {
    SPONG_FP_FUSED
    base k; eval_base_c(f, b, &k);
    double A2 = k.A * k.A, A3 = A2 * k.A;
    double asp = k.Bp / k.A - k.B * k.Ap / A2;
    double aspp = ((k.Bpp * k.A - k.B * k.App) / A2
                   - 2.0 * k.Ap * (k.Bp * k.A - k.B * k.Ap) / A3);
    double up = k.B * k.N / A2;
    double upp = ((k.Bp * k.N + k.B * k.Np) / A2
                  - 2.0 * k.B * k.N * k.Ap / A3);
    double Pv = up + k.Ap * w * w - 2.0 * k.A * w * asp;
    double D = 2.0 * k.A * w - asp * Pv;
    if (jac != NULL) {
        double P_b = upp + k.App * w * w - 2.0 * w * (k.Ap * asp + k.A * aspp);
        double D_b = 2.0 * k.Ap * w - aspp * Pv - asp * P_b;
        *jac = (P_b * D - Pv * D_b) / (D * D);
    }
    return Pv / D;
}

typedef double (*FJ)(const spong_continue_field *, double, double, double *);

/* Evaluation floors of the chart right-hand sides -- verbatim from
 * _native.c (floor_parts / slow_floor / fast_floor): the first-order
 * running-error bound, propagated operation by operation; fused as there. */
typedef double (*FLOOR)(const spong_continue_field *, double, double);

static void floor_parts(const spong_continue_field *k, double b, double w,
                        double *A_, double *EA_, double *asp_, double *Easp_,
                        double *Pv_, double *EP_) {
    SPONG_FP_FUSED
    double A = horner_c(k->A, k->nA, b), Ap = horner_c(k->Ap, k->nAp, b);
    double B = horner_c(k->B, k->nB, b), Bp = horner_c(k->Bp, k->nBp, b);
    double Nv = horner_c(k->N, k->nN, b);
    double EA = (double)k->nA * horner_abs_c(k->A, k->nA, b);
    double EAp = (double)k->nAp * horner_abs_c(k->Ap, k->nAp, b);
    double EB = (double)k->nB * horner_abs_c(k->B, k->nB, b);
    double EBp = (double)k->nBp * horner_abs_c(k->Bp, k->nBp, b);
    double EN = (double)k->nN * horner_abs_c(k->N, k->nN, b);
    double aA = fabs(A), A2 = A * A;
    double t1 = Bp / A;
    double Et1 = (EBp + fabs(t1) * EA) / aA + fabs(t1);
    double t2 = B * Ap / A2;
    double Et2 = (fabs(Ap) * EB + fabs(B) * EAp) / A2
                 + fabs(t2) * (2.0 * EA / aA + 1.0);
    double asp = t1 - t2;
    double Easp = Et1 + Et2 + fabs(asp);
    double up = B * Nv / A2;
    double Eup = (fabs(Nv) * EB + fabs(B) * EN) / A2
                 + fabs(up) * (2.0 * EA / aA + 1.0);
    double Pv = up + Ap * w * w - 2.0 * A * w * asp;
    double EP = Eup + w * w * EAp + fabs(Ap * w * w)
                + 2.0 * fabs(w) * (fabs(asp) * EA + aA * Easp)
                + 2.0 * fabs(A * w * asp) + fabs(Pv);
    *A_ = A; *EA_ = EA; *asp_ = asp; *Easp_ = Easp; *Pv_ = Pv; *EP_ = EP;
}

static double slow_floor(const spong_continue_field *f, double b, double w) {
    SPONG_FP_FUSED
    double A, EA, asp, Easp, Pv, EP;
    floor_parts(f, b, w, &A, &EA, &asp, &Easp, &Pv, &EP);
    double t = 2.0 * A * w / Pv;
    double Et = (2.0 * fabs(w) * EA + fabs(t) * EP) / fabs(Pv) + fabs(t);
    return DBL_EPSILON * (Et + Easp + fabs(t - asp));
}

static double fast_floor(const spong_continue_field *f, double w, double b) {
    SPONG_FP_FUSED
    double A, EA, asp, Easp, Pv, EP;
    floor_parts(f, b, w, &A, &EA, &asp, &Easp, &Pv, &EP);
    double D = 2.0 * A * w - asp * Pv;
    double ED = 2.0 * fabs(w) * EA + fabs(asp) * EP + fabs(Pv) * Easp
                + fabs(asp * Pv) + fabs(D);
    double fv = Pv / D;
    return DBL_EPSILON * ((EP + fabs(fv) * ED) / fabs(D) + fabs(fv));
}

/* 2-stage Gauss (IRK4-GL), verbatim from _native.c's gl4_step (which is
 * what Kernel.slow_step_gl4 / fast_step_gl4 call): the floor ladder's first
 * rung.  Fused, as _native.c is built. */
static double gl4_step(const spong_continue_field *ctx, FJ fj, FLOOR fl,
                       double x, double y, double h) {
    SPONG_FP_FUSED
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
        if (fl != NULL && it > 0
                && fabs(r1) <= NOISE_C * fl(ctx, x1, Y1)
                && fabs(r2) <= NOISE_C * fl(ctx, x2, Y2)) {
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

static double gl6_step(const spong_continue_field *ctx, FJ fj, FLOOR fl,
                       double x, double y, double h) {
    SPONG_FP_FUSED
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
    double x1 = x + c1 * h, x2 = x + c2 * h, x3 = x + c3 * h;
    double K1, K2, K3;
    int converged = 0;
    for (int pass = 0; pass < 2 && !converged; pass++) {
        K1 = fj(ctx, x, y, NULL); K2 = K1; K3 = K1;
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
            if (r < NEWTON_TOL * (1.0 + m)) { converged = 1; break; }
            if (fl != NULL && it > 0
                    && fabs(r1) <= NOISE_C * fl(ctx, x1, Y1)
                    && fabs(r2) <= NOISE_C * fl(ctx, x2, Y2)
                    && fabs(r3) <= NOISE_C * fl(ctx, x3, Y3)) {
                converged = 1; break;
            }
            double J1, J2, J3;
            (void)fj(ctx, x1, Y1, &J1);
            (void)fj(ctx, x2, Y2, &J2);
            (void)fj(ctx, x3, Y3, &J3);
            double m11 = 1.0 - h * a11 * J1, m12 = -h * a12 * J1, m13 = -h * a13 * J1;
            double m21 = -h * a21 * J2, m22 = 1.0 - h * a22 * J2, m23 = -h * a23 * J2;
            double m31 = -h * a31 * J3, m32 = -h * a32 * J3, m33 = 1.0 - h * a33 * J3;
            double n1 = fabs(m11);
            if (fabs(m12) > n1) n1 = fabs(m12);
            if (fabs(m13) > n1) n1 = fabs(m13);
            double n2 = fabs(m21);
            if (fabs(m22) > n2) n2 = fabs(m22);
            if (fabs(m23) > n2) n2 = fabs(m23);
            double n3 = fabs(m31);
            if (fabs(m32) > n3) n3 = fabs(m32);
            if (fabs(m33) > n3) n3 = fabs(m33);
            if (n1 == 0.0 || n2 == 0.0 || n3 == 0.0) return NAN;
            double Am[3][3] = {{m11 / n1, m12 / n1, m13 / n1},
                               {m21 / n2, m22 / n2, m23 / n2},
                               {m31 / n3, m32 / n3, m33 / n3}};
            double v[3] = {r1 / n1, r2 / n2, r3 / n3};
            double det = 1.0;
            for (int col = 0; col < 3; col++) {
                int p = col; double big = fabs(Am[col][col]);
                for (int row = col + 1; row < 3; row++)
                    if (fabs(Am[row][col]) > big) { big = fabs(Am[row][col]); p = row; }
                if (p != col) {
                    for (int k2 = 0; k2 < 3; k2++) {
                        double t = Am[col][k2]; Am[col][k2] = Am[p][k2]; Am[p][k2] = t;
                    }
                    double t = v[col]; v[col] = v[p]; v[p] = t;
                    det = -det;
                }
                double piv = Am[col][col];
                det *= piv;
                if (piv == 0.0) break;
                for (int row = col + 1; row < 3; row++) {
                    double f2 = Am[row][col] / piv;
                    for (int k2 = col; k2 < 3; k2++) Am[row][k2] -= f2 * Am[col][k2];
                    v[row] -= f2 * v[col];
                }
            }
            if (fabs(det) < STAGE_GUARD) return NAN;
            double d3 = v[2] / Am[2][2];
            double d2 = (v[1] - Am[1][2] * d3) / Am[1][1];
            double d1 = (v[0] - Am[0][1] * d2 - Am[0][2] * d3) / Am[0][0];
            double alpha = 1.0;
            if (pass == 1) {
                double phi = 0.5 * (r1 * r1 + r2 * r2 + r3 * r3);
                int accepted = 0;
                for (int ls = 0; ls <= 12; ls++) {
                    double C1 = K1 - alpha * d1, C2 = K2 - alpha * d2, C3 = K3 - alpha * d3;
                    double Z1 = y + h * (a11 * C1 + a12 * C2 + a13 * C3);
                    double Z2 = y + h * (a21 * C1 + a22 * C2 + a23 * C3);
                    double Z3 = y + h * (a31 * C1 + a32 * C2 + a33 * C3);
                    double q1 = C1 - fj(ctx, x1, Z1, NULL);
                    double q2 = C2 - fj(ctx, x2, Z2, NULL);
                    double q3 = C3 - fj(ctx, x3, Z3, NULL);
                    double phic = 0.5 * (q1 * q1 + q2 * q2 + q3 * q3);
                    if (isfinite(phic) && phic <= phi * (1.0 - 1e-4 * alpha)) {
                        accepted = 1; break;
                    }
                    alpha *= 0.5;
                }
                if (!accepted) break;
            }
            K1 -= alpha * d1; K2 -= alpha * d2; K3 -= alpha * d3;
        }
    }
    if (!converged) return NAN;
    return y + h * (5.0 / 18.0 * K1 + 4.0 / 9.0 * K2 + 5.0 / 18.0 * K3);
}

/* ------------------------------------------------------------------ *
 * capture                                                             *
 * ------------------------------------------------------------------ */

/* charts._segment_capture: whether a resolved chord enters a target
 * neighbourhood.  Note the STRICT inequality, and that a degenerate chord
 * measures from the END point -- both matter, and both were wrong in the
 * first draft. */
static int segment_capture(double a0, double b0, double a1, double b1,
                           double at, double bt, double radius) {
    SPONG_FP_EXACT
    double da = a1 - a0, db = b1 - b0;
    double denom = da * da + db * db;
    if (denom == 0.0) {
        return ((a1 - at) * (a1 - at) + (b1 - bt) * (b1 - bt))
               < radius * radius;
    }
    double t = ((at - a0) * da + (bt - b0) * db) / denom;
    if (t > 1.0) t = 1.0;
    if (t < 0.0) t = 0.0;
    return ((a0 + t * da - at) * (a0 + t * da - at)
            + (b0 + t * db - bt) * (b0 + t * db - bt)) < radius * radius;
}

/* ------------------------------------------------------------------ *
 * the segment                                                         *
 * ------------------------------------------------------------------ */

/* The last two ACCEPTED vertices are kept in locals, not read back out of
 * the output buffer.  The turn budget needs them, and reading them from the
 * buffer made the check silently inoperative whenever point_capacity was too
 * small -- so the capacity-probing pass computed a DIFFERENT, shorter curve
 * than the real one (270 vertices against 279 on dead-neuron-far-saddle-d3).
 * A sizing pass must not change the answer. */
#define EMIT(A_, B_)                                                      \
    do {                                                                  \
        if (n_pts < point_capacity) {                                     \
            points[2 * n_pts] = (A_); points[2 * n_pts + 1] = (B_);       \
        } else { overflow = 1; }                                          \
        prev0_a = prev1_a; prev0_b = prev1_b;                             \
        prev1_a = (A_);    prev1_b = (B_);                                \
        n_pts++;                                                          \
    } while (0)

#define FINISH(TERM_, REASON_)                                            \
    do {                                                                  \
        result->term = overflow ? SPONG_CONT_NEED_CAPACITY : (TERM_);     \
        result->delegate_reason = (REASON_);                              \
        result->switches = switches;                                      \
        result->b_end = b_end; result->w_end = w_end;                     \
        result->n_points = n_pts;                                         \
        result->steps_taken = taken; result->steps_rejected = rejected;   \
        return result->term;                                              \
    } while (0)

SPONG_API int spong_continue_curve(
        const spong_continue_field *f,
        double b0, double w0, int flow,
        const double *targets, size_t n_targets, double cap_r,
        const double box[4], double ds, double ds0,
        const double *shallow_gate, size_t max_steps,
        int centered_available,
        double *points, size_t point_capacity,
        spong_continue_result *result) {

    SPONG_FP_EXACT
    memset(result, 0, sizeof *result);
    /* spong_field and spong_continue_field have identical layout by design
     * (spong_gauss2.h); the normalized-arclength rescue reads the field
     * through that view. */
    const spong_field *gf = (const spong_field *)f;
    const double TMAX = turn_max();
    const double eps = DBL_EPSILON;
    // Budget the DISTANCE TRAVELLED, not the step count.  A step budget is
    // not invariant under the halving this loop does: the turn budget and the
    // descent-realization test drive `cur` down to continuation_floor =
    // cur0/128, so a branch crossing stiff country takes up to 128x more
    // steps than diagonal/ds predicts and a count derived from ds is
    // optimistic by exactly that factor.  Measured on 20 directed models,
    // abort_max_steps was 9 of 15 unfinished branches.  Arclength is what the
    // budget was always trying to express, and halving does not change it.
    const double arc_budget =
        8.0 * hypot(box[1] - box[0], box[3] - box[2]);
    double travelled = 0.0;

    double b = b0, w = w0;
    double b_end = b0, w_end = w0;
    double prev0_a = 0.0, prev0_b = 0.0, prev1_a = 0.0, prev1_b = 0.0;
    size_t n_pts = 0;
    int overflow = 0, switches = 0;
    uint64_t taken = 0, rejected = 0;

    EMIT(a_star(f, b) + w, b);

    double vb, vw;
    velocities(f, b, w, &vb, &vw);
    vb *= flow; vw *= flow;
    int slow = !(fabs(vw) > R_SWITCH * fabs(vb));

    /* launch ramp: start at the incoming chord, grow into ds */
    double cur = (ds0 > 0.0) ? fmin(ds, fmax(ds0, 1e-12)) : ds;
    const double continuation_floor = cur / 128.0;

    double recent[STALL_WINDOW + 1];
    size_t n_recent = 0;

    for (size_t step = 0; step < max_steps; step++) {
        velocities(f, b, w, &vb, &vw);
        vb *= flow; vw *= flow;
        double speed = fmax(fabs(vb), fabs(vw));
        if (speed < 1e-300) { b_end = b; w_end = w;
            FINISH(SPONG_CONT_ABORT_STATIONARY, SPONG_DELEGATE_NONE); }

        if (slow && fabs(vw) > R_SWITCH * fabs(vb)) { slow = 0; switches++; }
        else if (!slow && fabs(vb) > R_SWITCH * fabs(vw)) { slow = 1; switches++; }
        if (switches > MAX_SWITCHES) { b_end = b; w_end = w;
            FINISH(SPONG_CONT_ABORT_SWITCH_LIMIT, SPONG_DELEGATE_NONE); }

        if (flow > 0 && (shallow_gate == NULL
                         || (b - shallow_gate[0]) * shallow_gate[1] >= 0.0)) {
            if (depth_gauge_floor(f, b) >= KAPPA_HI) {
                double w1 = slaved_w1(f, b);
                if (fabs(w - w1) <= 0.05 * fabs(w1) + 1e-9 * (1.0 + fabs(w))) {
                    b_end = b; w_end = w;
                    FINISH(SPONG_CONT_ENTER_SHALLOW, SPONG_DELEGATE_NONE);
                }
                /* not slaved: watch for the hover two-cycle -- the
                 * discrete freeze of a non-L-stable method over shallow
                 * water -- and TRIM the stalled tail: it is a discrete
                 * artifact, and its hover altitude becomes the seam
                 * residual.  The window holds the last STALL_WINDOW
                 * values; the reference appends, pops the oldest, then
                 * compares against the (new) oldest. */
                recent[n_recent++] = b;
                if (n_recent > STALL_WINDOW) {
                    for (size_t i = 1; i <= STALL_WINDOW; i++)
                        recent[i - 1] = recent[i];
                    n_recent = STALL_WINDOW;
                    if (fabs(b - recent[0]) < 1.0 * ds) {
                        if (overflow) { b_end = b; w_end = w;
                            FINISH(SPONG_CONT_ENTER_SHALLOW, SPONG_DELEGATE_NONE); }
                        size_t n_trim = (n_pts - 1 < STALL_WINDOW)
                                        ? n_pts - 1 : STALL_WINDOW;
                        n_pts -= n_trim;
                        double a_bk = points[2 * (n_pts - 1)];
                        double b_bk = points[2 * (n_pts - 1) + 1];
                        b_end = b_bk; w_end = a_bk - a_star(f, b_bk);
                        FINISH(SPONG_CONT_ENTER_SHALLOW, SPONG_DELEGATE_NONE);
                    }
                }
            } else {
                n_recent = 0;
            }
        }

        double b_prev = b, w_prev = w;
        double b_new = b, w_new = w, a_prev = 0.0;
        int settled = 0;

        for (int retry = 0; retry < RETRY_ATTEMPTS; retry++) {
            double h;
            int failed = 0;
            if (slow) {
                h = cur / sqrt(1.0 + (vw / vb) * (vw / vb)) * (vb > 0 ? 1.0 : -1.0);
                w_new = gl6_step(f, slow_fj, slow_floor, b_prev, w_prev, h);
                b_new = b_prev + h;
            } else {
                h = cur / sqrt(1.0 + (vb / vw) * (vb / vw)) * (vw > 0 ? 1.0 : -1.0);
                b_new = gl6_step(f, fast_fj, fast_floor, w_prev, b_prev, h);
                w_new = w_prev + h;
            }
            if (!(isfinite(b_new) && isfinite(w_new))) failed = 1;

            if (!failed) {
                /* Descent realization: the accepted stage root must deliver a
                 * fixed fraction of its first-order expected change.  This is
                 * what selects the flow-connected root of a multi-root stage
                 * system; rejection is a step-size signal. */
                a_prev = a_star(f, b_prev) + w_prev;
                double a_test = a_star(f, b_new) + w_new;
                double da = a_test - a_prev, db = b_new - b_prev;
                double ga, gb;
                grad_loss(f, a_prev, b_prev, &ga, &gb);
                double Lp = loss(f, a_prev, b_prev);
                double expected = flow * (ga * da + gb * db);
                double actual = flow * (loss(f, a_test, b_new) - Lp);
                double slack = 64.0 * eps * (1.0 + fabs(Lp));
                if (!(isfinite(a_prev) && isfinite(a_test) && isfinite(expected)
                      && isfinite(actual) && isfinite(slack))
                    || expected >= 0.0
                    || actual > 1e-4 * expected + slack) failed = 1;
            }

            if (failed) {
                if (cur > continuation_floor) { cur *= 0.5; rejected++; continue; }
                /* FLOOR-FALLBACK LADDER.  At the spatial resolution floor,
                 * further halving moves into cancellation.  The two graph
                 * charts describe the same curve and GL4 has a different
                 * stage system, so exhaust those equivalent representations
                 * before declaring the state numerically undefined: (active
                 * chart, GL4), (other chart, GL6), (other chart, GL4), each
                 * under the same descent-realization test.  Note the test
                 * here is the reference's: no finiteness guard on
                 * expected/actual, just the two inequalities. */
                int rescued = 0;
                const int alt_slow[3] = { slow, !slow, !slow };
                const int alt_gl6[3]  = { 0, 1, 0 };
                for (int alt = 0; alt < 3 && !rescued; alt++) {
                    double bn, wn;
                    if (alt_slow[alt]) {
                        h = cur / sqrt(1.0 + (vw / vb) * (vw / vb))
                            * (vb > 0 ? 1.0 : -1.0);
                        wn = alt_gl6[alt] ? gl6_step(f, slow_fj, slow_floor, b_prev, w_prev, h)
                                          : gl4_step(f, slow_fj, slow_floor, b_prev, w_prev, h);
                        bn = b_prev + h;
                    } else {
                        h = cur / sqrt(1.0 + (vb / vw) * (vb / vw))
                            * (vw > 0 ? 1.0 : -1.0);
                        bn = alt_gl6[alt] ? gl6_step(f, fast_fj, fast_floor, w_prev, b_prev, h)
                                          : gl4_step(f, fast_fj, fast_floor, w_prev, b_prev, h);
                        wn = w_prev + h;
                    }
                    if (!(isfinite(bn) && isfinite(wn))) continue;
                    a_prev = a_star(f, b_prev) + w_prev;
                    double a_test = a_star(f, bn) + wn;
                    double da = a_test - a_prev, db = bn - b_prev;
                    double ga, gb;
                    grad_loss(f, a_prev, b_prev, &ga, &gb);
                    double Lp = loss(f, a_prev, b_prev);
                    double expected = flow * (ga * da + gb * db);
                    double actual = flow * (loss(f, a_test, bn) - Lp);
                    double slack = 64.0 * eps * (1.0 + fabs(Lp));
                    if (expected >= 0.0 || actual > 1e-4 * expected + slack)
                        continue;
                    b_new = bn; w_new = wn;
                    result->rescues[alt_slow[alt]
                        ? (alt_gl6[alt] ? SPONG_RESCUE_SLOW_GL6 : SPONG_RESCUE_SLOW_GL4)
                        : (alt_gl6[alt] ? SPONG_RESCUE_FAST_GL6 : SPONG_RESCUE_FAST_GL4)]++;
                    rescued = 1;
                }
                /* Both graph parameterizations can become singular at the
                 * same geometric point (most commonly on arrival).  The curve
                 * itself is still regular there: continue it by arclength in
                 * the 2-D normalized-gradient field, orders 8, 6, 4, with the
                 * same expected-change check selecting the local stage
                 * root. */
                if (!rescued) {
                    static const int orders[3] = { 8, 6, 4 };
                    for (int oi = 0; oi < 3 && !rescued; oi++) {
                        a_prev = a_star(f, b_prev) + w_prev;
                        double z[2] = { a_prev, b_prev }, out[2];
                        if (!spong_normalized_step(gf, z, -flow * cur,
                                                   orders[oi], out))
                            continue;
                        double a_try = out[0], b_try = out[1];
                        if (!(isfinite(a_try) && isfinite(b_try))) continue;
                        double da = a_try - a_prev, db = b_try - b_prev;
                        double ga, gb;
                        grad_loss(f, a_prev, b_prev, &ga, &gb);
                        double Lp = loss(f, a_prev, b_prev);
                        double expected = flow * (ga * da + gb * db);
                        double actual = flow * (loss(f, a_try, b_try) - Lp);
                        double slack = 64.0 * eps * (1.0 + fabs(Lp));
                        if (expected >= 0.0 || actual > 1e-4 * expected + slack)
                            continue;
                        b_new = b_try; w_new = a_try - a_star(f, b_try);
                        result->rescues[SPONG_RESCUE_NORMALIZED_GL8 + oi]++;
                        rescued = 1;
                    }
                }
                if (!rescued) {
                    b_end = b_prev; w_end = w_prev;
                    if (centered_available) {
                        /* the last rung needs the local jet: not on this ABI
                         * yet, so the whole segment goes back */
                        FINISH(SPONG_CONT_DELEGATE, SPONG_DELEGATE_CENTERED_CHART);
                    }
                    result->fail_b = b_prev; result->fail_w = w_prev;
                    result->fail_cur = cur; result->fail_h = h;
                    result->fail_vb = vb; result->fail_vw = vw;
                    result->fail_slow = slow; result->fail_retry = retry;
                    FINISH(SPONG_CONT_ABORT_STEP_FAILURE, SPONG_DELEGATE_NONE);
                }
                failed = 0;
            }

            if (n_pts >= 2) {
                double a_new = a_star(f, b_new) + w_new;
                double d1a = prev1_a - prev0_a, d1b = prev1_b - prev0_b;
                double d2a = a_new - prev1_a, d2b = b_new - prev1_b;
                double nn1 = sqrt(d1a * d1a + d1b * d1b);
                double nn2 = sqrt(d2a * d2a + d2b * d2b);
                if (nn1 > 1e-14 && nn2 > 1e-14
                    && (d1a * d2a + d1b * d2b) / (nn1 * nn2) < TMAX
                    && cur > continuation_floor) {
                    cur *= 0.5; rejected++; continue;
                }
            }
            settled = 1;
            break;
        }
        if (!settled) {
            /* Retry budget exhausted with cur still above the floor (the
             * ramped chord can sit far above the launch chord, so seven
             * halvings of ds need not reach cur0/128).  The reference falls
             * out of `for _retry in range(8)` and ACCEPTS the last attempt's
             * b_new, w_new -- a point that failed the descent-realization
             * test, or is non-finite (which then aborts as abort_nonfinite
             * below).  Observed: directed seed 1414065525, both stable
             * segments at the a*=0 saddle, taken=0 rejected=7.  Reproduced
             * here for parity, not endorsed; whether the reference should
             * instead descend the ladder is a separate decision, and this
             * comment is where it is flagged. */
            settled = 1;
        }

        b = b_new; w = w_new;
        double a = a_star(f, b) + w;
        cur = fmin(ds, cur * 1.06);
        taken++;
        travelled += hypot(a - a_prev, b - b_prev);
        if (travelled > arc_budget){ b_end = b; w_end = w;
            FINISH(SPONG_CONT_ABORT_MAX_STEPS, SPONG_DELEGATE_NONE); }

        if (!(isfinite(b) && isfinite(w))) { b_end = b; w_end = w;
            FINISH(SPONG_CONT_ABORT_NONFINITE, SPONG_DELEGATE_NONE); }

        for (size_t t = 0; t < n_targets; t++) {
            double at = targets[2 * t], bt = targets[2 * t + 1];
            if (!segment_capture(a_prev, b_prev, a, b, at, bt, cap_r)) continue;
            if (flow > 0) {
                double target_level = loss(f, at, bt);
                double current_level = loss(f, a_prev, b_prev);
                double slack = 128.0 * eps * (1.0 + fabs(current_level));
                if (target_level > current_level + slack) continue;
            }
            EMIT(a, b);
            if ((a - at) * (a - at) + (b - bt) * (b - bt) > 1e-24) EMIT(at, bt);
            b_end = b; w_end = w;
            FINISH(SPONG_CONT_CAPTURE, SPONG_DELEGATE_NONE);
        }
        EMIT(a, b);

        if (!(box[0] <= a && a <= box[1] && box[2] <= b && b <= box[3])) {
            b_end = b; w_end = w;
            FINISH(SPONG_CONT_BOX_EXIT, SPONG_DELEGATE_NONE);
        }
    }

    b_end = b; w_end = w;
    FINISH(SPONG_CONT_ABORT_MAX_STEPS, SPONG_DELEGATE_NONE);
}
