/*
 * Two-dimensional Gauss--Legendre collocation on the loss field.
 * See spong/spong_gauss2.h.  Bodies relocated verbatim from
 * src/spong/_native.c (irk2_step and its stage machinery, the two field
 * callbacks, the equilibrated small solve); only the context type changed
 * from the extension's Kernel object to the plain spong_field view.
 */

#include "spong/spong_gauss2.h"

#include <float.h>
#include <math.h>
#include <string.h>

static const double SQRT3 = 1.73205080756887729352744634150587237;
static const double SQRT15 = 3.87298334620741688517926539978239961;
static const double NEWTON_TOL = 1e-13;
static const int NEWTON_MAX = 30;

static double horner(const double *c, size_t n, double x) {
    double acc = 0.0;
    for (size_t i = n; i-- > 0;) {
        acc = acc * x + c[i];
    }
    return acc;
}

void spong_field_eval_base(
    const spong_field *k, double b,
    double *A, double *Ap, double *App,
    double *B, double *Bp, double *Bpp,
    double *N, double *Np)
{
    *A   = horner(k->A,   k->nA,   b);
    *Ap  = horner(k->Ap,  k->nAp,  b);
    *App = horner(k->App, k->nApp, b);
    *B   = horner(k->B,   k->nB,   b);
    *Bp  = horner(k->Bp,  k->nBp,  b);
    *Bpp = horner(k->Bpp, k->nBpp, b);
    *N   = horner(k->N,   k->nN,   b);
    *Np  = horner(k->Np,  k->nNp,  b);
}

double spong_field_loss(const spong_field *k, double a, double b) {
    double A = horner(k->A, k->nA, b), B = horner(k->B, k->nB, b);
    return k->C - 2.0*a*B + a*a*A;
}

void spong_field_gradient(const spong_field *k, double a, double b,
                          double g[2]) {
    double A = horner(k->A, k->nA, b), Ap = horner(k->Ap, k->nAp, b);
    double B = horner(k->B, k->nB, b), Bp = horner(k->Bp, k->nBp, b);
    g[0] = 2.0*(a*A - B);
    g[1] = -2.0*a*Bp + a*a*Ap;
}

void spong_field_hessian(const spong_field *k, double a, double b,
                         double H[2][2]) {
    double A = horner(k->A, k->nA, b), Ap = horner(k->Ap, k->nAp, b);
    double App = horner(k->App, k->nApp, b);
    double Bp = horner(k->Bp, k->nBp, b), Bpp = horner(k->Bpp, k->nBpp, b);
    H[0][0] = 2.0*A;
    H[0][1] = H[1][0] = -2.0*Bp + 2.0*a*Ap;
    H[1][1] = -2.0*a*Bpp + a*a*App;
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

int spong_normalized_fj(void *ctx, const double z[2],
                        double f[2], double J[2][2]) {
    const spong_field *k = (const spong_field *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    spong_field_eval_base(k, z[1], &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
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

int spong_potential_rate_fj(void *ctx, const double z[2],
                            double f[2], double J[2][2]) {
    const spong_field *k = (const spong_field *)ctx;
    double A, Ap, App, B, Bp, Bpp, Nv, Np;
    spong_field_eval_base(k, z[1], &A, &Ap, &App, &B, &Bp, &Bpp, &Nv, &Np);
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

static int irk2_evaluate(void *ctx, spong_vec_fj fj, const double z[2],
                         double h, int s, const double AT[4][4],
                         const double K[4][2], double R[8],
                         double Js[4][2][2],
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

int spong_irk2_step(void *ctx, spong_vec_fj fj, const double z[2], double h,
                    int order, double out[2]) {
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
    if (order != 4 && order != 6 && order != 8) return 0;
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

int spong_normalized_step(const spong_field *field, const double z[2],
                          double h, int order, double out[2]) {
    return spong_irk2_step((void *)field, spong_normalized_fj, z, h, order, out);
}

int spong_potential_step(const spong_field *field, const double z[2],
                         double h, int order, double out[2]) {
    return spong_irk2_step((void *)field, spong_potential_rate_fj, z, h, order,
                           out);
}
