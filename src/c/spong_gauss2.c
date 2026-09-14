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
/* Evaluation-floor convergence: a stage residual within NOISE_C times the
 * field's evaluation floor at that stage point is converged (after at least
 * one Newton correction).  gauss._NOISE_C in the Python reference. */
static const double NOISE_C = 4.0;

static double horner(const double *c, size_t n, double x) {
    double acc = 0.0;
    for (size_t i = n; i-- > 0;) {
        acc = acc * x + c[i];
    }
    return acc;
}

/* Horner on |c_i| at |x|: the running-error bound's building block. */
static double horner_abs(const double *c, size_t n, double x) {
    double acc = 0.0, ax = fabs(x);
    for (size_t i = n; i-- > 0;) {
        acc = acc * ax + fabs(c[i]);
    }
    return acc;
}

/* The gradient of L and the first-order running-error bound of each
 * component as computed (in units of eps): E(Horner of degree n) =
 * n*sum|c_i||x|^i; E(xy) = |x|E(y) + |y|E(x) + |xy|; E(x+y) = E(x) + E(y)
 * + |x+y|.  Cancellation in -2aB' + a^2 A' is charged where it occurs:
 * measured on directed seed 99912409 at a = -6651, two O(1e8) terms
 * cancelling to 1e3, the unit vector good to ~1e-10 and NEWTON_TOL = 1e-13
 * unreachable. */
static void gradient_floor(const spong_field *k, double a, double b,
                           double g[2], double Eg[2]) {
    double A = horner(k->A, k->nA, b), Ap = horner(k->Ap, k->nAp, b);
    double B = horner(k->B, k->nB, b), Bp = horner(k->Bp, k->nBp, b);
    double EA = (double)k->nA * horner_abs(k->A, k->nA, b);
    double EAp = (double)k->nAp * horner_abs(k->Ap, k->nAp, b);
    double EB = (double)k->nB * horner_abs(k->B, k->nB, b);
    double EBp = (double)k->nBp * horner_abs(k->Bp, k->nBp, b);
    double aa = fabs(a), a2 = a * a;
    g[0] = 2.0 * (a * A - B);
    g[1] = -2.0 * a * Bp + a2 * Ap;
    Eg[0] = 2.0 * (aa * EA + fabs(a * A) + EB + fabs(a * A - B));
    Eg[1] = 2.0 * (aa * EBp + fabs(a * Bp)) + a2 * EAp + fabs(a2 * Ap)
            + fabs(g[1]);
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

/* Floors of the two named fields: the running-error bound of the
 * normalization applied to the gradient's, times eps.  E(x/y) = (E(x) +
 * |x/y|E(y))/|y| + |x/y|; E(hypot) = (|g0|E(g0) + |g1|E(g1))/|g| + |g|;
 * E(g0^2 + g1^2) = 2|g0|E(g0) + 2|g1|E(g1) + 3q. */
int spong_normalized_floor(void *ctx, const double z[2], double eta[2]) {
    const spong_field *k = (const spong_field *)ctx;
    double g[2], Eg[2];
    gradient_floor(k, z[0], z[1], g, Eg);
    double ng = hypot(g[0], g[1]);
    if (!(ng > 1e-300) || !isfinite(ng)) return 0;
    double Eng = (fabs(g[0]) * Eg[0] + fabs(g[1]) * Eg[1]) / ng + ng;
    for (int d = 0; d < 2; d++) {
        double fd = g[d] / ng;
        eta[d] = DBL_EPSILON * ((Eg[d] + fabs(fd) * Eng) / ng + fabs(fd));
    }
    return isfinite(eta[0]) && isfinite(eta[1]);
}

int spong_potential_rate_floor(void *ctx, const double z[2], double eta[2]) {
    const spong_field *k = (const spong_field *)ctx;
    double g[2], Eg[2];
    gradient_floor(k, z[0], z[1], g, Eg);
    double q = g[0]*g[0] + g[1]*g[1];
    if (!(q > 1e-300) || !isfinite(q)) return 0;
    double Eq = 2.0 * fabs(g[0]) * Eg[0] + 2.0 * fabs(g[1]) * Eg[1] + 3.0 * q;
    for (int d = 0; d < 2; d++) {
        double fd = g[d] / q;
        eta[d] = DBL_EPSILON * ((Eg[d] + fabs(fd) * Eq) / q + fabs(fd));
    }
    return isfinite(eta[0]) && isfinite(eta[1]);
}

static int irk2_evaluate(void *ctx, spong_vec_fj fj, spong_vec_floor fl,
                         const double z[2],
                         double h, int s, const double AT[4][4],
                         const double K[4][2], double R[8],
                         double Js[4][2][2], double Eta[4][2],
                         double *scale, double *rmax, double *phi) {
    *scale = 1.0; *rmax = 0.0; *phi = 0.0;
    for (int i = 0; i < s; i++) {
        double Y[2] = {z[0], z[1]}, F[2];
        for (int d = 0; d < 2; d++)
            for (int j = 0; j < s; j++) Y[d] += h*AT[i][j]*K[j][d];
        if (!fj(ctx, Y, F, Js == NULL ? NULL : Js[i])) return 0;
        if (Eta != NULL) {
            if (fl == NULL || !fl(ctx, Y, Eta[i])) {
                Eta[i][0] = Eta[i][1] = 0.0;      /* no floor: old test only */
            }
        }
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

/* every stage residual within NOISE_C times its stage floor */
static int within_floor(int s, const double R[8], const double Eta[4][2]) {
    for (int i = 0; i < s; i++)
        for (int d = 0; d < 2; d++)
            if (!(fabs(R[2*i+d]) <= NOISE_C * Eta[i][d])) return 0;
    return 1;
}

/* CHORD REALISATION: |sum b_i K_i| / sum b_i |K_i|, the fraction of the
 * requested arc a converged stage configuration actually delivers.
 *
 * A converged configuration is not yet a step.  The alternating
 * configurations are GENUINE roots of the stage equations -- GL8 at
 * h = 1.5 on directed:99912409 converges to |R| = 6e-16 in three
 * iterations and delivers 0.652 of the chord -- and no residual test can
 * see them, because nothing about them is a residual failure.  Measured
 * against a Radau reference at rtol 1e-12 (scripts/chord_realisation_probe.py)
 * the endpoint of such a step lands 0.26 to 0.46 of an arclength away from
 * the flow, while its residual is at the evaluation floor.
 *
 * The short values are not arbitrary: in a hemstitch the stages are all
 * +-f, so what the weights can deliver is |sum eps_i b_i| over sign
 * patterns -- GL4 {1, 0}, GL6 {1, 4/9, 1/9}, GL8 {1, .6521, .3479,
 * .3043, 0}.  Every short ratio ever observed here is in that set, so the
 * gap below 1 is a property of the tableau and the gate below is its
 * midpoint, not a tuned constant.  1 - ratio also tracks the relative
 * endpoint error to within about 1.2x, so a sound step clears the gate by
 * orders of magnitude rather than marginally (measured: 1 - ratio <= 1e-9
 * on every sound row, against a gate deficit of 0.17).
 *
 * Dividing by sum b_i |K_i| rather than by |h| is what keeps this
 * dimensionless on a field that is not unit-speed: the potential-rate
 * field has |f| ~ 9e-4, where the |h| form reads zero for every root,
 * sound or not.  The weights are palindromic, so the ratio is invariant
 * under c -> 1-c and gating on it costs no anadromy.
 *
 * SCOPE, and it is narrow.  This is sound ONLY on a UNIT-SPEED field,
 * where |K_i| = 1 identically so that sum b_i |K_i| = 1 and the ratio
 * measures nothing but disagreement in DIRECTION.  On a field whose stage
 * magnitudes vary it is not a test at all: at h*lambda = -100 on an
 * anisotropic linear jet the CORRECT Gauss stages alternate in sign --
 * measured GL8 (+8.83e-2, -3.08e-2, +2.99e-2, -6.77e-2), ratio 0.0703;
 * GL4 0.0346; GL6 0.4310 -- because that is what A-stable collocation
 * does on a stiff decay, and tests/test_local.py catches the rejection.
 * A parasitic hemstitch root and a sound stiff step produce the SAME sign
 * pattern; only the unit-speed constraint separates them.  So
 * spong_normalized_step gates and nothing else does; the potential-rate
 * field wants its own measurement before it gets a test of its own.
 *
 * It also does NOT police step sizes: a step whose h is past the flow's
 * turning scale can deliver a full-length chord in the wrong place
 * (measured at h*rho(J) ~ 2e7), and that failure is loud -- halve h and
 * the answer moves -- so it belongs to step-size policy, and certification
 * holds if it can be had at any step size for which the computation makes
 * sense.  Refusal here returns 0, which is the signal the floor ladder and
 * the halving retries already consume.
 */
static int realises_chord(int s, const double *BT, const double K[4][2]) {
    double num[2] = {0.0, 0.0}, den = 0.0;
    for (int i = 0; i < s; i++) {
        num[0] += BT[i] * K[i][0];
        num[1] += BT[i] * K[i][1];
        den += BT[i] * hypot(K[i][0], K[i][1]);
    }
    if (!(den > 0.0) || !isfinite(den)) return 0;
    /* (1 + largest realisable short value)/2, midway across the gap. */
    const double gate = s == 2 ? 0.5
                      : (s == 3 ? 0.7222222222222222
                                : 0.8260362887156365);
    return hypot(num[0], num[1]) >= gate * den;
}

/* How often the gate above has refused a step, for cost/benefit measurement
 * across an ensemble.  ATOMIC: the tracer runs one segment per worker
 * thread with the GIL released, so a plain counter would lose increments
 * exactly when the number matters most.  C11 atomics where the toolchain
 * has them, the GCC/clang builtins otherwise; the library is C99, so
 * neither is assumed. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L \
    && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
static _Atomic unsigned long chord_rejections = 0;
#define SPONG_COUNT_REJECT() \
    atomic_fetch_add_explicit(&chord_rejections, 1UL, memory_order_relaxed)
#define SPONG_READ_REJECT() \
    atomic_load_explicit(&chord_rejections, memory_order_relaxed)
#define SPONG_RESET_REJECT() \
    atomic_exchange_explicit(&chord_rejections, 0UL, memory_order_relaxed)
#elif defined(__GNUC__)
static unsigned long chord_rejections = 0;
#define SPONG_COUNT_REJECT() \
    __atomic_fetch_add(&chord_rejections, 1UL, __ATOMIC_RELAXED)
#define SPONG_READ_REJECT() \
    __atomic_load_n(&chord_rejections, __ATOMIC_RELAXED)
#define SPONG_RESET_REJECT() \
    __atomic_exchange_n(&chord_rejections, 0UL, __ATOMIC_RELAXED)
#else
#warning "chord-rejection counter is not atomic on this toolchain"
static unsigned long chord_rejections = 0;
#define SPONG_COUNT_REJECT() (chord_rejections++)
#define SPONG_READ_REJECT()  (chord_rejections)
#define SPONG_RESET_REJECT() (chord_rejections)
#endif

unsigned long spong_chord_rejections(int reset) {
    return reset ? (unsigned long)SPONG_RESET_REJECT()
                 : (unsigned long)SPONG_READ_REJECT();
}

int spong_irk2_step(void *ctx, spong_vec_fj fj, const double z[2], double h,
                    int order, double out[2]) {
    return spong_irk2_step_gated(ctx, fj, NULL, z, h, order, out, 0);
}

/* clock_kind: 0 none; 1 potential-rate; 2 normalized arclength.  The clock
 * modes are private because they rely on ctx being a spong_field.  The
 * public arbitrary-field IRK entry points retain their old contract. */
static int irk2_step_core(void *ctx, spong_vec_fj fj, spong_vec_floor fl,
                          const double z[2], double h, int order,
                          double out[2], int unit_speed, int clock_kind,
                          double *tau) {
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
    double f0[2], K[4][2], K0[4][2], Js[4][2][2], Eta[4][2];
    if (!fj(ctx, z, f0, NULL)) return 0;
    for (int i = 0; i < s; i++) {
        K0[i][0] = f0[0]; K0[i][1] = f0[1];
    }
    int converged = 0;
    for (int pass = 0; pass < 2 && !converged; pass++) {
        memcpy(K, K0, sizeof(K));
        for (int it = 0; it < NEWTON_MAX; it++) {
            double R[8], scale, rmax, phi;
            if (!irk2_evaluate(ctx, fj, fl, z, h, s, AT, K, R, Js, Eta,
                               &scale, &rmax, &phi)) break;
            if (rmax < NEWTON_TOL * scale) { converged = 1; break; }
            if (fl != NULL && it > 0 && within_floor(s, R, Eta)) {
                converged = 1; break;
            }
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
                    if (irk2_evaluate(ctx, fj, NULL, z, h, s, AT, Kc, Rc, NULL,
                                      NULL, &sc, &rm, &phic)
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
    if (unit_speed && !realises_chord(s, BT, K)) {
        SPONG_COUNT_REJECT();
        return 0;
    }
    if (clock_kind != 0) {
        /* Reconstruct stage coordinates from the final K.  Eta above stores
         * evaluation-floor bounds, not positions, and a Newton iteration may
         * declare convergence immediately after its last correction.  This
         * explicit reconstruction therefore gives the stages belonging to
         * the K that produces the accepted endpoint without changing any
         * ordinary (clock-off) arithmetic. */
        double integral = 0.0;
        for (int i = 0; i < s; i++) {
            double clock_value;
            if (clock_kind == 1) {
                clock_value = K[i][0]*K[i][0] + K[i][1]*K[i][1];
            } else {
                double stage[2] = {z[0], z[1]};
                for (int d = 0; d < 2; d++)
                    for (int j = 0; j < s; j++)
                        stage[d] += h*AT[i][j]*K[j][d];
                double g[2];
                const spong_field *field = (const spong_field *)ctx;
                spong_field_gradient(field, stage[0], stage[1], g);
                double ng = hypot(g[0], g[1]);
                if (!(ng > 1e-300) || !isfinite(ng)) return 0;
                clock_value = 1.0/ng;
            }
            integral += BT[i]*clock_value;
        }
        *tau = fabs(h)*integral;
        if (!isfinite(*tau) || *tau < 0.0) return 0;
    }
    out[0] = z[0]; out[1] = z[1];
    for (int i = 0; i < s; i++) {
        out[0] += h * BT[i] * K[i][0];
        out[1] += h * BT[i] * K[i][1];
    }
    return isfinite(out[0]) && isfinite(out[1]);
}

int spong_irk2_step_gated(void *ctx, spong_vec_fj fj, spong_vec_floor fl,
                          const double z[2], double h,
                          int order, double out[2], int unit_speed) {
    return irk2_step_core(ctx, fj, fl, z, h, order, out, unit_speed, 0, NULL);
}

int spong_irk2_step_floored(void *ctx, spong_vec_fj fj, spong_vec_floor fl,
                            const double z[2], double h,
                            int order, double out[2]) {
    return spong_irk2_step_gated(ctx, fj, fl, z, h, order, out, 0);
}

int spong_normalized_step(const spong_field *field, const double z[2],
                          double h, int order, double out[2]) {
    /* The only unit-speed caller, and the only one that gates. */
    return spong_irk2_step_gated((void *)field, spong_normalized_fj,
                                 spong_normalized_floor, z, h, order, out, 1);
}

int spong_potential_step(const spong_field *field, const double z[2],
                         double h, int order, double out[2]) {
    return spong_irk2_step_floored((void *)field, spong_potential_rate_fj,
                                   spong_potential_rate_floor, z, h, order,
                                   out);
}

int spong_normalized_step_clock(const spong_field *field, const double z[2],
                                double h, int order, double out[2],
                                double *tau) {
    if (tau == NULL) return 0;
    return irk2_step_core((void *)field, spong_normalized_fj,
                          spong_normalized_floor, z, h, order, out, 1, 2,
                          tau);
}

int spong_potential_step_clock(const spong_field *field, const double z[2],
                               double h, int order, double out[2],
                               double *tau) {
    if (tau == NULL) return 0;
    return irk2_step_core((void *)field, spong_potential_rate_fj,
                          spong_potential_rate_floor, z, h, order, out, 0, 1,
                          tau);
}
