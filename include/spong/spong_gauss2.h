#ifndef SPONG_GAUSS2_H
#define SPONG_GAUSS2_H

/*
 * Two-dimensional Gauss--Legendre collocation steps on the loss field.
 *
 * Relocated from the CPython extension (src/spong/_native.c) with no change
 * of behaviour: the same tableaux, the same damped Newton stage solve, the
 * same equilibrated small dense solve with its backward-error certificate.
 * It lives here because the compute backend is a self-contained C99 library
 * -- the Python extension, MATLAB MEX, and the phone applications are
 * adapters over it -- and every potential-rate segment (spong_potential.h)
 * needs these steps without a Python object in sight.
 *
 * NO PYTHON.  No allocation.  Coefficients cross as raw ascending arrays in
 * spong_field, matching spong_continue_field.
 */

#include <stddef.h>

#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The eight ascending coefficient arrays and the loss constant.  Layout
 * identical to spong_continue_field so one view serves both. */
typedef struct {
    const double *A,  *Ap,  *App;
    const double *B,  *Bp,  *Bpp;
    const double *N,  *Np;
    size_t nA, nAp, nApp, nB, nBp, nBpp, nN, nNp;
    double C;
} spong_field;

/* Horner evaluation of all eight arrays at b. */
SPONG_API void spong_field_eval_base(
    const spong_field *field, double b,
    double *A, double *Ap, double *App,
    double *B, double *Bp, double *Bpp,
    double *N, double *Np);

/* Loss, gradient and Hessian of L(a,b) = C - 2aB(b) + a^2 A(b). */
SPONG_API double spong_field_loss(const spong_field *field, double a, double b);
SPONG_API void   spong_field_gradient(const spong_field *field, double a, double b,
                                      double g[2]);
SPONG_API void   spong_field_hessian(const spong_field *field, double a, double b,
                                     double H[2][2]);

/* A vector field with Jacobian: writes f and (if J != NULL) J at z; returns
 * 0 where the field is not evaluable (e.g. |grad L| below floor). */
typedef int (*spong_vec_fj)(void *ctx, const double z[2],
                            double f[2], double J[2][2]);

/* The field's evaluation floor at z: eta[d] bounds the rounding error of
 * f[d] as computed (first-order running-error bound times eps).  The stage
 * Newton in spong_irk2_step_floored accepts a residual within a small
 * multiple of it -- the absolute NEWTON_TOL is unreachable on a unit field
 * whose gradient carries cancellation (see gauss._NOISE_C in the Python
 * reference for the measurement). */
typedef int (*spong_vec_floor)(void *ctx, const double z[2], double eta[2]);

/* The two named fields on a spong_field context, and their floors. */
SPONG_API int spong_normalized_fj(void *ctx, const double z[2],
                                  double f[2], double J[2][2]);
SPONG_API int spong_potential_rate_fj(void *ctx, const double z[2],
                                      double f[2], double J[2][2]);
SPONG_API int spong_normalized_floor(void *ctx, const double z[2],
                                     double eta[2]);
SPONG_API int spong_potential_rate_floor(void *ctx, const double z[2],
                                         double eta[2]);

/* One implicit Gauss--Legendre step of the given order (4, 6 or 8) on any
 * field.  Returns 1 and writes out on convergence; 0 otherwise (a failed
 * stage solve is a step-size signal, not an error).  The floored variant
 * also converges when every stage residual is within the field's
 * evaluation floor at that stage point (fl may be NULL). */
SPONG_API int spong_irk2_step(void *ctx, spong_vec_fj fj, const double z[2],
                              double h, int order, double out[2]);
SPONG_API int spong_irk2_step_floored(void *ctx, spong_vec_fj fj,
                                      spong_vec_floor fl, const double z[2],
                                      double h, int order, double out[2]);

/* The same step with the CHORD-REALISATION admissibility test optionally
 * applied.  unit_speed = 1 asserts |f| = 1 on this field and enables the
 * test; on any other field it must be 0.  See the long comment in
 * spong_gauss2.c: the test refuses a converged stage configuration that
 * delivers only a fraction of the requested arc (the alternating roots,
 * which no residual test can see), and it is valid ONLY where the stage
 * magnitudes are pinned, because A-stable collocation on a stiff decay
 * produces the same alternating sign pattern legitimately. */
SPONG_API int spong_irk2_step_gated(void *ctx, spong_vec_fj fj,
                                    spong_vec_floor fl, const double z[2],
                                    double h, int order, double out[2],
                                    int unit_speed);

/* How many steps the chord-realisation gate has refused since the last
 * reset, for cost/benefit measurement.  reset != 0 reads and zeroes in one
 * atomic exchange.  Counts across all threads; the counter is process-wide,
 * not per-segment. */
SPONG_API unsigned long spong_chord_rejections(int reset);

/* Convenience: unit-speed and constant-potential-rate steps on the loss
 * field.  h > 0 ascends for the normalized field; for the potential field
 * h is the signed loss change. */
SPONG_API int spong_normalized_step(const spong_field *field, const double z[2],
                                    double h, int order, double out[2]);
SPONG_API int spong_potential_step(const spong_field *field, const double z[2],
                                   double h, int order, double out[2]);

#ifdef __cplusplus
}
#endif

#endif
