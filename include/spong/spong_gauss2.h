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

/* The two named fields on a spong_field context. */
SPONG_API int spong_normalized_fj(void *ctx, const double z[2],
                                  double f[2], double J[2][2]);
SPONG_API int spong_potential_rate_fj(void *ctx, const double z[2],
                                      double f[2], double J[2][2]);

/* One implicit Gauss--Legendre step of the given order (4, 6 or 8) on any
 * field.  Returns 1 and writes out on convergence; 0 otherwise (a failed
 * stage solve is a step-size signal, not an error). */
SPONG_API int spong_irk2_step(void *ctx, spong_vec_fj fj, const double z[2],
                              double h, int order, double out[2]);

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
