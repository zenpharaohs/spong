#ifndef SPONG_JET_H
#define SPONG_JET_H

/*
 * The centered critical-point jet: the finite gradient of the loss,
 * translated to a certified critical point, as a two-component bivariate
 * polynomial in (da, db).
 *
 * Relocated from the CPython extension's LocalKernel type
 * (src/spong/_native.c) with no change of behaviour: the same nested
 * Horner, the same two vector fields on it, the same Gauss--Legendre steps.
 * It lives here so the centered arrival (spong_arrival.h) -- and later the
 * centered continuation -- can run without a Python object; the extension's
 * LocalKernel is now an adapter over this struct.
 *
 * NO PYTHON.  No allocation.  The polynomial crosses as raw ascending
 * arrays: c[d][r] is the coefficient array in db of the da^r term of
 * component d, of length n[d][r]; rows[d] is the number of rows.
 *
 * CONTRACTION.  The gradient/Hessian evaluator and the steps are compiled
 * with the platform default (fused on arm64) and MUST stay so: the Python
 * oracle evaluates the jet through the extension, i.e. through this very
 * code, so fused is what the specification computes.  The potential alone
 * is exact (see spong_jet_potential) because it mirrors a pure-Python
 * evaluator and the goldens were frozen against that.
 */

#include <stddef.h>

#include "spong/spong_gauss2.h"
#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const double *const *c[2];     /* c[d][r]: ascending coefficients in db */
    const size_t *n[2];            /* n[d][r]: length of c[d][r]            */
    int rows[2];                   /* rows of each component                */
} spong_jet;

/* Gradient g (and, if H != NULL, the Hessian H) of the centered field at
 * z = (da, db).  H[d] holds the partials of component d in (da, db);
 * rounded cross-partials are NOT assumed identical. */
SPONG_API void spong_jet_poly(const spong_jet *jet, const double z[2],
                              double g[2], double H[2][2]);

/* The centered potential difference whose gradient is the jet: the
 * a-component integrated from (0, db) to (da, db), then the b-component
 * along a = 0 from (0, 0) to (0, db).  Evaluated with contraction OFF, in
 * the reference evaluator's operation order. */
SPONG_API double spong_jet_potential(const spong_jet *jet, double da,
                                     double db);

/* The two vector fields on a spong_jet context, for spong_irk2_step. */
SPONG_API int spong_jet_raw_fj(void *ctx, const double z[2],
                               double f[2], double J[2][2]);
SPONG_API int spong_jet_normalized_fj(void *ctx, const double z[2],
                                      double f[2], double J[2][2]);

/* One Gauss--Legendre step (order 4, 6 or 8) of the unnormalized and
 * unit-speed centered gradient flows.  Returns 1 and writes out on
 * convergence; 0 otherwise. */
SPONG_API int spong_jet_raw_step(const spong_jet *jet, const double z[2],
                                 double h, int order, double out[2]);
SPONG_API int spong_jet_normalized_step(const spong_jet *jet,
                                        const double z[2], double h,
                                        int order, double out[2]);

#ifdef __cplusplus
}
#endif

#endif /* SPONG_JET_H */
