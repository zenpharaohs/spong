#ifndef SPONG_GEOMETRY_H
#define SPONG_GEOMETRY_H

/* Portable batched measurements on computed invariant-manifold polylines. */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double angle_energy;
    uint64_t angle_resolved;
    uint64_t angle_unresolved;
    double backbone_residual;
} spong_curve_diagnostics_result;

/*
 * Coefficients are ascending powers. Points are packed (a,b) pairs.
 * The result combines the geometric angle certificate and the complementary
 * algebraic backbone-tail certificate in one coefficient-evaluation pass.
 */
SPONG_API int spong_curve_diagnostics(
    const double *A, size_t A_count,
    const double *Ap, size_t Ap_count,
    const double *B, size_t B_count,
    const double *Bp, size_t Bp_count,
    const double *points, size_t point_count,
    size_t start, double digit_budget,
    spong_curve_diagnostics_result *result);

#ifdef __cplusplus
}
#endif

#endif
