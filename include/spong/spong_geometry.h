#ifndef SPONG_GEOMETRY_H
#define SPONG_GEOMETRY_H

/* Portable batched measurements on computed invariant-manifold polylines. */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_resolution.h"
#include "spong/spong_smale.h"

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

typedef struct {
    int32_t status;           /* 0 measured, 1 nonfinite point, 2 zero chord,
                                3 stationary midpoint */
    double sin_squared;      /* rounded exact cross^2/(|chord|^2 |grad|^2) */
    int32_t cross_nonzero;    /* exact; a tiny nonzero ratio can round to 0 */
    int32_t flow_alignment;   /* exact sign of direction * chord.grad(mid) */
    int32_t loss_direction;   /* exact sign of direction * (L(end)-L(start)) */
} spong_level_normal_measurement;

/* Independent reference measurement, not a trajectory integrator or a
 * pass/fail tolerance policy. GMP evaluates exact rational coefficients at
 * the exact dyadic chord midpoint. direction is +1 for ascent, -1 for descent.
 * Output has point_count-1 entries; no segment is silently omitted.
 * L's additive constant cancels. Nonzero return means invalid input/allocation.
 * The displayed sin_squared is rounded; the signs and zero tests are exact.
 * No claim is made about unresolved geometry within a coordinate rounding cell.
 */
SPONG_API int spong_level_normal_reference(
    const spong_rational_input *A, size_t A_count,
    const spong_rational_input *B, size_t B_count,
    const double *points, size_t point_count, int32_t direction,
    spong_level_normal_measurement *out);

#ifdef __cplusplus
}
#endif

#endif
