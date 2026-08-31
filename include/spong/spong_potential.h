#ifndef SPONG_POTENTIAL_H
#define SPONG_POTENTIAL_H

/*
 * One constant-potential-rate segment of the continuation dispatcher.
 *
 * This is charts._potential_rate_prefix, _potential_rate_level_event and
 * _potential_rate_box_exit: march the flow by loss events with the field
 * grad L / |grad L|^2 (dL/dt = -1 under descent), so that a long narrow
 * valley is sampled by loss rather than by hundreds of thousands of
 * microscopic arclength chords.  Near a critical point that field is
 * singular; the step is bounded by the flow line's curvature
 * (kappa = |(I - t t^T) H t| / |grad L|, one Hessian) with a loose
 * distance guard, and where the loss cannot resolve the step at all (its
 * floor is ~4096*eps*(1+|L|)) the step is taken in arclength with the
 * unit-speed field.  Every accepted step is the two-half-step composition,
 * checked against the full step (Richardson), the loss change, and -- for
 * the ascent phase -- a cubic Hermite dense output that also supplies the
 * vertices between accepted steps.
 *
 * NO PYTHON.  Every quantity is a Horner evaluation on the eight
 * coefficient arrays of spong_field; the caller may release the GIL for
 * the whole call.  Coefficients cross as raw ascending arrays, results as a
 * plain struct, vertices as packed (a, b) pairs.
 *
 * The Python loops remain the executable specification and the parity
 * target: the corpus test records every potential-rate segment the zoo
 * produces and demands bit-identical vertices and counters.  For that to
 * be meaningful the Python loops evaluate L, grad L and H through the same
 * Horner kernels (Kernel.loss / gradient / hessian), not the model's
 * range-guarded evaluators.
 */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_gauss2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPONG_POT_PREFIX       = 0,   /* descent to one target minimum      */
    SPONG_POT_LEVEL_EVENT  = 1,   /* descent to the next candidate level */
    SPONG_POT_ASCENT       = 2    /* ascent to the box boundary          */
} spong_potential_mode;

/* Termination.  The strings are the Python phases' `term` values. */
typedef enum {
    SPONG_POT_NEAR_TARGET      = 0,   /* "near_target"      (prefix)      */
    SPONG_POT_CAPTURE          = 1,   /* "capture"                        */
    SPONG_POT_BOX_EXIT         = 2,   /* "box_exit"                       */
    SPONG_POT_LEVEL_EVENT_HIT  = 3,   /* "level_event"      (level event) */
    SPONG_POT_STEP_FAILURE     = 4,   /* "step_failure"                   */
    SPONG_POT_BUDGET           = 5,   /* "budget"                         */
    SPONG_POT_UNRESOLVED_FIELD = 6,   /* "unresolved_field" (ascent)      */
    SPONG_POT_UNAVAILABLE      = 7,   /* "unavailable"                    */
    /* points buffer too small; n_points holds the count required */
    SPONG_POT_NEED_CAPACITY    = 101
} spong_potential_term;

typedef struct {
    int    mode;                     /* spong_potential_mode */
    double a0, b0;
    /* prefix: exactly one (a, b); level event: the candidate minima;
     * ascent: none. */
    const double *targets;
    size_t n_targets;
    double cap_r;
    double box[4];                   /* a_lo, a_hi, b_lo, b_hi */
    double ds;                       /* ascent: the legacy chord */
    size_t n_levels;                 /* prefix / level event */
    size_t max_steps;                /* ascent */
    const double *critical;          /* packed (a, b), may be NULL */
    size_t n_critical;
    double critical_step_fraction;   /* charts.CRITICAL_STEP_FRACTION */
    int    primary_order;            /* charts.GEOMETRIC_IRK_PRIMARY */
} spong_potential_request;

typedef struct {
    int      term;                   /* spong_potential_term */
    size_t   n_points;               /* vertices written, or required */
    double   a_end, b_end;
    int      captured;               /* level event: a target was captured */
    double   captured_a, captured_b;
    double   event_level;            /* level event */
    double   level_step;             /* prefix: base */
    uint64_t accepted, rejected;
    uint64_t critical_capped, arclength_steps;
    uint64_t gl8_attempted, gl8_accepted;
    double   max_richardson;
    double   max_interpolation_error; /* ascent */
} spong_potential_result;

SPONG_API int spong_potential_rate_segment(
    const spong_field *field,
    const spong_potential_request *request,
    double *points, size_t point_capacity,
    spong_potential_result *result);

#ifdef __cplusplus
}
#endif

#endif
