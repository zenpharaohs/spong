#ifndef SPONG_ARRIVAL_H
#define SPONG_ARRIVAL_H

/*
 * The centered raw arrival: finishing a known connection with the regular
 * target-centered gradient flow.
 *
 * Constant-potential-rate and arclength parameterizations are singular at a
 * minimum; the unnormalized gradient flow of the centered jet is regular
 * there.  This entry point runs that flow from a point near the target to
 * capture, with the same step control, turn rejection and capture tests as
 * the Python phase (charts._centered_raw_arrival_python), which remains the
 * executable specification.  The parity corpus
 * (scripts/arrival_corpus.py) demands bit-identical vertices, endpoint,
 * term and counters.
 *
 * NO PYTHON.  No allocation: vertices go into a caller-supplied buffer.
 * When the buffer overflows the call returns SPONG_ARR_NEED_CAPACITY with
 * the exact count required in n_points.
 *
 * Policy constants that the specification computes in Python cross as
 * request fields, so the library holds no tunables and no transcendental
 * evaluation whose rounding could differ from the caller's:
 * critical-step turn rejection cosine, primary order.
 */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_jet.h"
#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPONG_ARR_CAPTURE           = 0,   /* "capture"           */
    SPONG_ARR_INVALID_POTENTIAL = 1,   /* "invalid_potential" */
    SPONG_ARR_STEP_FAILURE      = 2,   /* "step_failure"      */
    SPONG_ARR_BUDGET            = 3,   /* "budget"            */
    SPONG_ARR_UNAVAILABLE       = 4,   /* "unavailable"       */
    SPONG_ARR_NEED_CAPACITY     = 101
} spong_arrival_term;

typedef struct {
    double a0, b0;                   /* start, physical coordinates        */
    double at, bt;                   /* target minimum                      */
    double center_a, center_b;       /* the jet's centre                    */
    double slow, fast;               /* least and greatest Hessian eigenvalue */
    double cap_r;                    /* capture radius of the calling phase */
    size_t max_steps;
    double turn_reject;              /* cos(2 atan(CRITICAL_STEP_FRACTION)) */
    int    primary_order;            /* charts.GEOMETRIC_IRK_PRIMARY        */
} spong_arrival_request;

typedef struct {
    int      term;                   /* spong_arrival_term                  */
    size_t   n_points;               /* vertices written, or required       */
    double   a_end, b_end;
    uint64_t accepted, rejected, turn_rejected;
    uint64_t gl8_attempted, gl8_accepted;
    double   max_richardson;
    double   finish_radius;
    double   spectral_ratio;         /* fast / slow                         */
} spong_arrival_result;

/* Runs the arrival.  Returns result->term, or -1 for a malformed request.
 * points receives packed (a, b) pairs, physical coordinates. */
SPONG_API int spong_centered_arrival(
    const spong_jet *jet,
    const spong_arrival_request *request,
    double *points, size_t point_capacity,
    spong_arrival_result *result);

#ifdef __cplusplus
}
#endif

#endif /* SPONG_ARRIVAL_H */
