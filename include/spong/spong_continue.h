#ifndef SPONG_CONTINUE_H
#define SPONG_CONTINUE_H

/*
 * One engine segment of the continuation dispatcher.
 *
 * This is charts._continue_curve: march the descent/ascent flow through the
 * slow and fast graph charts from an initial deviation state, switching charts
 * on the velocity ratio, until capture, box exit, a shallow-water handoff, or
 * failure.  The Gauss stages it takes are the same ones the Python engine
 * already calls through Kernel.slow_step / fast_step; what moves here is the
 * loop around them.
 *
 * NO PYTHON.  Every quantity is a Horner evaluation on the eight coefficient
 * arrays, so the caller may release the GIL for the whole call.  That is the
 * point: it makes the branch, not the step, the unit that crosses the
 * boundary, which is what lets branches be traced concurrently.
 *
 * Coefficients are passed as raw ascending arrays rather than an opaque
 * handle, matching spong_curve_diagnostics in spong_geometry.h.  The library
 * then depends on nothing defined in the extension module.
 *
 * DELEGATION IS WHOLE-SEGMENT.  tests/corpus/continue_curve.json records
 * every engine segment the zoo produces, and across all 45 of them not one
 * reached the floor-fallback ladder, the normalized-arclength rescue, or the
 * stall trim.  Those paths exist because branches outside the zoo provoked
 * them, so a port of them could not be judged by anything we have.  Rather
 * than reimplement uncovered logic in a certificate path, this entry point
 * stops and returns SPONG_CONT_DELEGATE, and the caller re-runs the ENTIRE
 * segment in the reference implementation from the original arguments,
 * discarding whatever was written here.
 *
 * Resuming mid-segment was considered and rejected.  The loop state is not
 * (b, w): it also carries the active chart, the current chord `cur` with its
 * 1.06 ramp toward ds, the continuation_floor fixed from the INITIAL cur, and
 * the stall-detector window.  Handing all of that across a boundary to be
 * reconstructed is exactly the kind of near-equivalence that produces a
 * portrait nobody can explain.  Re-running is wasteful only in a case that
 * never occurs, and it is identical by construction.
 */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Termination.  The first eight mirror charts._continue_curve's `term`
 * strings exactly; the remainder are transport conditions, not outcomes. */
typedef enum {
    SPONG_CONT_CAPTURE             = 0,
    SPONG_CONT_BOX_EXIT            = 1,
    SPONG_CONT_ENTER_SHALLOW       = 2,
    SPONG_CONT_ABORT_STATIONARY    = 3,
    SPONG_CONT_ABORT_SWITCH_LIMIT  = 4,
    SPONG_CONT_ABORT_NONFINITE     = 5,
    SPONG_CONT_ABORT_STEP_FAILURE  = 6,
    SPONG_CONT_ABORT_MAX_STEPS     = 7,
    /* reached a path this port does not own; the caller discards any output
     * and re-runs the whole segment in the reference implementation */
    SPONG_CONT_DELEGATE            = 100,
    /* points buffer too small; n_points holds the count required */
    SPONG_CONT_NEED_CAPACITY       = 101
} spong_continue_term;

/* Why a call delegated.  Diagnostic only -- the caller re-runs identically
 * whatever the reason -- but it tells us which uncovered path a portrait
 * outside the zoo actually exercises, which is how the corpus grows. */
typedef enum {
    SPONG_DELEGATE_NONE            = 0,
    SPONG_DELEGATE_FLOOR_LADDER    = 1,  /* halving reached continuation_floor */
    SPONG_DELEGATE_STALL_TRIM      = 2,  /* hover two-cycle over the floor */
    SPONG_DELEGATE_CENTERED_CHART  = 3   /* centered rescue required */
} spong_continue_delegate_reason;

/* The eight ascending coefficient arrays, plus the loss constant.
 *
 * C is NOT recoverable from the others and is required: the descent
 * realization test scales its slack by |L| at the previous vertex, and the
 * capture level test compares absolute losses, so omitting it would silently
 * change which steps are accepted and which captures are allowed. */
typedef struct {
    const double *A,  *Ap,  *App;
    const double *B,  *Bp,  *Bpp;
    const double *N,  *Np;
    size_t nA, nAp, nApp, nB, nBp, nBpp, nN, nNp;
    double C;
} spong_continue_field;

typedef struct {
    int      term;                 /* spong_continue_term */
    int      delegate_reason;      /* spong_continue_delegate_reason */
    int      switches;             /* chart handoffs consumed */
    double   b_end, w_end;         /* deviation-chart state at return; not a
                                    * resume point -- see DELEGATION above */
    size_t   n_points;             /* vertices written, or required */
    uint64_t steps_taken;          /* accepted steps */
    uint64_t steps_rejected;       /* halvings, for step-control diagnosis */
} spong_continue_result;

/*
 * points receives packed (a, b) pairs -- PHYSICAL coordinates, matching what
 * the Python engine returns, not (b, w).  The first vertex is the initial
 * state.  If point_capacity is insufficient the call returns
 * SPONG_CONT_NEED_CAPACITY with n_points set to what is needed and the buffer
 * contents unspecified; engine segments on the zoo reach about 14k vertices,
 * so one retry is the worst case.
 *
 * targets is n_targets packed (a, b) pairs; capture is tested against every
 * one on each accepted chord.  Pass n_targets = 0 for stable branches, which
 * never capture.  The level guard on capture applies only when flow > 0.
 *
 * shallow_gate is NULL, or two doubles {value, sign}: while
 * (b - value) * sign < 0 the shallow handoff test is suppressed.
 *
 * ds0 <= 0 means "absent" (the reference implementation's None): the launch
 * chord is then ds.  continuation_floor is cur/128 taken from the LAUNCH
 * chord, not from ds -- a port deriving it from ds takes a different number
 * of halvings on branches with a materialized stub.
 */
SPONG_API int spong_continue_curve(
    const spong_continue_field *field,
    double b0, double w0,
    int flow,                                /* +1 descent, -1 ascent */
    const double *targets, size_t n_targets,
    double cap_r,
    const double box[4],                     /* a_lo, a_hi, b_lo, b_hi */
    double ds, double ds0,
    const double *shallow_gate,              /* may be NULL */
    size_t max_steps,
    double *points, size_t point_capacity,
    spong_continue_result *result);

#ifdef __cplusplus
}
#endif

#endif
