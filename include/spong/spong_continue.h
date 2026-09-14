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
 * DELEGATION IS THE LAST RUNG ONLY.  tests/corpus/continue_curve.json records
 * every engine segment the zoo produces, and across all 45 of them not one
 * reached the floor-fallback ladder, the normalized-arclength rescue, or the
 * stall trim.  The port originally returned SPONG_CONT_DELEGATE at any of
 * those and the caller re-ran the ENTIRE segment in the reference
 * implementation -- "free in a case that never occurs".  It occurred:
 * directed seed 1414065525 (2026-09-02) reached the floor ladder after 3.5M
 * native steps and the Python replay cost an hour.  The ladder, the
 * normalized rescue and the stall trim are now ported here, statement for
 * statement, with corpus entries recorded from the segments that reach them.
 *
 * What remains delegated is the centered-chart rescue, tried only after both
 * ladders fail and only when the caller HAS a centered local jet (the
 * reference's centered_local).  `centered_available` says whether it does:
 * with it the port returns DELEGATE(CENTERED_CHART) at exactly the point the
 * reference would try the jet; without it the port returns
 * ABORT_STEP_FAILURE itself, as the reference does.  Passing the jet through
 * this ABI retires that last rung.
 *
 * Delegation, where it survives, is still whole-segment: the caller discards
 * whatever was written here and re-runs from the original arguments.
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

/* Why a call delegated.  Only CENTERED_CHART is still issued; the other two
 * are retained so old diagnostics and the ctypes checker keep their names. */
typedef enum {
    SPONG_DELEGATE_NONE            = 0,
    SPONG_DELEGATE_FLOOR_LADDER    = 1,  /* historical: now ported */
    SPONG_DELEGATE_STALL_TRIM      = 2,  /* historical: now ported */
    SPONG_DELEGATE_CENTERED_CHART  = 3   /* centered rescue required */
} spong_continue_delegate_reason;

/* Floor-ladder rescues, in the order the reference tries them.  Counts of
 * each go to the caller's diagnostics under the reference's
 * floor_fallback_* keys. */
typedef enum {
    SPONG_RESCUE_SLOW_GL4 = 0, SPONG_RESCUE_SLOW_GL6 = 1,
    SPONG_RESCUE_FAST_GL4 = 2, SPONG_RESCUE_FAST_GL6 = 3,
    SPONG_RESCUE_NORMALIZED_GL8 = 4, SPONG_RESCUE_NORMALIZED_GL6 = 5,
    SPONG_RESCUE_NORMALIZED_GL4 = 6,
    SPONG_RESCUE_COUNT = 7
} spong_continue_rescue;

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
    uint64_t rescues[SPONG_RESCUE_COUNT];   /* floor-ladder rescues by kind */
    /* On ABORT_STEP_FAILURE: the reference's engine_diag["step_failure"]
     * record -- state and step at the failed attempt. */
    double   fail_b, fail_w, fail_cur, fail_h, fail_vb, fail_vw;
    int      fail_slow;            /* chart at failure: 1 slow, 0 fast */
    int      fail_retry;           /* halvings taken before the ladder */
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
 *
 * centered_available is nonzero when the caller holds a centered local jet
 * for this segment; see DELEGATION above.
 *
 * chart_order selects the PRIMARY chart stepper: 4 for the 2-stage Gauss
 * tableau, anything else (6 is the intended value) for the 3-stage one.  It
 * does NOT touch the floor-fallback ladder, which tries both orders on both
 * charts by construction and is the reference's behaviour at the resolution
 * floor.
 *
 * The parameter exists because the order was a compile-time fact and so
 * could not be varied in an experiment that means anything: a wall bisection
 * run at "GL4" and "GL6" through the Python module constant returned
 * bit-identical brackets, because the constant it varied orders the PLANE
 * steppers and this path was hard-coded GL6 either way.  An order knob that
 * only the Python oracle honours would answer a question about the oracle;
 * the backend is what phone and WebASM targets ship, so the knob belongs
 * here.
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
    int centered_available,
    int chart_order,                         /* 4, or 6 for the default */
    double *points, size_t point_capacity,
    spong_continue_result *result);

#ifdef __cplusplus
}
#endif

#endif
