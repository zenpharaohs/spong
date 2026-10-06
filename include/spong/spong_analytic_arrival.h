#ifndef SPONG_ANALYTIC_ARRIVAL_H
#define SPONG_ANALYTIC_ARRIVAL_H
#include <stddef.h>
#include "spong_resolution.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Native high-precision PROPOSAL, never a convergence/geometry certificate.
 * Exact gradient strings are rationals, packed [2][order+1][order+1].
 * Hessian is row major rational; frame and eigenvalues are decimal strings.
 * No Python objects or process-global GMP precision. Charts are immutable
 * and may be shared by concurrent matches. Caller owns all returned objects.
 */
typedef struct spong_arrival_chart spong_arrival_chart;
typedef struct {
    int order, decimal_digits;
    const char *const *gradient;
    const char *const *hessian; /* 4 */
    const char *const *frame;   /* 4, columns slow / fast */
    const char *const *rates;   /* 2, positive slow / fast */
} spong_arrival_chart_request;
typedef struct {
    size_t term_count, point_count;
    char **terms;              /* [term_count][3]: rate, a, b */
    char **parameters;         /* point_count-1 finite parameters */
    double *points;            /* [point_count][2], final point = target */
    char *amplitudes[2], *spectral_ratio, *divisor_margin;
    int slow_frobenius;
} spong_analytic_arrival;
/* Additive diagnostics API: no change to existing request/result layouts.
 * Owned decimal strings; inspect zeroes output, destroy releases strings.
 * These are numerical indicators, not interval error bounds. */
typedef struct {
    unsigned long working_bits;
    char *relative_divisor, *matrix_condition, *backward_error, *forward_error;
} spong_arrival_conditioning;
/* Per-call inverse diagnostics, also available on matching refusal. Residual
 * refers to the last attempted inverse target, which may be intermediate on
 * failure. Numerical indicators only. Destroy owned strings after any call. */
typedef struct {
    unsigned iterations, backtracks, stages;
    int continuation;
    char *max_condition, *max_forward_error, *residual;
} spong_arrival_matching;
SPONG_API void spong_arrival_matching_destroy(spong_arrival_matching *result);
SPONG_API int spong_arrival_chart_match_inspect(const spong_arrival_chart *chart,
    const char *const center[2],const double start[2],const double target[2],
    spong_analytic_arrival *result,spong_arrival_matching *info);
/* Exact rational entry coordinates, with start retained only for display.
 * Independent display binding and representation join checks are mandatory. */
SPONG_API int spong_arrival_chart_match_exact_inspect(const spong_arrival_chart *chart,
    const char *const center[2],const double start[2],const double target[2],
    const char *const exact_start[2],spong_analytic_arrival *result,
    spong_arrival_matching *info);
enum {
    SPONG_ARRIVAL_OK = 0, SPONG_ARRIVAL_INVALID = -1,
    SPONG_ARRIVAL_MEMORY = -2, SPONG_ARRIVAL_DIVISOR = -3,
    SPONG_ARRIVAL_MATCHING = -4, SPONG_ARRIVAL_RANGE = -5,
    SPONG_ARRIVAL_PRECISION = -6, SPONG_ARRIVAL_SOLVE = -7
};
SPONG_API const char *spong_analytic_arrival_error(int status);
SPONG_API int spong_arrival_chart_create(const spong_arrival_chart_request *request,
                                        spong_arrival_chart **chart);
SPONG_API void spong_arrival_chart_destroy(spong_arrival_chart *chart);
SPONG_API int spong_arrival_chart_inspect(const spong_arrival_chart *chart,
                                        spong_arrival_conditioning *result);
SPONG_API void spong_arrival_conditioning_destroy(spong_arrival_conditioning *result);
SPONG_API int spong_arrival_chart_match(const spong_arrival_chart *chart,
    const char *const center[2], const double start[2], const double target[2],
    spong_analytic_arrival *result);
SPONG_API void spong_analytic_arrival_destroy(spong_analytic_arrival *result);
#ifdef __cplusplus
}
#endif
#endif
