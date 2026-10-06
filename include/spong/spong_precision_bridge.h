#ifndef SPONG_PRECISION_BRIDGE_H
#define SPONG_PRECISION_BRIDGE_H
#include <stddef.h>
#include "spong_resolution.h"
/* Experimental native Radau-IIA bridge. Exact rational gradient coefficients
 * are in ascending [component][a power][b power] order, centered on the
 * ORIGINAL input. No critical residual is removed. Output cubics are proposed
 * smooth paths: independent original-model checks remain mandatory. */
typedef struct {
    unsigned rows, columns, decimal_digits, max_steps;
    const char *const *gradient;
    const char *const *rates; /* positive decimal slow, fast */
} spong_precision_bridge_request;
typedef struct {
    size_t steps;
    unsigned rejected, solve_failures, precision_refusals;
    unsigned long working_bits;
    int termination; /* 0 budget, 1 step unresolved */
    char **coefficients; /* steps * 8: two ascending cubics of offset */
    char *max_solve_condition;
} spong_precision_bridge_result;
SPONG_API int spong_precision_bridge(const spong_precision_bridge_request *,
                                    spong_precision_bridge_result *);
SPONG_API void spong_precision_bridge_destroy(spong_precision_bridge_result *);
#endif
