#ifndef SPONG_SECTION_H
#define SPONG_SECTION_H
#include "spong_gauss2.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Numerical event tracing, not a validated enclosure. No interpolated endpoint
 * is ever returned: every accepted vertex is a native GL8 integration result. */
typedef enum { SPONG_SECTION_OK=0, SPONG_SECTION_INVALID=1,
    SPONG_SECTION_WORK_LIMIT=2, SPONG_SECTION_CONDITIONING=3,
    SPONG_SECTION_STEP_FAILURE=4, SPONG_SECTION_ESCAPE=5 } spong_section_status;
typedef struct {
    double step, atol, rtol, max_radius;
    unsigned max_steps, max_halvings, max_refinements;
} spong_section_policy;
typedef struct {
    int status;
    size_t count;
    unsigned rejected_steps, refinements;
    double loss_residual, loss_roundoff_scale, max_step_error;
    char reason[128];
} spong_section_result;
SPONG_API spong_section_policy spong_section_default_policy(double step, unsigned max_steps);
/* vertices has capacity points, each containing (a,b). On refusal count may
 * describe a partial path, never a successful crossing. direction is +/-1. */
SPONG_API int spong_section_trace(const spong_field *field,const double start[2],
    int direction,double level,const spong_section_policy *policy,
    double *vertices,size_t capacity,spong_section_result *result);
#ifdef __cplusplus
}
#endif
#endif
