#ifndef SPONG_RHEOSTAT_H
#define SPONG_RHEOSTAT_H
#include "spong_exact.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Numerical proposals only. No result from this interface is a certificate.
 * Rational inputs are base-10 integers or numerator/positive-denominator.
 * The two saddle intervals must isolate roots of B (source=0) or
 * N=2 A B'-A' B (source=1), in source/target order. Native code verifies them.
 * Contexts own their storage, are independently usable, and are not reentrant.
 */
typedef struct spong_rheostat spong_rheostat;
typedef struct {
    uint32_t precision_bits; /* 128..512; internal guard bits added */
    uint32_t stages;         /* 2,3,4: GL4,6,8 */
    uint32_t steps;          /* 1..65536 nominal intervals per shot */
    uint32_t launch_order;   /* 2..20 */
    uint32_t max_newton;     /* 1..64 */
    uint32_t max_halvings;   /* 0..8 per nominal interval */
} spong_rheostat_policy;
typedef enum {
    SPONG_RHEOSTAT_OK=0, SPONG_RHEOSTAT_INVALID=1,
    SPONG_RHEOSTAT_WORK_LIMIT=2, SPONG_RHEOSTAT_CONDITIONING=3,
    SPONG_RHEOSTAT_NO_BRACKET=4, SPONG_RHEOSTAT_ALLOCATION=5,
    SPONG_RHEOSTAT_CHART=6
} spong_rheostat_status;
/* Fields: lambda,gap,slope,level, ua,ub,ua_lambda,ub_lambda,
 * sa,sb,sa_lambda,sb_lambda, u_loss_residual,s_loss_residual,
 * u_launch_residual,s_launch_residual, bracket_width,
 * u_projection_distance,s_projection_distance.
 * Printed digits are not accuracy claims. bracket_width belongs to the
 * numerical shooting map; it is zero for evaluate(), not an error bound.
 */
#define SPONG_RHEOSTAT_FIELDS 19
#define SPONG_RHEOSTAT_TEXT 192
typedef struct {
    int32_t status;
    char values[SPONG_RHEOSTAT_FIELDS][SPONG_RHEOSTAT_TEXT];
    uint64_t stage_solves, step_halvings, evaluations;
    double max_backward_error;
    char reason[128];
} spong_rheostat_result;
SPONG_API int spong_rheostat_create(
    const char *const *A, size_t na, const char *const *B, size_t nb,
    const char *C, const char *const bounds[4], const int32_t sources[2],
    const int32_t directions[2], const char *launch_radius,
    const char *level_fraction, const spong_rheostat_policy *policy,
    spong_rheostat **context, spong_rheostat_result *result);
SPONG_API void spong_rheostat_destroy(spong_rheostat *context);
/* Launch germs in base (alpha,b) coordinates. The point/sensitivity slots
 * contain germs instead of crossings; level is in base-loss units. */
SPONG_API int spong_rheostat_launch(spong_rheostat *context,
    const char *lambda, spong_rheostat_result *result);
SPONG_API int spong_rheostat_evaluate(spong_rheostat *context,
    const char *lambda, spong_rheostat_result *result);
SPONG_API int spong_rheostat_locate(spong_rheostat *context,
    const char *lower, const char *upper, const char *parameter_tolerance,
    uint32_t max_iterations, spong_rheostat_result *result);
/* Warm-start legacy brackets: pad, reevaluate, and attempt at most four
 * sensitivity-guided expansions inside explicit positive search bounds. */
SPONG_API int spong_rheostat_locate_seeded(spong_rheostat *context,
    const char *lower,const char *upper,const char *window_lower,
    const char *window_upper,const char *parameter_tolerance,
    uint32_t max_iterations,spong_rheostat_result *result);
#ifdef __cplusplus
}
#endif
#endif
