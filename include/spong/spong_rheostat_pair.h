#ifndef SPONG_RHEOSTAT_PAIR_H
#define SPONG_RHEOSTAT_PAIR_H
#include "spong_rheostat.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Exact polynomial family A=A0+eta*A1+eta^2*A2, B=B0+eta*B1, C fixed.
 * All A arrays have na entries, all B arrays nb; zero padding is explicit.
 * Four fixed isolating boxes identify (source,target) for each connection.
 * Every evaluation validates/refines those boxes against its rational model.
 * No callback, Python numerical decision, or nearest-saddle reselection occurs.
 */
typedef struct {
    const char *const *A[3]; size_t na;
    const char *const *B[2]; size_t nb;
    const char *C;
    const char *bounds[8];
    int32_t sources[4], directions[4];
    const char *launch_radius,*level_fraction;
} spong_rheostat_pair_family;
#define SPONG_PAIR_FIELDS 11
/* lambda,eta,gap0,gap1,scaled_correction_lambda,scaled_correction_eta,
 * row_scaled_determinant,eta_derivative_error0,eta_derivative_error1,
 * scaled_correction_norm,scaled_residual_norm. Numeric diagnostics, not bounds. */
typedef struct {
    int32_t status;
    char values[SPONG_PAIR_FIELDS][SPONG_RHEOSTAT_TEXT];
    unsigned iterations,evaluations,backtracks,derivative_refinements;
    double max_backward_error;
    char reason[128];
} spong_rheostat_pair_result;
SPONG_API int spong_rheostat_pair_locate(const spong_rheostat_pair_family *family,
    const char *const initial[2],const char *const lower[2],const char *const upper[2],
    const char *eta_difference_step,const char *scaled_tolerance,
    const spong_rheostat_policy *inner_policy,unsigned max_iterations,
    spong_rheostat_pair_result *result);
#ifdef __cplusplus
}
#endif
#endif
