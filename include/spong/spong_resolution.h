#ifndef SPONG_RESOLUTION_H
#define SPONG_RESOLUTION_H

/*
 * Stable, allocation-free C99 resolution policy ABI.
 *
 * The numerical engines and exact-algebra providers fill the analysis
 * structure.  Python, MATLAB, Swift/Objective-C, and other frontends call the
 * same decision functions and therefore cannot disagree about terminal state.
 */

#include <stdint.h>

#if defined(_WIN32) && defined(SPONG_BUILD_SHARED)
# if defined(SPONG_BUILDING_LIBRARY)
#  define SPONG_API __declspec(dllexport)
# else
#  define SPONG_API __declspec(dllimport)
# endif
#elif defined(__GNUC__) || defined(__clang__)
# define SPONG_API __attribute__((visibility("default")))
#else
# define SPONG_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPONG_ABI_VERSION UINT32_C(2)

typedef enum {
    SPONG_RESOLUTION_PROCEED = 0,
    SPONG_CERTIFIED_NON_MORSE = 1,
    SPONG_MORSE_NUMERICALLY_UNRESOLVED = 2,
    SPONG_CERTIFIED_PORTRAIT = 3
} spong_resolution_status;

typedef enum {
    SPONG_REASON_NONE = 0,
    SPONG_REASON_EXACT_NON_MORSE = 1,
    SPONG_REASON_MODEL_HYPOTHESIS = 2,
    SPONG_REASON_ROOT_COLLISION_MARGIN = 3,
    SPONG_REASON_HESSIAN_RESOLUTION = 4,
    SPONG_REASON_LOCAL_NONLINEARITY = 5,
    SPONG_REASON_BINARY64_COORDINATE_COLLISION = 6,
    SPONG_REASON_ARITHMETIC_FAILURE = 7,
    SPONG_REASON_BRANCH_ABORT = 8,
    SPONG_REASON_TOPOLOGY_UNRESOLVED = 9
} spong_resolution_reason;

enum {
    SPONG_POLICY_ROOT_COLLISION = UINT32_C(1) << 0,
    SPONG_POLICY_HESSIAN_CONDITION = UINT32_C(1) << 1,
    SPONG_POLICY_LOCAL_GAMMA = UINT32_C(1) << 2,
    SPONG_POLICY_DISTINCT_BINARY64 = UINT32_C(1) << 3
};

typedef struct {
    uint32_t enabled;
    double min_root_collision_margin_log2_eps;
    double max_hessian_condition_loss_bits;
    double max_gamma_target_product_log2;
} spong_resolution_policy;

typedef struct {
    int32_t exact_morse;
    int32_t A_positive;
    int32_t critical_coordinates_binary64_distinct;
    int32_t has_root_collision_margin;
    int32_t has_hessian_relative_nonsingularity;
    int32_t has_gamma_target_product;
    double root_collision_margin_log2_eps;
    double min_hessian_relative_nonsingularity;
    double max_gamma_target_product_log2;
} spong_morse_analysis;

typedef struct {
    int32_t status;          /* spong_resolution_status */
    int32_t primary_reason;  /* spong_resolution_reason */
    uint32_t reason_mask;
} spong_resolution_result;

SPONG_API uint32_t spong_abi_version(void);

SPONG_API int spong_resolution_preflight(
    const spong_morse_analysis *analysis,
    const spong_resolution_policy *policy,
    spong_resolution_result *out);

SPONG_API int spong_resolution_finalize(
    int32_t topology_certified, int32_t branch_aborted,
    spong_resolution_result *out);

SPONG_API const char *spong_resolution_status_name(int32_t status);
SPONG_API const char *spong_resolution_reason_name(int32_t reason);

#ifdef __cplusplus
}
#endif

#endif
