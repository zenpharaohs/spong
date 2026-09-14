#ifndef SPONG_SMALE_H
#define SPONG_SMALE_H

/*
 * Exact trapping-tube kernels used by the finite-plane Morse--Smale
 * certificate.  Frontends may supply floating trajectories as proposals,
 * but every value crossing this boundary is represented as a canonical
 * decimal rational and every load-bearing inequality is recomputed by the
 * library with GMP rationals.
 */

#include <stddef.h>
#include <stdint.h>

#include "spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *numerator;
    const char *denominator; /* nonzero, positive */
} spong_rational_input;

typedef struct {
    char *numerator;
    char *denominator;
} spong_owned_rational;

typedef struct {
    spong_owned_rational lower;
    spong_owned_rational upper;
} spong_owned_rational_interval;

typedef struct {
    const spong_rational_input *alpha; /* A(b), ascending order */
    size_t alpha_count;
    const spong_rational_input *beta; /* B(b), ascending order */
    size_t beta_count;
    const spong_rational_input *n; /* N=A'B-2B'A, ascending order */
    size_t n_count;
    spong_rational_input loss_constant;
} spong_exact_loss_pencil;

typedef struct {
    spong_rational_input level; /* proposal metadata; not trusted as exact */
    spong_rational_input b;
    spong_rational_input y;
} spong_b_parameter_centre;

typedef enum {
    SPONG_SMALE_OK = 0,
    SPONG_SMALE_INVALID_ARGUMENT = 1,
    SPONG_SMALE_PARSE_FAILURE = 2,
    SPONG_SMALE_ALLOCATION_FAILURE = 3,
    SPONG_SMALE_WORK_LIMIT = 4,
    SPONG_SMALE_MODEL_IDENTITY_FAILURE = 5,
    SPONG_SMALE_PROPOSAL_NONMONOTONE = 6,
    SPONG_SMALE_TUBE_UNRESOLVED = 7,
    SPONG_SMALE_TARGET_UNBRACKETED = 8,
    SPONG_SMALE_PROJECTION_UNRESOLVED = 9,
    SPONG_SMALE_INTERNAL_FAILURE = 10
} spong_smale_status;

typedef enum {
    SPONG_SMALE_REASON_NONE = 0,
    SPONG_SMALE_REASON_BAD_RATIONAL = 1,
    SPONG_SMALE_REASON_BAD_POLICY = 2,
    SPONG_SMALE_REASON_N_IDENTITY = 3,
    SPONG_SMALE_REASON_CENTRE_ORDER = 4,
    SPONG_SMALE_REASON_ENDPOINT_BITS = 5,
    SPONG_SMALE_REASON_TRANSVERSE_DIVISOR = 6,
    SPONG_SMALE_REASON_B_PARAMETER_SINGULAR = 7,
    SPONG_SMALE_REASON_CRITICAL_POINT = 8,
    SPONG_SMALE_REASON_FACE_INEQUALITY = 9,
    SPONG_SMALE_REASON_RADIUS_CAP = 10,
    SPONG_SMALE_REASON_SLAB_BUDGET = 11,
    SPONG_SMALE_REASON_TARGET_LEVEL = 12,
    SPONG_SMALE_REASON_PROJECTION_BUDGET = 13,
    SPONG_SMALE_REASON_TARGET_SHEET = 14,
    SPONG_SMALE_REASON_TARGET_MISSES_TUBE = 15,
    SPONG_SMALE_REASON_ALLOCATION = 16,
    SPONG_SMALE_REASON_INTERNAL = 17,
    SPONG_SMALE_REASON_ROOT_INTERVAL = 18,
    SPONG_SMALE_REASON_FRAME_SINGULAR = 19,
    SPONG_SMALE_REASON_CONE_UNRESOLVED = 20,
    SPONG_SMALE_REASON_FROBENIUS_UNRESOLVED = 21,
    SPONG_SMALE_REASON_SECTION_UNRESOLVED = 22,
    SPONG_SMALE_REASON_SECTION_SHEET = 23,
    SPONG_SMALE_REASON_FIXED_SHEET_DOMAIN = 24,
    SPONG_SMALE_REASON_INITIAL_PROJECTION = 25
} spong_smale_reason;

typedef enum {
    SPONG_CRITICAL_ROOT_B = 1,
    SPONG_CRITICAL_ROOT_N = 2
} spong_critical_root_source;

typedef struct {
    spong_rational_input lower;
    spong_rational_input upper;
} spong_rational_interval_input;

typedef struct {
    int32_t critical_source; /* spong_critical_root_source */
    int32_t orientation;     /* -1 or +1 */
    spong_rational_interval_input critical_b;
    spong_rational_input frame[4]; /* row-major 2x2 */
    spong_rational_input selected_map[6];
    spong_rational_input departing_eigenvalue;
    spong_rational_input desired_reach;
    uint32_t require_frobenius;
} spong_local_launch_request;

typedef struct {
    uint64_t max_reach_halvings;
    uint64_t max_slope_doublings;
    uint64_t tangent_bisections;
    uint64_t section_bisections;
    uint64_t frobenius_subdivisions;
    uint64_t max_rational_bits;
    uint32_t verify_n_identity;
} spong_local_launch_policy;

typedef struct {
    uint64_t input_rationals;
    uint64_t coefficient_intervals;
    uint64_t interval_evaluations;
    uint64_t cone_tests;
    uint64_t reach_halvings;
    uint64_t slope_doublings;
    uint64_t tangent_bisections;
    uint64_t section_bisections;
    uint64_t section_retries;
    uint64_t peak_rational_bits;
} spong_local_launch_work;

typedef struct {
    int32_t status;
    int32_t primary_reason;
    int32_t orientation;
    int32_t time_direction;
    uint32_t validated;
    uint32_t cone_power; /* 1 for linear; 2 for Frobenius graph cone */
    uint32_t b_section_validated;
    int32_t b_direction;
    spong_owned_rational reach;
    spong_owned_rational cone_slope;
    spong_owned_rational flow_margin;
    spong_owned_rational lower_face_margin;
    spong_owned_rational upper_face_margin;
    spong_owned_rational_interval tangent_slope;
    spong_owned_rational section_level;
    spong_owned_rational_interval section_b;
    spong_owned_rational_interval section_y;
    spong_owned_rational b_section_b;
    spong_owned_rational_interval b_section_y;
    spong_owned_rational_interval b_section_level;
    spong_local_launch_work work;
} spong_local_launch_result;

/*
 * Validate a local invariant-manifold launch directly from the exact loss
 * pencil.  The critical interval is independently checked to contain one
 * simple B- or N-root.  The rational frame and quadratic Poincare map are
 * replayed exactly; no transformed polynomial supplied by a frontend is
 * trusted.  The result is an exact loss-section rectangle on one y sheet.
 */
SPONG_API int spong_local_launch_decimal(const spong_exact_loss_pencil *pencil,
                                         const spong_local_launch_request *request,
                                         const spong_local_launch_policy *policy,
                                         spong_local_launch_result *result);

SPONG_API void spong_local_launch_result_destroy(spong_local_launch_result *result);

typedef struct {
    uint64_t max_inflations;
    uint64_t max_slab_bisections;
    uint64_t max_projection_bisections;
    uint64_t max_projection_subboxes;
    uint64_t max_rational_bits;
    uint64_t radius_round_bits;
    uint32_t verify_n_identity;
} spong_b_parameter_policy;

typedef struct {
    uint64_t input_rationals;
    uint64_t slabs_accepted;
    uint64_t slab_bisections;
    uint64_t inflation_steps;
    uint64_t interval_evaluations;
    uint64_t projection_subboxes;
    uint64_t projection_depth;
    uint64_t peak_rational_bits;
} spong_b_parameter_work;

typedef struct {
    int32_t status;         /* spong_smale_status */
    int32_t primary_reason; /* spong_smale_reason */
    int32_t b_direction;    /* -1 or +1 once admitted */
    uint32_t tube_validated;
    uint32_t projection_validated;
    spong_owned_rational_interval target_b;
    spong_owned_rational_interval target_y;
    spong_owned_rational minimum_face_margin;
    spong_owned_rational minimum_gradient_norm_squared;
    spong_b_parameter_work work;
} spong_b_parameter_result;

/*
 * Validate a scalar trapping tube for y=y(b), stop after the first slab
 * strictly bracketing target_level, and project that slab onto the exact
 * target fibre.  The centreline is proposal data only.  A successful result
 * proves every inward-face and regularity inequality needed from the supplied
 * initial y interval through the target fibre.
 *
 * max_radius may be NULL for no cap.  Returned strings are library-owned and
 * must be released with spong_b_parameter_result_destroy, including across
 * DLL boundaries.  A refusal never returns a partial positive certificate.
 */
SPONG_API int spong_b_parameter_handoff_decimal(
    const spong_exact_loss_pencil *pencil, const spong_b_parameter_centre *centres,
    size_t centre_count, const spong_rational_input *initial_y_lower,
    const spong_rational_input *initial_y_upper,
    const spong_rational_input *target_level, const spong_rational_input *max_radius,
    const spong_b_parameter_policy *policy, spong_b_parameter_result *result);

SPONG_API void spong_b_parameter_result_destroy(spong_b_parameter_result *result);

typedef struct {
    uint64_t max_inflations;
    uint64_t max_slab_bisections;
    uint64_t max_projection_bisections;
    uint64_t max_projection_subboxes;
    uint64_t max_rational_bits;
    uint64_t radius_round_bits;
    uint64_t sqrt_bits;
    uint32_t verify_n_identity;
} spong_sheet_tube_policy;

typedef struct {
    uint64_t input_rationals;
    uint64_t slabs_accepted;
    uint64_t slab_bisections;
    uint64_t inflation_steps;
    uint64_t interval_evaluations;
    uint64_t projection_subboxes;
    uint64_t projection_depth;
    uint64_t peak_rational_bits;
} spong_sheet_tube_work;

typedef struct {
    int32_t status;          /* spong_smale_status */
    int32_t primary_reason;  /* spong_smale_reason */
    int32_t level_direction; /* -1 or +1 once admitted */
    int32_t sheet;           /* -1 or +1 */
    uint32_t tube_validated;
    uint32_t projection_validated;
    spong_owned_rational_interval terminal_b;
    spong_owned_rational_interval terminal_y;
    spong_owned_rational minimum_face_margin;
    spong_owned_rational minimum_gradient_norm_squared;
    spong_sheet_tube_work work;
} spong_sheet_tube_result;

/*
 * Project the initial (b,y) rectangle onto one strict exact loss sheet, then
 * validate a scalar trapping tube b=b(level) through every supplied centre.
 * Centreline values are proposals only.  The backend reconstructs
 * y=sheet*sqrt(B(b)^2+(level-C)A(b)) and proves inward lateral faces,
 * full-slab sheet regularity, and exclusion of critical points exactly.
 */
SPONG_API int spong_sheet_flow_tube_decimal(
    const spong_exact_loss_pencil *pencil, const spong_b_parameter_centre *centres,
    size_t centre_count, const spong_rational_input *initial_b_lower,
    const spong_rational_input *initial_b_upper,
    const spong_rational_input *initial_y_lower,
    const spong_rational_input *initial_y_upper, int32_t sheet,
    const spong_rational_input *max_radius, const spong_sheet_tube_policy *policy,
    spong_sheet_tube_result *result);

SPONG_API void spong_sheet_tube_result_destroy(spong_sheet_tube_result *result);

SPONG_API const char *spong_smale_status_name(int32_t status);
SPONG_API const char *spong_smale_reason_name(int32_t reason);

#ifdef __cplusplus
}
#endif

#endif
