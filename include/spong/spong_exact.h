#ifndef SPONG_EXACT_H
#define SPONG_EXACT_H

#include <stddef.h>
#include <stdint.h>

#include "spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPONG_EXACT_OK = 0,
    SPONG_EXACT_INVALID_ARGUMENT = 1,
    SPONG_EXACT_ALLOCATION_FAILURE = 2,
    SPONG_EXACT_PARSE_FAILURE = 3,
    SPONG_EXACT_WORK_LIMIT = 4,
    SPONG_EXACT_INTERNAL_FAILURE = 5
} spong_exact_status;

typedef struct {
    uint64_t max_coefficient_bits;  /* zero means unlimited */
    uint64_t max_chain_coefficients;
    uint64_t max_prs_steps;
} spong_exact_policy;

typedef struct {
    uint64_t prs_steps;
    uint64_t chain_polynomials;
    uint64_t chain_coefficients;
    uint64_t peak_coefficient_bits;
} spong_exact_work;

typedef struct {
    int32_t status;  /* spong_exact_status */
    uint32_t distinct_real_roots;
    uint32_t repeated_real_roots;
    int32_t input_degree;
    int32_t squarefree_degree;
    spong_exact_work work;
} spong_sturm_analysis;

typedef struct spong_sturm_plan spong_sturm_plan;

/*
 * Analyze an ascending-order primitive integer polynomial.  Coefficients are
 * canonical base-10 strings so no frontend exposes or depends on GMP layout.
 * The polynomial need not already be squarefree.
 */
SPONG_API int spong_sturm_analyze_decimal(
    const char *const *coefficients,
    size_t coefficient_count,
    const spong_exact_policy *policy,
    spong_sturm_analysis *out);

/*
 * Persistent squarefree Sturm chain for bounded rational queries.  A NULL
 * lower numerator denotes -infinity; a NULL upper numerator denotes
 * +infinity.  Denominators must be positive canonical decimal integers.
 */
SPONG_API int spong_sturm_plan_create_decimal(
    const char *const *coefficients,
    size_t coefficient_count,
    const spong_exact_policy *policy,
    spong_sturm_plan **plan,
    spong_sturm_analysis *analysis);

SPONG_API void spong_sturm_plan_destroy(spong_sturm_plan *plan);

SPONG_API size_t spong_sturm_plan_chain_length(
    const spong_sturm_plan *plan);

SPONG_API uint64_t spong_sturm_plan_chain_coefficients(
    const spong_sturm_plan *plan);

SPONG_API int spong_sturm_plan_count(
    const spong_sturm_plan *plan,
    const char *lower_numerator,
    const char *lower_denominator,
    const char *upper_numerator,
    const char *upper_denominator,
    uint32_t *count);

/* Exact sign of the original (not squarefree-reduced) polynomial at x. */
SPONG_API int spong_sturm_plan_sign_at(
    const spong_sturm_plan *plan,
    const char *numerator,
    const char *denominator,
    int32_t *sign);

typedef struct {
    uint64_t max_subdivision_nodes;  /* zero means unlimited */
    uint64_t max_puncture_halvings;
    uint64_t max_endpoint_bits;
    uint64_t max_intervals;
} spong_isolation_policy;

typedef struct {
    int32_t status;  /* spong_exact_status */
    uint64_t subdivision_nodes;
    uint64_t variation_evaluations;
    uint64_t polynomial_evaluations;
    uint64_t puncture_halvings;
    uint64_t max_subdivision_depth;
    uint64_t max_endpoint_bits;
} spong_isolation_work;

typedef struct {
    char *lower_numerator;
    char *lower_denominator;
    char *upper_numerator;
    char *upper_denominator;
    uint32_t exact;
} spong_root_interval;

/*
 * Isolate every distinct real root using the plan's persistent chain.
 * Strings and the interval array are owned by the library and must be
 * released with spong_root_intervals_destroy, including across DLLs.
 */
SPONG_API int spong_sturm_plan_isolate(
    const spong_sturm_plan *plan,
    const spong_isolation_policy *policy,
    spong_root_interval **intervals,
    size_t *interval_count,
    spong_isolation_work *work);

SPONG_API void spong_root_intervals_destroy(
    spong_root_interval *intervals,
    size_t interval_count);

typedef struct {
    uint64_t max_bisections;   /* zero means unlimited */
    uint64_t max_endpoint_bits;
} spong_refinement_policy;

typedef struct {
    int32_t status;  /* spong_exact_status */
    uint64_t bisections;
    uint64_t max_endpoint_bits;
} spong_refinement_work;

/*
 * Refine an interval known to contain one distinct root until
 * width <= relative_width * (1 + |lower| + |upper|).
 */
SPONG_API int spong_sturm_plan_refine(
    const spong_sturm_plan *plan,
    const char *lower_numerator,
    const char *lower_denominator,
    const char *upper_numerator,
    const char *upper_denominator,
    const char *relative_width_numerator,
    const char *relative_width_denominator,
    const spong_refinement_policy *policy,
    spong_root_interval **interval,
    spong_refinement_work *work);

#ifdef __cplusplus
}
#endif

#endif
