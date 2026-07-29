#include "spong/spong_exact.h"

#include <assert.h>

static spong_sturm_analysis analyze(
        const char *const *coefficients, size_t n) {
    spong_exact_policy policy = {0, 0, 0};
    spong_sturm_analysis out;
    assert(spong_sturm_analyze_decimal(
        coefficients, n, &policy, &out) == 0);
    assert(out.status == SPONG_EXACT_OK);
    assert(out.work.chain_polynomials > 0);
    assert(out.work.peak_coefficient_bits > 0);
    return out;
}

int main(void) {
    const char *sqrt2[] = {"-2", "0", "1"};
    spong_sturm_analysis a = analyze(sqrt2, 3);
    assert(a.distinct_real_roots == 2);
    assert(a.repeated_real_roots == 0);
    assert(a.input_degree == 2);
    assert(a.squarefree_degree == 2);

    spong_exact_policy unlimited = {0, 0, 0};
    spong_sturm_plan *plan = 0;
    assert(spong_sturm_plan_create_decimal(
        sqrt2, 3, &unlimited, &plan, &a) == 0);
    uint32_t count = 0;
    assert(spong_sturm_plan_count(
        plan, "-2", "1", "0", "1", &count) == 0);
    assert(count == 1);
    assert(spong_sturm_plan_count(
        plan, "0", "1", "2", "1", &count) == 0);
    assert(count == 1);
    spong_isolation_policy isolation_policy = {0, 0, 0, 0};
    spong_isolation_work isolation_work;
    spong_root_interval *intervals = 0;
    size_t interval_count = 0;
    assert(spong_sturm_plan_isolate(
        plan, &isolation_policy, &intervals, &interval_count,
        &isolation_work) == 0);
    assert(isolation_work.status == SPONG_EXACT_OK);
    assert(interval_count == 2);
    spong_root_intervals_destroy(intervals, interval_count);
    spong_sturm_plan_destroy(plan);

    /* The interval convention is (lo, hi]: exclude zero, include one. */
    const char *endpoint_roots[] = {"0", "-1", "1"};
    assert(spong_sturm_plan_create_decimal(
        endpoint_roots, 3, &unlimited, &plan, &a) == 0);
    assert(spong_sturm_plan_count(
        plan, "0", "1", "1", "1", &count) == 0);
    assert(count == 1);
    int32_t sign = 0;
    assert(spong_sturm_plan_sign_at(
        plan, "1", "2", &sign) == 0);
    assert(sign < 0);
    spong_refinement_policy refinement_policy = {0, 0};
    spong_refinement_work refinement_work;
    spong_root_interval *refined = 0;
    assert(spong_sturm_plan_refine(
        plan, "1", "2", "3", "2", "1", "281474976710656",
        &refinement_policy, &refined, &refinement_work) == 0);
    assert(refinement_work.status == SPONG_EXACT_OK);
    assert(refinement_work.bisections > 0);
    spong_root_intervals_destroy(refined, 1);
    spong_sturm_plan_destroy(plan);

    /*
     * Exact zero with a root at 1e-6: the initial 2^-20 puncture contains
     * both, so isolation must halve and certify rather than lose the neighbor.
     */
    const char *close_to_zero[] = {"0", "-1", "1000000"};
    assert(spong_sturm_plan_create_decimal(
        close_to_zero, 3, &unlimited, &plan, &a) == 0);
    assert(spong_sturm_plan_isolate(
        plan, &isolation_policy, &intervals, &interval_count,
        &isolation_work) == 0);
    assert(interval_count == 2);
    assert(isolation_work.puncture_halvings >= 1);
    assert(intervals[0].exact == 1);
    spong_root_intervals_destroy(intervals, interval_count);
    spong_sturm_plan_destroy(plan);

    /* (x-1)^2 (x+2) = x^3 - 3x + 2. */
    const char *repeated_real[] = {"2", "-3", "0", "1"};
    a = analyze(repeated_real, 4);
    assert(a.distinct_real_roots == 2);
    assert(a.repeated_real_roots == 1);
    assert(a.squarefree_degree == 2);

    /* A repeated complex factor is not a real non-Morse event. */
    const char *repeated_complex[] = {"1", "0", "2", "0", "1"};
    a = analyze(repeated_complex, 5);
    assert(a.distinct_real_roots == 0);
    assert(a.repeated_real_roots == 0);

    spong_exact_policy tiny = {1, 0, 0};
    assert(spong_sturm_analyze_decimal(
        repeated_real, 4, &tiny, &a) != 0);
    assert(a.status == SPONG_EXACT_WORK_LIMIT);
    return 0;
}
