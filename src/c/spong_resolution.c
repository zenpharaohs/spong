#include "spong/spong_resolution.h"

#include <math.h>
#include <stddef.h>

static uint32_t reason_bit(spong_resolution_reason reason) {
    if (reason <= SPONG_REASON_NONE || reason > SPONG_REASON_TOPOLOGY_UNRESOLVED)
        return 0;
    return UINT32_C(1) << ((unsigned)reason - 1U);
}

static void refuse(spong_resolution_result *out,
                   spong_resolution_reason reason) {
    out->status = SPONG_MORSE_NUMERICALLY_UNRESOLVED;
    if (out->primary_reason == SPONG_REASON_NONE)
        out->primary_reason = reason;
    out->reason_mask |= reason_bit(reason);
}

uint32_t spong_abi_version(void) {
    return SPONG_ABI_VERSION;
}

int spong_resolution_preflight(
        const spong_morse_analysis *analysis,
        const spong_resolution_policy *policy,
        spong_resolution_result *out) {
    if (out == NULL)
        return -1;
    *out = (spong_resolution_result){
        SPONG_RESOLUTION_PROCEED, SPONG_REASON_NONE, 0
    };
    if (analysis == NULL || policy == NULL) {
        refuse(out, SPONG_REASON_ARITHMETIC_FAILURE);
        return -1;
    }
    if (!analysis->A_positive) {
        refuse(out, SPONG_REASON_MODEL_HYPOTHESIS);
        return 0;
    }
    if (!analysis->exact_morse) {
        out->status = SPONG_CERTIFIED_NON_MORSE;
        out->primary_reason = SPONG_REASON_EXACT_NON_MORSE;
        out->reason_mask = reason_bit(SPONG_REASON_EXACT_NON_MORSE);
        return 0;
    }
    if ((policy->enabled & SPONG_POLICY_DISTINCT_BINARY64)
            && !analysis->critical_coordinates_binary64_distinct)
        refuse(out, SPONG_REASON_BINARY64_COORDINATE_COLLISION);
    if ((policy->enabled & SPONG_POLICY_ROOT_COLLISION)
            && analysis->has_root_collision_margin
            && analysis->root_collision_margin_log2_eps
                < policy->min_root_collision_margin_log2_eps)
        refuse(out, SPONG_REASON_ROOT_COLLISION_MARGIN);
    if ((policy->enabled & SPONG_POLICY_HESSIAN_CONDITION)
            && analysis->has_hessian_relative_nonsingularity
            && analysis->min_hessian_relative_nonsingularity > 0.0
            && -log2(analysis->min_hessian_relative_nonsingularity)
                > policy->max_hessian_condition_loss_bits)
        refuse(out, SPONG_REASON_HESSIAN_RESOLUTION);
    if ((policy->enabled & SPONG_POLICY_LOCAL_GAMMA)
            && analysis->has_gamma_target_product
            && analysis->max_gamma_target_product_log2
                > policy->max_gamma_target_product_log2)
        refuse(out, SPONG_REASON_LOCAL_NONLINEARITY);
    return 0;
}

int spong_resolution_finalize(
        int32_t topology_certified, int32_t branch_aborted,
        spong_resolution_result *out) {
    if (out == NULL)
        return -1;
    *out = (spong_resolution_result){
        SPONG_CERTIFIED_PORTRAIT, SPONG_REASON_NONE, 0
    };
    if (!topology_certified) {
        spong_resolution_reason reason = (
            branch_aborted ? SPONG_REASON_BRANCH_ABORT
                           : SPONG_REASON_TOPOLOGY_UNRESOLVED);
        out->status = SPONG_MORSE_NUMERICALLY_UNRESOLVED;
        out->primary_reason = reason;
        out->reason_mask = reason_bit(reason);
    }
    return 0;
}

const char *spong_resolution_status_name(int32_t status) {
    switch (status) {
    case SPONG_RESOLUTION_PROCEED: return "proceed";
    case SPONG_CERTIFIED_NON_MORSE: return "certified_non_morse";
    case SPONG_MORSE_NUMERICALLY_UNRESOLVED:
        return "morse_numerically_unresolved";
    case SPONG_CERTIFIED_PORTRAIT: return "certified_portrait";
    default: return "unknown";
    }
}

const char *spong_resolution_reason_name(int32_t reason) {
    switch (reason) {
    case SPONG_REASON_NONE: return "none";
    case SPONG_REASON_EXACT_NON_MORSE: return "exact_non_morse";
    case SPONG_REASON_MODEL_HYPOTHESIS: return "model_hypothesis";
    case SPONG_REASON_ROOT_COLLISION_MARGIN:
        return "root_collision_margin";
    case SPONG_REASON_HESSIAN_RESOLUTION: return "hessian_resolution";
    case SPONG_REASON_LOCAL_NONLINEARITY: return "local_nonlinearity";
    case SPONG_REASON_BINARY64_COORDINATE_COLLISION:
        return "binary64_coordinate_collision";
    case SPONG_REASON_ARITHMETIC_FAILURE: return "arithmetic_failure";
    case SPONG_REASON_BRANCH_ABORT: return "branch_abort";
    case SPONG_REASON_TOPOLOGY_UNRESOLVED: return "topology_unresolved";
    default: return "unknown";
    }
}
