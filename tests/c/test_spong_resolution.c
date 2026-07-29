#include "spong/spong_resolution.h"

#include <assert.h>

int main(void) {
    spong_morse_analysis analysis = {
        1, 1, 1, 1, 1, 1, 12.0, 0x1p-20, 7.0
    };
    spong_resolution_policy policy = {
        SPONG_POLICY_ROOT_COLLISION | SPONG_POLICY_HESSIAN_CONDITION,
        20.0, 10.0, 100.0
    };
    assert(spong_abi_version() == SPONG_ABI_VERSION);
    spong_resolution_result preflight;
    assert(spong_resolution_preflight(&analysis, &policy, &preflight) == 0);
    assert(preflight.status == SPONG_MORSE_NUMERICALLY_UNRESOLVED);
    assert(preflight.primary_reason == SPONG_REASON_ROOT_COLLISION_MARGIN);
    assert(preflight.reason_mask
           & (UINT32_C(1) << (SPONG_REASON_HESSIAN_RESOLUTION-1)));

    analysis.exact_morse = 0;
    assert(spong_resolution_preflight(&analysis, &policy, &preflight) == 0);
    assert(preflight.status == SPONG_CERTIFIED_NON_MORSE);
    assert(preflight.primary_reason == SPONG_REASON_EXACT_NON_MORSE);

    spong_resolution_result final;
    assert(spong_resolution_finalize(1, 0, &final) == 0);
    assert(final.status == SPONG_CERTIFIED_PORTRAIT);
    assert(final.primary_reason == SPONG_REASON_NONE);
    return 0;
}
