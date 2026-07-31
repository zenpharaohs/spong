#include "spong/spong_topology.h"

#include <assert.h>
#include <math.h>

int main(void) {
    spong_topology_analysis analysis = {
        3, 12, 6, 6, 100, 1000, 0, 5000, 0, 0, 0, 0, 0
    };
    spong_topology_result result;
    assert(spong_topology_decide(&analysis, &result) == 0);
    assert(result.certified);
    assert(result.audit_complete);
    assert(result.branch_inventory_certified);
    assert(result.expected_stable == 6 && result.expected_unstable == 6);
    assert(result.primary_reason == SPONG_TOPOLOGY_REASON_NONE);

    analysis.forbidden_count = 1;
    analysis.uncertified_unstable_ends = 1;
    assert(spong_topology_decide(&analysis, &result) == 0);
    assert(!result.certified);
    assert(result.primary_reason == SPONG_TOPOLOGY_REASON_CONTACT);
    analysis.branch_aborted = 1;
    assert(spong_topology_decide(&analysis, &result) == 0);
    assert(!result.audit_complete);
    assert(result.primary_reason == SPONG_TOPOLOGY_REASON_BRANCH_ABORT);

    const double rising[] = {-1.0, -1.0, 1.0, 1.0};
    const double falling[] = {-1.0, 1.0, 1.0, -1.0};
    spong_contact_scan *pair = spong_contact_scan_create(
        rising, 2, falling, 2, 1e-12, 0);
    assert(pair != NULL);
    spong_contact_event event;
    assert(spong_contact_scan_next(pair, &event) == 1);
    assert(event.kind == SPONG_CONTACT_CROSS);
    assert(event.first_segment == 0 && event.second_segment == 0);
    assert(fabs(event.x) < 1e-15 && fabs(event.y) < 1e-15);
    assert(spong_contact_scan_next(pair, &event) == 0);
    spong_contact_scan_destroy(pair);

    const double loop[] = {
        -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0
    };
    spong_contact_scan *self = spong_contact_scan_create(
        loop, 4, NULL, 0, 1e-12, 1);
    assert(self != NULL);
    assert(spong_contact_scan_next(self, &event) == 1);
    assert(event.kind == SPONG_CONTACT_CROSS);
    assert(event.first_segment == 0 && event.second_segment == 2);
    assert(spong_contact_scan_next(self, &event) == 0);
    spong_contact_scan_destroy(self);
    return 0;
}
