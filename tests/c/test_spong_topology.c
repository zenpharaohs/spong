#include "spong/spong_topology.h"

#include <assert.h>
#include <math.h>

int main(void) {
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
