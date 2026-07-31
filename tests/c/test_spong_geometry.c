#include "spong/spong_geometry.h"

#include <assert.h>
#include <math.h>

int main(void) {
    /* L=a^2 gives grad L=(2a,0); this horizontal chord is exact. */
    const double A[] = {1.0};
    const double Ap[] = {0.0};
    const double B[] = {0.0};
    const double Bp[] = {0.0};
    const double radial[] = {
        1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0
    };
    spong_curve_diagnostics_result result;
    assert(spong_curve_diagnostics(
        A, 1, Ap, 1, B, 1, Bp, 1, radial, 4, 1, 1e3,
        &result) == 0);
    assert(result.angle_resolved == 2);
    assert(result.angle_unresolved == 0);
    assert(result.angle_energy < 1e-28);
    assert(result.backbone_residual == 0.0);
    return 0;
}
