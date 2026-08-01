#include "spong/spong_local.h"

#include <assert.h>
#include <math.h>

int main(void) {
    /* F(u,s) = (2u, -3s), packed [component][u][s]. */
    const char *normal[] = {
        "0", "0", "2", "0",
        "0", "-3", "0", "0"
    };
    const double identity_map[6] = {0, 0, 0, 0, 0, 0};
    spong_vector_polynomial result;
    assert(spong_poincare_pullback_decimal(
        normal, 2, 2, identity_map, "2", "-3", 192, &result)
        == SPONG_EXACT_OK);
    assert(result.u_count == 2);
    assert(result.s_count == 2);
    assert(result.coefficients[2] == 2.0);
    assert(result.coefficients[5] == -3.0);
    for (size_t i = 0; i < 8; ++i)
        assert(isfinite(result.coefficients[i]));
    spong_vector_polynomial_destroy(&result);
    return 0;
}
