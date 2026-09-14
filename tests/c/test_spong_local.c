#include "spong/spong_local.h"

#include <assert.h>
#include <math.h>

int main(void) {
    /* F(u,s) = (2u, -3s), packed [component][u][s]. */
    const char *normal[] = {"0", "0", "2", "0", "0", "-3", "0", "0"};
    const double identity_map[6] = {0, 0, 0, 0, 0, 0};
    spong_vector_polynomial result;
    assert(spong_poincare_pullback_decimal(normal, 2, 2, identity_map, "2", "-3", 192,
                                           &result) == SPONG_EXACT_OK);
    assert(result.u_count == 2);
    assert(result.s_count == 2);
    assert(result.coefficients[2] == 2.0);
    assert(result.coefficients[5] == -3.0);
    for (size_t i = 0; i < 8; ++i)
        assert(isfinite(result.coefficients[i]));
    spong_vector_polynomial_destroy(&result);

    /* The proposal itself is a public backend routine, not extension code. */
    static const double zero[] = {0.0};
    static const double two[] = {2.0};
    static const double minus_two_s[] = {0.0, -2.0};
    const double *g0[] = {zero, two};
    const double *g1[] = {minus_two_s};
    const size_t n0[] = {1, 1};
    const size_t n1[] = {2};
    spong_jet jet = {{g0, g1}, {n0, n1}, {2, 1}};
    const double frame[4] = {1, 0, 0, 1};
    const double map[6] = {0, 0, 0, 0, 0, 0};
    double x[65], graph[65];
    spong_poincare_graph_result proposal;
    assert(spong_poincare_graph_proposal(&jet, frame, map, 2.0, -2.0, 1e-4, 0.1, 1, 65,
                                         1e-13, 80, x, graph, &proposal) == 0);
    assert(proposal.finite);
    assert(proposal.relative_change == 0.0);
    assert(x[0] == 1e-4);
    for (int i = 0; i < 65; ++i)
        assert(graph[i] == 0.0);
    return 0;
}
