#ifndef SPONG_LOCAL_H
#define SPONG_LOCAL_H

#include <stddef.h>
#include <stdint.h>

#include "spong_exact.h"
#include "spong_jet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Solve row-major matrix * solution = rhs by row equilibration, partial
 * pivoting and back-substitution. Allocation-free; inputs are not modified.
 * Returns 0 on success, -1 for invalid/nonfinite input, -3 for a singular,
 * insufficiently resolved or nonfinite solve. On failure solution is untouched.
 * The scaled pivot product must be >= 1e-12 (the local chart admission bar).
 * A componentwise backward-error check uses the original equilibrated rows;
 * success is not a forward-error or manifold certificate. */
SPONG_API int spong_local_solve2(const double matrix[4], const double rhs[2],
                               double solution[2]);

typedef struct {
    size_t u_count;
    size_t s_count;
    double *coefficients; /* packed [component][u][s], two components */
} spong_vector_polynomial;

/*
 * Build adj(DT) F(T(z)) for the quadratic near-identity map
 *
 *   T_0 = u + h00 u^2 + h01 us + h02 s^2,
 *   T_1 = s + h10 u^2 + h11 us + h12 s^2.
 *
 * normal_coefficients is packed [component][u][s] as decimal strings.
 * Decimal strings and the two eigenvalues are evaluated at precision_bits;
 * the six selected_map coefficients are the exact binary64 map used later.
 * The result receives its one and only binary64 rounding on export.
 */
SPONG_API int spong_poincare_pullback_decimal(
    const char *const *normal_coefficients, size_t normal_u_count,
    size_t normal_s_count, const double selected_map[6],
    const char *unstable_eigenvalue, const char *stable_eigenvalue,
    uint64_t precision_bits, spong_vector_polynomial *result);

SPONG_API void spong_vector_polynomial_destroy(spong_vector_polynomial *polynomial);

typedef struct {
    size_t iterations;
    double relative_change;
    int finite;
} spong_poincare_graph_result;

/*
 * Floating proposal for the invariant graph in Poincare coordinates.
 *
 * This is deliberately not a certificate.  It is the fast Hadamard graph
 * fixed point used to propose the centre line for the exact graph-tube
 * verifier.  x and graph must each have graph_count entries.  The function
 * returns 0 for a well-formed request (including non-convergence, reported
 * by relative_change); -1 for invalid input and -2 on allocation failure.
 */
SPONG_API int
spong_poincare_graph_proposal(const spong_jet *jet, const double frame[4],
                              const double selected_map[6], double departing_eigenvalue,
                              double transverse_eigenvalue, double alpha0,
                              double alpha1, int orientation, size_t graph_count,
                              double tolerance, size_t max_iterations, double *x,
                              double *graph, spong_poincare_graph_result *result);

#ifdef __cplusplus
}
#endif

#endif
