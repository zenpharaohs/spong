#ifndef SPONG_LOCAL_H
#define SPONG_LOCAL_H

#include <stddef.h>
#include <stdint.h>

#include "spong_exact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t u_count;
    size_t s_count;
    double *coefficients;  /* packed [component][u][s], two components */
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
    const char *const *normal_coefficients,
    size_t normal_u_count,
    size_t normal_s_count,
    const double selected_map[6],
    const char *unstable_eigenvalue,
    const char *stable_eigenvalue,
    uint64_t precision_bits,
    spong_vector_polynomial *result);

SPONG_API void spong_vector_polynomial_destroy(
    spong_vector_polynomial *polynomial);

#ifdef __cplusplus
}
#endif

#endif
