/*
 * Centered critical-point jet.  See spong/spong_jet.h.
 *
 * spong_jet_poly, the two fields and the two steps are verbatim relocations
 * of local_poly, local_raw_fj, local_normalized_fj and the LocalKernel step
 * methods from src/spong/_native.c, and compile FUSED (platform default):
 * the Python oracle reaches them through the extension, so what they
 * compute IS the specification.  spong_jet_potential alone is exact -- it
 * mirrors LocalJet.potential, a pure-Python evaluator, statement for
 * statement, and the goldens were frozen against that arithmetic.
 */

#include "spong/spong_jet.h"

#include <math.h>

#if defined(__clang__)
#  define SPONG_FP_EXACT      _Pragma("clang fp contract(off)")
#  define SPONG_FP_EXACT_FN
#elif defined(__GNUC__)
#  define SPONG_FP_EXACT
#  define SPONG_FP_EXACT_FN   __attribute__((optimize("fp-contract=off")))
#else
#  define SPONG_FP_EXACT
#  define SPONG_FP_EXACT_FN
#endif

static double horner(const double *c, size_t n, double x) {
    double acc = 0.0;
    for (size_t i = n; i-- > 0;) {
        acc = acc * x + c[i];
    }
    return acc;
}

void spong_jet_poly(const spong_jet *k, const double z[2],
                    double g[2], double H[2][2]) {
    double da = z[0], db = z[1];
    for (int d = 0; d < 2; d++) {
        double value = 0.0, derivative_a = 0.0, derivative_b = 0.0;
        for (int r = k->rows[d]; r-- > 0;) {
            double row = horner(k->c[d][r], k->n[d][r], db);
            double drow = 0.0;
            for (size_t j = k->n[d][r]; j-- > 1;)
                drow = drow*db + (double)j*k->c[d][r][j];
            derivative_a = derivative_a*da + value;
            value = value*da + row;
            derivative_b = derivative_b*da + drow;
        }
        g[d] = value;
        if (H != NULL) {
            H[d][0] = derivative_a;
            H[d][1] = derivative_b;
        }
    }
}

/* LocalJet.potential (src/spong/local.py), in its operation order. */
SPONG_FP_EXACT_FN
double spong_jet_potential(const spong_jet *k, double da, double db) {
    SPONG_FP_EXACT
    double value = 0.0;
    double apow = da;
    for (int i = 0; i < k->rows[0]; i++) {
        const double *row = k->c[0][i];
        size_t n = k->n[0][i];
        double acc = 0.0;
        for (size_t j = n; j-- > 0;)
            acc = acc*db + row[j];
        value += apow*acc/(double)(i+1);
        apow *= da;
    }
    double bpow = db;
    if (k->rows[1] > 0) {
        const double *row = k->c[1][0];
        size_t n = k->n[1][0];
        for (size_t j = 0; j < n; j++) {
            value += bpow*row[j]/(double)(j+1);
            bpow *= db;
        }
    }
    return value;
}

int spong_jet_normalized_fj(void *ctx, const double z[2],
                            double f[2], double J[2][2]) {
    double g[2], H[2][2];
    spong_jet_poly((const spong_jet *)ctx, z, g, J == NULL ? NULL : H);
    double ng = hypot(g[0], g[1]);
    if (!(ng > 1e-300) || !isfinite(ng)) return 0;
    f[0] = g[0]/ng; f[1] = g[1]/ng;
    if (J != NULL) {
        /* g^T H columns; do not assume rounded cross-partials are identical. */
        double Hg[2] = {g[0]*H[0][0] + g[1]*H[1][0],
                        g[0]*H[0][1] + g[1]*H[1][1]};
        double ng3 = ng*ng*ng;
        for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++)
            J[r][c] = H[r][c]/ng - g[r]*Hg[c]/ng3;
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

int spong_jet_raw_fj(void *ctx, const double z[2],
                     double f[2], double J[2][2]) {
    double H[2][2];
    spong_jet_poly((const spong_jet *)ctx, z, f, J == NULL ? NULL : H);
    if (J != NULL) {
        J[0][0] = H[0][0]; J[0][1] = H[0][1];
        J[1][0] = H[1][0]; J[1][1] = H[1][1];
    }
    return isfinite(f[0]) && isfinite(f[1]);
}

int spong_jet_raw_step(const spong_jet *jet, const double z[2],
                       double h, int order, double out[2]) {
    return spong_irk2_step((void *)jet, spong_jet_raw_fj, z, h, order, out);
}

int spong_jet_normalized_step(const spong_jet *jet, const double z[2],
                              double h, int order, double out[2]) {
    return spong_irk2_step((void *)jet, spong_jet_normalized_fj, z, h,
                           order, out);
}
