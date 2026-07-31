#include "spong/spong_geometry.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

static double rounded_mul(double x, double y) {
    volatile double product = x*y;
    return product;
}

static double horner(const double *coefficients, size_t count, double x) {
    double value = 0.0;
    for (size_t i = count; i-- > 0;)
        value = rounded_mul(value, x)+coefficients[i];
    return value;
}

int spong_curve_diagnostics(
        const double *A, size_t A_count,
        const double *Ap, size_t Ap_count,
        const double *B, size_t B_count,
        const double *Bp, size_t Bp_count,
        const double *points, size_t point_count,
        size_t start, double digit_budget,
        spong_curve_diagnostics_result *result) {
    if (A == NULL || Ap == NULL || B == NULL || Bp == NULL
            || points == NULL || result == NULL
            || A_count == 0 || Ap_count == 0
            || B_count == 0 || Bp_count == 0
            || !isfinite(digit_budget) || digit_budget < 0.0)
        return -1;
    *result = (spong_curve_diagnostics_result){0.0, 0, 0, 0.0};
    if (start < 1) start = 1;
    if (point_count < 3 || start >= point_count-1) return 0;
    for (size_t k = start; k < point_count-1; ++k) {
        double a = points[2*k], b = points[2*k+1];
        double Aval = horner(A, A_count, b);
        double Apval = horner(Ap, Ap_count, b);
        double Bval = horner(B, B_count, b);
        double Bpval = horner(Bp, Bp_count, b);
        double ga = 2.0*(rounded_mul(a, Aval)-Bval);
        double gb_left = rounded_mul(rounded_mul(-2.0, a), Bpval);
        double gb_right = rounded_mul(rounded_mul(a, a), Apval);
        double gb = gb_left+gb_right;
        double ng = hypot(ga, gb);
        double da = points[2*(k+1)]-points[2*(k-1)];
        double db = points[2*(k+1)+1]-points[2*(k-1)+1];
        double nd = hypot(da, db);
        double scale_a = 2.0*(rounded_mul(fabs(a), Aval)+fabs(Bval));
        double scale_b = rounded_mul(
            rounded_mul(2.0, fabs(a)), fabs(Bpval));
        scale_b += rounded_mul(rounded_mul(a, a), fabs(Apval));
        double gradient_floor = 16.0*DBL_EPSILON*hypot(scale_a, scale_b);
        double threshold = fmax(1e-12, digit_budget*gradient_floor);
        int gradient_resolved = ng >= threshold;
        if (!gradient_resolved || nd < 1e-14) {
            ++result->angle_unresolved;
        } else {
            double gha = ga/ng, ghb = gb/ng;
            double projection = gha*da+ghb*db;
            double dpa = da-projection*gha;
            double dpb = db-projection*ghb;
            result->angle_energy += 0.5*(dpa*dpa+dpb*dpb);
            ++result->angle_resolved;
        }
        if (!gradient_resolved) {
            double astar = Bval/Aval;
            if (fabs(astar) >= 1e-300) {
                double residual = fabs(a-astar)/fabs(astar);
                if (residual > result->backbone_residual)
                    result->backbone_residual = residual;
            }
        }
    }
    return 0;
}
