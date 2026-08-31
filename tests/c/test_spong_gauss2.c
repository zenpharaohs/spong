/*
 * Standalone C test for spong/spong_gauss2.h: the relocated two-dimensional
 * Gauss--Legendre steps, exercised with no Python anywhere.
 *
 *   L = a^2 + b^2  (A = 1, B = 0, C = 0):  the normalized step from (1, 0)
 *   moves exactly h along a; the potential step lands on loss 1 + h.
 *   L = a^2 (1 + b^2):  a curved field; a full step and two half steps of
 *   the order-8 method must agree far inside the tolerance the potential-rate
 *   phases use, and the loss/gradient/Hessian helpers must match closed form.
 */

#include "spong/spong_gauss2.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        failures++;
        printf("FAIL: %s\n", what);
    }
}

int main(void) {
    double one = 1.0, zero = 0.0;
    spong_field quadratic = {
        &one, &zero, &zero, &zero, &zero, &zero, &zero, &zero,
        1, 1, 1, 1, 1, 1, 1, 1, 0.0 };
    double z[2] = {1.0, 0.0}, out[2];
    for (int order = 4; order <= 8; order += 2) {
        double h = 0.25;
        check(spong_normalized_step(&quadratic, z, h, order, out),
              "normalized step converges");
        check(fabs(out[0] - (1.0 + h)) <= 1e-12 && fabs(out[1]) <= 1e-12,
              "normalized step moves h along the gradient");
        check(spong_potential_step(&quadratic, z, h, order, out),
              "potential step converges");
        check(fabs(spong_field_loss(&quadratic, out[0], out[1]) - (1.0 + h))
              <= 1e-12,
              "potential step lands on the target loss");
    }

    double Acoef[3] = {1.0, 0.0, 1.0}, Ap[2] = {0.0, 2.0}, App[1] = {2.0};
    spong_field curved = {
        Acoef, Ap, App, &zero, &zero, &zero, &zero, &zero,
        3, 2, 1, 1, 1, 1, 1, 1, 0.0 };
    double z2[2] = {0.7, 0.4};
    double full[2], half[2], two[2];
    check(spong_normalized_step(&curved, z2, 0.02, 8, full),
          "curved: full step");
    check(spong_normalized_step(&curved, z2, 0.01, 8, half)
          && spong_normalized_step(&curved, half, 0.01, 8, two),
          "curved: two half steps");
    check(hypot(full[0] - two[0], full[1] - two[1]) <= 1e-13,
          "curved: full/two-half agreement");

    double g[2], H[2][2];
    spong_field_gradient(&curved, 0.7, 0.4, g);
    spong_field_hessian(&curved, 0.7, 0.4, H);
    /* L = a^2 (1 + b^2): L_a = 2a(1+b^2), L_b = 2 a^2 b,
     * L_aa = 2(1+b^2), L_ab = 4ab, L_bb = 2a^2 */
    check(fabs(g[0] - 2.0*0.7*(1.0 + 0.16)) <= 1e-15
          && fabs(g[1] - 2.0*0.49*0.4) <= 1e-15, "gradient closed form");
    check(fabs(H[0][0] - 2.0*1.16) <= 1e-15
          && fabs(H[0][1] - 4.0*0.7*0.4) <= 1e-15
          && fabs(H[1][1] - 2.0*0.49) <= 1e-15, "hessian closed form");
    check(fabs(spong_field_loss(&curved, 0.7, 0.4) - 0.49*1.16) <= 1e-15,
          "loss closed form");

    check(!spong_normalized_step(&quadratic, (double[2]){0.0, 0.0}, 0.1, 6,
                                 out),
          "normalized step refuses at a critical point");

    if (failures == 0) printf("spong_gauss2: all checks pass\n");
    return failures == 0 ? 0 : 1;
}
