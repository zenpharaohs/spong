/* Floating Poincare graph proposal.  The exact certificate lives in the
 * GMP-backed Smale module; this routine only supplies its centre line. */

#include "spong/spong_local.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

int spong_local_solve2(const double matrix[4], const double rhs[2], double out[2]) {
    if (!matrix || !rhs || !out) return -1;
    double A[2][2], r[2];
    for (int i = 0; i < 2; ++i) {
        if (!isfinite(matrix[2*i]) || !isfinite(matrix[2*i+1]) || !isfinite(rhs[i]))
            return -1;
        double scale = fmax(fabs(matrix[2*i]), fabs(matrix[2*i+1]));
        if (!(scale > 0.0)) return -3;
        A[i][0] = matrix[2*i]/scale;
        A[i][1] = matrix[2*i+1]/scale;
        r[i] = rhs[i]/scale;
        if (!isfinite(r[i])) return -3;
    }
    /* Retain the equilibrated originals for an independent residual check. */
    int p = fabs(A[1][0]) > fabs(A[0][0]) ? 1 : 0;
    int q = 1-p;
    double pivot = A[p][0];
    if (!(fabs(pivot) > 0.0)) return -3;
    double factor = A[q][0]/pivot;
    double trailing = fma(-factor, A[p][1], A[q][1]);
    /* Same admission threshold as the former scaled determinant, now
     * obtained from the elimination pivots rather than cross products. */
    if (!(fabs(pivot*trailing) >= 1e-12)) return -3;
    double y = fma(-factor, r[p], r[q])/trailing;
    double x = fma(-A[p][1], y, r[p])/pivot;
    if (!isfinite(x) || !isfinite(y)) return -3;
    for (int i = 0; i < 2; ++i) {
        /* Scale the unknowns and RHS together to keep this check finite. */
        double s = fmax(fmax(fabs(x), fabs(y)), fabs(r[i]));
        if (s == 0.0) continue;
        double u = x/s, v = y/s, b = r[i]/s;
        double residual = fma(A[i][0], u, fma(A[i][1], v, -b));
        double bound = fabs(A[i][0]*u) + fabs(A[i][1]*v) + fabs(b);
        if (!isfinite(residual) || fabs(residual) > 128.0*DBL_EPSILON*bound)
            return -3;
    }
    out[0] = x;
    out[1] = y;
    return 0;
}

int spong_poincare_graph_proposal(const spong_jet *jet, const double V[4],
                                  const double M[6], double ld, double lt,
                                  double alpha0, double alpha1, int sign, size_t n,
                                  double tol, size_t max_iter, double *x, double *graph,
                                  spong_poincare_graph_result *result) {
    if (result != NULL) {
        result->iterations = 0;
        result->relative_change = INFINITY;
        result->finite = 0;
    }
    if (jet == NULL || V == NULL || M == NULL || x == NULL || graph == NULL ||
        result == NULL || !(alpha0 > 0.0 && alpha1 > alpha0) ||
        (sign != -1 && sign != 1) || n < 5 || !(ld * lt < 0.0) || !(tol >= 0.0) ||
        max_iter == 0)
        return -1;

    double *h = calloc(n, sizeof(*h));
    double *hn = malloc(n * sizeof(*hn));
    double *Q = malloc(n * sizeof(*Q));
    if (h == NULL || hn == NULL || Q == NULL) {
        free(h);
        free(hn);
        free(Q);
        return -2;
    }

    double dt = log(alpha1 / alpha0) / (double)(n - 1);
    double rho = fabs(ld), nu = fabs(lt);
    double kappa = nu / rho, q = kappa * dt, decay = exp(-q);
    double w0, w1;
    if (fabs(q) < 1e-3) {
        /* Stable expansions of the exact exponential interpolation weights. */
        double q2 = q * q, q3 = q2 * q, q4 = q2 * q2;
        w1 = dt * (0.5 - q / 6.0 + q2 / 24.0 - q3 / 120.0 + q4 / 720.0);
        w0 = dt * (0.5 - q / 3.0 + q2 / 8.0 - q3 / 30.0 + q4 / 144.0);
    } else {
        w1 = dt * (q - 1.0 + decay) / (q * q);
        w0 = dt * (1.0 - decay * (1.0 + q)) / (q * q);
    }
    for (size_t i = 0; i < n; i++)
        x[i] = alpha0 * exp(dt * (double)i);

    double rel = INFINITY;
    size_t it;
    double orient = ld > 0.0 ? 1.0 : -1.0;
    int finite = 1;
    for (it = 1; it <= max_iter; it++) {
        Q[0] = 0.0;
        for (size_t i = 1; i < n; i++) {
            double u = sign * x[i], s = h[i];
            double U = u + M[0] * u * u + M[1] * u * s + M[2] * s * s;
            double S = s + M[3] * u * u + M[4] * u * s + M[5] * s * s;
            double z[2] = {V[0] * U + V[1] * S, V[2] * U + V[3] * S}, g[2];
            spong_jet_poly(jet, z, g, NULL);
            double ve0 = V[0] * g[0] + V[2] * g[1];
            double ve1 = V[1] * g[0] + V[3] * g[1];
            double j00 = 1 + 2 * M[0] * u + M[1] * s, j01 = M[1] * u + 2 * M[2] * s;
            double j10 = 2 * M[3] * u + M[4] * s, j11 = 1 + M[4] * u + 2 * M[5] * s;
            double det = j00 * j11 - j01 * j10;
            if (!(fabs(det) > 1e-14) || !isfinite(det)) {
                finite = 0;
                break;
            }
            double vu = (j11 * ve0 - j01 * ve1) / det;
            double vs = (-j10 * ve0 + j00 * ve1) / det;
            double vx = sign * orient * vu, vy = orient * vs;
            if (!(fabs(vx) > 1e-300) || !isfinite(vx)) {
                finite = 0;
                break;
            }
            /* t=log(x): h_t + (nu/rho)h = x*vy/vx + kappa*h. */
            Q[i] = x[i] * vy / vx + kappa * s;
        }
        if (!finite) {
            rel = INFINITY;
            break;
        }
        hn[0] = 0.0;
        double scale = 0.0, change = 0.0;
        for (size_t i = 1; i < n; i++) {
            hn[i] = decay * hn[i - 1] + w0 * Q[i - 1] + w1 * Q[i];
            scale = fmax(scale, fabs(hn[i]));
            change = fmax(change, fabs(hn[i] - h[i]));
        }
        rel = change / fmax(scale, 1e-300);
        double *tmp = h;
        h = hn;
        hn = tmp;
        if (rel < tol)
            break;
    }
    if (it > max_iter)
        it = max_iter;
    for (size_t i = 0; i < n; i++)
        graph[i] = h[i];
    result->iterations = it;
    result->relative_change = rel;
    result->finite = finite;
    free(h);
    free(hn);
    free(Q);
    return 0;
}
