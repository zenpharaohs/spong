/* Fused minibatch polynomial gradient and two-coordinate Adam demo kernel.
 * Samples are supplied by NumPy so each arm keeps its existing RNG stream.
 * No fast-math: non-finite iterates have experimental meaning here.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

enum { SPONG_ADAM_OK = 0, SPONG_ADAM_BAD_STATE = 1,
       SPONG_ADAM_BAD_GRADIENT = 2, SPONG_ADAM_BAD_STEP = 3 };
enum { SPONG_LR_CONSTANT = 0, SPONG_LR_INVERSE_SQRT = 1,
       SPONG_LR_COSINE = 2 };

static double polynomial(const double *c, int n, double x) {
    double value = 0.0;
    for (int k = n - 1; k >= 0; --k) value = value * x + c[k];
    return value;
}

static double learning_rate(int kind, double base, uint64_t t,
                            uint64_t warmup, uint64_t horizon,
                            double final_fraction) {
    if (kind == SPONG_LR_CONSTANT) return base;
    if (kind == SPONG_LR_INVERSE_SQRT) {
        if (t <= warmup) return base * (double)t / (double)warmup;
        return base * sqrt((double)warmup / (double)t);
    }
    if (t <= warmup) return base * (double)t / (double)warmup;
    const uint64_t span = horizon > warmup ? horizon - warmup : 1;
    double u = (double)(t - warmup) / (double)span;
    if (u > 1.0) u = 1.0;
    const double multiplier = final_fraction + (1.0 - final_fraction) *
        0.5 * (1.0 + cos(3.14159265358979323846264338327950288 * u));
    return base * multiplier;
}

/* Return 0 for a complete chunk, otherwise a failure code.  `done` counts
 * successful steps and `points` holds exactly their iterates.  On a failed
 * update, Adam's accumulators and t advance but z stays at the last finite
 * point, matching checked_step's terminal-state contract. */
int spong_adam_chunk(const double *f, int nf, const double *g, int ng,
                     const double *gp, int ngp, const double *samples,
                     int batch, int steps, double *z, double *first,
                     double *second, uint64_t *t, int schedule,
                     double base_lr, uint64_t warmup, uint64_t horizon,
                     double final_fraction, double b1, double b2,
                     double eps, double weight_decay, double *points,
                     int *done) {
    if (!f || !g || !gp || !samples || !z || !first || !second || !t ||
        !points || !done || nf < 1 || ng < 1 || ngp < 0 || batch < 1 ||
        steps < 1 || schedule < 0 || schedule > 2) return -1;
    *done = 0;
    for (int step = 0; step < steps; ++step) {
        const double a = z[0], b = z[1];
        if (!isfinite(a) || !isfinite(b)) return SPONG_ADAM_BAD_STATE;
        double sum_a = 0.0, sum_b = 0.0;
        const double *row = samples + (size_t)step * (size_t)batch;
        for (int j = 0; j < batch; ++j) {
            const double x = row[j];
            const double bx = b * x;
            const double gv = polynomial(g, ng, bx);
            const double res = polynomial(f, nf, x) - a * gv;
            sum_a += res * gv;
            sum_b += res * a * polynomial(gp, ngp, bx) * x;
        }
        const double grad_a = -2.0 * (sum_a / (double)batch);
        const double grad_b = -2.0 * (sum_b / (double)batch);
        if (!isfinite(grad_a) || !isfinite(grad_b))
            return SPONG_ADAM_BAD_GRADIENT;
        const double gradient[2] = {grad_a, grad_b};
        const uint64_t next_t = *t + 1;
        const double lr = learning_rate(schedule, base_lr, next_t,
                                        warmup, horizon, final_fraction);
        double next_z[2];
        for (int k = 0; k < 2; ++k) {
            first[k] = b1 * first[k] + (1.0 - b1) * gradient[k];
            second[k] = b2 * second[k] +
                        (1.0 - b2) * gradient[k] * gradient[k];
            const double mh = first[k] / (1.0 - pow(b1, (double)next_t));
            const double vh = second[k] / (1.0 - pow(b2, (double)next_t));
            next_z[k] = (1.0 - lr * weight_decay) * z[k] -
                        lr * mh / (sqrt(vh) + eps);
        }
        *t = next_t;
        if (!isfinite(next_z[0]) || !isfinite(next_z[1]))
            return SPONG_ADAM_BAD_STEP;
        z[0] = next_z[0]; z[1] = next_z[1];
        points[2 * step] = next_z[0];
        points[2 * step + 1] = next_z[1];
        *done = step + 1;
    }
    return SPONG_ADAM_OK;
}
