/*
 * Constant-potential-rate segments.  See spong/spong_potential.h.
 *
 * Each of the three runners below mirrors its Python phase statement for
 * statement -- the same tests in the same order with the same floating
 * expressions -- because the Python loop is the executable specification
 * and the corpus test demands bit-identical vertices and counters.
 */

#include "spong/spong_potential.h"

#include <float.h>
#include <math.h>
#include <string.h>

/*
 * FLOATING-POINT CONTRACTION IS OFF THROUGHOUT THIS FILE.
 *
 * Every expression here mirrors a Python statement in charts.py, where no
 * fused multiply-add is possible; the arithmetic that MUST match the fused
 * extension build -- the Gauss steps and the loss/gradient/Hessian
 * evaluators -- is not in this file but called through spong_gauss2, which
 * the Python oracle calls too.  So, unlike spong_continue.c, there is no
 * mixing: the whole file is exact, and a single fused cap or Hermite
 * expression is enough to flip an accept/reject and fail the corpus
 * (measured on arm64: 65 of 140 segments).
 *
 * Three mechanisms, because no one of them is honoured by every compiler:
 * the ISO pragma, gcc's file-scope optimize pragma, and the block-scope
 * clang pragma at the top of every function (the one spong_continue.c
 * established as reliable on Apple clang).  Compiler flags are not relied
 * on -- application builds will not carry them.
 */
#if defined(__clang__)
#  pragma STDC FP_CONTRACT OFF
#  define SPONG_FP_EXACT   _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#  pragma GCC optimize ("fp-contract=off")   /* gcc ignores the ISO pragma */
#  define SPONG_FP_EXACT
#else
#  define SPONG_FP_EXACT
#endif

typedef struct {
    const spong_field *field;
    const spong_potential_request *req;
    double *points;
    size_t capacity;
    size_t n;                 /* logical vertex count (may exceed capacity) */
    double geometry_floor;
    int orders[2];
    int clock_started;
    int clock_done;
    int clock_failed;
    spong_potential_result *out;
} segment;

/* ------------------------------------------------------------------ */
/* helpers mirroring charts.py                                         */
/* ------------------------------------------------------------------ */

static double fmax3(double a, double b, double c) {
    SPONG_FP_EXACT
    return fmax(fmax(a, b), c);
}

static int in_box(const double box[4], const double p[2]) {
    SPONG_FP_EXACT
    return box[0] <= p[0] && p[0] <= box[1] && box[2] <= p[1] && p[1] <= box[3];
}

/* charts._append_resolved_point: replace the last vertex when binary64 does
 * not resolve a new point; otherwise append.  Writes only inside the
 * buffer; the logical count keeps growing so the caller learns the
 * capacity it needs. */
static void append_resolved(segment *s, const double q[2]) {
    SPONG_FP_EXACT
    if (s->n > 0 && s->n <= s->capacity) {
        const double *p = &s->points[2*(s->n-1)];
        double scale = 1.0 + fmax(fmax(fabs(p[0]), fabs(p[1])),
                                  fmax(fabs(q[0]), fabs(q[1])));
        double floor_ = fmax(64.0*DBL_EPSILON*scale, s->geometry_floor);
        if (hypot(q[0]-p[0], q[1]-p[1]) <= floor_) {
            s->points[2*(s->n-1)] = q[0];
            s->points[2*(s->n-1)+1] = q[1];
            return;
        }
    } else if (s->n > s->capacity) {
        /* Past capacity we cannot compare against the last vertex; count
         * conservatively as an append.  The retry recomputes from scratch
         * with enough room, so the count only needs to be an upper bound. */
    }
    if (s->n < s->capacity) {
        s->points[2*s->n] = q[0];
        s->points[2*s->n+1] = q[1];
    }
    s->n++;
}

static void append_raw(segment *s, const double q[2]) {
    SPONG_FP_EXACT
    if (s->n < s->capacity) {
        s->points[2*s->n] = q[0];
        s->points[2*s->n+1] = q[1];
    }
    s->n++;
}

static int segment_capture(double a0, double b0, double a1, double b1,
                           double at, double bt, double radius) {
    SPONG_FP_EXACT
    double da = a1-a0, db = b1-b0;
    double denom = da*da+db*db;
    if (denom == 0.0)
        return (a1-at)*(a1-at)+(b1-bt)*(b1-bt) < radius*radius;
    double t = ((at-a0)*da+(bt-b0)*db)/denom;
    t = fmin(1.0, fmax(0.0, t));
    double ea = a0+t*da-at, eb = b0+t*db-bt;
    return ea*ea+eb*eb < radius*radius;
}

/* charts._full_and_two_half on either field.  When clock != 0, the three
 * calls also return collocation-stage quadrature for the full step and the
 * accepted two-half composition. */
static int full_and_two_half_clock(const spong_field *f, int potential,
                                   const double z[2], double h, int order,
                                   double full[2], double mid[2], double end[2],
                                   int clock, double *tau_full,
                                   double *tau_half) {
    SPONG_FP_EXACT
    int (*step)(const spong_field *, const double[2], double, int, double[2]) =
        potential ? spong_potential_step : spong_normalized_step;
    int (*clock_step)(const spong_field *, const double[2], double, int,
                      double[2], double *) =
        potential ? spong_potential_step_clock : spong_normalized_step_clock;
    double tau0 = 0.0, tau1 = 0.0, tau2 = 0.0;
    int ok_full = clock ? clock_step(f, z, h, order, full, &tau0)
                        : step(f, z, h, order, full);
    int ok_mid = clock ? clock_step(f, z, 0.5*h, order, mid, &tau1)
                       : step(f, z, 0.5*h, order, mid);
    if (!ok_full) { full[0] = full[1] = NAN; }
    if (!ok_mid) { mid[0] = mid[1] = NAN; }
    end[0] = mid[0]; end[1] = mid[1];
    if (isfinite(mid[0]) && isfinite(mid[1])) {
        int ok_end = clock ? clock_step(f, mid, 0.5*h, order, end, &tau2)
                           : step(f, mid, 0.5*h, order, end);
        if (!ok_end) { end[0] = end[1] = NAN; }
    }
    if (clock) {
        *tau_full = tau0;
        *tau_half = tau1+tau2;
    }
    return isfinite(full[0]) && isfinite(full[1])
        && isfinite(mid[0]) && isfinite(mid[1])
        && isfinite(end[0]) && isfinite(end[1])
        && (!clock || (isfinite(*tau_full) && isfinite(*tau_half)));
}

static int full_and_two_half(const spong_field *f, int potential,
                             const double z[2], double h, int order,
                             double full[2], double mid[2], double end[2]) {
    SPONG_FP_EXACT
    return full_and_two_half_clock(f, potential, z, h, order,
                                   full, mid, end, 0, NULL, NULL);
}

static int clock_error(const segment *s, int order,
                       double tau_full, double tau_half,
                       double *error, double *ratio) {
    SPONG_FP_EXACT
    double divisor = (double)((1u << order)-1u);
    *error = fabs(tau_half-tau_full)/divisor;
    double tolerance = s->req->clock_atol
        + s->req->clock_rtol*fabs(tau_half);
    *ratio = *error/fmax(tolerance, DBL_MIN);
    return isfinite(*error) && isfinite(*ratio) && *error <= tolerance;
}

static void record_clock_attempt(segment *s, double error, double ratio,
                                 int accepted) {
    SPONG_FP_EXACT
    if (accepted) {
        if (error > s->out->max_clock_error) s->out->max_clock_error = error;
        if (ratio > s->out->max_clock_error_ratio)
            s->out->max_clock_error_ratio = ratio;
    } else {
        s->out->clock_rejected++;
    }
}

/* charts._step_arclength_cap with the gradient already at hand. */
static double step_arclength_cap(const segment *s, const double z[2],
                                 const double g[2], double ng) {
    SPONG_FP_EXACT
    double cap = INFINITY;
    if (s->req->critical != NULL && s->req->n_critical > 0) {
        double dmin = INFINITY;
        for (size_t i = 0; i < s->req->n_critical; i++) {
            double d = hypot(s->req->critical[2*i]-z[0],
                             s->req->critical[2*i+1]-z[1]);
            if (d < dmin) dmin = d;
        }
        cap = 0.25*dmin;
    }
    if (!(isfinite(ng) && ng > 0.0)) return cap;
    double t0 = g[0]/ng, t1 = g[1]/ng;
    double H[2][2];
    spong_field_hessian(s->field, z[0], z[1], H);
    double w0 = H[0][0]*t0 + H[0][1]*t1;
    double w1 = H[1][0]*t0 + H[1][1]*t1;
    double wt = w0*t0 + w1*t1;
    double p0 = w0-wt*t0, p1 = w1-wt*t1;
    double kappa = hypot(p0, p1)/ng;
    if (isfinite(kappa) && kappa > 0.0)
        cap = fmin(cap, s->req->critical_step_fraction/kappa);
    return cap;
}

/* charts._arclength_step: returns 1 and writes half/mid on success.  The
 * clock-aware form returns 2 when the trajectory tests pass but the clock
 * defect does not, allowing the caller to distinguish clock rejection from
 * an unavailable geometric step. */
static int arclength_step_clock(const segment *s, const double z[2],
                                double arclength, double sign, int clock,
                                double half[2], double mid[2],
                                double *tau, double *clock_err,
                                double *clock_ratio) {
    SPONG_FP_EXACT
    double L0 = spong_field_loss(s->field, z[0], z[1]);
    double noise = 64.0*DBL_EPSILON*(1.0+fabs(z[0])+fabs(z[1]));
    for (int k = 0; k < 2; k++) {
        int order = s->orders[k];
        double full[2], m[2], end[2];
        double tau_full = 0.0, tau_half = 0.0;
        if (!full_and_two_half_clock(
                s->field, 0, z, sign*arclength, order, full, m, end,
                clock, &tau_full, &tau_half))
            continue;
        double chord = hypot(end[0]-z[0], end[1]-z[1]);
        if (chord == 0.0) continue;
        double richardson = hypot(full[0]-end[0], full[1]-end[1]);
        if (richardson > fmax(1e-6*chord, noise)) continue;
        double L1 = spong_field_loss(s->field, end[0], end[1]);
        double slack = 64.0*DBL_EPSILON*(1.0+fabs(L0));
        if (!isfinite(L1) || sign*(L1-L0) < -slack) continue;
        if (clock) {
            if (!clock_error(s, order, tau_full, tau_half,
                             clock_err, clock_ratio))
                return 2;
            *tau = tau_half;
        }
        half[0] = end[0]; half[1] = end[1];
        mid[0] = m[0]; mid[1] = m[1];
        return 1;
    }
    return 0;
}

static int arclength_step(const segment *s, const double z[2],
                          double arclength, double sign,
                          double half[2], double mid[2]) {
    SPONG_FP_EXACT
    double tau, error, ratio;
    return arclength_step_clock(s, z, arclength, sign, 0, half, mid,
                                &tau, &error, &ratio) == 1;
}

/* charts._cubic_hermite, componentwise in the Python operation order. */
static void cubic_hermite(const double z0[2], const double z1[2],
                          const double f0[2], const double f1[2],
                          double h, double sv, double out[2]) {
    SPONG_FP_EXACT
    double s2 = sv*sv, s3 = sv*sv*sv;
    double c1 = 2*s3-3*s2+1, c2 = s3-2*s2+sv, c3 = -2*s3+3*s2, c4 = s3-s2;
    for (int d = 0; d < 2; d++) {
        double acc = c1*z0[d];
        acc += (c2*h)*f0[d];
        acc += c3*z1[d];
        acc += (c4*h)*f1[d];
        out[d] = acc;
    }
}

static void finish(segment *s, int term, const double z[2]) {
    SPONG_FP_EXACT
    s->out->term = term;
    s->out->a_end = z[0];
    s->out->b_end = z[1];
    s->out->n_points = s->n;
    if (s->req->clock) {
        s->out->clock_started = s->clock_started && !s->clock_failed;
        s->out->clock_available = s->clock_done && !s->clock_failed;
    }
    if (s->n > s->capacity) s->out->term = SPONG_POT_NEED_CAPACITY;
}

enum {
    CLOCK_BOUNDARY_NONE = 0,
    CLOCK_BOUNDARY_START = 1,
    CLOCK_BOUNDARY_STOP = 2
};

static double clock_level_tolerance(double level) {
    SPONG_FP_EXACT
    return 256.0*DBL_EPSILON*(1.0+fabs(level));
}

/* Shorten a loss-parameter step to the next clock section.  This happens
 * after curvature policy has chosen the actual h, so later cap logic cannot
 * accidentally step over the section. */
static int limit_clock_loss_step(const segment *s, double level, double *h) {
    SPONG_FP_EXACT
    if (!s->req->clock || s->clock_done || s->clock_failed || !(*h < 0.0))
        return CLOCK_BOUNDARY_NONE;
    double boundary;
    int kind;
    if (!s->clock_started) {
        boundary = s->req->clock_start_level;
        kind = CLOCK_BOUNDARY_START;
    } else {
        boundary = s->req->clock_stop_level;
        kind = CLOCK_BOUNDARY_STOP;
    }
    if (level > boundary && level+*h <= boundary) {
        *h = boundary-level;
        return kind;
    }
    return CLOCK_BOUNDARY_NONE;
}

/* The potential-rate field has loss itself as its independent variable, so
 * the helper above lands on a section directly.  The singular-field rescue
 * is parameterized by arclength instead.  Only when that rescue crosses a
 * clock section, solve the scalar loss event by bisection in arclength and
 * retain the stage clock of the same accepted two-half endpoint. */
static int prefix_arclength_step(segment *s, const double z[2],
                                 double arclength, double half[2],
                                 double mid[2], int *boundary_kind,
                                 double *tau, double *error, double *ratio) {
    SPONG_FP_EXACT
    int clock = s->req->clock && s->clock_started && !s->clock_done
        && !s->clock_failed;
    *boundary_kind = CLOCK_BOUNDARY_NONE;
    *tau = *error = *ratio = 0.0;
    int status = arclength_step_clock(s, z, arclength, -1.0, clock,
                                      half, mid, tau, error, ratio);
    if (status == 2) {
        record_clock_attempt(s, *error, *ratio, 0);
        return 0;
    }
    if (status != 1 || !s->req->clock || s->clock_done || s->clock_failed)
        return status == 1;

    double boundary = s->clock_started ? s->req->clock_stop_level
                                       : s->req->clock_start_level;
    int kind = s->clock_started ? CLOCK_BOUNDARY_STOP : CLOCK_BOUNDARY_START;
    double end_level = spong_field_loss(s->field, half[0], half[1]);
    if (!(end_level <= boundary)) return 1;

    double lo = 0.0, hi = arclength;
    double best_half[2] = {half[0], half[1]};
    double best_mid[2] = {mid[0], mid[1]};
    double best_tau = *tau, best_error = *error, best_ratio = *ratio;
    double level_tol = clock_level_tolerance(boundary);
    for (int it = 0; it < 60; it++) {
        double trial_length = 0.5*(lo+hi);
        double trial_half[2], trial_mid[2];
        double trial_tau = 0.0, trial_error = 0.0, trial_ratio = 0.0;
        int trial = arclength_step_clock(
            s, z, trial_length, -1.0, clock, trial_half, trial_mid,
            &trial_tau, &trial_error, &trial_ratio);
        if (trial != 1) {
            /* Smaller arclength is the only useful direction after either
             * a stage or a clock failure.  The original crossing remains a
             * valid upper endpoint until a shorter one succeeds. */
            hi = trial_length;
            continue;
        }
        double trial_level = spong_field_loss(
            s->field, trial_half[0], trial_half[1]);
        if (trial_level <= boundary) {
            hi = trial_length;
            best_half[0] = trial_half[0]; best_half[1] = trial_half[1];
            best_mid[0] = trial_mid[0]; best_mid[1] = trial_mid[1];
            best_tau = trial_tau;
            best_error = trial_error;
            best_ratio = trial_ratio;
            if (fabs(trial_level-boundary) <= level_tol) break;
        } else {
            lo = trial_length;
        }
    }
    half[0] = best_half[0]; half[1] = best_half[1];
    mid[0] = best_mid[0]; mid[1] = best_mid[1];
    *tau = best_tau; *error = best_error; *ratio = best_ratio;
    *boundary_kind = kind;
    return 1;
}

/* ------------------------------------------------------------------ */
/* prefix: descent to one target minimum                               */
/* ------------------------------------------------------------------ */

static void run_prefix(segment *s) {
    SPONG_FP_EXACT
    const spong_potential_request *r = s->req;
    spong_potential_result *o = s->out;
    double z[2] = {r->a0, r->b0};
    append_raw(s, z);
    if (r->n_targets < 1) { finish(s, SPONG_POT_UNAVAILABLE, z); return; }
    double at = r->targets[0], bt = r->targets[1];
    double target_level = spong_field_loss(s->field, at, bt);
    double start_level = spong_field_loss(s->field, r->a0, r->b0);
    double gap0 = start_level-target_level;
    if (!(isfinite(gap0) && gap0 > 0.0)) {
        finish(s, SPONG_POT_UNAVAILABLE, z); return;
    }
    if (r->clock) {
        double start_tol = clock_level_tolerance(r->clock_start_level);
        if (!(isfinite(r->clock_start_level)
                && isfinite(r->clock_stop_level)
                && r->clock_start_level > r->clock_stop_level
                && r->clock_rtol >= 0.0 && isfinite(r->clock_rtol)
                && r->clock_atol >= 0.0 && isfinite(r->clock_atol)
                && (r->clock_rtol > 0.0 || r->clock_atol > 0.0))) {
            s->clock_failed = 1;
        } else if (fabs(start_level-r->clock_start_level) <= start_tol) {
            s->clock_started = 1;
        } else if (start_level < r->clock_start_level) {
            /* The prefix was launched below the requested regular section;
             * no silently truncated clock.  The trajectory still runs. */
            s->clock_failed = 1;
        }
    }
    size_t n_levels = r->n_levels > 0 ? r->n_levels : 1;
    double base = gap0/(double)n_levels;
    double near_gap = base/16384.0;
    double cur_level_step = base;
    o->level_step = base;
    int term = -1;
    uint64_t iteration = 0;
    double previous[2];
    for (;;) {
        if (!(iteration < (uint64_t)n_levels+1024+o->critical_capped+o->arclength_steps
              && iteration < 4*((uint64_t)n_levels+1024))) {
            term = SPONG_POT_BUDGET; break;
        }
        iteration++;
        double level = spong_field_loss(s->field, z[0], z[1]);
        double gap = level-target_level;
        if (gap <= near_gap) { term = SPONG_POT_NEAR_TARGET; break; }
        double h = -fmin(cur_level_step, 0.2*gap);
        double g[2];
        spong_field_gradient(s->field, z[0], z[1], g);
        double ng = hypot(g[0], g[1]);
        double arc_cap = step_arclength_cap(s, z, g, ng);
        double cap = arc_cap*ng;
        double loss_floor = 4096*DBL_EPSILON*(1.0+fabs(level));
        if (isfinite(cap) && cap > 0.0 && fabs(h) > cap) {
            o->critical_capped++;
            if (cap < loss_floor) {
                double half[2], mid[2];
                int clock_boundary = CLOCK_BOUNDARY_NONE;
                int timed = r->clock && s->clock_started && !s->clock_done
                    && !s->clock_failed;
                double tau_step = 0.0, clock_err = 0.0, clock_ratio = 0.0;
                if (prefix_arclength_step(
                        s, z, arc_cap, half, mid, &clock_boundary,
                        &tau_step, &clock_err, &clock_ratio)) {
                    o->arclength_steps++;
                    previous[0] = z[0]; previous[1] = z[1];
                    z[0] = half[0]; z[1] = half[1];
                    o->accepted++;
                    if (timed) {
                        record_clock_attempt(s, clock_err, clock_ratio, 1);
                        o->tau += tau_step;
                        o->clock_steps++;
                    }
                    if (clock_boundary == CLOCK_BOUNDARY_START)
                        s->clock_started = 1;
                    else if (clock_boundary == CLOCK_BOUNDARY_STOP)
                        s->clock_done = 1;
                    int exited = 0;
                    const double *samples[2] = {mid, z};
                    double zc[2] = {z[0], z[1]};
                    samples[1] = zc;
                    for (int k = 0; k < 2 && !exited; k++) {
                        const double *sample = samples[k];
                        if (!in_box(r->box, sample)) {
                            append_resolved(s, sample);
                            z[0] = sample[0]; z[1] = sample[1];
                            term = SPONG_POT_BOX_EXIT; exited = 1; break;
                        }
                        if (segment_capture(previous[0], previous[1],
                                            sample[0], sample[1],
                                            at, bt, r->cap_r)) {
                            double tz[2] = {at, bt};
                            append_resolved(s, sample);
                            append_resolved(s, tz);
                            z[0] = at; z[1] = bt;
                            term = SPONG_POT_CAPTURE; exited = 1; break;
                        }
                        append_resolved(s, sample);
                        previous[0] = sample[0]; previous[1] = sample[1];
                    }
                    if (exited) break;
                    continue;
                }
            }
            h = -fmax(cap, loss_floor);
        }
        int requested_boundary = limit_clock_loss_step(s, level, &h);
        double boundary_h = h;
        int have = 0;
        double zn[2] = {0, 0}, accepted_mid[2] = {0, 0};
        int accepted_boundary = CLOCK_BOUNDARY_NONE;
        int timed = r->clock && s->clock_started && !s->clock_done
            && !s->clock_failed;
        double accepted_tau = 0.0, accepted_clock_error = 0.0;
        double accepted_clock_ratio = 0.0;
        for (int retry = 0; retry < 12 && !have; retry++) {
            for (int k = 0; k < 2; k++) {
                int order = s->orders[k];
                if (order == 8) o->gl8_attempted++;
                double full[2], mid[2], half[2];
                double tau_full = 0.0, tau_half = 0.0;
                if (!full_and_two_half_clock(
                        s->field, 1, z, h, order, full, mid, half,
                        timed, &tau_full, &tau_half))
                    continue;
                double chord = hypot(half[0]-z[0], half[1]-z[1]);
                double richardson = hypot(full[0]-half[0], full[1]-half[1]);
                double new_level = spong_field_loss(s->field, half[0], half[1]);
                double loss_error = fabs((new_level-level)-h);
                if (new_level >= level
                        || richardson > 1e-6*fmax(chord, 1e-8)
                        || loss_error > 2e-5*fmax(fabs(h), 1e-12))
                    continue;
                double clock_err = 0.0, clock_ratio = 0.0;
                if (timed && !clock_error(s, order, tau_full, tau_half,
                                           &clock_err, &clock_ratio)) {
                    record_clock_attempt(s, clock_err, clock_ratio, 0);
                    continue;
                }
                zn[0] = half[0]; zn[1] = half[1];
                accepted_mid[0] = mid[0]; accepted_mid[1] = mid[1];
                if (r->clock && !s->clock_failed) {
                    if (requested_boundary != CLOCK_BOUNDARY_NONE
                            && h == boundary_h)
                        accepted_boundary = requested_boundary;
                    else if (!s->clock_started
                            && new_level <= r->clock_start_level)
                        accepted_boundary = CLOCK_BOUNDARY_START;
                    else if (s->clock_started && !s->clock_done
                             && new_level <= r->clock_stop_level)
                        accepted_boundary = CLOCK_BOUNDARY_STOP;
                }
                accepted_tau = tau_half;
                accepted_clock_error = clock_err;
                accepted_clock_ratio = clock_ratio;
                if (richardson > o->max_richardson)
                    o->max_richardson = richardson;
                if (order == 8) o->gl8_accepted++;
                have = 1;
                break;
            }
            if (!have) { h *= 0.5; o->rejected++; }
        }
        if (!have) { term = SPONG_POT_STEP_FAILURE; break; }
        previous[0] = z[0]; previous[1] = z[1];
        z[0] = zn[0]; z[1] = zn[1];
        o->accepted++;
        if (timed) {
            record_clock_attempt(s, accepted_clock_error,
                                 accepted_clock_ratio, 1);
            o->tau += accepted_tau;
            o->clock_steps++;
        }
        if (accepted_boundary == CLOCK_BOUNDARY_START)
            s->clock_started = 1;
        else if (accepted_boundary == CLOCK_BOUNDARY_STOP)
            s->clock_done = 1;
        cur_level_step = fmin(base, 1.5*fabs(h));
        {
            double zc[2] = {z[0], z[1]};
            const double *samples[2] = {accepted_mid, zc};
            for (int k = 0; k < 2; k++) {
                const double *sample = samples[k];
                if (!in_box(r->box, sample)) {
                    append_resolved(s, sample);
                    z[0] = sample[0]; z[1] = sample[1];
                    term = SPONG_POT_BOX_EXIT; break;
                }
                if (segment_capture(previous[0], previous[1],
                                    sample[0], sample[1], at, bt, r->cap_r)) {
                    double tz[2] = {at, bt};
                    append_resolved(s, sample);
                    append_resolved(s, tz);
                    z[0] = at; z[1] = bt;
                    term = SPONG_POT_CAPTURE; break;
                }
                append_resolved(s, sample);
                previous[0] = sample[0]; previous[1] = sample[1];
            }
        }
        if (term == SPONG_POT_BOX_EXIT || term == SPONG_POT_CAPTURE) break;
    }
    finish(s, term, z);
}

/* ------------------------------------------------------------------ */
/* level event: descent to the next candidate minimum level            */
/* ------------------------------------------------------------------ */

static void run_level_event(segment *s) {
    SPONG_FP_EXACT
    const spong_potential_request *r = s->req;
    spong_potential_result *o = s->out;
    double z[2] = {r->a0, r->b0};
    append_raw(s, z);
    double level0 = spong_field_loss(s->field, z[0], z[1]);
    double slack0 = 1024*DBL_EPSILON*(1.0+fabs(level0));
    double event_level = -INFINITY;
    int any = 0;
    for (size_t i = 0; i < r->n_targets; i++) {
        double v = spong_field_loss(s->field, r->targets[2*i], r->targets[2*i+1]);
        if (v < level0-slack0) {
            if (!any || v > event_level) event_level = v;
            any = 1;
        }
    }
    if (!any) { finish(s, SPONG_POT_UNAVAILABLE, z); return; }
    o->event_level = event_level;
    double gap0 = level0-event_level;
    size_t n_levels = r->n_levels > 0 ? r->n_levels : 1;
    double base = gap0/(double)n_levels;
    double crossing_floor = fmax(base/1024.0,
                                 4096*DBL_EPSILON*(1.0+fabs(event_level)));
    double cur_level_step = base;
    int term = SPONG_POT_BUDGET;
    uint64_t iteration = 0;
    double previous[2];
    while (iteration < 4*(uint64_t)n_levels+o->critical_capped+o->arclength_steps
           && iteration < 16*(uint64_t)n_levels) {
        iteration++;
        double level = spong_field_loss(s->field, z[0], z[1]);
        double gap = level-event_level;
        if (gap <= -crossing_floor) { term = SPONG_POT_LEVEL_EVENT_HIT; break; }
        double requested = (gap <= crossing_floor)
            ? fmax(2.0*gap, crossing_floor)
            : fmin(cur_level_step, 0.5*gap);
        double g[2];
        spong_field_gradient(s->field, z[0], z[1], g);
        double ng = hypot(g[0], g[1]);
        double arc_cap = step_arclength_cap(s, z, g, ng);
        double cap = arc_cap*ng;
        if (isfinite(cap) && cap > 0.0 && requested > cap
                && gap > crossing_floor) {
            o->critical_capped++;
            if (cap < crossing_floor) {
                double half[2], mid[2];
                if (arclength_step(s, z, arc_cap, -1.0, half, mid)) {
                    o->arclength_steps++;
                    previous[0] = z[0]; previous[1] = z[1];
                    z[0] = half[0]; z[1] = half[1];
                    o->accepted++;
                    append_resolved(s, mid);
                    append_resolved(s, z);
                    if (!in_box(r->box, z)) { term = SPONG_POT_BOX_EXIT; break; }
                    for (size_t i = 0; i < r->n_targets; i++) {
                        double at = r->targets[2*i], bt = r->targets[2*i+1];
                        if (segment_capture(previous[0], previous[1],
                                            z[0], z[1], at, bt, r->cap_r)) {
                            double tz[2] = {at, bt};
                            append_raw(s, tz);
                            o->captured = 1;
                            o->captured_a = at; o->captured_b = bt;
                            term = SPONG_POT_CAPTURE;
                            break;
                        }
                    }
                    if (o->captured) break;
                    continue;
                }
            }
            requested = fmax(cap, crossing_floor);
        }
        double h = -requested;
        int have = 0;
        double zn[2] = {0, 0}, accepted_mid[2] = {0, 0};
        for (int retry = 0; retry < 14 && !have; retry++) {
            for (int k = 0; k < 2; k++) {
                int order = s->orders[k];
                if (order == 8) o->gl8_attempted++;
                double full[2], mid[2], half[2];
                if (!full_and_two_half(s->field, 1, z, h, order,
                                       full, mid, half))
                    continue;
                double chord = hypot(half[0]-z[0], half[1]-z[1]);
                double richardson = hypot(full[0]-half[0], full[1]-half[1]);
                double new_level = spong_field_loss(s->field, half[0], half[1]);
                double loss_error = fabs((new_level-level)-h);
                if (new_level >= level
                        || richardson > 1e-6*fmax(chord, 1e-8)
                        || loss_error > 2e-5*fmax(fabs(h), 1e-12))
                    continue;
                zn[0] = half[0]; zn[1] = half[1];
                accepted_mid[0] = mid[0]; accepted_mid[1] = mid[1];
                if (richardson > o->max_richardson)
                    o->max_richardson = richardson;
                if (order == 8) o->gl8_accepted++;
                have = 1;
                break;
            }
            if (!have) { h *= 0.5; o->rejected++; }
        }
        if (!have) { term = SPONG_POT_STEP_FAILURE; break; }
        previous[0] = z[0]; previous[1] = z[1];
        z[0] = zn[0]; z[1] = zn[1];
        o->accepted++;
        cur_level_step = fmin(base, 1.5*fabs(h));
        append_resolved(s, accepted_mid);
        append_resolved(s, z);
        if (!in_box(r->box, z)) { term = SPONG_POT_BOX_EXIT; break; }
        for (size_t i = 0; i < r->n_targets; i++) {
            double at = r->targets[2*i], bt = r->targets[2*i+1];
            if (segment_capture(previous[0], previous[1], z[0], z[1],
                                at, bt, r->cap_r)) {
                double tz[2] = {at, bt};
                append_raw(s, tz);
                o->captured = 1;
                o->captured_a = at; o->captured_b = bt;
                term = SPONG_POT_CAPTURE;
                break;
            }
        }
        if (o->captured) break;
    }
    finish(s, term, z);
}

/* ------------------------------------------------------------------ */
/* ascent: stable branch outward to the box boundary                   */
/* ------------------------------------------------------------------ */

static void run_ascent(segment *s) {
    SPONG_FP_EXACT
    const spong_potential_request *r = s->req;
    spong_potential_result *o = s->out;
    double z[2] = {r->a0, r->b0};
    append_raw(s, z);
    double geometric_ds = 4.0*r->ds;
    double last_arc = -1.0;              /* < 0: none yet */
    int term = SPONG_POT_BUDGET;
    uint64_t iteration = 0;
    size_t max_steps = r->max_steps;
    while (iteration < (uint64_t)max_steps+o->critical_capped+o->arclength_steps
           && iteration < 4*(uint64_t)max_steps) {
        iteration++;
        double level = spong_field_loss(s->field, z[0], z[1]);
        double g[2];
        spong_field_gradient(s->field, z[0], z[1], g);
        double ng = hypot(g[0], g[1]);
        if (!(isfinite(level) && isfinite(ng) && ng > 0.0)) {
            term = SPONG_POT_UNRESOLVED_FIELD; break;
        }
        if (level >= r->stop_level) {
            /* The level bar: above it there are no critical points and no
             * possible connections, so the certificate is already decided
             * (one exact comparison at the audit).  Everything the census
             * measured beyond this line -- 400k-vertex crawls, ridge
             * stalls, overflow at |b| ~ 1e6 -- was tracing with nothing
             * left to certify. */
            term = SPONG_POT_LEVEL_STOP; break;
        }
        double nominal_arc = 16.0*geometric_ds;
        if (last_arc >= 0.0) nominal_arc = fmin(nominal_arc, 1.5*last_arc);
        double h = fmax(nominal_arc*ng, 4096*DBL_EPSILON*(1.0+fabs(level)));
        double arc_cap = step_arclength_cap(s, z, g, ng);
        double cap = arc_cap*ng;
        double loss_floor = 4096*DBL_EPSILON*(1.0+fabs(level));
        if (isfinite(cap) && cap > 0.0 && h > cap) {
            o->critical_capped++;
            if (cap < loss_floor) {
                double half[2], mid[2];
                if (arclength_step(s, z, arc_cap, +1.0, half, mid)) {
                    o->arclength_steps++;
                    last_arc = hypot(half[0]-z[0], half[1]-z[1]);
                    z[0] = half[0]; z[1] = half[1];
                    o->accepted++;
                    int exited = 0;
                    double zc[2] = {z[0], z[1]};
                    const double *samples[2] = {mid, zc};
                    for (int k = 0; k < 2; k++) {
                        const double *sample = samples[k];
                        append_resolved(s, sample);
                        if (!in_box(r->box, sample)) {
                            z[0] = sample[0]; z[1] = sample[1];
                            term = SPONG_POT_BOX_EXIT; exited = 1; break;
                        }
                    }
                    if (exited) break;
                    continue;
                }
            }
            h = fmax(cap, loss_floor);
        }
        int have = 0;
        double zn[2] = {0, 0};
        double f0[2] = {0, 0}, f1[2] = {0, 0};
        double h_used = 0.0, chord_used = 0.0, interp_used = 0.0;
        for (int retry = 0; retry < 14 && !have; retry++) {
            for (int k = 0; k < 2; k++) {
                int order = s->orders[k];
                if (order == 8) o->gl8_attempted++;
                double full[2], mid[2], half[2];
                if (!full_and_two_half(s->field, 1, z, h, order,
                                       full, mid, half))
                    continue;
                double chord = hypot(half[0]-z[0], half[1]-z[1]);
                double richardson = hypot(full[0]-half[0], full[1]-half[1]);
                double new_level = spong_field_loss(s->field, half[0], half[1]);
                double loss_error = fabs((new_level-level)-h);
                double g1[2];
                spong_field_gradient(s->field, half[0], half[1], g1);
                double q1 = g1[0]*g1[0]+g1[1]*g1[1];
                if (!(q1 > 0.0 && isfinite(q1))) continue;
                double fa[2] = {g[0]/(ng*ng), g[1]/(ng*ng)};
                double fb[2] = {g1[0]/q1, g1[1]/q1};
                double hermite_mid[2];
                cubic_hermite(z, half, fa, fb, h, 0.5, hermite_mid);
                double interpolation_error = hypot(hermite_mid[0]-mid[0],
                                                   hermite_mid[1]-mid[1]);
                double curve_tol = 2e-6*fmax3(chord, geometric_ds, 1e-8);
                if (new_level <= level
                        || richardson > 1e-6*fmax(chord, 1e-8)
                        || interpolation_error > curve_tol
                        || loss_error > 2e-5*fmax(fabs(h), 1e-12))
                    continue;
                zn[0] = half[0]; zn[1] = half[1];
                f0[0] = fa[0]; f0[1] = fa[1];
                f1[0] = fb[0]; f1[1] = fb[1];
                h_used = h; chord_used = chord; interp_used = interpolation_error;
                if (richardson > o->max_richardson)
                    o->max_richardson = richardson;
                if (order == 8) o->gl8_accepted++;
                have = 1;
                break;
            }
            if (!have) { h *= 0.5; o->rejected++; }
        }
        if (!have) { term = SPONG_POT_STEP_FAILURE; break; }
        double z0[2] = {z[0], z[1]};
        z[0] = zn[0]; z[1] = zn[1];
        last_arc = chord_used;
        if (interp_used > o->max_interpolation_error)
            o->max_interpolation_error = interp_used;
        double subdiv_d = ceil(chord_used/geometric_ds);
        size_t subdivisions = subdiv_d < 1.0 ? 1 : (size_t)subdiv_d;
        int exited = 0;
        for (size_t j = 1; j <= subdivisions; j++) {
            double p[2];
            cubic_hermite(z0, z, f0, f1, h_used, (double)j/(double)subdivisions, p);
            append_resolved(s, p);
            if (!in_box(r->box, p)) {
                z[0] = p[0]; z[1] = p[1];
                term = SPONG_POT_BOX_EXIT; exited = 1; break;
            }
        }
        o->accepted++;
        if (exited) break;
    }
    finish(s, term, z);
}

/* ------------------------------------------------------------------ */

int spong_potential_rate_segment(
    const spong_field *field,
    const spong_potential_request *request,
    double *points, size_t point_capacity,
    spong_potential_result *result)
{
    SPONG_FP_EXACT
    if (field == NULL || request == NULL || result == NULL) return -1;
    memset(result, 0, sizeof(*result));
    result->max_richardson = 0.0;
    result->max_interpolation_error = 0.0;
    result->event_level = NAN;
    result->level_step = NAN;
    result->captured_a = result->captured_b = NAN;
    segment s;
    s.field = field;
    s.req = request;
    s.points = points;
    s.capacity = points == NULL ? 0 : point_capacity;
    s.n = 0;
    s.clock_started = 0;
    s.clock_done = 0;
    s.clock_failed = 0;
    double box_scale = 0.0;
    for (int i = 0; i < 4; i++)
        if (fabs(request->box[i]) > box_scale) box_scale = fabs(request->box[i]);
    s.geometry_floor = 128.0*DBL_EPSILON*(1.0+box_scale);
    if (request->clock && request->mode != SPONG_POT_PREFIX) return -1;
    /* Order schedule: primary first, then the fallback.  4 -> (4, 6) is the
     * point of the table -- until it existed, GL4 was unreachable here, since
     * every non-8 primary collapsed to (6, 8) and "GL4" silently ran GL6.
     * GL8 is kept for the saturation check rather than as the authority: on
     * the nonnearest wall it accepts 10872 of 10874 steps (so it is not
     * falling back) yet places the wall 3.0e-12 from where 6-first places
     * it, with nothing to say it is the better answer.
     *
     * PURE MODE, for convergence studies: a NEGATIVE primary_order requests
     * both slots at |primary_order|, so a rejected step is retried at the
     * same order and then the step is halved -- it cannot switch method.
     * The fallback schedules are a state-dependent hybrid: a "GL4-first"
     * run may contain an unknown number of GL6 steps, and precisely near the
     * difficult saddle passage, which makes any h^4-vs-h^6 comparison
     * ambiguous.  Purity here is STRUCTURAL rather than measured, which is
     * why no per-order counter is needed to establish it.  Production keeps
     * the fallback schedules; those are robustness, not convergence. */
    int po = request->primary_order;
    if (po < 0) { s.orders[0] = -po; s.orders[1] = -po; }
    else if (po == 8) { s.orders[0] = 8; s.orders[1] = 6; }
    else if (po == 4) { s.orders[0] = 4; s.orders[1] = 6; }
    else              { s.orders[0] = 6; s.orders[1] = 8; }
    s.out = result;
    switch (request->mode) {
    case SPONG_POT_PREFIX:      run_prefix(&s); break;
    case SPONG_POT_LEVEL_EVENT: run_level_event(&s); break;
    case SPONG_POT_ASCENT:      run_ascent(&s); break;
    default:                    return -1;
    }
    return result->term;
}
