/*
 * Centered raw arrival.  See spong/spong_arrival.h.
 *
 * Mirrors charts._centered_raw_arrival_python statement for statement --
 * the same tests in the same order with the same floating expressions --
 * because the Python loop is the executable specification and the corpus
 * test demands bit-identical vertices and counters.
 */

#include "spong/spong_arrival.h"

#include <float.h>
#include <math.h>
#include <string.h>

/*
 * FLOATING-POINT CONTRACTION IS OFF THROUGHOUT THIS FILE, as in
 * spong_potential.c and for the same reason: every expression here mirrors
 * a Python statement, and the fused arithmetic -- the jet polynomial and
 * the Gauss steps -- is called through spong_jet, which the oracle calls
 * too.  A single fused expression flips an accept/reject.
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
    const spong_jet *jet;
    const spong_arrival_request *req;
    double *points;
    size_t capacity;
    size_t n;
    double geometry_floor;
    int orders[2];
    spong_arrival_result *out;
} arrival;

static double fmax3(double a, double b, double c) {
    SPONG_FP_EXACT
    return fmax(fmax(a, b), c);
}

/* charts._append_resolved_point (see spong_potential.c for the capacity
 * semantics: past capacity the logical count keeps growing). */
static void append_resolved(arrival *s, const double q[2]) {
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
    }
    if (s->n < s->capacity) {
        s->points[2*s->n] = q[0];
        s->points[2*s->n+1] = q[1];
    }
    s->n++;
}

static void append_raw(arrival *s, const double q[2]) {
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

/* charts._full_and_two_half on the raw centered field. */
static int full_and_two_half(const spong_jet *jet, const double z[2],
                             double h, int order,
                             double full[2], double mid[2], double end[2]) {
    SPONG_FP_EXACT
    int ok_full = spong_jet_raw_step(jet, z, h, order, full);
    int ok_mid = spong_jet_raw_step(jet, z, 0.5*h, order, mid);
    if (!ok_full) { full[0] = full[1] = NAN; }
    if (!ok_mid) { mid[0] = mid[1] = NAN; }
    end[0] = mid[0]; end[1] = mid[1];
    if (isfinite(mid[0]) && isfinite(mid[1])) {
        if (!spong_jet_raw_step(jet, mid, 0.5*h, order, end)) {
            end[0] = end[1] = NAN;
        }
    }
    return isfinite(full[0]) && isfinite(full[1])
        && isfinite(mid[0]) && isfinite(mid[1])
        && isfinite(end[0]) && isfinite(end[1]);
}

/* One raw-flow trial used only to locate a clock section after the ordinary
 * acceptance path has established that its proposed step crosses it.  The
 * independent variable here already IS unnormalized gradient-flow time, so
 * the elapsed clock is exactly dt; only the loss event needs localization. */
static int clock_trial(const arrival *s, const double z[2], double dt,
                       int order,
                       double value, double finish_r, double turn_reject,
                       int have_last, const double last_direction[2],
                       double half[2], double mid[2], double *richardson,
                       double *next_value) {
    SPONG_FP_EXACT
    double full[2];
    if (!full_and_two_half(s->jet, z, -dt, order, full, mid, half))
        return 0;
    double chord = hypot(half[0]-z[0], half[1]-z[1]);
    *richardson = hypot(full[0]-half[0], full[1]-half[1]);
    *next_value = spong_jet_potential(s->jet, half[0], half[1]);
    double tolerance = 2e-7*fmax3(chord, 0.05*finish_r, 1e-13);
    if (!isfinite(*next_value) || *next_value >= value
            || *richardson > tolerance)
        return 0;
    if (have_last && chord > 0.0) {
        double d0 = half[0]-z[0], d1 = half[1]-z[1];
        double cosine = (d0*last_direction[0]+d1*last_direction[1])
            /(chord*hypot(last_direction[0], last_direction[1]));
        if (cosine < turn_reject) return 0;
    }
    return 1;
}

static void finish(arrival *s, int term, const double physical[2]) {
    SPONG_FP_EXACT
    s->out->term = term;
    s->out->a_end = physical[0];
    s->out->b_end = physical[1];
    s->out->n_points = s->n;
    if (s->n > s->capacity) s->out->term = SPONG_ARR_NEED_CAPACITY;
}

static void run(arrival *s) {
    SPONG_FP_EXACT
    const spong_arrival_request *r = s->req;
    spong_arrival_result *o = s->out;
    const spong_jet *jet = s->jet;
    double at = r->at, bt = r->bt;
    double ca = r->center_a, cb = r->center_b;
    double start[2] = {r->a0, r->b0};
    append_raw(s, start);

    double finish_r = fmax(r->cap_r/64.0,
                           4096*DBL_EPSILON*(1.0+hypot(at, bt)));
    o->finish_radius = finish_r;
    double z[2] = {r->a0-ca, r->b0-cb};
    if (!(isfinite(r->slow) && isfinite(r->fast) && r->slow > 0.0)) {
        finish(s, SPONG_ARR_UNAVAILABLE, start); return;
    }
    double slow = r->slow, fast = r->fast;
    o->spectral_ratio = fast/slow;
    double dt = 0.25/fast;
    double dt_cap = 4.0/slow;
    double turn_reject = r->turn_reject;
    int have_last = 0;
    double last_direction[2] = {0.0, 0.0};
    int term = SPONG_ARR_BUDGET;
    double physical[2] = {r->a0, r->b0};
    int clock_started = r->clock && r->clock_started;
    int clock_done = 0, clock_failed = 0;
    if (r->clock && !(isfinite(r->clock_start_value)
            && isfinite(r->clock_stop_value)
            && r->clock_start_value > r->clock_stop_value))
        clock_failed = 1;
    for (size_t iteration = 0; iteration < r->max_steps; iteration++) {
        physical[0] = z[0]+ca; physical[1] = z[1]+cb;
        if (hypot(physical[0]-at, physical[1]-bt) < finish_r) {
            double tz[2] = {at, bt};
            append_raw(s, tz);
            physical[0] = at; physical[1] = bt;
            term = SPONG_ARR_CAPTURE; break;
        }
        double value = spong_jet_potential(jet, z[0], z[1]);
        if (!(isfinite(value) && value > 0.0)) {
            term = SPONG_ARR_INVALID_POTENTIAL; break;
        }
        if (r->clock && !clock_failed && !clock_started) {
            double tol = 256.0*DBL_EPSILON*(1.0+fabs(r->clock_start_value));
            if (fabs(value-r->clock_start_value) <= tol)
                clock_started = 1;
            else if (value < r->clock_start_value)
                clock_failed = 1;
        }
        if (r->clock && !clock_failed && clock_started && !clock_done) {
            double tol = 256.0*DBL_EPSILON*(1.0+fabs(r->clock_stop_value));
            if (fabs(value-r->clock_stop_value) <= tol)
                clock_done = 1;
            else if (value < r->clock_stop_value)
                clock_failed = 1;
        }
        int have = 0;
        double zn[2] = {0, 0}, accepted_mid[2] = {0, 0};
        double trial_dt = dt;
        int accepted_boundary = 0;  /* 1 start, 2 stop */
        for (int retry = 0; retry < 16 && !have; retry++) {
            double h = -trial_dt;
            for (int k = 0; k < 2; k++) {
                int order = s->orders[k];
                if (order == 8) o->gl8_attempted++;
                double full[2], mid[2], half[2];
                if (!full_and_two_half(jet, z, h, order, full, mid, half))
                    continue;
                double chord = hypot(half[0]-z[0], half[1]-z[1]);
                double richardson = hypot(full[0]-half[0], full[1]-half[1]);
                double next_value = spong_jet_potential(jet, half[0], half[1]);
                double tolerance = 2e-7*fmax3(chord, 0.05*finish_r, 1e-13);
                if (!isfinite(next_value) || next_value >= value
                        || richardson > tolerance)
                    continue;
                double d0 = half[0]-z[0], d1 = half[1]-z[1];
                if (have_last && chord > 0.0) {
                    double cosine = (d0*last_direction[0]+d1*last_direction[1])
                        /(chord*hypot(last_direction[0], last_direction[1]));
                    if (cosine < turn_reject) {
                        o->turn_rejected++;
                        continue;
                    }
                }
                int boundary = 0;
                double boundary_value = NAN;
                if (r->clock && !clock_failed && !clock_done) {
                    boundary = clock_started ? 2 : 1;
                    boundary_value = clock_started ? r->clock_stop_value
                                                   : r->clock_start_value;
                }
                if (boundary != 0 && next_value <= boundary_value) {
                    /* The accepted raw-time step crosses the next loss
                     * section.  Bisect in raw time with the same pure order
                     * and the same trajectory acceptance tests. */
                    double lo = 0.0, hi = trial_dt, best_dt = trial_dt;
                    double best_half[2] = {half[0], half[1]};
                    double best_mid[2] = {mid[0], mid[1]};
                    double best_richardson = richardson;
                    double best_value = next_value;
                    double level_tol = 256.0*DBL_EPSILON
                        *(1.0+fabs(boundary_value));
                    for (int event_it = 0; event_it < 60; event_it++) {
                        double event_dt = 0.5*(lo+hi);
                        double event_half[2], event_mid[2];
                        double event_richardson, event_value;
                        if (!clock_trial(s, z, event_dt, order, value, finish_r,
                                         turn_reject, have_last,
                                         last_direction, event_half, event_mid,
                                         &event_richardson, &event_value)) {
                            hi = event_dt;
                            continue;
                        }
                        if (event_value <= boundary_value) {
                            hi = event_dt;
                            best_dt = event_dt;
                            best_half[0] = event_half[0];
                            best_half[1] = event_half[1];
                            best_mid[0] = event_mid[0];
                            best_mid[1] = event_mid[1];
                            best_richardson = event_richardson;
                            best_value = event_value;
                            if (fabs(event_value-boundary_value) <= level_tol)
                                break;
                        } else {
                            lo = event_dt;
                        }
                    }
                    trial_dt = best_dt;
                    half[0] = best_half[0]; half[1] = best_half[1];
                    mid[0] = best_mid[0]; mid[1] = best_mid[1];
                    richardson = best_richardson;
                    next_value = best_value;
                    accepted_boundary = boundary;
                }
                zn[0] = half[0]; zn[1] = half[1];
                accepted_mid[0] = mid[0]; accepted_mid[1] = mid[1];
                if (richardson > o->max_richardson)
                    o->max_richardson = richardson;
                if (order == 8) o->gl8_accepted++;
                have = 1;
                break;
            }
            if (!have) { trial_dt *= 0.5; o->rejected++; }
        }
        if (!have) { term = SPONG_ARR_STEP_FAILURE; break; }
        double previous[2] = {z[0], z[1]};
        z[0] = zn[0]; z[1] = zn[1];
        last_direction[0] = z[0]-previous[0];
        last_direction[1] = z[1]-previous[1];
        have_last = 1;
        o->accepted++;
        if (r->clock && !clock_failed && clock_started && !clock_done) {
            o->tau += trial_dt;
            o->clock_steps++;
        }
        if (accepted_boundary == 1) clock_started = 1;
        else if (accepted_boundary == 2) clock_done = 1;
        dt = fmin(dt_cap, 1.5*trial_dt);
        double p0[2] = {previous[0]+ca, previous[1]+cb};
        int captured = 0;
        for (int k = 0; k < 2 && !captured; k++) {
            const double *sample = k == 0 ? accepted_mid : z;
            double p1[2] = {sample[0]+ca, sample[1]+cb};
            append_resolved(s, p1);
            physical[0] = p1[0]; physical[1] = p1[1];
            if (segment_capture(p0[0], p0[1], p1[0], p1[1], at, bt, finish_r)) {
                double tz[2] = {at, bt};
                append_raw(s, tz);
                physical[0] = at; physical[1] = bt;
                term = SPONG_ARR_CAPTURE; captured = 1; break;
            }
            p0[0] = p1[0]; p0[1] = p1[1];
        }
        if (captured) break;
    }
    if (r->clock) {
        o->clock_started = clock_started && !clock_failed;
        o->clock_available = clock_done && !clock_failed;
    }
    finish(s, term, physical);
}

int spong_centered_arrival(
    const spong_jet *jet,
    const spong_arrival_request *request,
    double *points, size_t point_capacity,
    spong_arrival_result *result)
{
    SPONG_FP_EXACT
    if (jet == NULL || request == NULL || result == NULL) return -1;
    memset(result, 0, sizeof(*result));
    result->finish_radius = NAN;
    result->spectral_ratio = NAN;
    arrival s;
    s.jet = jet;
    s.req = request;
    s.points = points;
    s.capacity = points == NULL ? 0 : point_capacity;
    s.n = 0;
    double scale = fmax(fmax(fabs(request->at), fabs(request->bt)),
                        fmax(fabs(request->a0), fabs(request->b0)));
    s.geometry_floor = 128.0*DBL_EPSILON*(1.0+scale);
    /* Order schedule; see the table in spong_potential.c.  4 -> (4, 6) so
     * that GL4 is reachable at all, and a NEGATIVE primary_order requests
     * pure |order| in both slots for convergence work. */
    int po = request->primary_order;
    if (po < 0) { s.orders[0] = -po; s.orders[1] = -po; }
    else if (po == 8) { s.orders[0] = 8; s.orders[1] = 6; }
    else if (po == 4) { s.orders[0] = 4; s.orders[1] = 6; }
    else              { s.orders[0] = 6; s.orders[1] = 8; }
    s.out = result;
    run(&s);
    return result->term;
}
