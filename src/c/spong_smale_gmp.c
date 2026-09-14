#include "spong/spong_smale.h"

#include <gmp.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mpq_t lo;
    mpq_t hi;
} qinterval;

typedef struct {
    size_t n;
    mpq_t *c;
} qpolynomial;

typedef struct {
    mpq_t level;
    mpq_t b;
    mpq_t y;
} qcentre;

typedef struct {
    qpolynomial A, B, Ap, Bp, N;
    mpq_t C;
    const spong_b_parameter_policy *policy;
    spong_b_parameter_result *result;
    uint64_t max_rational_bits;
    uint64_t *input_rationals;
    uint64_t *interval_evaluations;
    uint64_t *peak_rational_bits;
    int32_t *status;
    int32_t *primary_reason;
    int failed;
} context;

typedef struct {
    qcentre left;
    qcentre right;
    mpq_t left_radius;
    mpq_t right_radius;
    qinterval left_level;
    qinterval right_level;
    int initialized;
} target_slab;

static uint64_t qbits(const mpq_t value) {
    uint64_t n = (uint64_t)mpz_sizeinbase(mpq_numref(value), 2);
    uint64_t d = (uint64_t)mpz_sizeinbase(mpq_denref(value), 2);
    return n > d ? n : d;
}

static int observe(context *ctx, const mpq_t value) {
    uint64_t bits = qbits(value);
    if (bits > *ctx->peak_rational_bits)
        *ctx->peak_rational_bits = bits;
    if (ctx->max_rational_bits != 0 && bits > ctx->max_rational_bits) {
        ctx->failed = 1;
        *ctx->status = SPONG_SMALE_WORK_LIMIT;
        *ctx->primary_reason = SPONG_SMALE_REASON_ENDPOINT_BITS;
        return -1;
    }
    return 0;
}

static int parse_q(context *ctx, const spong_rational_input *input, mpq_t value) {
    if (input == NULL || input->numerator == NULL || input->denominator == NULL ||
        mpz_set_str(mpq_numref(value), input->numerator, 10) != 0 ||
        mpz_set_str(mpq_denref(value), input->denominator, 10) != 0 ||
        mpz_sgn(mpq_denref(value)) <= 0) {
        ctx->failed = 1;
        *ctx->status = SPONG_SMALE_PARSE_FAILURE;
        *ctx->primary_reason = SPONG_SMALE_REASON_BAD_RATIONAL;
        return -1;
    }
    mpq_canonicalize(value);
    (*ctx->input_rationals)++;
    return observe(ctx, value);
}

static void qi_init(qinterval *value) {
    mpq_init(value->lo);
    mpq_init(value->hi);
}

static void qi_clear(qinterval *value) {
    mpq_clear(value->lo);
    mpq_clear(value->hi);
}

static void qi_set(qinterval *out, const qinterval *value) {
    mpq_set(out->lo, value->lo);
    mpq_set(out->hi, value->hi);
}

static void qi_set_point(qinterval *out, const mpq_t value) {
    mpq_set(out->lo, value);
    mpq_set(out->hi, value);
}

static void qi_set_si(qinterval *out, long value) {
    mpq_set_si(out->lo, value, 1);
    mpq_set_si(out->hi, value, 1);
}

static int qi_contains_zero(const qinterval *value) {
    return mpq_sgn(value->lo) <= 0 && mpq_sgn(value->hi) >= 0;
}

static int qi_observe(context *ctx, const qinterval *value) {
    return observe(ctx, value->lo) != 0 || observe(ctx, value->hi) != 0 ? -1 : 0;
}

static int qi_add(context *ctx, qinterval *out, const qinterval *a,
                  const qinterval *b) {
    qinterval t;
    qi_init(&t);
    mpq_add(t.lo, a->lo, b->lo);
    mpq_add(t.hi, a->hi, b->hi);
    qi_set(out, &t);
    qi_clear(&t);
    return qi_observe(ctx, out);
}

static int qi_sub(context *ctx, qinterval *out, const qinterval *a,
                  const qinterval *b) {
    qinterval t;
    qi_init(&t);
    mpq_sub(t.lo, a->lo, b->hi);
    mpq_sub(t.hi, a->hi, b->lo);
    qi_set(out, &t);
    qi_clear(&t);
    return qi_observe(ctx, out);
}

static int qi_mul(context *ctx, qinterval *out, const qinterval *a,
                  const qinterval *b) {
    mpq_t products[4];
    for (int i = 0; i < 4; ++i)
        mpq_init(products[i]);
    mpq_mul(products[0], a->lo, b->lo);
    mpq_mul(products[1], a->lo, b->hi);
    mpq_mul(products[2], a->hi, b->lo);
    mpq_mul(products[3], a->hi, b->hi);
    mpq_set(out->lo, products[0]);
    mpq_set(out->hi, products[0]);
    for (int i = 1; i < 4; ++i) {
        if (mpq_cmp(products[i], out->lo) < 0)
            mpq_set(out->lo, products[i]);
        if (mpq_cmp(products[i], out->hi) > 0)
            mpq_set(out->hi, products[i]);
    }
    for (int i = 0; i < 4; ++i)
        mpq_clear(products[i]);
    return qi_observe(ctx, out);
}

static int qi_scale(context *ctx, qinterval *out, const qinterval *value,
                    const mpq_t scale) {
    qinterval point;
    qi_init(&point);
    qi_set_point(&point, scale);
    int status = qi_mul(ctx, out, value, &point);
    qi_clear(&point);
    return status;
}

static int qi_scale_si(context *ctx, qinterval *out, const qinterval *value,
                       long scale) {
    mpq_t q;
    mpq_init(q);
    mpq_set_si(q, scale, 1);
    int status = qi_scale(ctx, out, value, q);
    mpq_clear(q);
    return status;
}

static int qi_square(context *ctx, qinterval *out, const qinterval *value) {
    if (qi_contains_zero(value)) {
        mpq_set_ui(out->lo, 0, 1);
        mpq_mul(out->hi, value->lo, value->lo);
        mpq_t upper;
        mpq_init(upper);
        mpq_mul(upper, value->hi, value->hi);
        if (mpq_cmp(upper, out->hi) > 0)
            mpq_set(out->hi, upper);
        mpq_clear(upper);
    } else {
        mpq_mul(out->lo, value->lo, value->lo);
        mpq_mul(out->hi, value->hi, value->hi);
        if (mpq_cmp(out->lo, out->hi) > 0)
            mpq_swap(out->lo, out->hi);
    }
    return qi_observe(ctx, out);
}

static int qi_div(context *ctx, qinterval *out, const qinterval *numerator,
                  const qinterval *denominator) {
    if (qi_contains_zero(denominator))
        return -1;
    qinterval reciprocal;
    qi_init(&reciprocal);
    mpq_inv(reciprocal.lo, denominator->lo);
    mpq_inv(reciprocal.hi, denominator->hi);
    if (mpq_cmp(reciprocal.lo, reciprocal.hi) > 0)
        mpq_swap(reciprocal.lo, reciprocal.hi);
    int status = qi_mul(ctx, out, numerator, &reciprocal);
    qi_clear(&reciprocal);
    return status;
}

static int qp_init(qpolynomial *poly, size_t n) {
    memset(poly, 0, sizeof(*poly));
    n = n == 0 ? 1 : n;
    if (n > SIZE_MAX / sizeof(mpq_t))
        return -1;
    poly->c = (mpq_t *)calloc(n, sizeof(mpq_t));
    if (poly->c == NULL)
        return -1;
    poly->n = n;
    for (size_t i = 0; i < n; ++i)
        mpq_init(poly->c[i]);
    return 0;
}

static void qp_clear(qpolynomial *poly) {
    if (poly->c != NULL) {
        for (size_t i = 0; i < poly->n; ++i)
            mpq_clear(poly->c[i]);
        free(poly->c);
    }
    memset(poly, 0, sizeof(*poly));
}

static size_t qp_length(const qpolynomial *poly) {
    size_t n = poly->n;
    while (n > 1 && mpq_sgn(poly->c[n - 1]) == 0)
        --n;
    return n;
}

static int qp_observe(context *ctx, const qpolynomial *poly) {
    for (size_t i = 0; i < poly->n; ++i)
        if (observe(ctx, poly->c[i]) != 0)
            return -1;
    return 0;
}

static int qp_parse(context *ctx, qpolynomial *poly, const spong_rational_input *input,
                    size_t n) {
    if (input == NULL || n == 0 || qp_init(poly, n) != 0)
        return -1;
    for (size_t i = 0; i < n; ++i)
        if (parse_q(ctx, &input[i], poly->c[i]) != 0)
            return -1;
    return 0;
}

static int qp_add(context *ctx, qpolynomial *out, const qpolynomial *a,
                  const qpolynomial *b) {
    size_t n = a->n > b->n ? a->n : b->n;
    if (qp_init(out, n) != 0)
        return -1;
    for (size_t i = 0; i < n; ++i) {
        if (i < a->n && i < b->n)
            mpq_add(out->c[i], a->c[i], b->c[i]);
        else if (i < a->n)
            mpq_set(out->c[i], a->c[i]);
        else
            mpq_set(out->c[i], b->c[i]);
    }
    return qp_observe(ctx, out);
}

static int qp_scale(context *ctx, qpolynomial *out, const qpolynomial *in,
                    const mpq_t scale) {
    if (qp_init(out, in->n) != 0)
        return -1;
    for (size_t i = 0; i < in->n; ++i)
        mpq_mul(out->c[i], in->c[i], scale);
    return qp_observe(ctx, out);
}

static int qp_sub(context *ctx, qpolynomial *out, const qpolynomial *a,
                  const qpolynomial *b) {
    qpolynomial negative = {0};
    mpq_t minus_one;
    mpq_init(minus_one);
    mpq_set_si(minus_one, -1, 1);
    int status = qp_scale(ctx, &negative, b, minus_one);
    if (status == 0)
        status = qp_add(ctx, out, a, &negative);
    qp_clear(&negative);
    mpq_clear(minus_one);
    return status;
}

static int qp_mul(context *ctx, qpolynomial *out, const qpolynomial *a,
                  const qpolynomial *b) {
    size_t an = qp_length(a), bn = qp_length(b);
    if (an > SIZE_MAX - bn + 1 || qp_init(out, an + bn - 1) != 0)
        return -1;
    mpq_t product;
    mpq_init(product);
    for (size_t i = 0; i < an; ++i) {
        if (mpq_sgn(a->c[i]) == 0)
            continue;
        for (size_t j = 0; j < bn; ++j) {
            if (mpq_sgn(b->c[j]) == 0)
                continue;
            mpq_mul(product, a->c[i], b->c[j]);
            mpq_add(out->c[i + j], out->c[i + j], product);
        }
    }
    mpq_clear(product);
    return qp_observe(ctx, out);
}

static int qp_derivative(context *ctx, qpolynomial *out, const qpolynomial *in) {
    if (in->n <= 1)
        return qp_init(out, 1);
    if (qp_init(out, in->n - 1) != 0)
        return -1;
    for (size_t i = 1; i < in->n; ++i) {
        mpq_set(out->c[i - 1], in->c[i]);
        mpz_mul_ui(mpq_numref(out->c[i - 1]), mpq_numref(out->c[i - 1]),
                   (unsigned long)i);
        mpq_canonicalize(out->c[i - 1]);
    }
    return qp_observe(ctx, out);
}

static int qp_affine_compose(context *ctx, qpolynomial *out, const qpolynomial *in,
                             const mpq_t offset, const mpq_t scale) {
    qpolynomial result = {0}, affine = {0};
    if (qp_init(&result, 1) != 0 || qp_init(&affine, 2) != 0)
        goto failure;
    mpq_set(affine.c[0], offset);
    mpq_set(affine.c[1], scale);
    for (size_t k = qp_length(in); k-- > 0;) {
        qpolynomial product = {0}, next = {0}, constant = {0};
        if (qp_mul(ctx, &product, &result, &affine) != 0 || qp_init(&constant, 1) != 0)
            goto loop_failure;
        mpq_set(constant.c[0], in->c[k]);
        if (qp_add(ctx, &next, &product, &constant) != 0)
            goto loop_failure;
        qp_clear(&result);
        result = next;
        qp_clear(&product);
        qp_clear(&constant);
        continue;
    loop_failure:
        qp_clear(&product);
        qp_clear(&next);
        qp_clear(&constant);
        goto failure;
    }
    *out = result;
    qp_clear(&affine);
    return 0;
failure:
    qp_clear(&result);
    qp_clear(&affine);
    return -1;
}

static int qp_interval(context *ctx, qinterval *out, const qpolynomial *poly,
                       const qinterval *x) {
    qinterval value, coefficient;
    qi_init(&value);
    qi_init(&coefficient);
    qi_set_si(&value, 0);
    for (size_t k = qp_length(poly); k-- > 0;) {
        if (qi_mul(ctx, &value, &value, x) != 0)
            goto failure;
        qi_set_point(&coefficient, poly->c[k]);
        if (qi_add(ctx, &value, &value, &coefficient) != 0)
            goto failure;
    }
    qi_set(out, &value);
    qi_clear(&value);
    qi_clear(&coefficient);
    (*ctx->interval_evaluations)++;
    return 0;
failure:
    qi_clear(&value);
    qi_clear(&coefficient);
    return -1;
}

static int qp_centered_interval(context *ctx, qinterval *out, const qpolynomial *poly,
                                const qinterval *x) {
    mpq_t centre, radius, one;
    mpq_inits(centre, radius, one, NULL);
    mpq_add(centre, x->lo, x->hi);
    mpq_div_2exp(centre, centre, 1);
    mpq_sub(radius, x->hi, x->lo);
    mpq_div_2exp(radius, radius, 1);
    mpq_set_ui(one, 1, 1);
    qpolynomial shifted = {0};
    qinterval delta;
    qi_init(&delta);
    mpq_neg(delta.lo, radius);
    mpq_set(delta.hi, radius);
    int status = qp_affine_compose(ctx, &shifted, poly, centre, one);
    if (status == 0)
        status = qp_interval(ctx, out, &shifted, &delta);
    qp_clear(&shifted);
    qi_clear(&delta);
    mpq_clears(centre, radius, one, NULL);
    return status;
}

static int qp_equal(const qpolynomial *a, const qpolynomial *b) {
    size_t an = qp_length(a), bn = qp_length(b);
    if (an != bn)
        return 0;
    for (size_t i = 0; i < an; ++i)
        if (mpq_cmp(a->c[i], b->c[i]) != 0)
            return 0;
    return 1;
}

static void qc_init(qcentre *point) {
    mpq_inits(point->level, point->b, point->y, NULL);
}

static void qc_clear(qcentre *point) {
    mpq_clears(point->level, point->b, point->y, NULL);
}

static void qc_set(qcentre *out, const qcentre *in) {
    mpq_set(out->level, in->level);
    mpq_set(out->b, in->b);
    mpq_set(out->y, in->y);
}

static void target_slab_init(target_slab *slab) {
    memset(slab, 0, sizeof(*slab));
    qc_init(&slab->left);
    qc_init(&slab->right);
    mpq_inits(slab->left_radius, slab->right_radius, NULL);
    qi_init(&slab->left_level);
    qi_init(&slab->right_level);
}

static void target_slab_clear(target_slab *slab) {
    qc_clear(&slab->left);
    qc_clear(&slab->right);
    mpq_clears(slab->left_radius, slab->right_radius, NULL);
    qi_clear(&slab->left_level);
    qi_clear(&slab->right_level);
}

static void set_refusal(context *ctx, int status, int reason) {
    if (!ctx->failed) {
        ctx->failed = 1;
        *ctx->status = status;
        *ctx->primary_reason = reason;
    }
}

static int hard_failure(const context *ctx) {
    return *ctx->status == SPONG_SMALE_WORK_LIMIT ||
           *ctx->status == SPONG_SMALE_ALLOCATION_FAILURE ||
           *ctx->status == SPONG_SMALE_PARSE_FAILURE ||
           *ctx->status == SPONG_SMALE_INTERNAL_FAILURE;
}

static void clear_local_refusal(context *ctx) {
    ctx->failed = 0;
    *ctx->status = SPONG_SMALE_OK;
    *ctx->primary_reason = SPONG_SMALE_REASON_NONE;
}

static int level_box(context *ctx, qinterval *out, const qinterval *b,
                     const qinterval *y) {
    qinterval A, B, ys, bs, numerator, quotient, C;
    qi_init(&A);
    qi_init(&B);
    qi_init(&ys);
    qi_init(&bs);
    qi_init(&numerator);
    qi_init(&quotient);
    qi_init(&C);
    int status = qp_centered_interval(ctx, &A, &ctx->A, b);
    if (status != 0 || qi_contains_zero(&A)) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                        SPONG_SMALE_REASON_TRANSVERSE_DIVISOR);
        goto done;
    }
    if (qp_centered_interval(ctx, &B, &ctx->B, b) != 0 || qi_square(ctx, &ys, y) != 0 ||
        qi_square(ctx, &bs, &B) != 0 || qi_sub(ctx, &numerator, &ys, &bs) != 0 ||
        qi_div(ctx, &quotient, &numerator, &A) != 0)
        goto done;
    qi_set_point(&C, ctx->C);
    if (qi_add(ctx, out, &C, &quotient) != 0)
        goto done;
    status = 0;
done:
    qi_clear(&A);
    qi_clear(&B);
    qi_clear(&ys);
    qi_clear(&bs);
    qi_clear(&numerator);
    qi_clear(&quotient);
    qi_clear(&C);
    return status;
}

static int regularity_box(context *ctx, const qinterval *b, const qinterval *y,
                          qinterval *norm2) {
    qinterval A, B, Ap, N, A2, t, q, loss_b, ys, four_ys, lbs;
    qi_init(&A);
    qi_init(&B);
    qi_init(&Ap);
    qi_init(&N);
    qi_init(&A2);
    qi_init(&t);
    qi_init(&q);
    qi_init(&loss_b);
    qi_init(&ys);
    qi_init(&four_ys);
    qi_init(&lbs);
    int status = qp_centered_interval(ctx, &A, &ctx->A, b);
    if (status != 0 || qi_contains_zero(&A)) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                        SPONG_SMALE_REASON_TRANSVERSE_DIVISOR);
        goto done;
    }
    if (qp_centered_interval(ctx, &B, &ctx->B, b) != 0 ||
        qp_centered_interval(ctx, &Ap, &ctx->Ap, b) != 0 ||
        qp_centered_interval(ctx, &N, &ctx->N, b) != 0 ||
        qi_mul(ctx, &t, &Ap, y) != 0 || qi_add(ctx, &t, &N, &t) != 0 ||
        qi_add(ctx, &q, &B, y) != 0 || qi_mul(ctx, &q, &q, &t) != 0)
        goto done;
    if (qi_contains_zero(&q)) {
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    SPONG_SMALE_REASON_B_PARAMETER_SINGULAR);
        goto done;
    }
    if (qi_square(ctx, &A2, &A) != 0 || qi_div(ctx, &loss_b, &q, &A2) != 0 ||
        qi_square(ctx, &ys, y) != 0 || qi_scale_si(ctx, &four_ys, &ys, 4) != 0 ||
        qi_square(ctx, &lbs, &loss_b) != 0 || qi_add(ctx, norm2, &four_ys, &lbs) != 0)
        goto done;
    if (qi_contains_zero(norm2)) {
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    SPONG_SMALE_REASON_CRITICAL_POINT);
        goto done;
    }
    status = 0;
done:
    qi_clear(&A);
    qi_clear(&B);
    qi_clear(&Ap);
    qi_clear(&N);
    qi_clear(&A2);
    qi_clear(&t);
    qi_clear(&q);
    qi_clear(&loss_b);
    qi_clear(&ys);
    qi_clear(&four_ys);
    qi_clear(&lbs);
    return status;
}

typedef struct {
    qinterval dy_db;
    qinterval level;
    qinterval loss_b;
    qinterval norm2;
} face_result;

static void face_init(face_result *face) {
    qi_init(&face->dy_db);
    qi_init(&face->level);
    qi_init(&face->loss_b);
    qi_init(&face->norm2);
}

static void face_clear(face_result *face) {
    qi_clear(&face->dy_db);
    qi_clear(&face->level);
    qi_clear(&face->loss_b);
    qi_clear(&face->norm2);
}

static int face_transport(context *ctx, const mpq_t b0, const mpq_t b1, const mpq_t y0,
                          const mpq_t y1, face_result *out) {
    mpq_t db, dy, two, four;
    mpq_inits(db, dy, two, four, NULL);
    mpq_sub(db, b1, b0);
    mpq_sub(dy, y1, y0);
    mpq_set_ui(two, 2, 1);
    mpq_set_ui(four, 4, 1);
    qpolynomial A = {0}, B = {0}, Ap = {0}, Bp = {0}, N = {0}, Y = {0};
    qpolynomial By = {0}, Apy = {0}, Nplus = {0}, Q = {0};
    qpolynomial A2 = {0}, A4 = {0}, ApBy = {0}, ABp = {0}, R = {0};
    qpolynomial A4Y = {0}, RQ = {0}, D = {0}, denom = {0};
    qpolynomial Y2 = {0}, Y2A4 = {0}, Q2 = {0}, normnum = {0};
    qpolynomial CA = {0}, B2 = {0}, levelnum = {0}, temp = {0};
    qinterval unit, Ai, deni, normi, Qi, A2i, A4i, Di, leveli;
    qi_init(&unit);
    qi_init(&Ai);
    qi_init(&deni);
    qi_init(&normi);
    qi_init(&Qi);
    qi_init(&A2i);
    qi_init(&A4i);
    qi_init(&Di);
    qi_init(&leveli);
    mpq_set_ui(unit.lo, 0, 1);
    mpq_set_ui(unit.hi, 1, 1);
    int status = -1;

    if (qp_affine_compose(ctx, &A, &ctx->A, b0, db) != 0 ||
        qp_affine_compose(ctx, &B, &ctx->B, b0, db) != 0 ||
        qp_affine_compose(ctx, &Ap, &ctx->Ap, b0, db) != 0 ||
        qp_affine_compose(ctx, &Bp, &ctx->Bp, b0, db) != 0 ||
        qp_affine_compose(ctx, &N, &ctx->N, b0, db) != 0 || qp_init(&Y, 2) != 0)
        goto done;
    mpq_set(Y.c[0], y0);
    mpq_set(Y.c[1], dy);
    if (qp_add(ctx, &By, &B, &Y) != 0 || qp_mul(ctx, &Apy, &Ap, &Y) != 0 ||
        qp_add(ctx, &Nplus, &N, &Apy) != 0 || qp_mul(ctx, &Q, &By, &Nplus) != 0 ||
        qp_mul(ctx, &A2, &A, &A) != 0 || qp_mul(ctx, &A4, &A2, &A2) != 0 ||
        qp_mul(ctx, &ApBy, &Ap, &By) != 0 || qp_mul(ctx, &ABp, &A, &Bp) != 0 ||
        qp_sub(ctx, &R, &ApBy, &ABp) != 0 || qp_mul(ctx, &A4Y, &A4, &Y) != 0 ||
        qp_scale(ctx, &temp, &A4Y, two) != 0 || qp_mul(ctx, &RQ, &R, &Q) != 0 ||
        qp_add(ctx, &D, &temp, &RQ) != 0)
        goto done;
    qp_clear(&temp);
    if (qp_mul(ctx, &denom, &A, &Q) != 0 || qp_mul(ctx, &Y2, &Y, &Y) != 0 ||
        qp_mul(ctx, &Y2A4, &Y2, &A4) != 0 || qp_scale(ctx, &temp, &Y2A4, four) != 0 ||
        qp_mul(ctx, &Q2, &Q, &Q) != 0 || qp_add(ctx, &normnum, &temp, &Q2) != 0)
        goto done;
    qp_clear(&temp);
    if (qp_scale(ctx, &CA, &A, ctx->C) != 0 || qp_mul(ctx, &B2, &B, &B) != 0 ||
        qp_sub(ctx, &temp, &Y2, &B2) != 0 || qp_add(ctx, &levelnum, &CA, &temp) != 0)
        goto done;

    if (qp_centered_interval(ctx, &Ai, &A, &unit) != 0 || qi_contains_zero(&Ai)) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                        SPONG_SMALE_REASON_TRANSVERSE_DIVISOR);
        goto done;
    }
    if (qp_centered_interval(ctx, &deni, &denom, &unit) != 0 ||
        qi_contains_zero(&deni)) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                        SPONG_SMALE_REASON_B_PARAMETER_SINGULAR);
        goto done;
    }
    if (qp_centered_interval(ctx, &normi, &normnum, &unit) != 0 ||
        qi_contains_zero(&normi)) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                        SPONG_SMALE_REASON_CRITICAL_POINT);
        goto done;
    }
    if (qp_centered_interval(ctx, &Di, &D, &unit) != 0 ||
        qi_div(ctx, &out->dy_db, &Di, &deni) != 0 ||
        qp_centered_interval(ctx, &leveli, &levelnum, &unit) != 0 ||
        qi_div(ctx, &out->level, &leveli, &Ai) != 0 ||
        qp_centered_interval(ctx, &Qi, &Q, &unit) != 0 ||
        qp_centered_interval(ctx, &A2i, &A2, &unit) != 0 ||
        qi_div(ctx, &out->loss_b, &Qi, &A2i) != 0 ||
        qp_centered_interval(ctx, &A4i, &A4, &unit) != 0 ||
        qi_div(ctx, &out->norm2, &normi, &A4i) != 0)
        goto done;
    status = 0;
done:
    qp_clear(&A);
    qp_clear(&B);
    qp_clear(&Ap);
    qp_clear(&Bp);
    qp_clear(&N);
    qp_clear(&Y);
    qp_clear(&By);
    qp_clear(&Apy);
    qp_clear(&Nplus);
    qp_clear(&Q);
    qp_clear(&A2);
    qp_clear(&A4);
    qp_clear(&ApBy);
    qp_clear(&ABp);
    qp_clear(&R);
    qp_clear(&A4Y);
    qp_clear(&RQ);
    qp_clear(&D);
    qp_clear(&denom);
    qp_clear(&Y2);
    qp_clear(&Y2A4);
    qp_clear(&Q2);
    qp_clear(&normnum);
    qp_clear(&CA);
    qp_clear(&B2);
    qp_clear(&levelnum);
    qp_clear(&temp);
    qi_clear(&unit);
    qi_clear(&Ai);
    qi_clear(&deni);
    qi_clear(&normi);
    qi_clear(&Qi);
    qi_clear(&A2i);
    qi_clear(&A4i);
    qi_clear(&Di);
    qi_clear(&leveli);
    mpq_clears(db, dy, two, four, NULL);
    return status;
}

static void qmax_set(mpq_t out, const mpq_t value) {
    if (mpq_cmp(value, out) > 0)
        mpq_set(out, value);
}

static int ceil_dyadic_bits(context *ctx, mpq_t out, const mpq_t value, uint64_t bits) {
    mpz_t numerator, quotient;
    mpz_inits(numerator, quotient, NULL);
    mpz_mul_2exp(numerator, mpq_numref(value), (mp_bitcnt_t)bits);
    mpz_cdiv_q(quotient, numerator, mpq_denref(value));
    mpq_set_z(out, quotient);
    mpz_set_ui(mpq_denref(out), 1);
    mpz_mul_2exp(mpq_denref(out), mpq_denref(out), (mp_bitcnt_t)bits);
    mpq_canonicalize(out);
    mpz_clears(numerator, quotient, NULL);
    return observe(ctx, out);
}

static int ceil_dyadic(context *ctx, mpq_t out, const mpq_t value) {
    return ceil_dyadic_bits(ctx, out, value, ctx->policy->radius_round_bits);
}

static void update_minimum(mpq_t minimum, int *has_minimum, const mpq_t value) {
    if (!*has_minimum || mpq_cmp(value, minimum) < 0) {
        mpq_set(minimum, value);
        *has_minimum = 1;
    }
}

typedef struct {
    context *ctx;
    mpq_srcptr target;
    mpq_srcptr radius_cap;
    mpq_t minimum_face;
    mpq_t minimum_norm;
    int has_face;
    int has_norm;
    target_slab *target_slab;
} march_state;

/* 1 means target slab found, 0 means accepted through right, -1 refusal. */
static int march_segment(march_state *state, const qcentre *left, const qcentre *right,
                         uint64_t depth, mpq_t radius, qinterval *left_level) {
    context *ctx = state->ctx;
    mpq_t h, slope, next_radius, lower0, lower1, upper0, upper1;
    mpq_t lower_slope, upper_slope, lower_margin, upper_margin;
    mpq_t growth, candidate, required, temp, factor;
    mpq_inits(h, slope, next_radius, lower0, lower1, upper0, upper1, lower_slope,
              upper_slope, lower_margin, upper_margin, growth, candidate, required,
              temp, factor, NULL);
    mpq_sub(h, right->b, left->b);
    if (ctx->result->b_direction < 0)
        mpq_neg(h, h);
    mpq_sub(slope, right->y, left->y);
    mpq_div(slope, slope, h);
    mpq_set(next_radius, radius);
    qinterval interior_b, interior_y, norm, lower_velocity, upper_velocity;
    qinterval right_y, right_level;
    qi_init(&interior_b);
    qi_init(&interior_y);
    qi_init(&norm);
    qi_init(&lower_velocity);
    qi_init(&upper_velocity);
    qi_init(&right_y);
    qi_init(&right_level);
    face_result lower, upper;
    face_init(&lower);
    face_init(&upper);
    int accepted = 0;
    int failure_reason = SPONG_SMALE_REASON_FACE_INEQUALITY;

    for (uint64_t inflation = 0; inflation < ctx->policy->max_inflations; ++inflation) {
        ctx->result->work.inflation_steps++;
        if (observe(ctx, left->b) != 0 || observe(ctx, right->b) != 0 ||
            observe(ctx, left->y) != 0 || observe(ctx, right->y) != 0 ||
            observe(ctx, radius) != 0 || observe(ctx, next_radius) != 0)
            goto refusal;
        if (mpq_cmp(left->b, right->b) < 0) {
            mpq_set(interior_b.lo, left->b);
            mpq_set(interior_b.hi, right->b);
        } else {
            mpq_set(interior_b.lo, right->b);
            mpq_set(interior_b.hi, left->b);
        }
        mpq_sub(lower0, left->y, radius);
        mpq_sub(lower1, right->y, next_radius);
        mpq_add(upper0, left->y, radius);
        mpq_add(upper1, right->y, next_radius);
        mpq_set(interior_y.lo, mpq_cmp(lower0, lower1) < 0 ? lower0 : lower1);
        mpq_set(interior_y.hi, mpq_cmp(upper0, upper1) > 0 ? upper0 : upper1);

        ctx->failed = 0;
        if (regularity_box(ctx, &interior_b, &interior_y, &norm) != 0) {
            if (hard_failure(ctx))
                goto refusal;
            failure_reason = ctx->result->primary_reason;
            clear_local_refusal(ctx);
            break;
        }
        if (face_transport(ctx, left->b, right->b, lower0, lower1, &lower) != 0) {
            if (ctx->result->status == SPONG_SMALE_OK)
                set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                            SPONG_SMALE_REASON_ALLOCATION);
            if (hard_failure(ctx))
                goto refusal;
            failure_reason = ctx->result->primary_reason;
            clear_local_refusal(ctx);
            break;
        }
        if (face_transport(ctx, left->b, right->b, upper0, upper1, &upper) != 0) {
            if (ctx->result->status == SPONG_SMALE_OK)
                set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                            SPONG_SMALE_REASON_ALLOCATION);
            if (hard_failure(ctx))
                goto refusal;
            failure_reason = ctx->result->primary_reason;
            clear_local_refusal(ctx);
            break;
        }
        if (ctx->result->b_direction > 0) {
            qi_set(&lower_velocity, &lower.dy_db);
            qi_set(&upper_velocity, &upper.dy_db);
        } else {
            mpq_neg(lower_velocity.lo, lower.dy_db.hi);
            mpq_neg(lower_velocity.hi, lower.dy_db.lo);
            mpq_neg(upper_velocity.lo, upper.dy_db.hi);
            mpq_neg(upper_velocity.hi, upper.dy_db.lo);
        }
        mpq_sub(temp, next_radius, radius);
        mpq_div(temp, temp, h);
        mpq_sub(lower_slope, slope, temp);
        mpq_add(upper_slope, slope, temp);
        mpq_sub(lower_margin, lower_velocity.lo, lower_slope);
        mpq_sub(upper_margin, upper_slope, upper_velocity.hi);
        mpq_sub(growth, slope, lower_velocity.lo);
        if (mpq_sgn(growth) < 0)
            mpq_set_ui(growth, 0, 1);
        mpq_sub(temp, upper_velocity.hi, slope);
        qmax_set(growth, temp);
        mpq_set_ui(factor, 65, 64);
        mpq_mul(candidate, h, growth);
        mpq_mul(candidate, candidate, factor);
        mpq_add(candidate, candidate, radius);
        if (ceil_dyadic(ctx, required, candidate) != 0)
            goto refusal;
        if (mpq_cmp(required, next_radius) < 0)
            mpq_set(required, next_radius);
        if (mpq_sgn(lower_margin) >= 0 && mpq_sgn(upper_margin) >= 0) {
            accepted = 1;
            update_minimum(state->minimum_face, &state->has_face, lower_margin);
            update_minimum(state->minimum_face, &state->has_face, upper_margin);
            update_minimum(state->minimum_norm, &state->has_norm, norm.lo);
            update_minimum(state->minimum_norm, &state->has_norm, lower.norm2.lo);
            update_minimum(state->minimum_norm, &state->has_norm, upper.norm2.lo);
            break;
        }
        if (mpq_cmp(required, next_radius) == 0)
            break;
        mpq_set(next_radius, required);
        if (state->radius_cap != NULL && mpq_cmp(next_radius, state->radius_cap) > 0) {
            failure_reason = SPONG_SMALE_REASON_RADIUS_CAP;
            break;
        }
    }

    if (!accepted) {
        if (depth < ctx->policy->max_slab_bisections) {
            qcentre middle;
            qc_init(&middle);
            mpq_add(middle.level, left->level, right->level);
            mpq_div_2exp(middle.level, middle.level, 1);
            mpq_add(middle.b, left->b, right->b);
            mpq_div_2exp(middle.b, middle.b, 1);
            mpq_add(middle.y, left->y, right->y);
            mpq_div_2exp(middle.y, middle.y, 1);
            ctx->result->work.slab_bisections++;
            int first =
                march_segment(state, left, &middle, depth + 1, radius, left_level);
            if (first != 0) {
                qc_clear(&middle);
                accepted = first;
                goto return_code;
            }
            int second =
                march_segment(state, &middle, right, depth + 1, radius, left_level);
            qc_clear(&middle);
            if (second != 0) {
                accepted = second;
                goto return_code;
            }
            accepted = 0;
            goto return_code;
        }
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    failure_reason == SPONG_SMALE_REASON_FACE_INEQUALITY
                        ? SPONG_SMALE_REASON_SLAB_BUDGET
                        : failure_reason);
        goto refusal;
    }

    mpq_set(radius, next_radius);
    mpq_sub(right_y.lo, right->y, radius);
    mpq_add(right_y.hi, right->y, radius);
    ctx->failed = 0;
    {
        qinterval right_b;
        qi_init(&right_b);
        qi_set_point(&right_b, right->b);
        if (level_box(ctx, &right_level, &right_b, &right_y) != 0) {
            qi_clear(&right_b);
            goto refusal;
        }
        qi_clear(&right_b);
    }
    ctx->result->work.slabs_accepted++;
    if ((mpq_cmp(left_level->lo, state->target) > 0 &&
         mpq_cmp(state->target, right_level.hi) > 0) ||
        (mpq_cmp(left_level->hi, state->target) < 0 &&
         mpq_cmp(state->target, right_level.lo) < 0)) {
        target_slab *slab = state->target_slab;
        qc_set(&slab->left, left);
        qc_set(&slab->right, right);
        mpq_set(slab->left_radius, radius);
        /* radius was already advanced.  The incoming left radius is needed;
         * preserve it via lower/upper face endpoint separation. */
        mpq_sub(slab->left_radius, left->y, lower0);
        mpq_set(slab->right_radius, radius);
        qi_set(&slab->left_level, left_level);
        qi_set(&slab->right_level, &right_level);
        slab->initialized = 1;
        qi_set(left_level, &right_level);
        accepted = 1;
        goto return_code;
    }
    qi_set(left_level, &right_level);
    accepted = 0;
    goto return_code;

refusal:
    accepted = -1;
return_code:
    face_clear(&lower);
    face_clear(&upper);
    qi_clear(&interior_b);
    qi_clear(&interior_y);
    qi_clear(&norm);
    qi_clear(&lower_velocity);
    qi_clear(&upper_velocity);
    qi_clear(&right_y);
    qi_clear(&right_level);
    mpq_clears(h, slope, next_radius, lower0, lower1, upper0, upper1, lower_slope,
               upper_slope, lower_margin, upper_margin, growth, candidate, required,
               temp, factor, NULL);
    return accepted;
}

static int sqrt_interval(context *ctx, qinterval *out, const qinterval *value,
                         uint64_t bits) {
    if (mpq_sgn(value->lo) < 0 || bits > (uint64_t)ULONG_MAX / 2)
        return -1;
    mpz_t scaled, quotient, root, square, denominator;
    mpz_inits(scaled, quotient, root, square, denominator, NULL);
    mpz_mul_2exp(scaled, mpq_numref(value->lo), (mp_bitcnt_t)(2 * bits));
    mpz_fdiv_q(quotient, scaled, mpq_denref(value->lo));
    mpz_sqrt(root, quotient);
    mpq_set_z(out->lo, root);
    mpz_set_ui(mpq_denref(out->lo), 1);
    mpz_mul_2exp(mpq_denref(out->lo), mpq_denref(out->lo), (mp_bitcnt_t)bits);
    mpq_canonicalize(out->lo);

    mpz_mul_2exp(scaled, mpq_numref(value->hi), (mp_bitcnt_t)(2 * bits));
    mpz_fdiv_q(quotient, scaled, mpq_denref(value->hi));
    mpz_sqrt(root, quotient);
    mpz_mul(square, root, root);
    mpz_mul(denominator, square, mpq_denref(value->hi));
    if (mpz_cmp(denominator, scaled) != 0)
        mpz_add_ui(root, root, 1);
    mpq_set_z(out->hi, root);
    mpz_set_ui(mpq_denref(out->hi), 1);
    mpz_mul_2exp(mpq_denref(out->hi), mpq_denref(out->hi), (mp_bitcnt_t)bits);
    mpq_canonicalize(out->hi);
    mpz_clears(scaled, quotient, root, square, denominator, NULL);
    return qi_observe(ctx, out);
}

static int slab_box(context *ctx, const target_slab *slab, const qinterval *parameter,
                    qinterval *b, qinterval *y) {
    qinterval base, scale, lower, upper;
    qi_init(&base);
    qi_init(&scale);
    qi_init(&lower);
    qi_init(&upper);
    mpq_t delta, lower0, lower1, upper0, upper1;
    mpq_inits(delta, lower0, lower1, upper0, upper1, NULL);
    qi_set_point(&base, slab->left.b);
    mpq_sub(delta, slab->right.b, slab->left.b);
    qi_scale(ctx, &scale, parameter, delta);
    qi_add(ctx, b, &base, &scale);

    mpq_sub(lower0, slab->left.y, slab->left_radius);
    mpq_sub(lower1, slab->right.y, slab->right_radius);
    mpq_add(upper0, slab->left.y, slab->left_radius);
    mpq_add(upper1, slab->right.y, slab->right_radius);
    qi_set_point(&base, lower0);
    mpq_sub(delta, lower1, lower0);
    qi_scale(ctx, &scale, parameter, delta);
    qi_add(ctx, &lower, &base, &scale);
    qi_set_point(&base, upper0);
    mpq_sub(delta, upper1, upper0);
    qi_scale(ctx, &scale, parameter, delta);
    qi_add(ctx, &upper, &base, &scale);
    mpq_set(y->lo, mpq_cmp(lower.lo, upper.lo) < 0 ? lower.lo : upper.lo);
    mpq_set(y->hi, mpq_cmp(lower.hi, upper.hi) > 0 ? lower.hi : upper.hi);
    int status = qi_observe(ctx, b) != 0 || qi_observe(ctx, y) != 0 ? -1 : 0;
    qi_clear(&base);
    qi_clear(&scale);
    qi_clear(&lower);
    qi_clear(&upper);
    mpq_clears(delta, lower0, lower1, upper0, upper1, NULL);
    return status;
}

static int project_target(context *ctx, const target_slab *slab, const mpq_t target,
                          qinterval *target_b, qinterval *target_y) {
    uint64_t cap = ctx->policy->max_projection_subboxes;
    if (cap == 0 || cap > SIZE_MAX / sizeof(qinterval)) {
        set_refusal(ctx, SPONG_SMALE_INVALID_ARGUMENT, SPONG_SMALE_REASON_BAD_POLICY);
        return -1;
    }
    qinterval *pieces = (qinterval *)calloc((size_t)cap, sizeof(qinterval));
    qinterval *survivors = (qinterval *)calloc((size_t)cap, sizeof(qinterval));
    if (pieces == NULL || survivors == NULL) {
        free(pieces);
        free(survivors);
        set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE, SPONG_SMALE_REASON_ALLOCATION);
        return -1;
    }
    for (uint64_t i = 0; i < cap; ++i) {
        qi_init(&pieces[i]);
        qi_init(&survivors[i]);
    }
    mpq_set_ui(pieces[0].lo, 0, 1);
    mpq_set_ui(pieces[0].hi, 1, 1);
    uint64_t count = 1, survivor_count = 0, depth = 0;
    qinterval b, y, level;
    qi_init(&b);
    qi_init(&y);
    qi_init(&level);

    for (depth = 0; depth <= ctx->policy->max_projection_bisections; ++depth) {
        survivor_count = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (slab_box(ctx, slab, &pieces[i], &b, &y) != 0)
                goto failure;
            ctx->failed = 0;
            if (level_box(ctx, &level, &b, &y) != 0) {
                if (hard_failure(ctx))
                    goto failure;
                clear_local_refusal(ctx);
                continue;
            }
            if (mpq_cmp(level.lo, target) <= 0 && mpq_cmp(target, level.hi) <= 0) {
                qi_set(&survivors[survivor_count++], &pieces[i]);
            }
        }
        if (survivor_count == 0) {
            set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                        SPONG_SMALE_REASON_TARGET_MISSES_TUBE);
            goto failure;
        }
        if (depth == ctx->policy->max_projection_bisections)
            break;
        if (survivor_count > cap / 2) {
            set_refusal(ctx, SPONG_SMALE_WORK_LIMIT,
                        SPONG_SMALE_REASON_PROJECTION_BUDGET);
            goto failure;
        }
        count = 0;
        for (uint64_t i = 0; i < survivor_count; ++i) {
            mpq_t middle;
            mpq_init(middle);
            mpq_add(middle, survivors[i].lo, survivors[i].hi);
            mpq_div_2exp(middle, middle, 1);
            mpq_set(pieces[count].lo, survivors[i].lo);
            mpq_set(pieces[count++].hi, middle);
            mpq_set(pieces[count].lo, middle);
            mpq_set(pieces[count++].hi, survivors[i].hi);
            mpq_clear(middle);
        }
    }

    ctx->result->work.projection_subboxes = survivor_count;
    ctx->result->work.projection_depth = depth;
    int first = 1;
    qinterval y_hull;
    qi_init(&y_hull);
    for (uint64_t i = 0; i < survivor_count; ++i) {
        slab_box(ctx, slab, &survivors[i], &b, &y);
        if (first) {
            qi_set(target_b, &b);
            qi_set(&y_hull, &y);
            first = 0;
        } else {
            if (mpq_cmp(b.lo, target_b->lo) < 0)
                mpq_set(target_b->lo, b.lo);
            if (mpq_cmp(b.hi, target_b->hi) > 0)
                mpq_set(target_b->hi, b.hi);
            if (mpq_cmp(y.lo, y_hull.lo) < 0)
                mpq_set(y_hull.lo, y.lo);
            if (mpq_cmp(y.hi, y_hull.hi) > 0)
                mpq_set(y_hull.hi, y.hi);
        }
    }
    int sheet = mpq_sgn(y_hull.lo) > 0 ? 1 : mpq_sgn(y_hull.hi) < 0 ? -1 : 0;
    if (sheet == 0) {
        qi_clear(&y_hull);
        set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                    SPONG_SMALE_REASON_TARGET_SHEET);
        goto failure;
    }

    qpolynomial B2 = {0}, scaled_A = {0}, S = {0};
    mpq_t scale;
    mpq_init(scale);
    mpq_sub(scale, target, ctx->C);
    qinterval Si, Ai, Bi, Api, Ni, temp, Q, A2, Lb, Lb2, y2, four_y2, norm;
    qi_init(&Si);
    qi_init(&Ai);
    qi_init(&Bi);
    qi_init(&Api);
    qi_init(&Ni);
    qi_init(&temp);
    qi_init(&Q);
    qi_init(&A2);
    qi_init(&Lb);
    qi_init(&Lb2);
    qi_init(&y2);
    qi_init(&four_y2);
    qi_init(&norm);
    if (qp_mul(ctx, &B2, &ctx->B, &ctx->B) != 0 ||
        qp_scale(ctx, &scaled_A, &ctx->A, scale) != 0 ||
        qp_add(ctx, &S, &B2, &scaled_A) != 0 ||
        qp_centered_interval(ctx, &Si, &S, target_b) != 0 || mpq_sgn(Si.lo) <= 0 ||
        sqrt_interval(ctx, target_y, &Si, ctx->policy->radius_round_bits) != 0) {
        set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                    SPONG_SMALE_REASON_TARGET_SHEET);
        goto target_cleanup;
    }
    if (sheet < 0) {
        mpq_neg(temp.lo, target_y->hi);
        mpq_neg(temp.hi, target_y->lo);
        qi_set(target_y, &temp);
    }
    if (qp_centered_interval(ctx, &Ai, &ctx->A, target_b) != 0 ||
        qi_contains_zero(&Ai) ||
        qp_centered_interval(ctx, &Bi, &ctx->B, target_b) != 0 ||
        qp_centered_interval(ctx, &Api, &ctx->Ap, target_b) != 0 ||
        qp_centered_interval(ctx, &Ni, &ctx->N, target_b) != 0 ||
        qi_mul(ctx, &temp, &Api, target_y) != 0 ||
        qi_add(ctx, &temp, &Ni, &temp) != 0 || qi_add(ctx, &Q, &Bi, target_y) != 0 ||
        qi_mul(ctx, &Q, &Q, &temp) != 0 || qi_square(ctx, &A2, &Ai) != 0 ||
        qi_div(ctx, &Lb, &Q, &A2) != 0 || qi_square(ctx, &Lb2, &Lb) != 0 ||
        qi_square(ctx, &y2, target_y) != 0 || qi_scale_si(ctx, &four_y2, &y2, 4) != 0 ||
        qi_add(ctx, &norm, &four_y2, &Lb2) != 0 || qi_contains_zero(&norm)) {
        set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                    SPONG_SMALE_REASON_CRITICAL_POINT);
        goto target_cleanup;
    }
    if (mpq_cmp(target_y->hi, y_hull.lo) < 0 || mpq_cmp(target_y->lo, y_hull.hi) > 0) {
        set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                    SPONG_SMALE_REASON_TARGET_MISSES_TUBE);
        goto target_cleanup;
    }
    qi_clear(&y_hull);
    qp_clear(&B2);
    qp_clear(&scaled_A);
    qp_clear(&S);
    mpq_clear(scale);
    qi_clear(&Si);
    qi_clear(&Ai);
    qi_clear(&Bi);
    qi_clear(&Api);
    qi_clear(&Ni);
    qi_clear(&temp);
    qi_clear(&Q);
    qi_clear(&A2);
    qi_clear(&Lb);
    qi_clear(&Lb2);
    qi_clear(&y2);
    qi_clear(&four_y2);
    qi_clear(&norm);
    for (uint64_t i = 0; i < cap; ++i) {
        qi_clear(&pieces[i]);
        qi_clear(&survivors[i]);
    }
    free(pieces);
    free(survivors);
    qi_clear(&b);
    qi_clear(&y);
    qi_clear(&level);
    return 0;

target_cleanup:
    qi_clear(&y_hull);
    qp_clear(&B2);
    qp_clear(&scaled_A);
    qp_clear(&S);
    mpq_clear(scale);
    qi_clear(&Si);
    qi_clear(&Ai);
    qi_clear(&Bi);
    qi_clear(&Api);
    qi_clear(&Ni);
    qi_clear(&temp);
    qi_clear(&Q);
    qi_clear(&A2);
    qi_clear(&Lb);
    qi_clear(&Lb2);
    qi_clear(&y2);
    qi_clear(&four_y2);
    qi_clear(&norm);
failure:
    for (uint64_t i = 0; i < cap; ++i) {
        qi_clear(&pieces[i]);
        qi_clear(&survivors[i]);
    }
    free(pieces);
    free(survivors);
    qi_clear(&b);
    qi_clear(&y);
    qi_clear(&level);
    return -1;
}

static char *integer_string(const mpz_t value) {
    size_t digits = mpz_sizeinbase(value, 10);
    char *text = (char *)malloc(digits + 3);
    if (text != NULL)
        mpz_get_str(text, 10, value);
    return text;
}

static int export_q(spong_owned_rational *out, const mpq_t value) {
    out->numerator = integer_string(mpq_numref(value));
    out->denominator = integer_string(mpq_denref(value));
    return out->numerator == NULL || out->denominator == NULL ? -1 : 0;
}

static int export_interval(spong_owned_rational_interval *out, const qinterval *value) {
    return export_q(&out->lower, value->lo) != 0 ||
                   export_q(&out->upper, value->hi) != 0
               ? -1
               : 0;
}

void spong_b_parameter_result_destroy(spong_b_parameter_result *result) {
    if (result == NULL)
        return;
    free(result->target_b.lower.numerator);
    free(result->target_b.lower.denominator);
    free(result->target_b.upper.numerator);
    free(result->target_b.upper.denominator);
    free(result->target_y.lower.numerator);
    free(result->target_y.lower.denominator);
    free(result->target_y.upper.numerator);
    free(result->target_y.upper.denominator);
    free(result->minimum_face_margin.numerator);
    free(result->minimum_face_margin.denominator);
    free(result->minimum_gradient_norm_squared.numerator);
    free(result->minimum_gradient_norm_squared.denominator);
    memset(result, 0, sizeof(*result));
}

static void context_clear(context *ctx) {
    qp_clear(&ctx->A);
    qp_clear(&ctx->B);
    qp_clear(&ctx->Ap);
    qp_clear(&ctx->Bp);
    qp_clear(&ctx->N);
    mpq_clear(ctx->C);
}

int spong_b_parameter_handoff_decimal(
    const spong_exact_loss_pencil *pencil, const spong_b_parameter_centre *centres,
    size_t centre_count, const spong_rational_input *initial_y_lower,
    const spong_rational_input *initial_y_upper,
    const spong_rational_input *target_level, const spong_rational_input *max_radius,
    const spong_b_parameter_policy *policy, spong_b_parameter_result *result) {
    if (result == NULL)
        return -1;
    memset(result, 0, sizeof(*result));
    result->status = SPONG_SMALE_INVALID_ARGUMENT;
    result->primary_reason = SPONG_SMALE_REASON_BAD_POLICY;
    if (pencil == NULL || centres == NULL || centre_count < 2 ||
        initial_y_lower == NULL || initial_y_upper == NULL || target_level == NULL ||
        policy == NULL || pencil->alpha == NULL || pencil->alpha_count == 0 ||
        pencil->beta == NULL || pencil->beta_count == 0 || pencil->n == NULL ||
        pencil->n_count == 0 || policy->max_inflations == 0 ||
        policy->max_projection_subboxes == 0 || policy->max_rational_bits < 64 ||
        policy->radius_round_bits < 1 ||
        policy->radius_round_bits >= policy->max_rational_bits)
        return -1;

    context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.policy = policy;
    ctx.result = result;
    ctx.max_rational_bits = policy->max_rational_bits;
    ctx.input_rationals = &result->work.input_rationals;
    ctx.interval_evaluations = &result->work.interval_evaluations;
    ctx.peak_rational_bits = &result->work.peak_rational_bits;
    ctx.status = &result->status;
    ctx.primary_reason = &result->primary_reason;
    mpq_init(ctx.C);
    result->status = SPONG_SMALE_OK;
    result->primary_reason = SPONG_SMALE_REASON_NONE;
    if (qp_parse(&ctx, &ctx.A, pencil->alpha, pencil->alpha_count) != 0 ||
        qp_parse(&ctx, &ctx.B, pencil->beta, pencil->beta_count) != 0 ||
        qp_parse(&ctx, &ctx.N, pencil->n, pencil->n_count) != 0 ||
        parse_q(&ctx, &pencil->loss_constant, ctx.C) != 0 ||
        qp_derivative(&ctx, &ctx.Ap, &ctx.A) != 0 ||
        qp_derivative(&ctx, &ctx.Bp, &ctx.B) != 0)
        goto failure;

    if (policy->verify_n_identity) {
        qpolynomial ApB = {0}, BpA = {0}, twice = {0}, expected = {0};
        mpq_t two;
        mpq_init(two);
        mpq_set_ui(two, 2, 1);
        int identity = qp_mul(&ctx, &ApB, &ctx.Ap, &ctx.B) == 0 &&
                       qp_mul(&ctx, &BpA, &ctx.Bp, &ctx.A) == 0 &&
                       qp_scale(&ctx, &twice, &BpA, two) == 0 &&
                       qp_sub(&ctx, &expected, &ApB, &twice) == 0 &&
                       qp_equal(&expected, &ctx.N);
        qp_clear(&ApB);
        qp_clear(&BpA);
        qp_clear(&twice);
        qp_clear(&expected);
        mpq_clear(two);
        if (!identity) {
            result->status = SPONG_SMALE_MODEL_IDENTITY_FAILURE;
            result->primary_reason = SPONG_SMALE_REASON_N_IDENTITY;
            goto failure;
        }
    }

    qcentre *points = (qcentre *)calloc(centre_count, sizeof(qcentre));
    if (points == NULL) {
        result->status = SPONG_SMALE_ALLOCATION_FAILURE;
        result->primary_reason = SPONG_SMALE_REASON_ALLOCATION;
        goto failure;
    }
    for (size_t i = 0; i < centre_count; ++i)
        qc_init(&points[i]);
    size_t parsed_points = 0;
    for (; parsed_points < centre_count; ++parsed_points) {
        if (parse_q(&ctx, &centres[parsed_points].level, points[parsed_points].level) !=
                0 ||
            parse_q(&ctx, &centres[parsed_points].b, points[parsed_points].b) != 0 ||
            parse_q(&ctx, &centres[parsed_points].y, points[parsed_points].y) != 0)
            goto point_failure;
    }
    int direction = mpq_cmp(points[1].b, points[0].b);
    direction = direction > 0 ? 1 : direction < 0 ? -1 : 0;
    if (direction == 0) {
        result->status = SPONG_SMALE_PROPOSAL_NONMONOTONE;
        result->primary_reason = SPONG_SMALE_REASON_CENTRE_ORDER;
        goto point_failure;
    }
    for (size_t i = 1; i < centre_count; ++i) {
        int sign = mpq_cmp(points[i].b, points[i - 1].b);
        sign = sign > 0 ? 1 : sign < 0 ? -1 : 0;
        if (sign != direction) {
            result->status = SPONG_SMALE_PROPOSAL_NONMONOTONE;
            result->primary_reason = SPONG_SMALE_REASON_CENTRE_ORDER;
            goto point_failure;
        }
    }
    result->b_direction = direction;

    mpq_t initial_lo, initial_hi, target, radius, radius_cap;
    mpq_inits(initial_lo, initial_hi, target, radius, radius_cap, NULL);
    if (parse_q(&ctx, initial_y_lower, initial_lo) != 0 ||
        parse_q(&ctx, initial_y_upper, initial_hi) != 0 ||
        parse_q(&ctx, target_level, target) != 0 ||
        (max_radius != NULL && parse_q(&ctx, max_radius, radius_cap) != 0)) {
        mpq_clears(initial_lo, initial_hi, target, radius, radius_cap, NULL);
        goto point_failure;
    }
    if (mpq_cmp(initial_lo, initial_hi) >= 0 ||
        (max_radius != NULL && mpq_sgn(radius_cap) <= 0)) {
        result->status = SPONG_SMALE_INVALID_ARGUMENT;
        result->primary_reason = SPONG_SMALE_REASON_BAD_POLICY;
        mpq_clears(initial_lo, initial_hi, target, radius, radius_cap, NULL);
        goto point_failure;
    }
    mpq_add(points[0].y, initial_lo, initial_hi);
    mpq_div_2exp(points[0].y, points[0].y, 1);
    mpq_sub(radius, initial_hi, initial_lo);
    mpq_div_2exp(radius, radius, 1);
    qinterval left_y, left_b, left_level;
    qi_init(&left_y);
    qi_init(&left_b);
    qi_init(&left_level);
    mpq_set(left_y.lo, initial_lo);
    mpq_set(left_y.hi, initial_hi);
    qi_set_point(&left_b, points[0].b);
    ctx.failed = 0;
    if (level_box(&ctx, &left_level, &left_b, &left_y) != 0) {
        qi_clear(&left_y);
        qi_clear(&left_b);
        qi_clear(&left_level);
        mpq_clears(initial_lo, initial_hi, target, radius, radius_cap, NULL);
        goto point_failure;
    }
    qi_clear(&left_y);
    qi_clear(&left_b);

    target_slab slab;
    target_slab_init(&slab);
    march_state march;
    memset(&march, 0, sizeof(march));
    march.ctx = &ctx;
    march.target = target;
    march.radius_cap = max_radius == NULL ? NULL : radius_cap;
    march.target_slab = &slab;
    mpq_init(march.minimum_face);
    mpq_init(march.minimum_norm);
    int found = 0;
    for (size_t i = 0; i + 1 < centre_count && found == 0; ++i)
        found =
            march_segment(&march, &points[i], &points[i + 1], 0, radius, &left_level);
    if (found < 0)
        goto march_failure;
    if (found == 0 || !slab.initialized) {
        result->status = SPONG_SMALE_TARGET_UNBRACKETED;
        result->primary_reason = SPONG_SMALE_REASON_TARGET_LEVEL;
        goto march_failure;
    }
    result->tube_validated = 1;
    qinterval target_b_box, target_y_box;
    qi_init(&target_b_box);
    qi_init(&target_y_box);
    ctx.failed = 0;
    if (project_target(&ctx, &slab, target, &target_b_box, &target_y_box) != 0) {
        qi_clear(&target_b_box);
        qi_clear(&target_y_box);
        goto march_failure;
    }
    if (export_interval(&result->target_b, &target_b_box) != 0 ||
        export_interval(&result->target_y, &target_y_box) != 0 ||
        export_q(&result->minimum_face_margin, march.minimum_face) != 0 ||
        export_q(&result->minimum_gradient_norm_squared, march.minimum_norm) != 0) {
        qi_clear(&target_b_box);
        qi_clear(&target_y_box);
        result->status = SPONG_SMALE_ALLOCATION_FAILURE;
        result->primary_reason = SPONG_SMALE_REASON_ALLOCATION;
        goto march_failure;
    }
    qi_clear(&target_b_box);
    qi_clear(&target_y_box);
    result->projection_validated = 1;
    result->status = SPONG_SMALE_OK;
    result->primary_reason = SPONG_SMALE_REASON_NONE;

march_failure:
    mpq_clear(march.minimum_face);
    mpq_clear(march.minimum_norm);
    target_slab_clear(&slab);
    qi_clear(&left_level);
    mpq_clears(initial_lo, initial_hi, target, radius, radius_cap, NULL);
point_failure:
    for (size_t i = 0; i < centre_count; ++i)
        qc_clear(&points[i]);
    free(points);
failure:
    if (result->status != SPONG_SMALE_OK)
        result->projection_validated = 0;
    context_clear(&ctx);
    return result->status == SPONG_SMALE_OK ? 0 : -1;
}

/* Dense bivariate rational polynomials used only for the correlated
 * (level,b) quadrilateral of a fixed-sheet slab.  Degrees are small for the
 * SPONG loss pencil, and dense storage keeps the exact replay simple. */
typedef struct {
    size_t nx, ny;
    mpq_t *c;
} qpoly2;

static size_t q2_index(const qpoly2 *poly, size_t i, size_t j) {
    return i * poly->ny + j;
}

static int q2_init(qpoly2 *poly, size_t nx, size_t ny) {
    memset(poly, 0, sizeof(*poly));
    nx = nx == 0 ? 1 : nx;
    ny = ny == 0 ? 1 : ny;
    if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / sizeof(mpq_t))
        return -1;
    poly->c = (mpq_t *)calloc(nx * ny, sizeof(mpq_t));
    if (poly->c == NULL)
        return -1;
    poly->nx = nx;
    poly->ny = ny;
    for (size_t i = 0; i < nx * ny; ++i)
        mpq_init(poly->c[i]);
    return 0;
}

static void q2_clear(qpoly2 *poly) {
    if (poly->c != NULL) {
        for (size_t i = 0; i < poly->nx * poly->ny; ++i)
            mpq_clear(poly->c[i]);
        free(poly->c);
    }
    memset(poly, 0, sizeof(*poly));
}

static int q2_observe(context *ctx, const qpoly2 *poly) {
    for (size_t i = 0; i < poly->nx * poly->ny; ++i)
        if (observe(ctx, poly->c[i]) != 0)
            return -1;
    return 0;
}

static int q2_add(context *ctx, qpoly2 *out, const qpoly2 *a, const qpoly2 *b) {
    size_t nx = a->nx > b->nx ? a->nx : b->nx;
    size_t ny = a->ny > b->ny ? a->ny : b->ny;
    qpoly2 result = {0};
    if (q2_init(&result, nx, ny) != 0)
        return -1;
    for (size_t i = 0; i < nx; ++i)
        for (size_t j = 0; j < ny; ++j) {
            if (i < a->nx && j < a->ny)
                mpq_add(result.c[q2_index(&result, i, j)],
                        result.c[q2_index(&result, i, j)], a->c[q2_index(a, i, j)]);
            if (i < b->nx && j < b->ny)
                mpq_add(result.c[q2_index(&result, i, j)],
                        result.c[q2_index(&result, i, j)], b->c[q2_index(b, i, j)]);
        }
    if (q2_observe(ctx, &result) != 0) {
        q2_clear(&result);
        return -1;
    }
    *out = result;
    return 0;
}

static int q2_scale(context *ctx, qpoly2 *out, const qpoly2 *poly, const mpq_t scale) {
    qpoly2 result = {0};
    if (q2_init(&result, poly->nx, poly->ny) != 0)
        return -1;
    for (size_t i = 0; i < poly->nx * poly->ny; ++i)
        mpq_mul(result.c[i], poly->c[i], scale);
    if (q2_observe(ctx, &result) != 0) {
        q2_clear(&result);
        return -1;
    }
    *out = result;
    return 0;
}

static int q2_mul(context *ctx, qpoly2 *out, const qpoly2 *a, const qpoly2 *b) {
    if (a->nx > SIZE_MAX - b->nx + 1 || a->ny > SIZE_MAX - b->ny + 1)
        return -1;
    qpoly2 result = {0};
    if (q2_init(&result, a->nx + b->nx - 1, a->ny + b->ny - 1) != 0)
        return -1;
    mpq_t product;
    mpq_init(product);
    for (size_t i = 0; i < a->nx; ++i)
        for (size_t j = 0; j < a->ny; ++j)
            for (size_t k = 0; k < b->nx; ++k)
                for (size_t l = 0; l < b->ny; ++l) {
                    mpq_mul(product, a->c[q2_index(a, i, j)], b->c[q2_index(b, k, l)]);
                    mpq_add(result.c[q2_index(&result, i + k, j + l)],
                            result.c[q2_index(&result, i + k, j + l)], product);
                }
    mpq_clear(product);
    if (q2_observe(ctx, &result) != 0) {
        q2_clear(&result);
        return -1;
    }
    *out = result;
    return 0;
}

static int q2_compose_univariate(context *ctx, qpoly2 *out, const qpolynomial *poly,
                                 const qpoly2 *argument) {
    qpoly2 result = {0};
    if (q2_init(&result, 1, 1) != 0)
        return -1;
    for (size_t k = qp_length(poly); k-- > 0;) {
        qpoly2 product = {0}, constant = {0}, next = {0};
        if (q2_mul(ctx, &product, &result, argument) != 0 ||
            q2_init(&constant, 1, 1) != 0)
            goto loop_failure;
        mpq_set(constant.c[0], poly->c[k]);
        if (q2_add(ctx, &next, &product, &constant) != 0)
            goto loop_failure;
        q2_clear(&result);
        result = next;
        q2_clear(&product);
        q2_clear(&constant);
        continue;
    loop_failure:
        q2_clear(&product);
        q2_clear(&constant);
        q2_clear(&next);
        q2_clear(&result);
        return -1;
    }
    *out = result;
    return 0;
}

static int q2_interval(context *ctx, qinterval *out, const qpoly2 *poly,
                       const qinterval *x, const qinterval *y) {
    qinterval sum, term, power;
    qi_init(&sum);
    qi_init(&term);
    qi_init(&power);
    qi_set_si(&sum, 0);
    for (size_t i = 0; i < poly->nx; ++i)
        for (size_t j = 0; j < poly->ny; ++j) {
            qi_set_point(&term, poly->c[q2_index(poly, i, j)]);
            for (size_t k = 0; k < i; ++k) {
                qi_set(&power, &term);
                if (qi_mul(ctx, &term, &power, x) != 0)
                    goto failure;
            }
            for (size_t k = 0; k < j; ++k) {
                qi_set(&power, &term);
                if (qi_mul(ctx, &term, &power, y) != 0)
                    goto failure;
            }
            qi_set(&power, &sum);
            if (qi_add(ctx, &sum, &power, &term) != 0)
                goto failure;
        }
    qi_set(out, &sum);
    (*ctx->interval_evaluations)++;
    qi_clear(&sum);
    qi_clear(&term);
    qi_clear(&power);
    return 0;
failure:
    qi_clear(&sum);
    qi_clear(&term);
    qi_clear(&power);
    return -1;
}

static int q2_affine_in_first(context *ctx, qpoly2 *out, const mpq_t first,
                              const mpq_t second) {
    qpoly2 value = {0};
    if (q2_init(&value, 2, 1) != 0)
        return -1;
    mpq_add(value.c[q2_index(&value, 0, 0)], first, second);
    mpq_div_2exp(value.c[q2_index(&value, 0, 0)], value.c[q2_index(&value, 0, 0)], 1);
    mpq_sub(value.c[q2_index(&value, 1, 0)], second, first);
    if (q2_observe(ctx, &value) != 0) {
        q2_clear(&value);
        return -1;
    }
    *out = value;
    return 0;
}

static int sheet_domain_interval(context *ctx, const mpq_t level0, const mpq_t level1,
                                 const mpq_t lower_b0, const mpq_t lower_b1,
                                 const mpq_t upper_b0, const mpq_t upper_b1,
                                 qinterval *S_out, qinterval *A_out) {
    qpoly2 level = {0}, lower = {0}, upper = {0}, difference = {0};
    qpoly2 r = {0}, product = {0}, b = {0}, A = {0}, B = {0};
    qpoly2 constant = {0}, ell_minus = {0}, ellA = {0}, B2 = {0}, S = {0};
    qinterval box;
    qi_init(&box);
    mpq_set_si(box.lo, -1, 2);
    mpq_set_si(box.hi, 1, 2);
    mpq_t minus_one;
    mpq_init(minus_one);
    mpq_set_si(minus_one, -1, 1);
    int status = -1;
    if (q2_affine_in_first(ctx, &level, level0, level1) != 0 ||
        q2_affine_in_first(ctx, &lower, lower_b0, lower_b1) != 0 ||
        q2_affine_in_first(ctx, &upper, upper_b0, upper_b1) != 0 ||
        q2_scale(ctx, &difference, &lower, minus_one) != 0)
        goto done;
    {
        qpoly2 next = {0};
        if (q2_add(ctx, &next, &upper, &difference) != 0)
            goto done;
        q2_clear(&difference);
        difference = next;
    }
    if (q2_init(&r, 1, 2) != 0)
        goto done;
    mpq_set_si(r.c[q2_index(&r, 0, 0)], 1, 2);
    mpq_set_ui(r.c[q2_index(&r, 0, 1)], 1, 1);
    if (q2_mul(ctx, &product, &r, &difference) != 0 ||
        q2_add(ctx, &b, &lower, &product) != 0 ||
        q2_compose_univariate(ctx, &A, &ctx->A, &b) != 0 ||
        q2_compose_univariate(ctx, &B, &ctx->B, &b) != 0 ||
        q2_init(&constant, 1, 1) != 0)
        goto done;
    mpq_neg(constant.c[0], ctx->C);
    if (q2_add(ctx, &ell_minus, &level, &constant) != 0 ||
        q2_mul(ctx, &ellA, &ell_minus, &A) != 0 || q2_mul(ctx, &B2, &B, &B) != 0 ||
        q2_add(ctx, &S, &B2, &ellA) != 0 ||
        q2_interval(ctx, S_out, &S, &box, &box) != 0 ||
        q2_interval(ctx, A_out, &A, &box, &box) != 0)
        goto done;
    status = 0;
done:
    q2_clear(&level);
    q2_clear(&lower);
    q2_clear(&upper);
    q2_clear(&difference);
    q2_clear(&r);
    q2_clear(&product);
    q2_clear(&b);
    q2_clear(&A);
    q2_clear(&B);
    q2_clear(&constant);
    q2_clear(&ell_minus);
    q2_clear(&ellA);
    q2_clear(&B2);
    q2_clear(&S);
    qi_clear(&box);
    mpq_clear(minus_one);
    return status;
}

typedef struct {
    qinterval db_dlevel;
    qinterval y;
    qinterval norm2;
} sheet_face_result;

static void sheet_face_init(sheet_face_result *face) {
    qi_init(&face->db_dlevel);
    qi_init(&face->y);
    qi_init(&face->norm2);
}

static void sheet_face_clear(sheet_face_result *face) {
    qi_clear(&face->db_dlevel);
    qi_clear(&face->y);
    qi_clear(&face->norm2);
}

static int sheet_face_transport(context *ctx, const mpq_t level0, const mpq_t level1,
                                const mpq_t b0, const mpq_t b1, int sheet,
                                uint64_t sqrt_bits, sheet_face_result *out) {
    mpq_t db, dl;
    mpq_inits(db, dl, NULL);
    mpq_sub(db, b1, b0);
    mpq_sub(dl, level1, level0);
    qpolynomial A = {0}, B = {0}, Ap = {0}, N = {0}, ell = {0};
    qpolynomial ell_minus = {0}, B2 = {0}, ellA = {0}, S = {0};
    qinterval unit, Ai, Bi, Api, Ni, Si, root, temp, q, A2, loss_b;
    qinterval ys, four_ys, lbs;
    qi_init(&unit);
    qi_init(&Ai);
    qi_init(&Bi);
    qi_init(&Api);
    qi_init(&Ni);
    qi_init(&Si);
    qi_init(&root);
    qi_init(&temp);
    qi_init(&q);
    qi_init(&A2);
    qi_init(&loss_b);
    qi_init(&ys);
    qi_init(&four_ys);
    qi_init(&lbs);
    mpq_set_ui(unit.lo, 0, 1);
    mpq_set_ui(unit.hi, 1, 1);
    int status = -1;
    if (qp_affine_compose(ctx, &A, &ctx->A, b0, db) != 0 ||
        qp_affine_compose(ctx, &B, &ctx->B, b0, db) != 0 ||
        qp_affine_compose(ctx, &Ap, &ctx->Ap, b0, db) != 0 ||
        qp_affine_compose(ctx, &N, &ctx->N, b0, db) != 0 || qp_init(&ell, 2) != 0)
        goto done;
    mpq_set(ell.c[0], level0);
    mpq_set(ell.c[1], dl);
    {
        qpolynomial constant = {0};
        if (qp_init(&constant, 1) != 0)
            goto done;
        mpq_set(constant.c[0], ctx->C);
        int built = qp_sub(ctx, &ell_minus, &ell, &constant);
        qp_clear(&constant);
        if (built != 0)
            goto done;
    }
    if (qp_mul(ctx, &B2, &B, &B) != 0 || qp_mul(ctx, &ellA, &ell_minus, &A) != 0 ||
        qp_add(ctx, &S, &B2, &ellA) != 0 ||
        qp_centered_interval(ctx, &Si, &S, &unit) != 0 ||
        qp_centered_interval(ctx, &Ai, &A, &unit) != 0 ||
        qp_centered_interval(ctx, &Bi, &B, &unit) != 0 ||
        qp_centered_interval(ctx, &Api, &Ap, &unit) != 0 ||
        qp_centered_interval(ctx, &Ni, &N, &unit) != 0)
        goto done;
    if (mpq_sgn(Si.lo) <= 0) {
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    SPONG_SMALE_REASON_FIXED_SHEET_DOMAIN);
        goto done;
    }
    if (qi_contains_zero(&Ai)) {
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    SPONG_SMALE_REASON_TRANSVERSE_DIVISOR);
        goto done;
    }
    if (sqrt_interval(ctx, &root, &Si, sqrt_bits) != 0)
        goto done;
    if (sheet > 0) {
        qi_set(&out->y, &root);
    } else {
        mpq_neg(out->y.lo, root.hi);
        mpq_neg(out->y.hi, root.lo);
    }
    if (qi_mul(ctx, &temp, &Api, &out->y) != 0 || qi_add(ctx, &temp, &Ni, &temp) != 0 ||
        qi_add(ctx, &q, &Bi, &out->y) != 0 || qi_mul(ctx, &q, &q, &temp) != 0 ||
        qi_square(ctx, &A2, &Ai) != 0 || qi_div(ctx, &loss_b, &q, &A2) != 0 ||
        qi_square(ctx, &ys, &out->y) != 0 || qi_scale_si(ctx, &four_ys, &ys, 4) != 0 ||
        qi_square(ctx, &lbs, &loss_b) != 0 ||
        qi_add(ctx, &out->norm2, &four_ys, &lbs) != 0)
        goto done;
    if (qi_contains_zero(&out->norm2)) {
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    SPONG_SMALE_REASON_CRITICAL_POINT);
        goto done;
    }
    if (qi_div(ctx, &out->db_dlevel, &loss_b, &out->norm2) != 0)
        goto done;
    status = 0;
done:
    if (status != 0 && !ctx->failed)
        set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE, SPONG_SMALE_REASON_ALLOCATION);
    qp_clear(&A);
    qp_clear(&B);
    qp_clear(&Ap);
    qp_clear(&N);
    qp_clear(&ell);
    qp_clear(&ell_minus);
    qp_clear(&B2);
    qp_clear(&ellA);
    qp_clear(&S);
    qi_clear(&unit);
    qi_clear(&Ai);
    qi_clear(&Bi);
    qi_clear(&Api);
    qi_clear(&Ni);
    qi_clear(&Si);
    qi_clear(&root);
    qi_clear(&temp);
    qi_clear(&q);
    qi_clear(&A2);
    qi_clear(&loss_b);
    qi_clear(&ys);
    qi_clear(&four_ys);
    qi_clear(&lbs);
    mpq_clears(db, dl, NULL);
    return status;
}

static int intervals_overlap(const qinterval *a, const qinterval *b) {
    return mpq_cmp(a->hi, b->lo) >= 0 && mpq_cmp(b->hi, a->lo) >= 0;
}

static void interval_array_clear(qinterval *values, size_t count) {
    if (values != NULL) {
        for (size_t i = 0; i < count; ++i)
            qi_clear(&values[i]);
        free(values);
    }
}

static qinterval *interval_array_alloc(size_t count) {
    if (count == 0 || count > SIZE_MAX / sizeof(qinterval))
        return NULL;
    qinterval *values = (qinterval *)calloc(count, sizeof(qinterval));
    if (values == NULL)
        return NULL;
    for (size_t i = 0; i < count; ++i)
        qi_init(&values[i]);
    return values;
}

static int project_initial_sheet(context *ctx, const spong_sheet_tube_policy *policy,
                                 spong_sheet_tube_result *result, const mpq_t level,
                                 const qinterval *input_b, const qinterval *input_y,
                                 int sheet, qinterval *projected_b,
                                 qinterval *projected_y) {
    if ((sheet > 0 && mpq_sgn(input_y->lo) <= 0) ||
        (sheet < 0 && mpq_sgn(input_y->hi) >= 0)) {
        set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                    SPONG_SMALE_REASON_TARGET_SHEET);
        return -1;
    }
    qinterval y2;
    qi_init(&y2);
    if (qi_square(ctx, &y2, input_y) != 0) {
        qi_clear(&y2);
        return -1;
    }
    qpolynomial B2 = {0}, ell_minus = {0}, ellA = {0}, S = {0};
    mpq_t shift;
    mpq_init(shift);
    mpq_sub(shift, level, ctx->C);
    if (qp_mul(ctx, &B2, &ctx->B, &ctx->B) != 0 ||
        qp_scale(ctx, &ellA, &ctx->A, shift) != 0 || qp_add(ctx, &S, &B2, &ellA) != 0) {
        if (!ctx->failed)
            set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                        SPONG_SMALE_REASON_ALLOCATION);
        goto failure;
    }
    qinterval *pieces = interval_array_alloc(1);
    size_t piece_count = 1;
    if (pieces == NULL) {
        set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE, SPONG_SMALE_REASON_ALLOCATION);
        goto failure;
    }
    qi_set(&pieces[0], input_b);
    for (uint64_t depth = 0; depth <= policy->max_projection_bisections; ++depth) {
        qinterval *survivors = interval_array_alloc(piece_count);
        if (survivors == NULL) {
            set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                        SPONG_SMALE_REASON_ALLOCATION);
            interval_array_clear(pieces, piece_count);
            goto failure;
        }
        size_t survivor_count = 0;
        for (size_t i = 0; i < piece_count; ++i) {
            qinterval range;
            qi_init(&range);
            int evaluated = qp_centered_interval(ctx, &range, &S, &pieces[i]);
            if (evaluated == 0 && intervals_overlap(&range, &y2)) {
                qi_set(&survivors[survivor_count], &pieces[i]);
                survivor_count++;
            }
            qi_clear(&range);
            if (evaluated != 0) {
                interval_array_clear(survivors, piece_count);
                interval_array_clear(pieces, piece_count);
                goto failure;
            }
        }
        interval_array_clear(pieces, piece_count);
        pieces = NULL;
        if (survivor_count == 0) {
            interval_array_clear(survivors, piece_count);
            set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                        SPONG_SMALE_REASON_INITIAL_PROJECTION);
            goto failure;
        }
        mpq_set(projected_b->lo, survivors[0].lo);
        mpq_set(projected_b->hi, survivors[0].hi);
        for (size_t i = 1; i < survivor_count; ++i) {
            if (mpq_cmp(survivors[i].lo, projected_b->lo) < 0)
                mpq_set(projected_b->lo, survivors[i].lo);
            if (mpq_cmp(survivors[i].hi, projected_b->hi) > 0)
                mpq_set(projected_b->hi, survivors[i].hi);
        }
        sheet_face_result field;
        sheet_face_init(&field);
        ctx->failed = 0;
        int regular =
            sheet_face_transport(ctx, level, level, projected_b->lo, projected_b->hi,
                                 sheet, policy->sqrt_bits, &field);
        if (regular == 0) {
            qi_set(projected_y, &field.y);
            sheet_face_clear(&field);
            result->work.projection_subboxes = survivor_count;
            result->work.projection_depth = depth;
            interval_array_clear(survivors, piece_count);
            result->projection_validated = 1;
            qp_clear(&B2);
            qp_clear(&ell_minus);
            qp_clear(&ellA);
            qp_clear(&S);
            mpq_clear(shift);
            qi_clear(&y2);
            return 0;
        }
        sheet_face_clear(&field);
        if (hard_failure(ctx)) {
            interval_array_clear(survivors, piece_count);
            goto failure;
        }
        clear_local_refusal(ctx);
        if (depth == policy->max_projection_bisections) {
            interval_array_clear(survivors, piece_count);
            set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                        SPONG_SMALE_REASON_INITIAL_PROJECTION);
            goto failure;
        }
        if (survivor_count > policy->max_projection_subboxes / 2) {
            interval_array_clear(survivors, piece_count);
            set_refusal(ctx, SPONG_SMALE_PROJECTION_UNRESOLVED,
                        SPONG_SMALE_REASON_PROJECTION_BUDGET);
            goto failure;
        }
        pieces = interval_array_alloc(2 * survivor_count);
        if (pieces == NULL) {
            interval_array_clear(survivors, piece_count);
            set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                        SPONG_SMALE_REASON_ALLOCATION);
            goto failure;
        }
        mpq_t middle;
        mpq_init(middle);
        for (size_t i = 0; i < survivor_count; ++i) {
            mpq_add(middle, survivors[i].lo, survivors[i].hi);
            mpq_div_2exp(middle, middle, 1);
            mpq_set(pieces[2 * i].lo, survivors[i].lo);
            mpq_set(pieces[2 * i].hi, middle);
            mpq_set(pieces[2 * i + 1].lo, middle);
            mpq_set(pieces[2 * i + 1].hi, survivors[i].hi);
        }
        mpq_clear(middle);
        interval_array_clear(survivors, piece_count);
        piece_count = 2 * survivor_count;
    }
failure:
    qp_clear(&B2);
    qp_clear(&ell_minus);
    qp_clear(&ellA);
    qp_clear(&S);
    mpq_clear(shift);
    qi_clear(&y2);
    return -1;
}

typedef struct {
    context *ctx;
    const spong_sheet_tube_policy *policy;
    spong_sheet_tube_result *result;
    mpq_srcptr radius_cap;
    mpq_t minimum_face;
    mpq_t minimum_norm;
    int has_face;
    int has_norm;
} sheet_march_state;

static int sheet_march_segment(sheet_march_state *state, const qcentre *left,
                               const qcentre *right, uint64_t depth, mpq_t radius,
                               qinterval *terminal_y) {
    context *ctx = state->ctx;
    mpq_t h, slope, next_radius, lower0, lower1, upper0, upper1;
    mpq_t lower_slope, upper_slope, lower_margin, upper_margin;
    mpq_t growth, candidate, required, temp, factor;
    mpq_inits(h, slope, next_radius, lower0, lower1, upper0, upper1, lower_slope,
              upper_slope, lower_margin, upper_margin, growth, candidate, required,
              temp, factor, NULL);
    mpq_sub(h, right->level, left->level);
    if (state->result->level_direction < 0)
        mpq_neg(h, h);
    mpq_sub(slope, right->b, left->b);
    mpq_div(slope, slope, h);
    mpq_set(next_radius, radius);
    qinterval interior_S, interior_A, lower_velocity, upper_velocity;
    qi_init(&interior_S);
    qi_init(&interior_A);
    qi_init(&lower_velocity);
    qi_init(&upper_velocity);
    sheet_face_result lower, upper, endpoint;
    sheet_face_init(&lower);
    sheet_face_init(&upper);
    sheet_face_init(&endpoint);
    int accepted = 0;
    int failure_reason = SPONG_SMALE_REASON_FACE_INEQUALITY;

    for (uint64_t inflation = 0; inflation < state->policy->max_inflations;
         ++inflation) {
        state->result->work.inflation_steps++;
        if (observe(ctx, left->level) != 0 || observe(ctx, right->level) != 0 ||
            observe(ctx, left->b) != 0 || observe(ctx, right->b) != 0 ||
            observe(ctx, radius) != 0 || observe(ctx, next_radius) != 0)
            goto refusal;
        mpq_sub(lower0, left->b, radius);
        mpq_sub(lower1, right->b, next_radius);
        mpq_add(upper0, left->b, radius);
        mpq_add(upper1, right->b, next_radius);

        ctx->failed = 0;
        if (sheet_domain_interval(ctx, left->level, right->level, lower0, lower1,
                                  upper0, upper1, &interior_S, &interior_A) != 0) {
            if (!ctx->failed)
                set_refusal(ctx, SPONG_SMALE_ALLOCATION_FAILURE,
                            SPONG_SMALE_REASON_ALLOCATION);
            goto refusal;
        }
        if (mpq_sgn(interior_S.lo) <= 0) {
            failure_reason = SPONG_SMALE_REASON_FIXED_SHEET_DOMAIN;
            break;
        }
        if (qi_contains_zero(&interior_A)) {
            failure_reason = SPONG_SMALE_REASON_TRANSVERSE_DIVISOR;
            break;
        }
        if (sheet_face_transport(ctx, left->level, right->level, lower0, lower1,
                                 state->result->sheet, state->policy->sqrt_bits,
                                 &lower) != 0) {
            if (hard_failure(ctx))
                goto refusal;
            failure_reason = *ctx->primary_reason;
            clear_local_refusal(ctx);
            break;
        }
        if (sheet_face_transport(ctx, left->level, right->level, upper0, upper1,
                                 state->result->sheet, state->policy->sqrt_bits,
                                 &upper) != 0) {
            if (hard_failure(ctx))
                goto refusal;
            failure_reason = *ctx->primary_reason;
            clear_local_refusal(ctx);
            break;
        }
        if (state->result->level_direction > 0) {
            qi_set(&lower_velocity, &lower.db_dlevel);
            qi_set(&upper_velocity, &upper.db_dlevel);
        } else {
            mpq_neg(lower_velocity.lo, lower.db_dlevel.hi);
            mpq_neg(lower_velocity.hi, lower.db_dlevel.lo);
            mpq_neg(upper_velocity.lo, upper.db_dlevel.hi);
            mpq_neg(upper_velocity.hi, upper.db_dlevel.lo);
        }
        mpq_sub(temp, next_radius, radius);
        mpq_div(temp, temp, h);
        mpq_sub(lower_slope, slope, temp);
        mpq_add(upper_slope, slope, temp);
        mpq_sub(lower_margin, lower_velocity.lo, lower_slope);
        mpq_sub(upper_margin, upper_slope, upper_velocity.hi);
        mpq_sub(growth, slope, lower_velocity.lo);
        if (mpq_sgn(growth) < 0)
            mpq_set_ui(growth, 0, 1);
        mpq_sub(temp, upper_velocity.hi, slope);
        qmax_set(growth, temp);
        mpq_set_ui(factor, 65, 64);
        mpq_mul(candidate, h, growth);
        mpq_mul(candidate, candidate, factor);
        mpq_add(candidate, candidate, radius);
        if (ceil_dyadic_bits(ctx, required, candidate,
                             state->policy->radius_round_bits) != 0)
            goto refusal;
        if (mpq_cmp(required, next_radius) < 0)
            mpq_set(required, next_radius);
        if (mpq_sgn(lower_margin) >= 0 && mpq_sgn(upper_margin) >= 0) {
            accepted = 1;
            update_minimum(state->minimum_face, &state->has_face, lower_margin);
            update_minimum(state->minimum_face, &state->has_face, upper_margin);
            mpq_mul_2exp(temp, interior_S.lo, 2);
            update_minimum(state->minimum_norm, &state->has_norm, temp);
            update_minimum(state->minimum_norm, &state->has_norm, lower.norm2.lo);
            update_minimum(state->minimum_norm, &state->has_norm, upper.norm2.lo);
            break;
        }
        if (mpq_cmp(required, next_radius) == 0)
            break;
        mpq_set(next_radius, required);
        if (state->radius_cap != NULL && mpq_cmp(next_radius, state->radius_cap) > 0) {
            failure_reason = SPONG_SMALE_REASON_RADIUS_CAP;
            break;
        }
    }

    if (!accepted) {
        if (depth < state->policy->max_slab_bisections) {
            qcentre middle;
            qc_init(&middle);
            mpq_add(middle.level, left->level, right->level);
            mpq_div_2exp(middle.level, middle.level, 1);
            mpq_add(middle.b, left->b, right->b);
            mpq_div_2exp(middle.b, middle.b, 1);
            mpq_add(middle.y, left->y, right->y);
            mpq_div_2exp(middle.y, middle.y, 1);
            state->result->work.slab_bisections++;
            if (sheet_march_segment(state, left, &middle, depth + 1, radius,
                                    terminal_y) != 0 ||
                sheet_march_segment(state, &middle, right, depth + 1, radius,
                                    terminal_y) != 0) {
                qc_clear(&middle);
                goto refusal;
            }
            qc_clear(&middle);
            accepted = 1;
            goto done;
        }
        set_refusal(ctx, SPONG_SMALE_TUBE_UNRESOLVED,
                    failure_reason == SPONG_SMALE_REASON_FACE_INEQUALITY
                        ? SPONG_SMALE_REASON_SLAB_BUDGET
                        : failure_reason);
        goto refusal;
    }

    mpq_set(radius, next_radius);
    mpq_sub(lower0, right->b, radius);
    mpq_add(upper0, right->b, radius);
    ctx->failed = 0;
    if (sheet_face_transport(ctx, right->level, right->level, lower0, upper0,
                             state->result->sheet, state->policy->sqrt_bits,
                             &endpoint) != 0)
        goto refusal;
    qi_set(terminal_y, &endpoint.y);
    state->result->work.slabs_accepted++;
    accepted = 1;
    goto done;

refusal:
    accepted = -1;
done:
    sheet_face_clear(&lower);
    sheet_face_clear(&upper);
    sheet_face_clear(&endpoint);
    qi_clear(&interior_S);
    qi_clear(&interior_A);
    qi_clear(&lower_velocity);
    qi_clear(&upper_velocity);
    mpq_clears(h, slope, next_radius, lower0, lower1, upper0, upper1, lower_slope,
               upper_slope, lower_margin, upper_margin, growth, candidate, required,
               temp, factor, NULL);
    return accepted < 0 ? -1 : 0;
}

void spong_sheet_tube_result_destroy(spong_sheet_tube_result *result) {
    if (result == NULL)
        return;
    free(result->terminal_b.lower.numerator);
    free(result->terminal_b.lower.denominator);
    free(result->terminal_b.upper.numerator);
    free(result->terminal_b.upper.denominator);
    free(result->terminal_y.lower.numerator);
    free(result->terminal_y.lower.denominator);
    free(result->terminal_y.upper.numerator);
    free(result->terminal_y.upper.denominator);
    free(result->minimum_face_margin.numerator);
    free(result->minimum_face_margin.denominator);
    free(result->minimum_gradient_norm_squared.numerator);
    free(result->minimum_gradient_norm_squared.denominator);
    memset(result, 0, sizeof(*result));
}

int spong_sheet_flow_tube_decimal(const spong_exact_loss_pencil *pencil,
                                  const spong_b_parameter_centre *centres,
                                  size_t centre_count,
                                  const spong_rational_input *initial_b_lower,
                                  const spong_rational_input *initial_b_upper,
                                  const spong_rational_input *initial_y_lower,
                                  const spong_rational_input *initial_y_upper,
                                  int32_t sheet, const spong_rational_input *max_radius,
                                  const spong_sheet_tube_policy *policy,
                                  spong_sheet_tube_result *result) {
    if (result == NULL)
        return -1;
    memset(result, 0, sizeof(*result));
    result->status = SPONG_SMALE_INVALID_ARGUMENT;
    result->primary_reason = SPONG_SMALE_REASON_BAD_POLICY;
    if (pencil == NULL || centres == NULL || centre_count < 2 ||
        initial_b_lower == NULL || initial_b_upper == NULL || initial_y_lower == NULL ||
        initial_y_upper == NULL || policy == NULL || (sheet != -1 && sheet != 1) ||
        pencil->alpha == NULL || pencil->alpha_count == 0 || pencil->beta == NULL ||
        pencil->beta_count == 0 || pencil->n == NULL || pencil->n_count == 0 ||
        policy->max_inflations == 0 || policy->max_projection_subboxes == 0 ||
        policy->max_rational_bits < 64 || policy->sqrt_bits == 0 ||
        policy->radius_round_bits < 1 ||
        policy->radius_round_bits >= policy->max_rational_bits)
        return -1;

    context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.max_rational_bits = policy->max_rational_bits;
    ctx.input_rationals = &result->work.input_rationals;
    ctx.interval_evaluations = &result->work.interval_evaluations;
    ctx.peak_rational_bits = &result->work.peak_rational_bits;
    ctx.status = &result->status;
    ctx.primary_reason = &result->primary_reason;
    mpq_init(ctx.C);
    result->status = SPONG_SMALE_OK;
    result->primary_reason = SPONG_SMALE_REASON_NONE;
    result->sheet = sheet;
    if (qp_parse(&ctx, &ctx.A, pencil->alpha, pencil->alpha_count) != 0 ||
        qp_parse(&ctx, &ctx.B, pencil->beta, pencil->beta_count) != 0 ||
        qp_parse(&ctx, &ctx.N, pencil->n, pencil->n_count) != 0 ||
        parse_q(&ctx, &pencil->loss_constant, ctx.C) != 0 ||
        qp_derivative(&ctx, &ctx.Ap, &ctx.A) != 0 ||
        qp_derivative(&ctx, &ctx.Bp, &ctx.B) != 0)
        goto failure;

    if (policy->verify_n_identity) {
        qpolynomial ApB = {0}, BpA = {0}, twice = {0}, expected = {0};
        mpq_t two;
        mpq_init(two);
        mpq_set_ui(two, 2, 1);
        int identity = qp_mul(&ctx, &ApB, &ctx.Ap, &ctx.B) == 0 &&
                       qp_mul(&ctx, &BpA, &ctx.Bp, &ctx.A) == 0 &&
                       qp_scale(&ctx, &twice, &BpA, two) == 0 &&
                       qp_sub(&ctx, &expected, &ApB, &twice) == 0 &&
                       qp_equal(&expected, &ctx.N);
        qp_clear(&ApB);
        qp_clear(&BpA);
        qp_clear(&twice);
        qp_clear(&expected);
        mpq_clear(two);
        if (!identity) {
            result->status = SPONG_SMALE_MODEL_IDENTITY_FAILURE;
            result->primary_reason = SPONG_SMALE_REASON_N_IDENTITY;
            goto failure;
        }
    }

    qcentre *points = (qcentre *)calloc(centre_count, sizeof(qcentre));
    if (points == NULL) {
        result->status = SPONG_SMALE_ALLOCATION_FAILURE;
        result->primary_reason = SPONG_SMALE_REASON_ALLOCATION;
        goto failure;
    }
    for (size_t i = 0; i < centre_count; ++i)
        qc_init(&points[i]);
    for (size_t i = 0; i < centre_count; ++i) {
        if (parse_q(&ctx, &centres[i].level, points[i].level) != 0 ||
            parse_q(&ctx, &centres[i].b, points[i].b) != 0 ||
            parse_q(&ctx, &centres[i].y, points[i].y) != 0)
            goto point_failure;
    }
    int direction = mpq_cmp(points[1].level, points[0].level);
    direction = direction > 0 ? 1 : direction < 0 ? -1 : 0;
    if (direction == 0) {
        result->status = SPONG_SMALE_PROPOSAL_NONMONOTONE;
        result->primary_reason = SPONG_SMALE_REASON_CENTRE_ORDER;
        goto point_failure;
    }
    for (size_t i = 1; i < centre_count; ++i) {
        int sign = mpq_cmp(points[i].level, points[i - 1].level);
        sign = sign > 0 ? 1 : sign < 0 ? -1 : 0;
        if (sign != direction) {
            result->status = SPONG_SMALE_PROPOSAL_NONMONOTONE;
            result->primary_reason = SPONG_SMALE_REASON_CENTRE_ORDER;
            goto point_failure;
        }
    }
    result->level_direction = direction;

    mpq_t b_lo, b_hi, y_lo, y_hi, radius, radius_cap;
    mpq_inits(b_lo, b_hi, y_lo, y_hi, radius, radius_cap, NULL);
    if (parse_q(&ctx, initial_b_lower, b_lo) != 0 ||
        parse_q(&ctx, initial_b_upper, b_hi) != 0 ||
        parse_q(&ctx, initial_y_lower, y_lo) != 0 ||
        parse_q(&ctx, initial_y_upper, y_hi) != 0 ||
        (max_radius != NULL && parse_q(&ctx, max_radius, radius_cap) != 0)) {
        mpq_clears(b_lo, b_hi, y_lo, y_hi, radius, radius_cap, NULL);
        goto point_failure;
    }
    if (mpq_cmp(b_lo, b_hi) >= 0 || mpq_cmp(y_lo, y_hi) >= 0 ||
        (max_radius != NULL && mpq_sgn(radius_cap) <= 0)) {
        result->status = SPONG_SMALE_INVALID_ARGUMENT;
        result->primary_reason = SPONG_SMALE_REASON_BAD_POLICY;
        mpq_clears(b_lo, b_hi, y_lo, y_hi, radius, radius_cap, NULL);
        goto point_failure;
    }
    qinterval initial_b, initial_y, projected_b, terminal_y;
    qi_init(&initial_b);
    qi_init(&initial_y);
    qi_init(&projected_b);
    qi_init(&terminal_y);
    mpq_set(initial_b.lo, b_lo);
    mpq_set(initial_b.hi, b_hi);
    mpq_set(initial_y.lo, y_lo);
    mpq_set(initial_y.hi, y_hi);
    ctx.failed = 0;
    if (project_initial_sheet(&ctx, policy, result, points[0].level, &initial_b,
                              &initial_y, sheet, &projected_b, &terminal_y) != 0)
        goto march_failure;
    mpq_add(points[0].b, projected_b.lo, projected_b.hi);
    mpq_div_2exp(points[0].b, points[0].b, 1);
    mpq_sub(radius, projected_b.hi, projected_b.lo);
    mpq_div_2exp(radius, radius, 1);

    sheet_march_state march;
    memset(&march, 0, sizeof(march));
    march.ctx = &ctx;
    march.policy = policy;
    march.result = result;
    march.radius_cap = max_radius == NULL ? NULL : radius_cap;
    mpq_init(march.minimum_face);
    mpq_init(march.minimum_norm);
    for (size_t i = 0; i + 1 < centre_count; ++i)
        if (sheet_march_segment(&march, &points[i], &points[i + 1], 0, radius,
                                &terminal_y) != 0)
            goto tube_failure;
    result->tube_validated = 1;
    qinterval terminal_b;
    qi_init(&terminal_b);
    mpq_sub(terminal_b.lo, points[centre_count - 1].b, radius);
    mpq_add(terminal_b.hi, points[centre_count - 1].b, radius);
    if (export_interval(&result->terminal_b, &terminal_b) != 0 ||
        export_interval(&result->terminal_y, &terminal_y) != 0 ||
        export_q(&result->minimum_face_margin, march.minimum_face) != 0 ||
        export_q(&result->minimum_gradient_norm_squared, march.minimum_norm) != 0) {
        result->status = SPONG_SMALE_ALLOCATION_FAILURE;
        result->primary_reason = SPONG_SMALE_REASON_ALLOCATION;
    }
    qi_clear(&terminal_b);

tube_failure:
    mpq_clear(march.minimum_face);
    mpq_clear(march.minimum_norm);
march_failure:
    qi_clear(&initial_b);
    qi_clear(&initial_y);
    qi_clear(&projected_b);
    qi_clear(&terminal_y);
    mpq_clears(b_lo, b_hi, y_lo, y_hi, radius, radius_cap, NULL);
point_failure:
    for (size_t i = 0; i < centre_count; ++i)
        qc_clear(&points[i]);
    free(points);
failure:
    if (result->status != SPONG_SMALE_OK)
        result->tube_validated = 0;
    context_clear(&ctx);
    return result->status == SPONG_SMALE_OK ? 0 : -1;
}

const char *spong_smale_status_name(int32_t status) {
    switch (status) {
    case SPONG_SMALE_OK:
        return "validated";
    case SPONG_SMALE_INVALID_ARGUMENT:
        return "invalid_argument";
    case SPONG_SMALE_PARSE_FAILURE:
        return "parse_failure";
    case SPONG_SMALE_ALLOCATION_FAILURE:
        return "allocation_failure";
    case SPONG_SMALE_WORK_LIMIT:
        return "work_limit";
    case SPONG_SMALE_MODEL_IDENTITY_FAILURE:
        return "model_identity_failure";
    case SPONG_SMALE_PROPOSAL_NONMONOTONE:
        return "proposal_nonmonotone";
    case SPONG_SMALE_TUBE_UNRESOLVED:
        return "tube_unresolved";
    case SPONG_SMALE_TARGET_UNBRACKETED:
        return "target_unbracketed";
    case SPONG_SMALE_PROJECTION_UNRESOLVED:
        return "projection_unresolved";
    case SPONG_SMALE_INTERNAL_FAILURE:
        return "internal_failure";
    default:
        return "unknown";
    }
}

const char *spong_smale_reason_name(int32_t reason) {
    switch (reason) {
    case SPONG_SMALE_REASON_NONE:
        return "none";
    case SPONG_SMALE_REASON_BAD_RATIONAL:
        return "bad_rational";
    case SPONG_SMALE_REASON_BAD_POLICY:
        return "bad_policy";
    case SPONG_SMALE_REASON_N_IDENTITY:
        return "n_identity";
    case SPONG_SMALE_REASON_CENTRE_ORDER:
        return "centre_order";
    case SPONG_SMALE_REASON_ENDPOINT_BITS:
        return "endpoint_bits";
    case SPONG_SMALE_REASON_TRANSVERSE_DIVISOR:
        return "transverse_divisor";
    case SPONG_SMALE_REASON_B_PARAMETER_SINGULAR:
        return "b_parameter_singular";
    case SPONG_SMALE_REASON_CRITICAL_POINT:
        return "critical_point";
    case SPONG_SMALE_REASON_FACE_INEQUALITY:
        return "face_inequality";
    case SPONG_SMALE_REASON_RADIUS_CAP:
        return "radius_cap";
    case SPONG_SMALE_REASON_SLAB_BUDGET:
        return "slab_budget";
    case SPONG_SMALE_REASON_TARGET_LEVEL:
        return "target_level";
    case SPONG_SMALE_REASON_PROJECTION_BUDGET:
        return "projection_budget";
    case SPONG_SMALE_REASON_TARGET_SHEET:
        return "target_sheet";
    case SPONG_SMALE_REASON_TARGET_MISSES_TUBE:
        return "target_misses_tube";
    case SPONG_SMALE_REASON_ALLOCATION:
        return "allocation";
    case SPONG_SMALE_REASON_INTERNAL:
        return "internal";
    case SPONG_SMALE_REASON_ROOT_INTERVAL:
        return "root_interval";
    case SPONG_SMALE_REASON_FRAME_SINGULAR:
        return "frame_singular";
    case SPONG_SMALE_REASON_CONE_UNRESOLVED:
        return "cone_unresolved";
    case SPONG_SMALE_REASON_FROBENIUS_UNRESOLVED:
        return "frobenius_unresolved";
    case SPONG_SMALE_REASON_SECTION_UNRESOLVED:
        return "section_unresolved";
    case SPONG_SMALE_REASON_SECTION_SHEET:
        return "section_sheet";
    case SPONG_SMALE_REASON_FIXED_SHEET_DOMAIN:
        return "fixed_sheet_domain";
    case SPONG_SMALE_REASON_INITIAL_PROJECTION:
        return "initial_projection";
    default:
        return "unknown";
    }
}
