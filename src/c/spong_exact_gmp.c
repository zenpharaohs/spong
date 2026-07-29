#include "spong/spong_exact.h"

#include <gmp.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mpq_t *c;
    size_t n;
    size_t cap;
} qpoly;

typedef struct {
    spong_exact_policy policy;
    spong_exact_work work;
    int failed;
} exact_context;

struct spong_sturm_plan {
    qpoly input;
    qpoly squarefree;
    qpoly *chain;
    size_t chain_length;
    spong_exact_work work;
};

static int observe_poly(exact_context *ctx, const qpoly *p, int count_work);

static int qpoly_init(qpoly *p, size_t cap) {
    p->c = NULL;
    p->n = 0;
    p->cap = 0;
    if (cap == 0)
        return 0;
    p->c = (mpq_t *)calloc(cap, sizeof(mpq_t));
    if (p->c == NULL)
        return -1;
    for (size_t i = 0; i < cap; ++i)
        mpq_init(p->c[i]);
    p->cap = cap;
    return 0;
}

static void qpoly_clear(qpoly *p) {
    if (p->c != NULL) {
        for (size_t i = 0; i < p->cap; ++i)
            mpq_clear(p->c[i]);
        free(p->c);
    }
    p->c = NULL;
    p->n = p->cap = 0;
}

static void qpoly_trim(qpoly *p) {
    while (p->n > 0 && mpq_sgn(p->c[p->n-1]) == 0)
        --p->n;
}

static int qpoly_copy(qpoly *dst, const qpoly *src) {
    if (qpoly_init(dst, src->n) != 0)
        return -1;
    dst->n = src->n;
    for (size_t i = 0; i < src->n; ++i)
        mpq_set(dst->c[i], src->c[i]);
    return 0;
}

static int qpoly_derivative(qpoly *dst, const qpoly *src) {
    size_t n = src->n > 1 ? src->n-1 : 0;
    if (qpoly_init(dst, n) != 0)
        return -1;
    dst->n = n;
    for (size_t i = 1; i < src->n; ++i) {
        mpq_set(dst->c[i-1], src->c[i]);
        mpz_mul_ui(
            mpq_numref(dst->c[i-1]), mpq_numref(dst->c[i-1]),
            (unsigned long)i);
        mpq_canonicalize(dst->c[i-1]);
    }
    qpoly_trim(dst);
    return 0;
}

/* Multiply by a positive rational to obtain primitive integer coefficients. */
static void qpoly_primitive(qpoly *p) {
    qpoly_trim(p);
    if (p->n == 0)
        return;
    mpz_t lcm, gcd, factor, value;
    mpz_inits(lcm, gcd, factor, value, NULL);
    mpz_set_ui(lcm, 1);
    for (size_t i = 0; i < p->n; ++i)
        mpz_lcm(lcm, lcm, mpq_denref(p->c[i]));
    mpz_set_ui(gcd, 0);
    for (size_t i = 0; i < p->n; ++i) {
        mpz_divexact(factor, lcm, mpq_denref(p->c[i]));
        mpz_mul(value, mpq_numref(p->c[i]), factor);
        mpz_gcd(gcd, gcd, value);
    }
    if (mpz_sgn(gcd) == 0)
        mpz_set_ui(gcd, 1);
    for (size_t i = 0; i < p->n; ++i) {
        mpz_divexact(factor, lcm, mpq_denref(p->c[i]));
        mpz_mul(value, mpq_numref(p->c[i]), factor);
        mpz_divexact(value, value, gcd);
        mpq_set_z(p->c[i], value);
    }
    mpz_clears(lcm, gcd, factor, value, NULL);
}

static int qpoly_divrem(const qpoly *f, const qpoly *g,
                        qpoly *quotient, qpoly *remainder,
                        exact_context *ctx) {
    if (g->n == 0)
        return -1;
    size_t qn = f->n >= g->n ? f->n-g->n+1 : 0;
    if (qpoly_init(quotient, qn) != 0)
        return -1;
    quotient->n = qn;
    if (qpoly_copy(remainder, f) != 0) {
        qpoly_clear(quotient);
        return -1;
    }
    mpq_t ratio, product;
    mpq_inits(ratio, product, NULL);
    while (remainder->n >= g->n && remainder->n != 0) {
        size_t k = remainder->n-g->n;
        mpq_div(ratio, remainder->c[remainder->n-1], g->c[g->n-1]);
        mpq_set(quotient->c[k], ratio);
        for (size_t j = 0; j < g->n; ++j) {
            mpq_mul(product, ratio, g->c[j]);
            mpq_sub(remainder->c[k+j], remainder->c[k+j], product);
        }
        qpoly_trim(remainder);
        if (observe_poly(ctx, quotient, 0) != 0
                || observe_poly(ctx, remainder, 0) != 0) {
            mpq_clears(ratio, product, NULL);
            return -1;
        }
    }
    qpoly_trim(quotient);
    mpq_clears(ratio, product, NULL);
    return 0;
}

static int qpoly_gcd(qpoly *out, const qpoly *p, const qpoly *q,
                     exact_context *ctx) {
    qpoly a = {0}, b = {0};
    if (qpoly_copy(&a, p) != 0 || qpoly_copy(&b, q) != 0)
        goto allocation_failure;
    qpoly_primitive(&a);
    qpoly_primitive(&b);
    if (observe_poly(ctx, &a, 1) != 0
            || observe_poly(ctx, &b, 1) != 0)
        goto work_failure;
    while (b.n != 0) {
        qpoly quotient = {0}, remainder = {0};
        if (qpoly_divrem(&a, &b, &quotient, &remainder, ctx) != 0) {
            qpoly_clear(&quotient);
            qpoly_clear(&remainder);
            if (ctx->failed)
                goto work_failure;
            goto allocation_failure;
        }
        qpoly_clear(&quotient);
        qpoly_primitive(&remainder);
        if (observe_poly(ctx, &remainder, 1) != 0) {
            qpoly_clear(&remainder);
            goto work_failure;
        }
        qpoly_clear(&a);
        a = b;
        b = remainder;
        ++ctx->work.prs_steps;
        if (ctx->policy.max_prs_steps
                && ctx->work.prs_steps > ctx->policy.max_prs_steps) {
            ctx->failed = SPONG_EXACT_WORK_LIMIT;
            qpoly_clear(&a);
            qpoly_clear(&b);
            return -1;
        }
    }
    qpoly_clear(&b);
    *out = a;
    return 0;

work_failure:
    qpoly_clear(&a);
    qpoly_clear(&b);
    return -1;

allocation_failure:
    qpoly_clear(&a);
    qpoly_clear(&b);
    ctx->failed = SPONG_EXACT_ALLOCATION_FAILURE;
    return -1;
}

static uint64_t coefficient_bits(const mpq_t value) {
    uint64_t nbits = (uint64_t)mpz_sizeinbase(mpq_numref(value), 2);
    uint64_t dbits = (uint64_t)mpz_sizeinbase(mpq_denref(value), 2);
    return nbits > dbits ? nbits : dbits;
}

static int observe_poly(exact_context *ctx, const qpoly *p, int count_work) {
    if (count_work) {
        ++ctx->work.chain_polynomials;
        ctx->work.chain_coefficients += (uint64_t)p->n;
    }
    for (size_t i = 0; i < p->n; ++i) {
        uint64_t bits = coefficient_bits(p->c[i]);
        if (bits > ctx->work.peak_coefficient_bits)
            ctx->work.peak_coefficient_bits = bits;
    }
    if ((ctx->policy.max_chain_coefficients
            && ctx->work.chain_coefficients
                > ctx->policy.max_chain_coefficients)
            || (ctx->policy.max_coefficient_bits
                && ctx->work.peak_coefficient_bits
                    > ctx->policy.max_coefficient_bits)) {
        ctx->failed = SPONG_EXACT_WORK_LIMIT;
        return -1;
    }
    return 0;
}

static int sign_at_infinity(const qpoly *p, int positive) {
    if (p->n == 0)
        return 0;
    int sign = mpq_sgn(p->c[p->n-1]);
    if (!positive && ((p->n-1) & 1U))
        sign = -sign;
    return sign;
}

static int variations_at_infinity(qpoly *chain, size_t n, int positive) {
    int previous = 0, variations = 0;
    for (size_t i = 0; i < n; ++i) {
        int sign = sign_at_infinity(&chain[i], positive);
        if (sign != 0) {
            if (previous != 0 && sign != previous)
                ++variations;
            previous = sign;
        }
    }
    return variations;
}

static void sturm_chain_clear(qpoly *chain, size_t n) {
    for (size_t i = 0; i < n; ++i)
        qpoly_clear(&chain[i]);
    free(chain);
}

static int sturm_chain_build(const qpoly *p, exact_context *ctx,
                             qpoly **chain_out, size_t *n_out) {
    qpoly a = {0}, b = {0};
    qpoly *chain = NULL;
    size_t n = 0, cap = 0;
    if (qpoly_copy(&a, p) != 0 || qpoly_derivative(&b, &a) != 0)
        goto failure;
    qpoly_primitive(&a);
    qpoly_primitive(&b);
    while (a.n != 0) {
        if (n == cap) {
            size_t next = cap ? 2*cap : 4;
            qpoly *grown = (qpoly *)realloc(chain, next*sizeof(qpoly));
            if (grown == NULL)
                goto failure;
            chain = grown;
            memset(chain+cap, 0, (next-cap)*sizeof(qpoly));
            cap = next;
        }
        chain[n++] = a;
        a = (qpoly){0};
        if (observe_poly(ctx, &chain[n-1], 1) != 0)
            goto failure;
        if (b.n == 0)
            break;
        qpoly quotient = {0}, remainder = {0};
        if (qpoly_divrem(
                &chain[n-1], &b, &quotient, &remainder, ctx) != 0) {
            qpoly_clear(&quotient);
            qpoly_clear(&remainder);
            goto failure;
        }
        qpoly_clear(&quotient);
        for (size_t i = 0; i < remainder.n; ++i)
            mpq_neg(remainder.c[i], remainder.c[i]);
        qpoly_primitive(&remainder);
        a = b;
        b = remainder;
        ++ctx->work.prs_steps;
        if (ctx->policy.max_prs_steps
                && ctx->work.prs_steps > ctx->policy.max_prs_steps) {
            ctx->failed = SPONG_EXACT_WORK_LIMIT;
            goto failure;
        }
    }
    qpoly_clear(&a);
    qpoly_clear(&b);
    *chain_out = chain;
    *n_out = n;
    return 0;

failure:
    if (!ctx->failed)
        ctx->failed = SPONG_EXACT_ALLOCATION_FAILURE;
    qpoly_clear(&a);
    qpoly_clear(&b);
    sturm_chain_clear(chain, n);
    return -1;
}

static int sturm_count_squarefree(const qpoly *p, exact_context *ctx,
                                  uint32_t *count) {
    qpoly *chain = NULL;
    size_t n = 0;
    if (sturm_chain_build(p, ctx, &chain, &n) != 0)
        return -1;
    *count = (uint32_t)(
        variations_at_infinity(chain, n, 0)
        - variations_at_infinity(chain, n, 1));
    sturm_chain_clear(chain, n);
    return 0;
}

int spong_sturm_analyze_decimal(
        const char *const *coefficients,
        size_t coefficient_count,
        const spong_exact_policy *policy,
        spong_sturm_analysis *out) {
    if (out == NULL)
        return -1;
    memset(out, 0, sizeof(*out));
    out->status = SPONG_EXACT_INVALID_ARGUMENT;
    out->input_degree = -1;
    out->squarefree_degree = -1;
    if (coefficients == NULL || policy == NULL || coefficient_count == 0)
        return -1;

    exact_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.policy = *policy;
    if (policy->max_chain_coefficients
            && coefficient_count > policy->max_chain_coefficients) {
        out->status = SPONG_EXACT_WORK_LIMIT;
        return -1;
    }
    qpoly p = {0}, derivative = {0}, gcd = {0}, squarefree = {0};
    if (qpoly_init(&p, coefficient_count) != 0)
        goto failure;
    p.n = coefficient_count;
    for (size_t i = 0; i < coefficient_count; ++i) {
        if (coefficients[i] == NULL) {
            ctx.failed = SPONG_EXACT_PARSE_FAILURE;
            goto failure;
        }
        const char *digits = coefficients[i];
        if (*digits == '-' || *digits == '+')
            ++digits;
        while (*digits == '0')
            ++digits;
        size_t digit_count = strlen(digits);
        /*
         * A nonzero decimal integer with d digits has more than d bits once
         * d>1.  This deliberately loose pre-parse ceiling bounds hostile
         * allocations before GMP sees the string; the exact bit test follows.
         */
        if (policy->max_coefficient_bits
                && digit_count > policy->max_coefficient_bits) {
            ctx.failed = SPONG_EXACT_WORK_LIMIT;
            goto failure;
        }
        if (mpz_set_str(mpq_numref(p.c[i]), coefficients[i], 10) != 0) {
            ctx.failed = SPONG_EXACT_PARSE_FAILURE;
            goto failure;
        }
        mpz_set_ui(mpq_denref(p.c[i]), 1);
    }
    qpoly_trim(&p);
    if (p.n == 0) {
        ctx.failed = SPONG_EXACT_INVALID_ARGUMENT;
        goto failure;
    }
    qpoly_primitive(&p);
    if (observe_poly(&ctx, &p, 1) != 0)
        goto failure;
    out->input_degree = (int32_t)(p.n-1);
    if (qpoly_derivative(&derivative, &p) != 0)
        goto failure;
    if (qpoly_gcd(&gcd, &p, &derivative, &ctx) != 0)
        goto failure;

    if (gcd.n > 1) {
        qpoly gcd_squarefree = {0}, gd = {0}, gg = {0}, quotient = {0}, rem = {0};
        if (qpoly_derivative(&gd, &gcd) != 0
                || qpoly_gcd(&gg, &gcd, &gd, &ctx) != 0
                || qpoly_divrem(
                    &gcd, &gg, &quotient, &rem, &ctx) != 0)
            goto repeated_failure;
        if (rem.n != 0) {
            ctx.failed = SPONG_EXACT_INTERNAL_FAILURE;
            goto repeated_failure;
        }
        qpoly_clear(&rem);
        qpoly_clear(&gg);
        qpoly_clear(&gd);
        gcd_squarefree = quotient;
        qpoly_primitive(&gcd_squarefree);
        if (sturm_count_squarefree(
                &gcd_squarefree, &ctx, &out->repeated_real_roots) != 0) {
            qpoly_clear(&gcd_squarefree);
            goto failure;
        }
        qpoly_clear(&gcd_squarefree);
        goto repeated_done;
repeated_failure:
        qpoly_clear(&gd);
        qpoly_clear(&gg);
        qpoly_clear(&quotient);
        qpoly_clear(&rem);
        goto failure;
    }
repeated_done:
    {
        qpoly remainder = {0};
        if (qpoly_divrem(
                &p, &gcd, &squarefree, &remainder, &ctx) != 0) {
            qpoly_clear(&remainder);
            goto failure;
        }
        if (remainder.n != 0) {
            ctx.failed = SPONG_EXACT_INTERNAL_FAILURE;
            qpoly_clear(&remainder);
            goto failure;
        }
        qpoly_clear(&remainder);
    }
    qpoly_primitive(&squarefree);
    out->squarefree_degree = squarefree.n
        ? (int32_t)(squarefree.n-1) : -1;
    if (sturm_count_squarefree(
            &squarefree, &ctx, &out->distinct_real_roots) != 0)
        goto failure;
    out->status = SPONG_EXACT_OK;
    out->work = ctx.work;
    qpoly_clear(&p);
    qpoly_clear(&derivative);
    qpoly_clear(&gcd);
    qpoly_clear(&squarefree);
    return 0;

failure:
    if (!ctx.failed)
        ctx.failed = SPONG_EXACT_ALLOCATION_FAILURE;
    out->status = ctx.failed;
    out->work = ctx.work;
    qpoly_clear(&p);
    qpoly_clear(&derivative);
    qpoly_clear(&gcd);
    qpoly_clear(&squarefree);
    return -1;
}

static int parse_integer_poly(const char *const *coefficients, size_t n,
                              exact_context *ctx, qpoly *p) {
    if (qpoly_init(p, n) != 0) {
        ctx->failed = SPONG_EXACT_ALLOCATION_FAILURE;
        return -1;
    }
    p->n = n;
    for (size_t i = 0; i < n; ++i) {
        if (coefficients[i] == NULL) {
            ctx->failed = SPONG_EXACT_PARSE_FAILURE;
            return -1;
        }
        const char *digits = coefficients[i];
        if (*digits == '-' || *digits == '+')
            ++digits;
        while (*digits == '0')
            ++digits;
        if (ctx->policy.max_coefficient_bits
                && strlen(digits) > ctx->policy.max_coefficient_bits) {
            ctx->failed = SPONG_EXACT_WORK_LIMIT;
            return -1;
        }
        if (mpz_set_str(mpq_numref(p->c[i]), coefficients[i], 10) != 0) {
            ctx->failed = SPONG_EXACT_PARSE_FAILURE;
            return -1;
        }
        mpz_set_ui(mpq_denref(p->c[i]), 1);
    }
    qpoly_trim(p);
    if (p->n == 0) {
        ctx->failed = SPONG_EXACT_INVALID_ARGUMENT;
        return -1;
    }
    qpoly_primitive(p);
    return observe_poly(ctx, p, 1);
}

static int squarefree_from_integer_poly(const qpoly *p, exact_context *ctx,
                                        qpoly *squarefree) {
    qpoly derivative = {0}, gcd = {0}, remainder = {0};
    if (qpoly_derivative(&derivative, p) != 0
            || qpoly_gcd(&gcd, p, &derivative, ctx) != 0
            || qpoly_divrem(p, &gcd, squarefree, &remainder, ctx) != 0)
        goto failure;
    if (remainder.n != 0) {
        ctx->failed = SPONG_EXACT_INTERNAL_FAILURE;
        goto failure;
    }
    qpoly_primitive(squarefree);
    qpoly_clear(&derivative);
    qpoly_clear(&gcd);
    qpoly_clear(&remainder);
    return 0;
failure:
    qpoly_clear(&derivative);
    qpoly_clear(&gcd);
    qpoly_clear(&remainder);
    return -1;
}

int spong_sturm_plan_create_decimal(
        const char *const *coefficients,
        size_t coefficient_count,
        const spong_exact_policy *policy,
        spong_sturm_plan **plan_out,
        spong_sturm_analysis *analysis) {
    if (plan_out == NULL || analysis == NULL)
        return -1;
    *plan_out = NULL;
    if (spong_sturm_analyze_decimal(
            coefficients, coefficient_count, policy, analysis) != 0)
        return -1;
    spong_sturm_plan *plan = (spong_sturm_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        analysis->status = SPONG_EXACT_ALLOCATION_FAILURE;
        return -1;
    }
    exact_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.policy = *policy;
    ctx.work = analysis->work;
    qpoly input = {0};
    if (parse_integer_poly(
            coefficients, coefficient_count, &ctx, &input) != 0
            || squarefree_from_integer_poly(
                &input, &ctx, &plan->squarefree) != 0
            || sturm_chain_build(
                &plan->squarefree, &ctx,
                &plan->chain, &plan->chain_length) != 0) {
        qpoly_clear(&input);
        spong_sturm_plan_destroy(plan);
        analysis->status = ctx.failed
            ? ctx.failed : SPONG_EXACT_ALLOCATION_FAILURE;
        analysis->work = ctx.work;
        return -1;
    }
    plan->input = input;
    input = (qpoly){0};
    plan->work = ctx.work;
    analysis->work = ctx.work;
    *plan_out = plan;
    return 0;
}

void spong_sturm_plan_destroy(spong_sturm_plan *plan) {
    if (plan == NULL)
        return;
    qpoly_clear(&plan->input);
    qpoly_clear(&plan->squarefree);
    sturm_chain_clear(plan->chain, plan->chain_length);
    free(plan);
}

size_t spong_sturm_plan_chain_length(const spong_sturm_plan *plan) {
    return plan != NULL ? plan->chain_length : 0;
}

uint64_t spong_sturm_plan_chain_coefficients(
        const spong_sturm_plan *plan) {
    uint64_t count = 0;
    if (plan != NULL) {
        for (size_t i = 0; i < plan->chain_length; ++i)
            count += (uint64_t)plan->chain[i].n;
    }
    return count;
}

static int parse_rational(const char *numerator, const char *denominator,
                          mpq_t value) {
    if (numerator == NULL || denominator == NULL
            || mpz_set_str(mpq_numref(value), numerator, 10) != 0
            || mpz_set_str(mpq_denref(value), denominator, 10) != 0
            || mpz_sgn(mpq_denref(value)) <= 0)
        return -1;
    mpq_canonicalize(value);
    return 0;
}

static int qpoly_sign_at(const qpoly *p, const mpq_t x) {
    if (p->n == 0)
        return 0;
    mpq_t value;
    mpq_init(value);
    mpq_set(value, p->c[p->n-1]);
    for (size_t i = p->n-1; i-- > 0;) {
        mpq_mul(value, value, x);
        mpq_add(value, value, p->c[i]);
    }
    int sign = mpq_sgn(value);
    mpq_clear(value);
    return sign;
}

static int variations_at_point(
        qpoly *chain, size_t n, const mpq_t x) {
    int previous = 0, variations = 0;
    for (size_t i = 0; i < n; ++i) {
        int sign = qpoly_sign_at(&chain[i], x);
        if (sign != 0) {
            if (previous != 0 && sign != previous)
                ++variations;
            previous = sign;
        }
    }
    return variations;
}

int spong_sturm_plan_count(
        const spong_sturm_plan *plan,
        const char *lower_numerator,
        const char *lower_denominator,
        const char *upper_numerator,
        const char *upper_denominator,
        uint32_t *count) {
    if (plan == NULL || count == NULL)
        return -1;
    int lower_variations, upper_variations;
    mpq_t lower, upper;
    mpq_inits(lower, upper, NULL);
    if (lower_numerator == NULL) {
        lower_variations = variations_at_infinity(
            plan->chain, plan->chain_length, 0);
    } else {
        if (parse_rational(lower_numerator, lower_denominator, lower) != 0)
            goto invalid;
        lower_variations = variations_at_point(
            plan->chain, plan->chain_length, lower);
    }
    if (upper_numerator == NULL) {
        upper_variations = variations_at_infinity(
            plan->chain, plan->chain_length, 1);
    } else {
        if (parse_rational(upper_numerator, upper_denominator, upper) != 0)
            goto invalid;
        upper_variations = variations_at_point(
            plan->chain, plan->chain_length, upper);
    }
    if (lower_numerator != NULL && upper_numerator != NULL
            && mpq_cmp(lower, upper) > 0)
        goto invalid;
    *count = (uint32_t)(lower_variations-upper_variations);
    mpq_clears(lower, upper, NULL);
    return 0;
invalid:
    mpq_clears(lower, upper, NULL);
    return -1;
}

int spong_sturm_plan_sign_at(
        const spong_sturm_plan *plan,
        const char *numerator,
        const char *denominator,
        int32_t *sign) {
    if (plan == NULL || sign == NULL)
        return -1;
    mpq_t x;
    mpq_init(x);
    if (parse_rational(numerator, denominator, x) != 0) {
        mpq_clear(x);
        return -1;
    }
    *sign = (int32_t)qpoly_sign_at(&plan->input, x);
    mpq_clear(x);
    return 0;
}

typedef struct {
    mpq_t lo;
    mpq_t hi;
    int v_lo;
    int v_hi;
    uint64_t depth;
    uint32_t exact;
} isolation_node;

static void isolation_node_clear(isolation_node *node) {
    mpq_clear(node->lo);
    mpq_clear(node->hi);
}

static int isolation_array_append(
        isolation_node **array, size_t *length, size_t *capacity,
        const mpq_t lo, const mpq_t hi, int v_lo, int v_hi,
        uint64_t depth, uint32_t exact) {
    if (*length == *capacity) {
        size_t next = *capacity ? 2*(*capacity) : 16;
        if (next < *capacity || next > SIZE_MAX/sizeof(isolation_node))
            return -1;
        isolation_node *grown = (isolation_node *)realloc(
            *array, next*sizeof(isolation_node));
        if (grown == NULL)
            return -1;
        *array = grown;
        *capacity = next;
    }
    isolation_node *node = &(*array)[(*length)++];
    mpq_init(node->lo);
    mpq_init(node->hi);
    mpq_set(node->lo, lo);
    mpq_set(node->hi, hi);
    node->v_lo = v_lo;
    node->v_hi = v_hi;
    node->depth = depth;
    node->exact = exact;
    return 0;
}

static uint64_t rational_bits(const mpq_t x) {
    uint64_t n = (uint64_t)mpz_sizeinbase(mpq_numref(x), 2);
    uint64_t d = (uint64_t)mpz_sizeinbase(mpq_denref(x), 2);
    return n > d ? n : d;
}

static int observe_isolation_node(
        const isolation_node *node, const spong_isolation_policy *policy,
        spong_isolation_work *work) {
    ++work->subdivision_nodes;
    if (node->depth > work->max_subdivision_depth)
        work->max_subdivision_depth = node->depth;
    uint64_t lo_bits = rational_bits(node->lo);
    uint64_t hi_bits = rational_bits(node->hi);
    uint64_t bits = lo_bits > hi_bits ? lo_bits : hi_bits;
    if (bits > work->max_endpoint_bits)
        work->max_endpoint_bits = bits;
    if ((policy->max_subdivision_nodes
            && work->subdivision_nodes > policy->max_subdivision_nodes)
            || (policy->max_endpoint_bits
                && bits > policy->max_endpoint_bits)) {
        work->status = SPONG_EXACT_WORK_LIMIT;
        return -1;
    }
    return 0;
}

static int isolation_variations(
        const spong_sturm_plan *plan, const mpq_t x,
        spong_isolation_work *work) {
    ++work->variation_evaluations;
    return variations_at_point(plan->chain, plan->chain_length, x);
}

static int isolation_compare(const void *left, const void *right) {
    const isolation_node *a = (const isolation_node *)left;
    const isolation_node *b = (const isolation_node *)right;
    return mpq_cmp(a->lo, b->lo);
}

static char *integer_string(const mpz_t value) {
    size_t digits = mpz_sizeinbase(value, 10);
    char *text = (char *)malloc(digits+3);
    if (text != NULL)
        mpz_get_str(text, 10, value);
    return text;
}

void spong_root_intervals_destroy(
        spong_root_interval *intervals, size_t interval_count) {
    if (intervals == NULL)
        return;
    for (size_t i = 0; i < interval_count; ++i) {
        free(intervals[i].lower_numerator);
        free(intervals[i].lower_denominator);
        free(intervals[i].upper_numerator);
        free(intervals[i].upper_denominator);
    }
    free(intervals);
}

int spong_sturm_plan_isolate(
        const spong_sturm_plan *plan,
        const spong_isolation_policy *policy,
        spong_root_interval **intervals_out,
        size_t *interval_count_out,
        spong_isolation_work *work) {
    if (intervals_out == NULL || interval_count_out == NULL || work == NULL)
        return -1;
    *intervals_out = NULL;
    *interval_count_out = 0;
    memset(work, 0, sizeof(*work));
    work->status = SPONG_EXACT_INVALID_ARGUMENT;
    if (plan == NULL || policy == NULL || plan->squarefree.n == 0)
        return -1;

    isolation_node *stack = NULL, *roots = NULL;
    size_t stack_n = 0, stack_cap = 0, roots_n = 0, roots_cap = 0;
    mpq_t bound, maximum, ratio, negative_bound;
    mpq_inits(bound, maximum, ratio, negative_bound, NULL);
    mpq_set_ui(maximum, 0, 1);
    for (size_t i = 0; i+1 < plan->squarefree.n; ++i) {
        mpq_abs(ratio, plan->squarefree.c[i]);
        if (mpq_cmp(ratio, maximum) > 0)
            mpq_set(maximum, ratio);
    }
    mpq_abs(ratio, plan->squarefree.c[plan->squarefree.n-1]);
    mpq_div(bound, maximum, ratio);
    mpq_set_ui(ratio, 1, 1);
    mpq_add(bound, bound, ratio);
    mpq_neg(negative_bound, bound);
    int v_lo = isolation_variations(plan, negative_bound, work);
    int v_hi = isolation_variations(plan, bound, work);
    if (isolation_array_append(
            &stack, &stack_n, &stack_cap, negative_bound, bound,
            v_lo, v_hi, 0, 0) != 0)
        goto allocation_failure;

    while (stack_n != 0) {
        isolation_node node = stack[--stack_n];
        if (observe_isolation_node(&node, policy, work) != 0) {
            isolation_node_clear(&node);
            goto failure;
        }
        int count = node.v_lo-node.v_hi;
        if (count == 0) {
            isolation_node_clear(&node);
            continue;
        }
        if (count == 1) {
            if (policy->max_intervals
                    && roots_n+1 > policy->max_intervals) {
                work->status = SPONG_EXACT_WORK_LIMIT;
                isolation_node_clear(&node);
                goto failure;
            }
            if (isolation_array_append(
                    &roots, &roots_n, &roots_cap, node.lo, node.hi,
                    node.v_lo, node.v_hi, node.depth, 0) != 0) {
                isolation_node_clear(&node);
                goto allocation_failure;
            }
            isolation_node_clear(&node);
            continue;
        }

        mpq_t mid;
        mpq_init(mid);
        mpq_add(mid, node.lo, node.hi);
        mpq_div_2exp(mid, mid, 1);
        ++work->polynomial_evaluations;
        if (qpoly_sign_at(&plan->squarefree, mid) == 0) {
            if (policy->max_intervals
                    && roots_n+1 > policy->max_intervals) {
                work->status = SPONG_EXACT_WORK_LIMIT;
                mpq_clear(mid);
                isolation_node_clear(&node);
                goto failure;
            }
            if (isolation_array_append(
                    &roots, &roots_n, &roots_cap, mid, mid, 0, 0,
                    node.depth, 1) != 0) {
                mpq_clear(mid);
                isolation_node_clear(&node);
                goto allocation_failure;
            }
            mpq_t eps, left, right;
            mpq_inits(eps, left, right, NULL);
            mpq_sub(eps, node.hi, node.lo);
            mpq_div_2exp(eps, eps, 20);
            int left_v = 0, right_v = 0;
            for (;;) {
                mpq_sub(left, mid, eps);
                mpq_add(right, mid, eps);
                work->polynomial_evaluations += 2;
                int left_sign = qpoly_sign_at(&plan->squarefree, left);
                int right_sign = qpoly_sign_at(&plan->squarefree, right);
                left_v = isolation_variations(plan, left, work);
                right_v = isolation_variations(plan, right, work);
                if (left_sign != 0 && right_sign != 0
                        && left_v-right_v == 1)
                    break;
                ++work->puncture_halvings;
                if (policy->max_puncture_halvings
                        && work->puncture_halvings
                            > policy->max_puncture_halvings) {
                    work->status = SPONG_EXACT_WORK_LIMIT;
                    mpq_clears(eps, left, right, NULL);
                    mpq_clear(mid);
                    isolation_node_clear(&node);
                    goto failure;
                }
                mpq_div_2exp(eps, eps, 1);
                uint64_t bits = rational_bits(eps);
                if (bits > work->max_endpoint_bits)
                    work->max_endpoint_bits = bits;
                if (policy->max_endpoint_bits
                        && bits > policy->max_endpoint_bits) {
                    work->status = SPONG_EXACT_WORK_LIMIT;
                    mpq_clears(eps, left, right, NULL);
                    mpq_clear(mid);
                    isolation_node_clear(&node);
                    goto failure;
                }
            }
            /* LIFO: push right first so the left subtree is processed first. */
            if (isolation_array_append(
                    &stack, &stack_n, &stack_cap, right, node.hi,
                    right_v, node.v_hi, node.depth+1, 0) != 0
                    || isolation_array_append(
                        &stack, &stack_n, &stack_cap, node.lo, left,
                        node.v_lo, left_v, node.depth+1, 0) != 0) {
                mpq_clears(eps, left, right, NULL);
                mpq_clear(mid);
                isolation_node_clear(&node);
                goto allocation_failure;
            }
            mpq_clears(eps, left, right, NULL);
        } else {
            int v_mid = isolation_variations(plan, mid, work);
            if (isolation_array_append(
                    &stack, &stack_n, &stack_cap, mid, node.hi,
                    v_mid, node.v_hi, node.depth+1, 0) != 0
                    || isolation_array_append(
                        &stack, &stack_n, &stack_cap, node.lo, mid,
                        node.v_lo, v_mid, node.depth+1, 0) != 0) {
                mpq_clear(mid);
                isolation_node_clear(&node);
                goto allocation_failure;
            }
        }
        mpq_clear(mid);
        isolation_node_clear(&node);
    }

    qsort(roots, roots_n, sizeof(*roots), isolation_compare);
    spong_root_interval *result = (spong_root_interval *)calloc(
        roots_n, sizeof(*result));
    if (roots_n != 0 && result == NULL)
        goto allocation_failure;
    for (size_t i = 0; i < roots_n; ++i) {
        result[i].lower_numerator = integer_string(mpq_numref(roots[i].lo));
        result[i].lower_denominator = integer_string(mpq_denref(roots[i].lo));
        result[i].upper_numerator = integer_string(mpq_numref(roots[i].hi));
        result[i].upper_denominator = integer_string(mpq_denref(roots[i].hi));
        result[i].exact = roots[i].exact;
        if (result[i].lower_numerator == NULL
                || result[i].lower_denominator == NULL
                || result[i].upper_numerator == NULL
                || result[i].upper_denominator == NULL) {
            spong_root_intervals_destroy(result, roots_n);
            goto allocation_failure;
        }
    }
    for (size_t i = 0; i < roots_n; ++i)
        isolation_node_clear(&roots[i]);
    free(roots);
    free(stack);
    mpq_clears(bound, maximum, ratio, negative_bound, NULL);
    work->status = SPONG_EXACT_OK;
    *intervals_out = result;
    *interval_count_out = roots_n;
    return 0;

allocation_failure:
    work->status = SPONG_EXACT_ALLOCATION_FAILURE;
failure:
    for (size_t i = 0; i < stack_n; ++i)
        isolation_node_clear(&stack[i]);
    for (size_t i = 0; i < roots_n; ++i)
        isolation_node_clear(&roots[i]);
    free(stack);
    free(roots);
    mpq_clears(bound, maximum, ratio, negative_bound, NULL);
    return -1;
}

static spong_root_interval *export_one_interval(
        const mpq_t lo, const mpq_t hi, uint32_t exact) {
    spong_root_interval *result = (spong_root_interval *)calloc(
        1, sizeof(*result));
    if (result == NULL)
        return NULL;
    result->lower_numerator = integer_string(mpq_numref(lo));
    result->lower_denominator = integer_string(mpq_denref(lo));
    result->upper_numerator = integer_string(mpq_numref(hi));
    result->upper_denominator = integer_string(mpq_denref(hi));
    result->exact = exact;
    if (result->lower_numerator == NULL
            || result->lower_denominator == NULL
            || result->upper_numerator == NULL
            || result->upper_denominator == NULL) {
        spong_root_intervals_destroy(result, 1);
        return NULL;
    }
    return result;
}

int spong_sturm_plan_refine(
        const spong_sturm_plan *plan,
        const char *lower_numerator,
        const char *lower_denominator,
        const char *upper_numerator,
        const char *upper_denominator,
        const char *relative_width_numerator,
        const char *relative_width_denominator,
        const spong_refinement_policy *policy,
        spong_root_interval **interval,
        spong_refinement_work *work) {
    if (interval == NULL || work == NULL)
        return -1;
    *interval = NULL;
    memset(work, 0, sizeof(*work));
    work->status = SPONG_EXACT_INVALID_ARGUMENT;
    if (plan == NULL || policy == NULL)
        return -1;
    mpq_t lo, hi, rel, width, target, scale, temp, mid;
    mpq_inits(lo, hi, rel, width, target, scale, temp, mid, NULL);
    if (parse_rational(lower_numerator, lower_denominator, lo) != 0
            || parse_rational(upper_numerator, upper_denominator, hi) != 0
            || parse_rational(
                relative_width_numerator, relative_width_denominator,
                rel) != 0
            || mpq_cmp(lo, hi) > 0 || mpq_sgn(rel) <= 0)
        goto failure;
    {
        uint64_t lo_bits = rational_bits(lo);
        uint64_t hi_bits = rational_bits(hi);
        uint64_t bits = lo_bits > hi_bits ? lo_bits : hi_bits;
        work->max_endpoint_bits = bits;
        if (policy->max_endpoint_bits
                && bits > policy->max_endpoint_bits) {
            work->status = SPONG_EXACT_WORK_LIMIT;
            goto failure;
        }
    }
    if (mpq_cmp(lo, hi) == 0) {
        if (qpoly_sign_at(&plan->squarefree, lo) != 0)
            goto failure;
        *interval = export_one_interval(lo, hi, 1);
        if (*interval == NULL)
            goto allocation_failure;
        work->status = SPONG_EXACT_OK;
        mpq_clears(lo, hi, rel, width, target, scale, temp, mid, NULL);
        return 0;
    }
    int v_lo = variations_at_point(plan->chain, plan->chain_length, lo);
    int v_hi = variations_at_point(plan->chain, plan->chain_length, hi);
    int sign_lo = qpoly_sign_at(&plan->squarefree, lo);
    if (v_lo-v_hi != 1 || sign_lo == 0
            || qpoly_sign_at(&plan->squarefree, hi) == 0)
        goto failure;

    for (;;) {
        mpq_sub(width, hi, lo);
        mpq_abs(scale, lo);
        mpq_abs(temp, hi);
        mpq_add(scale, scale, temp);
        mpq_set_ui(temp, 1, 1);
        mpq_add(scale, scale, temp);
        mpq_mul(target, rel, scale);
        if (mpq_cmp(width, target) <= 0)
            break;
        ++work->bisections;
        if (policy->max_bisections
                && work->bisections > policy->max_bisections) {
            work->status = SPONG_EXACT_WORK_LIMIT;
            goto failure;
        }
        mpq_add(mid, lo, hi);
        mpq_div_2exp(mid, mid, 1);
        int sign_mid = qpoly_sign_at(&plan->squarefree, mid);
        if (sign_mid == 0) {
            mpq_set(lo, mid);
            mpq_set(hi, mid);
            break;
        }
        if (sign_mid == sign_lo) {
            mpq_set(lo, mid);
            sign_lo = sign_mid;
        } else {
            mpq_set(hi, mid);
        }
        uint64_t lo_bits = rational_bits(lo);
        uint64_t hi_bits = rational_bits(hi);
        uint64_t bits = lo_bits > hi_bits ? lo_bits : hi_bits;
        if (bits > work->max_endpoint_bits)
            work->max_endpoint_bits = bits;
        if (policy->max_endpoint_bits
                && bits > policy->max_endpoint_bits) {
            work->status = SPONG_EXACT_WORK_LIMIT;
            goto failure;
        }
    }
    {
        uint32_t exact = mpq_cmp(lo, hi) == 0;
        *interval = export_one_interval(lo, hi, exact);
        if (*interval == NULL)
            goto allocation_failure;
    }
    {
        uint64_t lo_bits = rational_bits(lo);
        uint64_t hi_bits = rational_bits(hi);
        uint64_t bits = lo_bits > hi_bits ? lo_bits : hi_bits;
        if (bits > work->max_endpoint_bits)
            work->max_endpoint_bits = bits;
    }
    work->status = SPONG_EXACT_OK;
    mpq_clears(lo, hi, rel, width, target, scale, temp, mid, NULL);
    return 0;

allocation_failure:
    work->status = SPONG_EXACT_ALLOCATION_FAILURE;
failure:
    mpq_clears(lo, hi, rel, width, target, scale, temp, mid, NULL);
    return -1;
}
