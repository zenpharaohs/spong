/* Exact local invariant-manifold launch certificate. */

#include "spong/spong_smale.h"

#include <gmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mpq_t lo, hi;
} li;
typedef struct {
    size_t n;
    mpq_t *c;
} lpoly;
typedef struct {
    size_t nu, ns;
    li *c;
} lp2;

typedef struct {
    const spong_local_launch_policy *policy;
    spong_local_launch_result *result;
    int failed;
} lcontext;

static uint64_t lq_bits(const mpq_t q) {
    uint64_t n = (uint64_t)mpz_sizeinbase(mpq_numref(q), 2);
    uint64_t d = (uint64_t)mpz_sizeinbase(mpq_denref(q), 2);
    return n > d ? n : d;
}

static int lobserve(lcontext *ctx, const mpq_t q) {
    uint64_t bits = lq_bits(q);
    if (bits > ctx->result->work.peak_rational_bits)
        ctx->result->work.peak_rational_bits = bits;
    if (ctx->policy->max_rational_bits && bits > ctx->policy->max_rational_bits) {
        ctx->failed = 1;
        ctx->result->status = SPONG_SMALE_WORK_LIMIT;
        ctx->result->primary_reason = SPONG_SMALE_REASON_ENDPOINT_BITS;
        return -1;
    }
    return 0;
}

static int lparse(lcontext *ctx, const spong_rational_input *in, mpq_t q) {
    if (in == NULL || in->numerator == NULL || in->denominator == NULL ||
        mpz_set_str(mpq_numref(q), in->numerator, 10) != 0 ||
        mpz_set_str(mpq_denref(q), in->denominator, 10) != 0 ||
        mpz_sgn(mpq_denref(q)) <= 0) {
        ctx->failed = 1;
        ctx->result->status = SPONG_SMALE_PARSE_FAILURE;
        ctx->result->primary_reason = SPONG_SMALE_REASON_BAD_RATIONAL;
        return -1;
    }
    mpq_canonicalize(q);
    ctx->result->work.input_rationals++;
    return lobserve(ctx, q);
}

static void li_init(li *x) {
    mpq_init(x->lo);
    mpq_init(x->hi);
}
static void li_clear(li *x) {
    mpq_clear(x->lo);
    mpq_clear(x->hi);
}
static void li_set(li *z, const li *x) {
    mpq_set(z->lo, x->lo);
    mpq_set(z->hi, x->hi);
}
static void li_point(li *z, const mpq_t x) {
    mpq_set(z->lo, x);
    mpq_set(z->hi, x);
}
static void li_si(li *z, long x) {
    mpq_set_si(z->lo, x, 1);
    mpq_set_si(z->hi, x, 1);
}
static int li_zero(const li *x) { return mpq_sgn(x->lo) <= 0 && mpq_sgn(x->hi) >= 0; }
static int li_exact_zero(const li *x) {
    return mpq_sgn(x->lo) == 0 && mpq_sgn(x->hi) == 0;
}
static int li_observe(lcontext *c, const li *x) {
    return lobserve(c, x->lo) || lobserve(c, x->hi) ? -1 : 0;
}

static int li_add(lcontext *c, li *z, const li *x, const li *y) {
    mpq_add(z->lo, x->lo, y->lo);
    mpq_add(z->hi, x->hi, y->hi);
    return li_observe(c, z);
}
static int li_sub(lcontext *c, li *z, const li *x, const li *y) {
    mpq_sub(z->lo, x->lo, y->hi);
    mpq_sub(z->hi, x->hi, y->lo);
    return li_observe(c, z);
}
static int li_neg(lcontext *c, li *z, const li *x) {
    mpq_t lo, hi;
    mpq_init(lo);
    mpq_init(hi);
    mpq_neg(lo, x->hi);
    mpq_neg(hi, x->lo);
    mpq_set(z->lo, lo);
    mpq_set(z->hi, hi);
    mpq_clear(lo);
    mpq_clear(hi);
    return li_observe(c, z);
}
static int li_mul(lcontext *c, li *z, const li *x, const li *y) {
    mpq_t p[4];
    for (int i = 0; i < 4; i++)
        mpq_init(p[i]);
    mpq_mul(p[0], x->lo, y->lo);
    mpq_mul(p[1], x->lo, y->hi);
    mpq_mul(p[2], x->hi, y->lo);
    mpq_mul(p[3], x->hi, y->hi);
    mpq_set(z->lo, p[0]);
    mpq_set(z->hi, p[0]);
    for (int i = 1; i < 4; i++) {
        if (mpq_cmp(p[i], z->lo) < 0)
            mpq_set(z->lo, p[i]);
        if (mpq_cmp(p[i], z->hi) > 0)
            mpq_set(z->hi, p[i]);
    }
    for (int i = 0; i < 4; i++)
        mpq_clear(p[i]);
    return li_observe(c, z);
}
static int li_scale(lcontext *c, li *z, const li *x, const mpq_t q) {
    li p;
    li_init(&p);
    li_point(&p, q);
    int s = li_mul(c, z, x, &p);
    li_clear(&p);
    return s;
}
static int li_scale_si(lcontext *c, li *z, const li *x, long q) {
    mpq_t s;
    mpq_init(s);
    mpq_set_si(s, q, 1);
    int r = li_scale(c, z, x, s);
    mpq_clear(s);
    return r;
}
static int li_div(lcontext *c, li *z, const li *x, const li *y) {
    if (li_zero(y))
        return -1;
    li r;
    li_init(&r);
    mpq_inv(r.lo, y->lo);
    mpq_inv(r.hi, y->hi);
    if (mpq_cmp(r.lo, r.hi) > 0)
        mpq_swap(r.lo, r.hi);
    int s = li_mul(c, z, x, &r);
    li_clear(&r);
    return s;
}
static int li_square(lcontext *c, li *z, const li *x) {
    if (li_zero(x)) {
        mpq_set_ui(z->lo, 0, 1);
        mpq_mul(z->hi, x->lo, x->lo);
        mpq_t q;
        mpq_init(q);
        mpq_mul(q, x->hi, x->hi);
        if (mpq_cmp(q, z->hi) > 0)
            mpq_set(z->hi, q);
        mpq_clear(q);
    } else {
        mpq_mul(z->lo, x->lo, x->lo);
        mpq_mul(z->hi, x->hi, x->hi);
        if (mpq_cmp(z->lo, z->hi) > 0)
            mpq_swap(z->lo, z->hi);
    }
    return li_observe(c, z);
}
static int li_pow(lcontext *c, li *z, const li *x, size_t n) {
    li a, b;
    li_init(&a);
    li_init(&b);
    li_si(&a, 1);
    li_set(&b, x);
    while (n) {
        if (n & 1) {
            if (li_mul(c, &a, &a, &b))
                goto fail;
        }
        n >>= 1;
        if (n && li_mul(c, &b, &b, &b))
            goto fail;
    }
    li_set(z, &a);
    li_clear(&a);
    li_clear(&b);
    return 0;
fail:
    li_clear(&a);
    li_clear(&b);
    return -1;
}

static int lpoly_init(lpoly *p, size_t n) {
    memset(p, 0, sizeof(*p));
    if (n == 0)
        n = 1;
    if (n > SIZE_MAX / sizeof(mpq_t))
        return -1;
    p->c = calloc(n, sizeof(mpq_t));
    if (!p->c)
        return -1;
    p->n = n;
    for (size_t i = 0; i < n; i++)
        mpq_init(p->c[i]);
    return 0;
}
static void lpoly_clear(lpoly *p) {
    if (p->c) {
        for (size_t i = 0; i < p->n; i++)
            mpq_clear(p->c[i]);
        free(p->c);
    }
    memset(p, 0, sizeof(*p));
}
static size_t lpoly_len(const lpoly *p) {
    size_t n = p->n;
    while (n > 1 && mpq_sgn(p->c[n - 1]) == 0)
        n--;
    return n;
}
static int lpoly_parse(lcontext *c, lpoly *p, const spong_rational_input *in,
                       size_t n) {
    if (!in || !n || lpoly_init(p, n))
        return -1;
    for (size_t i = 0; i < n; i++)
        if (lparse(c, &in[i], p->c[i]))
            return -1;
    return 0;
}
static int lpoly_deriv(lcontext *c, lpoly *z, const lpoly *p) {
    if (lpoly_init(z, p->n > 1 ? p->n - 1 : 1))
        return -1;
    for (size_t i = 1; i < p->n; i++) {
        mpq_set(z->c[i - 1], p->c[i]);
        mpz_mul_ui(mpq_numref(z->c[i - 1]), mpq_numref(z->c[i - 1]), (unsigned long)i);
        mpq_canonicalize(z->c[i - 1]);
        if (lobserve(c, z->c[i - 1]))
            return -1;
    }
    return 0;
}
static int lpoly_eval_q(lcontext *c, mpq_t z, const lpoly *p, const mpq_t x) {
    mpq_set_ui(z, 0, 1);
    for (size_t k = lpoly_len(p); k-- > 0;) {
        mpq_mul(z, z, x);
        mpq_add(z, z, p->c[k]);
        if (lobserve(c, z))
            return -1;
    }
    return 0;
}
static int lpoly_interval(lcontext *c, li *z, const lpoly *p, const li *x) {
    li a, b;
    li_init(&a);
    li_init(&b);
    li_si(&a, 0);
    for (size_t k = lpoly_len(p); k-- > 0;) {
        if (li_mul(c, &a, &a, x))
            goto fail;
        li_point(&b, p->c[k]);
        if (li_add(c, &a, &a, &b))
            goto fail;
    }
    li_set(z, &a);
    c->result->work.interval_evaluations++;
    li_clear(&a);
    li_clear(&b);
    return 0;
fail:
    li_clear(&a);
    li_clear(&b);
    return -1;
}

static int lp2_init(lp2 *p, size_t nu, size_t ns) {
    memset(p, 0, sizeof(*p));
    if (!nu)
        nu = 1;
    if (!ns)
        ns = 1;
    if (nu > SIZE_MAX / ns || nu * ns > SIZE_MAX / sizeof(li))
        return -1;
    p->c = calloc(nu * ns, sizeof(li));
    if (!p->c)
        return -1;
    p->nu = nu;
    p->ns = ns;
    for (size_t i = 0; i < nu * ns; i++)
        li_init(&p->c[i]);
    return 0;
}
static void lp2_clear(lp2 *p) {
    if (p->c) {
        for (size_t i = 0; i < p->nu * p->ns; i++)
            li_clear(&p->c[i]);
        free(p->c);
    }
    memset(p, 0, sizeof(*p));
}
static li *lp2_at(lp2 *p, size_t i, size_t j) { return &p->c[i * p->ns + j]; }
static const li *lp2_cat(const lp2 *p, size_t i, size_t j) {
    return &p->c[i * p->ns + j];
}
static int lp2_accum(lcontext *c, li *z, const li *x) {
    li t;
    li_init(&t);
    int s = li_add(c, &t, z, x);
    if (!s)
        li_set(z, &t);
    li_clear(&t);
    return s;
}
static int lp2_add_scaled(lcontext *c, lp2 *z, const lp2 *p, const mpq_t scale) {
    if (mpq_sgn(scale) == 0)
        return 0;
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            if (li_exact_zero(lp2_cat(p, i, j)))
                continue;
            li t;
            li_init(&t);
            if (li_scale(c, &t, lp2_cat(p, i, j), scale) ||
                lp2_accum(c, lp2_at(z, i, j), &t)) {
                li_clear(&t);
                return -1;
            }
            li_clear(&t);
        }
    return 0;
}
static int lp2_lincomb(lcontext *c, lp2 *z, const lp2 *a, const mpq_t x, const lp2 *b,
                       const mpq_t y) {
    size_t nu = a->nu > b->nu ? a->nu : b->nu, ns = a->ns > b->ns ? a->ns : b->ns;
    if (lp2_init(z, nu, ns))
        return -1;
    return lp2_add_scaled(c, z, a, x) || lp2_add_scaled(c, z, b, y) ? -1 : 0;
}
static int lp2_mul(lcontext *c, lp2 *z, const lp2 *a, const lp2 *b) {
    if (a->nu > SIZE_MAX - b->nu + 1 || a->ns > SIZE_MAX - b->ns + 1 ||
        lp2_init(z, a->nu + b->nu - 1, a->ns + b->ns - 1))
        return -1;
    for (size_t i = 0; i < a->nu; i++)
        for (size_t j = 0; j < a->ns; j++)
            for (size_t k = 0; k < b->nu; k++)
                for (size_t l = 0; l < b->ns; l++) {
                    if (li_exact_zero(lp2_cat(a, i, j)) ||
                        li_exact_zero(lp2_cat(b, k, l)))
                        continue;
                    li t;
                    li_init(&t);
                    if (li_mul(c, &t, lp2_cat(a, i, j), lp2_cat(b, k, l)) ||
                        lp2_accum(c, lp2_at(z, i + k, j + l), &t)) {
                        li_clear(&t);
                        return -1;
                    }
                    li_clear(&t);
                }
    return 0;
}
static int lp2_one(lp2 *p) {
    if (lp2_init(p, 1, 1))
        return -1;
    li_si(lp2_at(p, 0, 0), 1);
    return 0;
}
static int lp2_powers(lcontext *c, const lp2 *p, size_t n, lp2 **out) {
    lp2 *a = calloc(n + 1, sizeof(*a));
    if (!a)
        return -1;
    if (lp2_one(&a[0])) {
        free(a);
        return -1;
    }
    for (size_t i = 1; i <= n; i++)
        if (lp2_mul(c, &a[i], &a[i - 1], p)) {
            for (size_t k = 0; k <= n; k++)
                lp2_clear(&a[k]);
            free(a);
            return -1;
        }
    *out = a;
    return 0;
}
static void lp2_powers_clear(lp2 *p, size_t n) {
    if (p) {
        for (size_t i = 0; i <= n; i++)
            lp2_clear(&p[i]);
        free(p);
    }
}
static int lp2_substitute(lcontext *c, lp2 *z, const lp2 *p, const lp2 *u,
                          const lp2 *s) {
    lp2 *up = NULL, *sp = NULL;
    if (lp2_powers(c, u, p->nu - 1, &up) || lp2_powers(c, s, p->ns - 1, &sp))
        goto fail;
    size_t nu = 1, ns = 1;
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            size_t a = up[i].nu + sp[j].nu - 1, b = up[i].ns + sp[j].ns - 1;
            if (a > nu)
                nu = a;
            if (b > ns)
                ns = b;
        }
    if (lp2_init(z, nu, ns))
        goto fail;
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            if (li_exact_zero(lp2_cat(p, i, j)))
                continue;
            lp2 term = {0};
            if (lp2_mul(c, &term, &up[i], &sp[j]))
                goto fail_z;
            mpq_t one;
            mpq_init(one);
            mpq_set_ui(one, 1, 1);
            for (size_t a = 0; a < term.nu; a++)
                for (size_t b = 0; b < term.ns; b++) {
                    if (li_exact_zero(lp2_cat(&term, a, b)))
                        continue;
                    li q;
                    li_init(&q);
                    if (li_mul(c, &q, lp2_cat(&term, a, b), lp2_cat(p, i, j)) ||
                        lp2_accum(c, lp2_at(z, a, b), &q)) {
                        li_clear(&q);
                        mpq_clear(one);
                        lp2_clear(&term);
                        goto fail_z;
                    }
                    li_clear(&q);
                }
            mpq_clear(one);
            lp2_clear(&term);
        }
    lp2_powers_clear(up, p->nu - 1);
    lp2_powers_clear(sp, p->ns - 1);
    return 0;
fail_z:
    lp2_clear(z);
fail:
    lp2_powers_clear(up, p->nu - 1);
    lp2_powers_clear(sp, p->ns - 1);
    return -1;
}
static int lp2_partial(lcontext *c, lp2 *z, const lp2 *p, int dim) {
    size_t nu = dim == 0 ? (p->nu > 1 ? p->nu - 1 : 1) : p->nu,
           ns = dim == 1 ? (p->ns > 1 ? p->ns - 1 : 1) : p->ns;
    if (lp2_init(z, nu, ns))
        return -1;
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            size_t e = dim == 0 ? i : j;
            if (!e)
                continue;
            size_t a = dim == 0 ? i - 1 : i, b = dim == 1 ? j - 1 : j;
            if (li_scale_si(c, lp2_at(z, a, b), lp2_cat(p, i, j), (long)e))
                return -1;
        }
    return 0;
}
static int lp2_cone(lcontext *c, li *z, const lp2 *p, const li *t, const li *v,
                    int orient, int scaled, int time) {
    li sum, tp, vp, q;
    li_init(&sum);
    li_init(&tp);
    li_init(&vp);
    li_init(&q);
    li_si(&sum, 0);
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            if (li_exact_zero(lp2_cat(p, i, j)))
                continue;
            size_t e = i + j;
            if (scaled && e == 0)
                continue;
            if (scaled)
                e--;
            if (li_pow(c, &tp, t, e) || li_pow(c, &vp, v, j) ||
                li_mul(c, &q, &tp, &vp) || li_mul(c, &q, &q, lp2_cat(p, i, j)))
                goto fail;
            int sg = (i & 1) ? orient : 1;
            if (scaled)
                sg *= time;
            if (sg < 0 && li_neg(c, &q, &q))
                goto fail;
            if (li_add(c, &sum, &sum, &q))
                goto fail;
        }
    li_set(z, &sum);
    c->result->work.interval_evaluations++;
    li_clear(&sum);
    li_clear(&tp);
    li_clear(&vp);
    li_clear(&q);
    return 0;
fail:
    li_clear(&sum);
    li_clear(&tp);
    li_clear(&vp);
    li_clear(&q);
    return -1;
}

static int shift_intervals(lcontext *c, const lpoly *p, const li *root, li **out) {
    li *a = calloc(p->n, sizeof(*a));
    if (!a)
        return -1;
    for (size_t i = 0; i < p->n; i++)
        li_init(&a[i]);
    lpoly d = {0};
    if (lpoly_init(&d, p->n))
        goto fail;
    for (size_t i = 0; i < p->n; i++)
        mpq_set(d.c[i], p->c[i]);
    mpz_t fact;
    mpz_init_set_ui(fact, 1);
    mpq_t inv;
    mpq_init(inv);
    for (size_t k = 0; k < p->n; k++) {
        if (k > 1)
            mpz_mul_ui(fact, fact, (unsigned long)k);
        if (lpoly_interval(c, &a[k], &d, root))
            goto loop_fail;
        mpq_set_ui(inv, 1, 1);
        mpz_set(mpq_denref(inv), fact);
        mpq_canonicalize(inv);
        if (li_scale(c, &a[k], &a[k], inv))
            goto loop_fail;
        if (k + 1 < p->n) {
            lpoly next = {0};
            if (lpoly_deriv(c, &next, &d))
                goto loop_fail;
            lpoly_clear(&d);
            d = next;
        }
    }
    lpoly_clear(&d);
    mpz_clear(fact);
    mpq_clear(inv);
    *out = a;
    return 0;
loop_fail:
    lpoly_clear(&d);
    mpz_clear(fact);
    mpq_clear(inv);
fail:
    for (size_t i = 0; i < p->n; i++)
        li_clear(&a[i]);
    free(a);
    return -1;
}
static void shift_clear(li *a, size_t n) {
    if (a) {
        for (size_t i = 0; i < n; i++)
            li_clear(&a[i]);
        free(a);
    }
}

typedef struct {
    lpoly A, B, N, Ap, Bp;
    mpq_t C;
    li ac, bc, critical_loss;
    lp2 fu, fs, loss, y, da, db, jac;
} local_field;
static void local_field_init(local_field *f) {
    memset(f, 0, sizeof(*f));
    mpq_init(f->C);
    li_init(&f->ac);
    li_init(&f->bc);
    li_init(&f->critical_loss);
}
static void local_field_clear(local_field *f) {
    lpoly_clear(&f->A);
    lpoly_clear(&f->B);
    lpoly_clear(&f->N);
    lpoly_clear(&f->Ap);
    lpoly_clear(&f->Bp);
    mpq_clear(f->C);
    li_clear(&f->ac);
    li_clear(&f->bc);
    li_clear(&f->critical_loss);
    lp2_clear(&f->fu);
    lp2_clear(&f->fs);
    lp2_clear(&f->loss);
    lp2_clear(&f->y);
    lp2_clear(&f->da);
    lp2_clear(&f->db);
    lp2_clear(&f->jac);
}

static int verify_identity(lcontext *c, const local_field *f) {
    size_t n = f->Ap.n + f->B.n - 1, m = f->Bp.n + f->A.n - 1, k = n > m ? n : m;
    mpq_t *v = calloc(k, sizeof(mpq_t));
    if (!v)
        return -1;
    for (size_t i = 0; i < k; i++)
        mpq_init(v[i]);
    mpq_t q;
    mpq_init(q);
    for (size_t i = 0; i < f->Ap.n; i++)
        for (size_t j = 0; j < f->B.n; j++) {
            mpq_mul(q, f->Ap.c[i], f->B.c[j]);
            mpq_add(v[i + j], v[i + j], q);
        }
    for (size_t i = 0; i < f->Bp.n; i++)
        for (size_t j = 0; j < f->A.n; j++) {
            mpq_mul(q, f->Bp.c[i], f->A.c[j]);
            mpq_mul_2exp(q, q, 1);
            mpq_sub(v[i + j], v[i + j], q);
        }
    size_t vn = k;
    while (vn > 1 && mpq_sgn(v[vn - 1]) == 0)
        vn--;
    size_t nn = lpoly_len(&f->N);
    int ok = vn == nn;
    for (size_t i = 0; i < vn && ok; i++)
        ok = mpq_cmp(v[i], f->N.c[i]) == 0;
    mpq_clear(q);
    for (size_t i = 0; i < k; i++)
        mpq_clear(v[i]);
    free(v);
    if (!ok) {
        c->result->status = SPONG_SMALE_MODEL_IDENTITY_FAILURE;
        c->result->primary_reason = SPONG_SMALE_REASON_N_IDENTITY;
        return -1;
    }
    return 0;
}

static int verify_root(lcontext *c, const lpoly *p, const li *root) {
    mpq_t a, b;
    mpq_init(a);
    mpq_init(b);
    lpoly d = {0};
    li di;
    li_init(&di);
    int ok = 0;
    if (lpoly_eval_q(c, a, p, root->lo) || lpoly_eval_q(c, b, p, root->hi) ||
        lpoly_deriv(c, &d, p) || lpoly_interval(c, &di, &d, root))
        goto done;
    if (li_zero(&di))
        goto done;
    if (mpq_cmp(root->lo, root->hi) == 0)
        ok = mpq_sgn(a) == 0;
    else
        ok = (mpq_sgn(a) <= 0 && mpq_sgn(b) >= 0) ||
             (mpq_sgn(a) >= 0 && mpq_sgn(b) <= 0);
done:
    mpq_clear(a);
    mpq_clear(b);
    lpoly_clear(&d);
    li_clear(&di);
    if (!ok) {
        c->result->status = SPONG_SMALE_INVALID_ARGUMENT;
        c->result->primary_reason = SPONG_SMALE_REASON_ROOT_INTERVAL;
        return -1;
    }
    return 0;
}

static int build_field(lcontext *c, local_field *f, const spong_exact_loss_pencil *p,
                       const spong_local_launch_request *r) {
    if (lpoly_parse(c, &f->A, p->alpha, p->alpha_count) ||
        lpoly_parse(c, &f->B, p->beta, p->beta_count) ||
        lpoly_parse(c, &f->N, p->n, p->n_count) || lparse(c, &p->loss_constant, f->C) ||
        lpoly_deriv(c, &f->Ap, &f->A) || lpoly_deriv(c, &f->Bp, &f->B) ||
        lparse(c, &r->critical_b.lower, f->bc.lo) ||
        lparse(c, &r->critical_b.upper, f->bc.hi))
        return -1;
    if (mpq_cmp(f->bc.lo, f->bc.hi) > 0)
        return -1;
    if (c->policy->verify_n_identity && verify_identity(c, f))
        return -1;
    if (verify_root(c, r->critical_source == SPONG_CRITICAL_ROOT_B ? &f->B : &f->N,
                    &f->bc))
        return -1;
    li *A = NULL, *B = NULL, *Ap = NULL, *Bp = NULL;
    li Ai, Bi, tmp, tmp2;
    li_init(&Ai);
    li_init(&Bi);
    li_init(&tmp);
    li_init(&tmp2);
    if (shift_intervals(c, &f->A, &f->bc, &A) ||
        shift_intervals(c, &f->B, &f->bc, &B) ||
        shift_intervals(c, &f->Ap, &f->bc, &Ap) ||
        shift_intervals(c, &f->Bp, &f->bc, &Bp))
        goto fail;
    if (lpoly_interval(c, &Ai, &f->A, &f->bc) || mpq_sgn(Ai.lo) <= 0 ||
        lpoly_interval(c, &Bi, &f->B, &f->bc) || li_div(c, &f->ac, &Bi, &Ai))
        goto fail;
    /* Independently verify the point is a saddle.  For this loss pencil the
     * exact Hessian entries at the algebraic centre are
     * 2A, 2(aA'-B'), and a^2 A''-2aB''. */
    {
        li haa, hab, hbb, det, square, product;
        li_init(&haa);
        li_init(&hab);
        li_init(&hbb);
        li_init(&det);
        li_init(&square);
        li_init(&product);
        int bad = li_scale_si(c, &haa, &A[0], 2) || li_mul(c, &hab, &f->ac, &Ap[0]) ||
                  li_sub(c, &hab, &hab, &Bp[0]) || li_scale_si(c, &hab, &hab, 2);
        li_si(&hbb, 0);
        if (!bad && f->Ap.n > 1) {
            bad = li_square(c, &hbb, &f->ac) || li_mul(c, &hbb, &hbb, &Ap[1]);
        }
        if (!bad && f->Bp.n > 1) {
            li term;
            li_init(&term);
            bad = li_mul(c, &term, &f->ac, &Bp[1]) || li_scale_si(c, &term, &term, 2) ||
                  li_sub(c, &hbb, &hbb, &term);
            li_clear(&term);
        }
        if (!bad)
            bad = li_mul(c, &product, &haa, &hbb) || li_square(c, &square, &hab) ||
                  li_sub(c, &det, &product, &square);
        int saddle = !bad && mpq_sgn(det.hi) < 0;
        li_clear(&haa);
        li_clear(&hab);
        li_clear(&hbb);
        li_clear(&det);
        li_clear(&square);
        li_clear(&product);
        if (!saddle) {
            c->result->status = SPONG_SMALE_INVALID_ARGUMENT;
            c->result->primary_reason = SPONG_SMALE_REASON_CRITICAL_POINT;
            goto fail;
        }
    }
    size_t gans = f->A.n > f->B.n ? f->A.n : f->B.n,
           gbns = f->Ap.n > f->Bp.n ? f->Ap.n : f->Bp.n;
    lp2 ga = {0}, gb = {0}, U = {0}, S = {0}, pga = {0}, pgb = {0};
    if (lp2_init(&ga, 2, gans) || lp2_init(&gb, 3, gbns))
        goto polyfail;
    for (size_t j = 0; j < gans; j++) {
        li aj, bj;
        li_init(&aj);
        li_init(&bj);
        li_si(&aj, 0);
        li_si(&bj, 0);
        if (j < f->A.n)
            li_set(&aj, &A[j]);
        if (j < f->B.n)
            li_set(&bj, &B[j]);
        if (j && (li_mul(c, &tmp, &f->ac, &aj) || li_sub(c, &tmp, &tmp, &bj) ||
                  li_scale_si(c, lp2_at(&ga, 0, j), &tmp, 2))) {
            li_clear(&aj);
            li_clear(&bj);
            goto polyfail;
        }
        if (li_scale_si(c, lp2_at(&ga, 1, j), &aj, 2)) {
            li_clear(&aj);
            li_clear(&bj);
            goto polyfail;
        }
        li_clear(&aj);
        li_clear(&bj);
    }
    for (size_t j = 0; j < gbns; j++) {
        li aj, bj;
        li_init(&aj);
        li_init(&bj);
        li_si(&aj, 0);
        li_si(&bj, 0);
        if (j < f->Ap.n)
            li_set(&aj, &Ap[j]);
        if (j < f->Bp.n)
            li_set(&bj, &Bp[j]);
        if (j) {
            if (li_mul(c, &tmp, &f->ac, &f->ac) || li_mul(c, &tmp, &tmp, &aj) ||
                li_mul(c, &tmp2, &f->ac, &bj) || li_scale_si(c, &tmp2, &tmp2, 2) ||
                li_sub(c, lp2_at(&gb, 0, j), &tmp, &tmp2)) {
                li_clear(&aj);
                li_clear(&bj);
                goto polyfail;
            }
        }
        if (li_mul(c, &tmp, &f->ac, &aj) || li_scale_si(c, &tmp, &tmp, 2) ||
            li_scale_si(c, &tmp2, &bj, 2) ||
            li_sub(c, lp2_at(&gb, 1, j), &tmp, &tmp2)) {
            li_clear(&aj);
            li_clear(&bj);
            goto polyfail;
        }
        li_set(lp2_at(&gb, 2, j), &aj);
        li_clear(&aj);
        li_clear(&bj);
    }
    if (lp2_init(&f->loss, ga.nu + 1, (ga.ns > (gb.ns + 1) ? ga.ns : gb.ns + 1)))
        goto polyfail;
    for (size_t i = 0; i < ga.nu; i++)
        for (size_t j = 0; j < ga.ns; j++) {
            mpq_t q;
            mpq_init(q);
            mpq_set_ui(q, 1, (unsigned long)(i + 1));
            li_scale(c, &tmp, lp2_cat(&ga, i, j), q);
            lp2_accum(c, lp2_at(&f->loss, i + 1, j), &tmp);
            mpq_clear(q);
        }
    for (size_t j = 0; j < gb.ns; j++) {
        mpq_t q;
        mpq_init(q);
        mpq_set_ui(q, 1, (unsigned long)(j + 1));
        li_scale(c, &tmp, lp2_cat(&gb, 0, j), q);
        lp2_accum(c, lp2_at(&f->loss, 0, j + 1), &tmp);
        mpq_clear(q);
    }
    mpq_t frame[4], map[6], one;
    for (int i = 0; i < 4; i++)
        mpq_init(frame[i]);
    for (int i = 0; i < 6; i++)
        mpq_init(map[i]);
    mpq_init(one);
    mpq_set_ui(one, 1, 1);
    for (int i = 0; i < 4; i++)
        if (lparse(c, &r->frame[i], frame[i]))
            goto coordsfail;
    for (int i = 0; i < 6; i++)
        if (lparse(c, &r->selected_map[i], map[i]))
            goto coordsfail;
    mpq_mul(tmp.lo, frame[0], frame[3]);
    mpq_mul(tmp.hi, frame[1], frame[2]);
    mpq_sub(tmp.lo, tmp.lo, tmp.hi);
    if (mpq_sgn(tmp.lo) == 0) {
        c->result->status = SPONG_SMALE_INVALID_ARGUMENT;
        c->result->primary_reason = SPONG_SMALE_REASON_FRAME_SINGULAR;
        goto coordsfail;
    }
    if (lp2_init(&U, 3, 3) || lp2_init(&S, 3, 3))
        goto coordsfail;
    li_si(lp2_at(&U, 1, 0), 1);
    li_si(lp2_at(&S, 0, 1), 1);
    li_point(lp2_at(&U, 2, 0), map[0]);
    li_point(lp2_at(&U, 1, 1), map[1]);
    li_point(lp2_at(&U, 0, 2), map[2]);
    li_point(lp2_at(&S, 2, 0), map[3]);
    li_point(lp2_at(&S, 1, 1), map[4]);
    li_point(lp2_at(&S, 0, 2), map[5]);
    if (lp2_lincomb(c, &f->da, &U, frame[0], &S, frame[1]) ||
        lp2_lincomb(c, &f->db, &U, frame[2], &S, frame[3]) ||
        lp2_substitute(c, &pga, &ga, &f->da, &f->db) ||
        lp2_substitute(c, &pgb, &gb, &f->da, &f->db))
        goto coordsfail;
    if (lp2_init(&f->y, pga.nu, pga.ns))
        goto coordsfail;
    mpq_t half;
    mpq_init(half);
    mpq_set_ui(half, 1, 2);
    if (lp2_add_scaled(c, &f->y, &pga, half)) {
        mpq_clear(half);
        goto coordsfail;
    }
    mpq_clear(half);
    {
        lp2 transformed = {0};
        if (lp2_substitute(c, &transformed, &f->loss, &f->da, &f->db))
            goto coordsfail;
        lp2_clear(&f->loss);
        f->loss = transformed;
    }
    lp2 da_u = {0}, da_s = {0}, db_u = {0}, db_s = {0}, x = {0}, y = {0};
    if (lp2_partial(c, &da_u, &f->da, 0) || lp2_partial(c, &da_s, &f->da, 1) ||
        lp2_partial(c, &db_u, &f->db, 0) || lp2_partial(c, &db_s, &f->db, 1))
        goto derivfail;
    if (lp2_mul(c, &x, &da_u, &db_s) || lp2_mul(c, &y, &da_s, &db_u))
        goto derivfail;
    mpq_t neg;
    mpq_init(neg);
    mpq_set_si(neg, -1, 1);
    if (lp2_lincomb(c, &f->jac, &x, one, &y, neg)) {
        mpq_clear(neg);
        goto derivfail;
    }
    lp2_clear(&x);
    lp2_clear(&y);
    int sgn = mpq_sgn(tmp.lo) > 0 ? 1 : -1;
    mpq_set_si(neg, -sgn, 1);
    mpq_t pos;
    mpq_init(pos);
    mpq_set_si(pos, sgn, 1);
    if (lp2_mul(c, &x, &db_s, &pga) || lp2_mul(c, &y, &da_s, &pgb) ||
        lp2_lincomb(c, &f->fu, &x, pos, &y, neg))
        goto signedfail;
    lp2_clear(&x);
    lp2_clear(&y);
    if (lp2_mul(c, &x, &db_u, &pga) || lp2_mul(c, &y, &da_u, &pgb)) {
        goto signedfail;
    }
    mpq_set_si(neg, -sgn, 1);
    if (lp2_lincomb(c, &f->fs, &x, neg, &y, pos))
        goto signedfail;
    {
        lp2 old = f->jac;
        memset(&f->jac, 0, sizeof(f->jac));
        mpq_set_si(pos, sgn, 1);
        if (lp2_init(&f->jac, old.nu, old.ns) ||
            lp2_add_scaled(c, &f->jac, &old, pos)) {
            lp2_clear(&old);
            goto signedfail;
        }
        lp2_clear(&old);
    }
    li Bsq, quot, Ci;
    li_init(&Bsq);
    li_init(&quot);
    li_init(&Ci);
    li_point(&Ci, f->C);
    if (li_mul(c, &Bsq, &Bi, &Bi) || li_div(c, &quot, &Bsq, &Ai) ||
        li_sub(c, &f->critical_loss, &Ci, &quot)) {
        li_clear(&Bsq);
        li_clear(&quot);
        li_clear(&Ci);
        goto signedfail;
    }
    li_clear(&Bsq);
    li_clear(&quot);
    li_clear(&Ci);
    c->result->work.coefficient_intervals = f->fu.nu * f->fu.ns + f->fs.nu * f->fs.ns +
                                            f->loss.nu * f->loss.ns + f->y.nu * f->y.ns;
    mpq_clear(pos);
    mpq_clear(neg);
    lp2_clear(&x);
    lp2_clear(&y);
    lp2_clear(&da_u);
    lp2_clear(&da_s);
    lp2_clear(&db_u);
    lp2_clear(&db_s);
    for (int i = 0; i < 4; i++)
        mpq_clear(frame[i]);
    for (int i = 0; i < 6; i++)
        mpq_clear(map[i]);
    mpq_clear(one);
    lp2_clear(&ga);
    lp2_clear(&gb);
    lp2_clear(&U);
    lp2_clear(&S);
    lp2_clear(&pga);
    lp2_clear(&pgb);
    shift_clear(A, f->A.n);
    shift_clear(B, f->B.n);
    shift_clear(Ap, f->Ap.n);
    shift_clear(Bp, f->Bp.n);
    li_clear(&Ai);
    li_clear(&Bi);
    li_clear(&tmp);
    li_clear(&tmp2);
    return 0;
signedfail:
    mpq_clear(pos);
    mpq_clear(neg);
    lp2_clear(&x);
    lp2_clear(&y);
derivfail:
    lp2_clear(&da_u);
    lp2_clear(&da_s);
    lp2_clear(&db_u);
    lp2_clear(&db_s);
coordsfail:
    for (int i = 0; i < 4; i++)
        mpq_clear(frame[i]);
    for (int i = 0; i < 6; i++)
        mpq_clear(map[i]);
    mpq_clear(one);
polyfail:
    lp2_clear(&ga);
    lp2_clear(&gb);
    lp2_clear(&U);
    lp2_clear(&S);
    lp2_clear(&pga);
    lp2_clear(&pgb);
fail:
    shift_clear(A, f->A.n);
    shift_clear(B, f->B.n);
    shift_clear(Ap, f->Ap.n);
    shift_clear(Bp, f->Bp.n);
    li_clear(&Ai);
    li_clear(&Bi);
    li_clear(&tmp);
    li_clear(&tmp2);
    return -1;
}

static int cone_eval(lcontext *c, li *z, const lp2 *p, const li *t, const li *v, int o,
                     int time) {
    return lp2_cone(c, z, p, t, v, o, 1, time);
}
static int value_eval(lcontext *c, li *z, const lp2 *p, const li *t, const li *v,
                      int o) {
    return lp2_cone(c, z, p, t, v, o, 0, 1);
}

typedef struct {
    mpq_t reach, slope, flow, lower, upper;
    li tangent;
    int power;
} accepted_cone;
static void cone_init(accepted_cone *x) {
    mpq_inits(x->reach, x->slope, x->flow, x->lower, x->upper, NULL);
    li_init(&x->tangent);
    x->power = 0;
}
static void cone_clear(accepted_cone *x) {
    mpq_clears(x->reach, x->slope, x->flow, x->lower, x->upper, NULL);
    li_clear(&x->tangent);
}

static int linear_cone(lcontext *c, const local_field *f, const mpq_t reach,
                       const mpq_t slope, int o, int time, accepted_cone *out) {
    li t, v, j, ft, lfs, lft, ufs, uft, q;
    li_init(&t);
    li_init(&v);
    li_init(&j);
    li_init(&ft);
    li_init(&lfs);
    li_init(&lft);
    li_init(&ufs);
    li_init(&uft);
    li_init(&q);
    mpq_set_ui(t.lo, 0, 1);
    mpq_set(t.hi, reach);
    mpq_neg(v.lo, slope);
    mpq_set(v.hi, slope);
    int ok = !lp2_cone(c, &j, &f->jac, &t, &v, o, 0, 1) &&
             !cone_eval(c, &ft, &f->fu, &t, &v, o, time);
    if (o < 0)
        li_neg(c, &ft, &ft);
    li_point(&v, slope);
    if (ok)
        ok = !cone_eval(c, &ufs, &f->fs, &t, &v, o, time) &&
             !cone_eval(c, &uft, &f->fu, &t, &v, o, time);
    if (o < 0)
        li_neg(c, &uft, &uft);
    mpq_neg(v.lo, slope);
    mpq_set(v.hi, v.lo);
    if (ok)
        ok = !cone_eval(c, &lfs, &f->fs, &t, &v, o, time) &&
             !cone_eval(c, &lft, &f->fu, &t, &v, o, time);
    if (o < 0)
        li_neg(c, &lft, &lft);
    if (ok) {
        mpq_mul(q.lo, slope, lft.lo);
        mpq_add(q.lo, q.lo, lfs.lo);
        mpq_mul(q.hi, slope, uft.lo);
        mpq_sub(q.hi, q.hi, ufs.hi);
        ok = mpq_sgn(j.lo) > 0 && mpq_sgn(ft.lo) > 0 && mpq_sgn(q.lo) > 0 &&
             mpq_sgn(q.hi) > 0;
        if (ok) {
            mpq_set(out->reach, reach);
            mpq_set(out->slope, slope);
            mpq_set(out->flow, ft.lo);
            mpq_set(out->lower, q.lo);
            mpq_set(out->upper, q.hi);
            out->power = 1;
        }
    }
    li_clear(&t);
    li_clear(&v);
    li_clear(&j);
    li_clear(&ft);
    li_clear(&lfs);
    li_clear(&lft);
    li_clear(&ufs);
    li_clear(&uft);
    li_clear(&q);
    return ok ? 0 : 1;
}

static int tangent_defect(lcontext *c, const local_field *f, const mpq_t slope, int o,
                          int time, li *out) {
    li zero, v, U, V, q;
    li_init(&zero);
    li_init(&v);
    li_init(&U);
    li_init(&V);
    li_init(&q);
    li_si(&zero, 0);
    li_point(&v, slope);
    int s = cone_eval(c, &U, &f->fu, &zero, &v, o, time) ||
            cone_eval(c, &V, &f->fs, &zero, &v, o, time);
    if (o < 0)
        li_neg(c, &U, &U);
    if (!s) {
        li_scale(c, &q, &U, slope);
        s = li_sub(c, out, &V, &q);
    }
    li_clear(&zero);
    li_clear(&v);
    li_clear(&U);
    li_clear(&V);
    li_clear(&q);
    return s;
}
static int tangent_bracket(lcontext *c, const local_field *f, const mpq_t linear, int o,
                           int time, li *out) {
    mpq_neg(out->lo, linear);
    mpq_set(out->hi, linear);
    li a, b;
    li_init(&a);
    li_init(&b);
    if (tangent_defect(c, f, out->lo, o, time, &a) ||
        tangent_defect(c, f, out->hi, o, time, &b)) {
        li_clear(&a);
        li_clear(&b);
        return -1;
    }
    if (mpq_sgn(a.lo) <= 0 || mpq_sgn(b.hi) >= 0) {
        li_clear(&a);
        li_clear(&b);
        return 0;
    }
    mpq_t mid;
    mpq_init(mid);
    for (uint64_t k = 0; k < c->policy->tangent_bisections; k++) {
        mpq_add(mid, out->lo, out->hi);
        mpq_div_2exp(mid, mid, 1);
        if (tangent_defect(c, f, mid, o, time, &a)) {
            mpq_clear(mid);
            li_clear(&a);
            li_clear(&b);
            return -1;
        }
        c->result->work.tangent_bisections++;
        if (mpq_sgn(a.lo) > 0)
            mpq_set(out->lo, mid);
        else if (mpq_sgn(a.hi) < 0)
            mpq_set(out->hi, mid);
        else if (li_exact_zero(&a)) {
            mpq_set(out->lo, mid);
            mpq_set(out->hi, mid);
            break;
        } else
            break;
    }
    mpq_clear(mid);
    li_clear(&a);
    li_clear(&b);
    return 0;
}

typedef struct {
    size_t n;
    li *c;
} lipoly;
static int lip_init(lipoly *p, size_t n) {
    memset(p, 0, sizeof(*p));
    if (!n)
        n = 1;
    p->c = calloc(n, sizeof(li));
    if (!p->c)
        return -1;
    p->n = n;
    for (size_t i = 0; i < n; i++)
        li_init(&p->c[i]);
    return 0;
}
static void lip_clear(lipoly *p) {
    if (p->c) {
        for (size_t i = 0; i < p->n; i++)
            li_clear(&p->c[i]);
        free(p->c);
    }
    memset(p, 0, sizeof(*p));
}
static int lip_add(lcontext *c, lipoly *z, const lipoly *a, const lipoly *b) {
    size_t n = a->n > b->n ? a->n : b->n;
    if (lip_init(z, n))
        return -1;
    for (size_t i = 0; i < n; i++) {
        if (i < a->n)
            li_set(&z->c[i], &a->c[i]);
        if (i < b->n && lp2_accum(c, &z->c[i], &b->c[i]))
            return -1;
    }
    return 0;
}
static int lip_scale(lcontext *c, lipoly *z, const lipoly *p, const li *q) {
    if (lip_init(z, p->n))
        return -1;
    for (size_t i = 0; i < p->n; i++) {
        if (li_exact_zero(&p->c[i]))
            continue;
        if (li_mul(c, &z->c[i], &p->c[i], q))
            return -1;
    }
    return 0;
}
static int lip_mul(lcontext *c, lipoly *z, const lipoly *a, const lipoly *b) {
    if (lip_init(z, a->n + b->n - 1))
        return -1;
    for (size_t i = 0; i < a->n; i++)
        for (size_t j = 0; j < b->n; j++) {
            if (li_exact_zero(&a->c[i]) || li_exact_zero(&b->c[j]))
                continue;
            li q;
            li_init(&q);
            if (li_mul(c, &q, &a->c[i], &b->c[j]) || lp2_accum(c, &z->c[i + j], &q)) {
                li_clear(&q);
                return -1;
            }
            li_clear(&q);
        }
    return 0;
}
static int lip_power(lcontext *c, lipoly *z, const lipoly *p, size_t n) {
    lipoly a = {0};
    if (lip_init(&a, 1))
        return -1;
    li_si(&a.c[0], 1);
    for (size_t i = 0; i < n; i++) {
        lipoly next = {0};
        if (lip_mul(c, &next, &a, p)) {
            lip_clear(&a);
            return -1;
        }
        lip_clear(&a);
        a = next;
    }
    *z = a;
    return 0;
}
static int lip_eval_punctured(lcontext *c, li *z, const lipoly *p, const li *t) {
    size_t first = 0;
    while (first + 1 < p->n && li_exact_zero(&p->c[first]))
        first++;
    li a;
    li_init(&a);
    li_si(&a, 0);
    for (size_t k = p->n; k-- > first;) {
        if (li_mul(c, &a, &a, t) || li_add(c, &a, &a, &p->c[k])) {
            li_clear(&a);
            return -1;
        }
    }
    li_set(z, &a);
    c->result->work.interval_evaluations++;
    li_clear(&a);
    return 0;
}

static int scaled_affine_face(lcontext *c, lipoly *z, const lp2 *p, const mpq_t tangent,
                              const mpq_t curvature, int o, int time) {
    lipoly v = {0};
    if (lip_init(&v, 2))
        return -1;
    li_point(&v.c[0], tangent);
    li_point(&v.c[1], curvature);
    size_t max_degree = 0;
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++)
            if (!li_exact_zero(lp2_cat(p, i, j))) {
                size_t degree = i + 2 * j - 1;
                if (degree > max_degree)
                    max_degree = degree;
            }
    if (lip_init(z, max_degree + 1)) {
        lip_clear(&v);
        return -1;
    }
    for (size_t i = 0; i < p->nu; i++)
        for (size_t j = 0; j < p->ns; j++) {
            const li *coefficient = lp2_cat(p, i, j);
            if (li_exact_zero(coefficient) || i + j == 0)
                continue;
            lipoly power = {0};
            if (lip_power(c, &power, &v, j))
                goto fail;
            size_t shift = i + j - 1;
            int sign = ((i & 1) ? o : 1) * time;
            li scale;
            li_init(&scale);
            if (sign > 0)
                li_set(&scale, coefficient);
            else if (li_neg(c, &scale, coefficient)) {
                li_clear(&scale);
                lip_clear(&power);
                goto fail;
            }
            for (size_t k = 0; k < power.n; k++) {
                li q;
                li_init(&q);
                if (li_mul(c, &q, &power.c[k], &scale) ||
                    lp2_accum(c, &z->c[k + shift], &q)) {
                    li_clear(&q);
                    li_clear(&scale);
                    lip_clear(&power);
                    goto fail;
                }
                li_clear(&q);
            }
            li_clear(&scale);
            lip_clear(&power);
        }
    lip_clear(&v);
    return 0;
fail:
    lip_clear(&v);
    lip_clear(z);
    return -1;
}

static int frobenius_defect(lcontext *c, lipoly *z, const local_field *f,
                            const mpq_t tangent, const mpq_t curvature, int o, int time,
                            int upper) {
    lipoly U = {0}, V = {0}, slope = {0}, product = {0}, negative = {0};
    if (scaled_affine_face(c, &U, &f->fu, tangent, curvature, o, time) ||
        scaled_affine_face(c, &V, &f->fs, tangent, curvature, o, time) ||
        lip_init(&slope, 2))
        goto fail;
    if (o < 0) {
        li minus;
        li_init(&minus);
        li_si(&minus, -1);
        if (lip_scale(c, &negative, &U, &minus)) {
            li_clear(&minus);
            goto fail;
        }
        li_clear(&minus);
        lip_clear(&U);
        U = negative;
        memset(&negative, 0, sizeof(negative));
    }
    li_point(&slope.c[0], tangent);
    mpq_t twice;
    mpq_init(twice);
    mpq_mul_2exp(twice, curvature, 1);
    li_point(&slope.c[1], twice);
    mpq_clear(twice);
    if (lip_mul(c, &product, &slope, &U))
        goto fail;
    li minus;
    li_init(&minus);
    li_si(&minus, -1);
    int status;
    if (upper) {
        if (lip_scale(c, &negative, &V, &minus)) {
            li_clear(&minus);
            goto fail;
        }
        status = lip_add(c, z, &product, &negative);
    } else {
        if (lip_scale(c, &negative, &product, &minus)) {
            li_clear(&minus);
            goto fail;
        }
        status = lip_add(c, z, &V, &negative);
    }
    li_clear(&minus);
    lip_clear(&U);
    lip_clear(&V);
    lip_clear(&slope);
    lip_clear(&product);
    lip_clear(&negative);
    return status;
fail:
    lip_clear(&U);
    lip_clear(&V);
    lip_clear(&slope);
    lip_clear(&product);
    lip_clear(&negative);
    return -1;
}

static int frobenius_cone(lcontext *c, const local_field *f,
                          const accepted_cone *linear, int o, int time,
                          accepted_cone *out) {
    if (tangent_bracket(c, f, linear->slope, o, time, &out->tangent))
        return -1;
    mpq_t al, au, max, K, negative_K;
    mpq_inits(al, au, max, K, negative_K, NULL);
    mpq_add(al, out->tangent.lo, linear->slope);
    mpq_sub(au, linear->slope, out->tangent.hi);
    mpq_set(max, mpq_cmp(al, au) < 0 ? al : au);
    if (mpq_sgn(max) <= 0)
        goto no;
    mpq_div(max, max, linear->reach);
    mpq_div_2exp(K, max, 16);
    uint64_t parts =
        c->policy->frobenius_subdivisions ? c->policy->frobenius_subdivisions : 1;
    for (int attempt = 0; attempt < 17; attempt++) {
        lipoly lower_defect = {0}, upper_defect = {0};
        mpq_neg(negative_K, K);
        if (frobenius_defect(c, &lower_defect, f, out->tangent.lo, negative_K, o, time,
                             0) ||
            frobenius_defect(c, &upper_defect, f, out->tangent.hi, K, o, time, 1)) {
            lip_clear(&lower_defect);
            lip_clear(&upper_defect);
            goto failure;
        }
        int good = 1, have = 0;
        mpq_t mf, ml, mu;
        mpq_inits(mf, ml, mu, NULL);
        for (uint64_t k = 0; k < parts && good; k++) {
            li t, r, v, U, lower, upper;
            li_init(&t);
            li_init(&r);
            li_init(&v);
            li_init(&U);
            li_init(&lower);
            li_init(&upper);
            mpq_set(t.lo, linear->reach);
            mpz_mul_ui(mpq_numref(t.lo), mpq_numref(t.lo), (unsigned long)k);
            mpz_mul_ui(mpq_denref(t.lo), mpq_denref(t.lo), (unsigned long)parts);
            mpq_canonicalize(t.lo);
            mpq_set(t.hi, linear->reach);
            mpz_mul_ui(mpq_numref(t.hi), mpq_numref(t.hi), (unsigned long)(k + 1));
            mpz_mul_ui(mpq_denref(t.hi), mpq_denref(t.hi), (unsigned long)parts);
            mpq_canonicalize(t.hi);
            mpq_neg(r.lo, K);
            mpq_set(r.hi, K);
            if (li_mul(c, &v, &t, &r) || li_add(c, &v, &v, &out->tangent) ||
                cone_eval(c, &U, &f->fu, &t, &v, o, time)) {
                good = 0;
                goto slabdone;
            }
            if (o < 0)
                li_neg(c, &U, &U);
            if (mpq_sgn(U.lo) <= 0) {
                good = 0;
                goto slabdone;
            }
            if (lip_eval_punctured(c, &lower, &lower_defect, &t) ||
                lip_eval_punctured(c, &upper, &upper_defect, &t) ||
                mpq_sgn(lower.lo) <= 0 || mpq_sgn(upper.lo) <= 0) {
                good = 0;
                goto slabdone;
            }
            if (!have || mpq_cmp(U.lo, mf) < 0)
                mpq_set(mf, U.lo);
            if (!have || mpq_cmp(lower.lo, ml) < 0)
                mpq_set(ml, lower.lo);
            if (!have || mpq_cmp(upper.lo, mu) < 0)
                mpq_set(mu, upper.lo);
            have = 1;
        slabdone:
            li_clear(&t);
            li_clear(&r);
            li_clear(&v);
            li_clear(&U);
            li_clear(&lower);
            li_clear(&upper);
        }
        lip_clear(&lower_defect);
        lip_clear(&upper_defect);
        if (good && have) {
            mpq_set(out->reach, linear->reach);
            mpq_set(out->slope, K);
            mpq_set(out->flow, mf);
            mpq_set(out->lower, ml);
            mpq_set(out->upper, mu);
            out->power = 2;
            mpq_clears(mf, ml, mu, NULL);
            mpq_clears(al, au, max, K, negative_K, NULL);
            return 0;
        }
        mpq_clears(mf, ml, mu, NULL);
        mpq_mul_2exp(K, K, 1);
    }
no:
    mpq_clears(al, au, max, K, negative_K, NULL);
    return 1;
failure:
    mpq_clears(al, au, max, K, negative_K, NULL);
    return -1;
}

static int graph_value(lcontext *c, li *z, const lp2 *p, const mpq_t tvalue,
                       const accepted_cone *cone, int o) {
    li t, v, r;
    li_init(&t);
    li_init(&v);
    li_init(&r);
    li_point(&t, tvalue);
    if (cone->power == 1) {
        mpq_neg(v.lo, cone->slope);
        mpq_set(v.hi, cone->slope);
    } else {
        mpq_neg(r.lo, cone->slope);
        mpq_set(r.hi, cone->slope);
        if (li_mul(c, &v, &t, &r) || li_add(c, &v, &v, &cone->tangent)) {
            li_clear(&t);
            li_clear(&v);
            li_clear(&r);
            return -1;
        }
    }
    int s = value_eval(c, z, p, &t, &v, o);
    li_clear(&t);
    li_clear(&v);
    li_clear(&r);
    return s;
}
static int graph_interval(lcontext *c, li *z, const lp2 *p, const li *t,
                          const accepted_cone *cone, int o) {
    li v, r;
    li_init(&v);
    li_init(&r);
    if (cone->power == 1) {
        mpq_neg(v.lo, cone->slope);
        mpq_set(v.hi, cone->slope);
    } else {
        mpq_neg(r.lo, cone->slope);
        mpq_set(r.hi, cone->slope);
        if (li_mul(c, &v, t, &r) || li_add(c, &v, &v, &cone->tangent)) {
            li_clear(&v);
            li_clear(&r);
            return -1;
        }
    }
    int s = value_eval(c, z, p, t, &v, o);
    li_clear(&v);
    li_clear(&r);
    return s;
}

static int section(lcontext *c, const local_field *f, const accepted_cone *cone, int o,
                   int time, mpq_t level, li *bout, li *yout) {
    mpq_t inner, mid, signed_level;
    mpq_inits(inner, mid, signed_level, NULL);
    li lo, hi, x, tbox, Abox;
    li_init(&lo);
    li_init(&hi);
    li_init(&x);
    li_init(&tbox);
    li_init(&Abox);
    if (graph_value(c, &hi, &f->loss, cone->reach, cone, o))
        goto fail;
    li_add(c, &hi, &hi, &f->critical_loss);
    int found = 0;
    for (int e = 1; e <= 8; e++) {
        mpq_set(inner, cone->reach);
        mpq_div_2exp(inner, inner, (mp_bitcnt_t)e);
        if (graph_value(c, &lo, &f->loss, inner, cone, o))
            goto fail;
        li_add(c, &lo, &lo, &f->critical_loss);
        if (time > 0 && mpq_cmp(hi.lo, lo.hi) > 0) {
            mpq_add(level, lo.hi, hi.lo);
            mpq_div_2exp(level, level, 1);
            found = 1;
            break;
        }
        if (time < 0 && mpq_cmp(lo.lo, hi.hi) > 0) {
            mpq_add(level, hi.hi, lo.lo);
            mpq_div_2exp(level, level, 1);
            found = 1;
            break;
        }
    }
    if (!found)
        goto unresolved;
    mpq_set(signed_level, level);
    if (time < 0)
        mpq_neg(signed_level, signed_level);
    mpq_t lower_safe, upper_probe, lower_probe, upper_safe;
    mpq_inits(lower_safe, upper_probe, lower_probe, upper_safe, NULL);
    mpq_set(lower_safe, inner);
    mpq_set(upper_probe, cone->reach);
    mpq_set(lower_probe, inner);
    mpq_set(upper_safe, cone->reach);
    for (uint64_t k = 0; k < c->policy->section_bisections; k++) {
        mpq_add(mid, lower_safe, upper_probe);
        mpq_div_2exp(mid, mid, 1);
        if (graph_value(c, &x, &f->loss, mid, cone, o))
            goto bisfail;
        li_add(c, &x, &x, &f->critical_loss);
        if (time < 0)
            li_neg(c, &x, &x);
        if (mpq_cmp(x.hi, signed_level) < 0)
            mpq_set(lower_safe, mid);
        else
            mpq_set(upper_probe, mid);
        c->result->work.section_bisections++;
    }
    for (uint64_t k = 0; k < c->policy->section_bisections; k++) {
        mpq_add(mid, lower_probe, upper_safe);
        mpq_div_2exp(mid, mid, 1);
        if (graph_value(c, &x, &f->loss, mid, cone, o))
            goto bisfail;
        li_add(c, &x, &x, &f->critical_loss);
        if (time < 0)
            li_neg(c, &x, &x);
        if (mpq_cmp(x.lo, signed_level) > 0)
            mpq_set(upper_safe, mid);
        else
            mpq_set(lower_probe, mid);
        c->result->work.section_bisections++;
    }
    if (mpq_cmp(lower_safe, upper_safe) >= 0)
        goto bisunresolved;
    mpq_set(tbox.lo, lower_safe);
    mpq_set(tbox.hi, upper_safe);
    if (graph_interval(c, &x, &f->db, &tbox, cone, o) || li_add(c, bout, &f->bc, &x) ||
        graph_interval(c, yout, &f->y, &tbox, cone, o) ||
        lpoly_interval(c, &Abox, &f->A, bout) || li_zero(&Abox))
        goto bisunresolved;
    if (li_zero(yout))
        goto sheetunresolved;
    mpq_clears(lower_safe, upper_probe, lower_probe, upper_safe, NULL);
    mpq_clears(inner, mid, signed_level, NULL);
    li_clear(&lo);
    li_clear(&hi);
    li_clear(&x);
    li_clear(&tbox);
    li_clear(&Abox);
    return 0;
sheetunresolved:
    c->result->status = SPONG_SMALE_TUBE_UNRESOLVED;
    c->result->primary_reason = SPONG_SMALE_REASON_SECTION_SHEET;
    mpq_clears(lower_safe, upper_probe, lower_probe, upper_safe, NULL);
    mpq_clears(inner, mid, signed_level, NULL);
    li_clear(&lo);
    li_clear(&hi);
    li_clear(&x);
    li_clear(&tbox);
    li_clear(&Abox);
    return 1;
bisunresolved:
    mpq_clears(lower_safe, upper_probe, lower_probe, upper_safe, NULL);
unresolved:
    c->result->status = SPONG_SMALE_TUBE_UNRESOLVED;
    c->result->primary_reason = SPONG_SMALE_REASON_SECTION_UNRESOLVED;
    mpq_clears(inner, mid, signed_level, NULL);
    li_clear(&lo);
    li_clear(&hi);
    li_clear(&x);
    li_clear(&tbox);
    li_clear(&Abox);
    return 1;
bisfail:
    mpq_clears(lower_safe, upper_probe, lower_probe, upper_safe, NULL);
fail:
    mpq_clears(inner, mid, signed_level, NULL);
    li_clear(&lo);
    li_clear(&hi);
    li_clear(&x);
    li_clear(&tbox);
    li_clear(&Abox);
    return -1;
}

static int strict_dyadic_between(lcontext *c, mpq_t out, const mpq_t lower,
                                 const mpq_t upper) {
    mpz_t scaled, lower_integer, upper_integer, numerator;
    mpz_inits(scaled, lower_integer, upper_integer, numerator, NULL);
    for (uint64_t bits = 64; bits <= c->policy->max_rational_bits; bits *= 2) {
        mpz_mul_2exp(scaled, mpq_numref(lower), (mp_bitcnt_t)bits);
        mpz_fdiv_q(lower_integer, scaled, mpq_denref(lower));
        mpz_add_ui(lower_integer, lower_integer, 1);
        mpz_mul_2exp(scaled, mpq_numref(upper), (mp_bitcnt_t)bits);
        mpz_cdiv_q(upper_integer, scaled, mpq_denref(upper));
        mpz_sub_ui(upper_integer, upper_integer, 1);
        if (mpz_cmp(lower_integer, upper_integer) > 0)
            continue;
        mpz_add(numerator, lower_integer, upper_integer);
        mpz_fdiv_q_2exp(numerator, numerator, 1);
        mpq_set_z(out, numerator);
        mpz_set_ui(mpq_denref(out), 1);
        mpz_mul_2exp(mpq_denref(out), mpq_denref(out), (mp_bitcnt_t)bits);
        mpq_canonicalize(out);
        if (mpq_cmp(out, upper) < 0) {
            mpz_clears(scaled, lower_integer, upper_integer, numerator, NULL);
            return lobserve(c, out);
        }
        if (bits > c->policy->max_rational_bits / 2)
            break;
    }
    mpz_clears(scaled, lower_integer, upper_integer, numerator, NULL);
    return 1;
}

/* Cut a regular fixed-b section directly from the validated Frobenius cone.
 * This is the complementary local chart used when the loss sheet is too
 * curved close to y=0.  No numerical graph is load-bearing: the cone itself
 * encloses the invariant germ, and the physical b-flow is checked on the
 * complete selected cone slab. */
static int cone_b_section(lcontext *c, const local_field *f, const accepted_cone *cone,
                          int o, int time, mpq_t section_b, li *section_y,
                          li *section_level, int *b_direction) {
    if (cone->power != 2)
        return 1;
    lp2 db_u = {0}, db_s = {0}, first_term = {0}, second_term = {0};
    lp2 b_flow = {0};
    mpq_t one;
    mpq_init(one);
    mpq_set_ui(one, 1, 1);
    if (lp2_partial(c, &db_u, &f->db, 0) || lp2_partial(c, &db_s, &f->db, 1) ||
        lp2_mul(c, &first_term, &db_u, &f->fu) ||
        lp2_mul(c, &second_term, &db_s, &f->fs) ||
        lp2_lincomb(c, &b_flow, &first_term, one, &second_term, one))
        goto fail;

    mpq_t t0, t1, scale, signed_b, mid;
    mpq_t lower_safe, upper_probe, lower_probe, upper_safe;
    mpq_inits(t0, t1, scale, signed_b, mid, lower_safe, upper_probe, lower_probe,
              upper_safe, NULL);
    li first, second, flow, tbox, value;
    li_init(&first);
    li_init(&second);
    li_init(&flow);
    li_init(&tbox);
    li_init(&value);
    const unsigned long subdivisions = 64;
    const uint64_t bisections =
        c->policy->section_bisections < 32 ? c->policy->section_bisections : 32;
    int status = 1;
    for (unsigned long index = subdivisions; index-- > 0;) {
        mpq_set_ui(scale, index, subdivisions);
        mpq_mul(t0, cone->reach, scale);
        mpq_set_ui(scale, index + 1, subdivisions);
        mpq_mul(t1, cone->reach, scale);
        if (graph_value(c, &first, &f->db, t0, cone, o) ||
            li_add(c, &first, &first, &f->bc) ||
            graph_value(c, &second, &f->db, t1, cone, o) ||
            li_add(c, &second, &second, &f->bc))
            goto arithmetic_fail;
        int direction = 0;
        if (mpq_cmp(second.lo, first.hi) > 0) {
            direction = 1;
            int dyadic = strict_dyadic_between(c, section_b, first.hi, second.lo);
            if (dyadic < 0)
                goto arithmetic_fail;
            if (dyadic > 0)
                continue;
        } else if (mpq_cmp(second.hi, first.lo) < 0) {
            direction = -1;
            int dyadic = strict_dyadic_between(c, section_b, second.hi, first.lo);
            if (dyadic < 0)
                goto arithmetic_fail;
            if (dyadic > 0)
                continue;
        } else {
            continue;
        }
        mpq_set(tbox.lo, t0);
        mpq_set(tbox.hi, t1);
        if (graph_interval(c, &flow, &b_flow, &tbox, cone, o) ||
            li_scale_si(c, &flow, &flow, direction * time))
            goto arithmetic_fail;
        if (mpq_sgn(flow.lo) <= 0)
            continue;
        mpq_set(signed_b, section_b);
        if (direction < 0)
            mpq_neg(signed_b, signed_b);
        mpq_set(lower_safe, t0);
        mpq_set(upper_probe, t1);
        mpq_set(lower_probe, t0);
        mpq_set(upper_safe, t1);
        for (uint64_t k = 0; k < bisections; ++k) {
            mpq_add(mid, lower_safe, upper_probe);
            mpq_div_2exp(mid, mid, 1);
            if (graph_value(c, &value, &f->db, mid, cone, o) ||
                li_add(c, &value, &value, &f->bc) ||
                (direction < 0 && li_neg(c, &value, &value)))
                goto arithmetic_fail;
            if (mpq_cmp(value.hi, signed_b) < 0)
                mpq_set(lower_safe, mid);
            else
                mpq_set(upper_probe, mid);
            c->result->work.section_bisections++;
        }
        for (uint64_t k = 0; k < bisections; ++k) {
            mpq_add(mid, lower_probe, upper_safe);
            mpq_div_2exp(mid, mid, 1);
            if (graph_value(c, &value, &f->db, mid, cone, o) ||
                li_add(c, &value, &value, &f->bc) ||
                (direction < 0 && li_neg(c, &value, &value)))
                goto arithmetic_fail;
            if (mpq_cmp(value.lo, signed_b) > 0)
                mpq_set(upper_safe, mid);
            else
                mpq_set(lower_probe, mid);
            c->result->work.section_bisections++;
        }
        if (mpq_cmp(lower_safe, upper_safe) >= 0)
            continue;
        mpq_set(tbox.lo, lower_safe);
        mpq_set(tbox.hi, upper_safe);
        if (graph_interval(c, section_y, &f->y, &tbox, cone, o) ||
            graph_interval(c, section_level, &f->loss, &tbox, cone, o) ||
            li_add(c, section_level, section_level, &f->critical_loss))
            goto arithmetic_fail;
        if (li_zero(section_y))
            continue;
        *b_direction = direction;
        status = 0;
        break;
    }
    goto section_fail;
arithmetic_fail:
    status = -1;
section_fail:
    li_clear(&first);
    li_clear(&second);
    li_clear(&flow);
    li_clear(&tbox);
    li_clear(&value);
    mpq_clears(t0, t1, scale, signed_b, mid, lower_safe, upper_probe, lower_probe,
               upper_safe, NULL);
    lp2_clear(&db_u);
    lp2_clear(&db_s);
    lp2_clear(&first_term);
    lp2_clear(&second_term);
    lp2_clear(&b_flow);
    mpq_clear(one);
    return status;
fail:
    lp2_clear(&db_u);
    lp2_clear(&db_s);
    lp2_clear(&first_term);
    lp2_clear(&second_term);
    lp2_clear(&b_flow);
    mpq_clear(one);
    return -1;
}

static char *lstr(const mpz_t z) {
    size_t n = mpz_sizeinbase(z, 10);
    char *s = malloc(n + 3);
    if (s)
        mpz_get_str(s, 10, z);
    return s;
}
static int lexport(spong_owned_rational *o, const mpq_t q) {
    o->numerator = lstr(mpq_numref(q));
    o->denominator = lstr(mpq_denref(q));
    return !o->numerator || !o->denominator ? -1 : 0;
}
static int lexport_i(spong_owned_rational_interval *o, const li *x) {
    return lexport(&o->lower, x->lo) || lexport(&o->upper, x->hi) ? -1 : 0;
}
static void lfree_q(spong_owned_rational *q) {
    free(q->numerator);
    free(q->denominator);
}
static void lfree_i(spong_owned_rational_interval *x) {
    lfree_q(&x->lower);
    lfree_q(&x->upper);
}

void spong_local_launch_result_destroy(spong_local_launch_result *r) {
    if (!r)
        return;
    lfree_q(&r->reach);
    lfree_q(&r->cone_slope);
    lfree_q(&r->flow_margin);
    lfree_q(&r->lower_face_margin);
    lfree_q(&r->upper_face_margin);
    lfree_i(&r->tangent_slope);
    lfree_q(&r->section_level);
    lfree_i(&r->section_b);
    lfree_i(&r->section_y);
    lfree_q(&r->b_section_b);
    lfree_i(&r->b_section_y);
    lfree_i(&r->b_section_level);
    memset(r, 0, sizeof(*r));
}

int spong_local_launch_decimal(const spong_exact_loss_pencil *p,
                               const spong_local_launch_request *r,
                               const spong_local_launch_policy *policy,
                               spong_local_launch_result *out) {
    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    out->status = SPONG_SMALE_INVALID_ARGUMENT;
    out->primary_reason = SPONG_SMALE_REASON_BAD_POLICY;
    if (!p || !r || !policy || !p->alpha || !p->alpha_count || !p->beta ||
        !p->beta_count || !p->n || !p->n_count ||
        (r->critical_source != SPONG_CRITICAL_ROOT_B &&
         r->critical_source != SPONG_CRITICAL_ROOT_N) ||
        (r->orientation != -1 && r->orientation != 1) || !policy->max_slope_doublings ||
        policy->max_rational_bits < 64 || !policy->section_bisections)
        return -1;
    lcontext c = {policy, out, 0};
    local_field f;
    local_field_init(&f);
    accepted_cone linear, chosen;
    cone_init(&linear);
    cone_init(&chosen);
    mpq_t ld, reach, slope, maxs, initial, slope_start;
    mpq_inits(ld, reach, slope, maxs, initial, slope_start, NULL);
    li bout, yout;
    li_init(&bout);
    li_init(&yout);
    li b_section_y, b_section_level;
    li_init(&b_section_y);
    li_init(&b_section_level);
    mpq_t level, b_section_b;
    mpq_inits(level, b_section_b, NULL);
    out->status = SPONG_SMALE_OK;
    out->primary_reason = SPONG_SMALE_REASON_NONE;
    out->orientation = r->orientation;
    if (build_field(&c, &f, p, r) || lparse(&c, &r->departing_eigenvalue, ld) ||
        lparse(&c, &r->desired_reach, reach))
        goto fail;
    if (mpq_sgn(ld) == 0 || mpq_sgn(reach) <= 0)
        goto fail;
    int time = mpq_sgn(ld) > 0 ? 1 : -1;
    out->time_direction = time;
    mpq_set_ui(maxs, 1, 2);
    mpq_set_ui(initial, 1, 1);
    mpq_div_2exp(initial, initial, 96);
    mpq_set(slope_start, initial);
    int certified = 0;
    int last_reason = SPONG_SMALE_REASON_CONE_UNRESOLVED;
    for (uint64_t h = 0; h <= policy->max_reach_halvings && !certified; h++) {
        int closed = 0;
        mpq_set(slope, slope_start);
        for (uint64_t d = 0;
             d <= policy->max_slope_doublings && mpq_cmp(slope, maxs) <= 0; d++) {
            out->work.cone_tests++;
            if (d > out->work.slope_doublings)
                out->work.slope_doublings = d;
            int s = linear_cone(&c, &f, reach, slope, r->orientation, time, &linear);
            if (s < 0)
                goto fail;
            if (s == 0) {
                closed = 1;
                break;
            }
            mpq_mul_2exp(slope, slope, 1);
        }
        if (closed) {
            int ready = 1;
            if (r->require_frobenius) {
                int s = frobenius_cone(&c, &f, &linear, r->orientation, time, &chosen);
                if (s < 0)
                    goto fail;
                if (s > 0) {
                    ready = 0;
                    last_reason = SPONG_SMALE_REASON_FROBENIUS_UNRESOLVED;
                }
            } else {
                mpq_set(chosen.reach, linear.reach);
                mpq_set(chosen.slope, linear.slope);
                mpq_set(chosen.flow, linear.flow);
                mpq_set(chosen.lower, linear.lower);
                mpq_set(chosen.upper, linear.upper);
                li_set(&chosen.tangent, &linear.tangent);
                chosen.power = 1;
            }
            if (ready) {
                int s =
                    section(&c, &f, &chosen, r->orientation, time, level, &bout, &yout);
                if (s < 0)
                    goto fail;
                if (s == 0)
                    certified = 1;
                else {
                    last_reason = out->primary_reason;
                    out->work.section_bisections = 0;
                    out->work.section_retries++;
                    out->status = SPONG_SMALE_OK;
                    out->primary_reason = SPONG_SMALE_REASON_NONE;
                }
            }
        }
        if (!certified) {
            if (closed) {
                mpq_set(slope_start, linear.slope);
                mpq_div_2exp(slope_start, slope_start, 2);
                if (mpq_cmp(slope_start, initial) < 0)
                    mpq_set(slope_start, initial);
            } else
                mpq_set(slope_start, initial);
            mpq_div_2exp(reach, reach, 1);
            out->work.reach_halvings = h + 1;
        }
    }
    if (!certified) {
        out->status = SPONG_SMALE_TUBE_UNRESOLVED;
        out->primary_reason = last_reason;
        goto fail;
    }
    {
        int b_direction = 0;
        int b_status =
            cone_b_section(&c, &f, &chosen, r->orientation, time, b_section_b,
                           &b_section_y, &b_section_level, &b_direction);
        if (b_status < 0)
            goto fail;
        if (b_status == 0) {
            out->b_section_validated = 1;
            out->b_direction = b_direction;
        }
    }
    if (lexport(&out->reach, chosen.reach) || lexport(&out->cone_slope, chosen.slope) ||
        lexport(&out->flow_margin, chosen.flow) ||
        lexport(&out->lower_face_margin, chosen.lower) ||
        lexport(&out->upper_face_margin, chosen.upper) ||
        lexport_i(&out->tangent_slope, &chosen.tangent) ||
        lexport(&out->section_level, level) || lexport_i(&out->section_b, &bout) ||
        lexport_i(&out->section_y, &yout) ||
        (out->b_section_validated &&
         (lexport(&out->b_section_b, b_section_b) ||
          lexport_i(&out->b_section_y, &b_section_y) ||
          lexport_i(&out->b_section_level, &b_section_level)))) {
        out->status = SPONG_SMALE_ALLOCATION_FAILURE;
        out->primary_reason = SPONG_SMALE_REASON_ALLOCATION;
        goto fail;
    }
    out->cone_power = (uint32_t)chosen.power;
    out->validated = 1;
    out->status = SPONG_SMALE_OK;
    out->primary_reason = SPONG_SMALE_REASON_NONE;
    local_field_clear(&f);
    cone_clear(&linear);
    cone_clear(&chosen);
    mpq_clears(ld, reach, slope, maxs, initial, slope_start, level, b_section_b, NULL);
    li_clear(&bout);
    li_clear(&yout);
    li_clear(&b_section_y);
    li_clear(&b_section_level);
    return 0;
fail:
    if (out->status == SPONG_SMALE_OK) {
        out->status = SPONG_SMALE_INTERNAL_FAILURE;
        out->primary_reason = SPONG_SMALE_REASON_INTERNAL;
    }
    out->validated = 0;
    local_field_clear(&f);
    cone_clear(&linear);
    cone_clear(&chosen);
    mpq_clears(ld, reach, slope, maxs, initial, slope_start, level, b_section_b, NULL);
    li_clear(&bout);
    li_clear(&yout);
    li_clear(&b_section_y);
    li_clear(&b_section_level);
    return -1;
}
