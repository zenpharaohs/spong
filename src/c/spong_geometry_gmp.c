#include "spong/spong_geometry.h"

#include <gmp.h>
#include <math.h>
#include <stdlib.h>

static void clear_poly(mpq_t *p, size_t count) {
    if (p == NULL) return;
    for (size_t i = 0; i < count; ++i) mpq_clear(p[i]);
    free(p);
}

static mpq_t *read_poly(const spong_rational_input *input, size_t count) {
    if (count > SIZE_MAX/sizeof(mpq_t)) return NULL;
    mpq_t *p = calloc(count, sizeof(mpq_t));
    if (p == NULL) return NULL;
    for (size_t i = 0; i < count; ++i) mpq_init(p[i]);
    for (size_t i = 0; i < count; ++i) {
        if (input[i].numerator == NULL || input[i].denominator == NULL
                || mpz_set_str(mpq_numref(p[i]), input[i].numerator, 10)
                || mpz_set_str(mpq_denref(p[i]), input[i].denominator, 10)
                || mpz_sgn(mpq_denref(p[i])) <= 0) {
            clear_poly(p, count);
            return NULL;
        }
        mpq_canonicalize(p[i]);
    }
    return p;
}

/* Simultaneous exact Horner value and derivative. */
static void eval(mpq_t value, mpq_t derivative, mpq_t *p, size_t count,
                 const mpq_t x) {
    mpq_set_ui(value, 0, 1);
    mpq_set_ui(derivative, 0, 1);
    for (size_t i = count; i-- > 0;) {
        mpq_mul(derivative, derivative, x);
        mpq_add(derivative, derivative, value);
        mpq_mul(value, value, x);
        mpq_add(value, value, p[i]);
    }
}

static void loss(mpq_t out, const mpq_t a, const mpq_t b,
                 mpq_t *A, size_t na, mpq_t *B, size_t nb,
                 mpq_t v, mpq_t derivative, mpq_t tmp) {
    eval(v, derivative, A, na, b);
    mpq_mul(out, a, a);
    mpq_mul(out, out, v);
    eval(v, derivative, B, nb, b);
    mpq_mul(tmp, a, v);
    mpq_mul_2exp(tmp, tmp, 1);
    mpq_sub(out, out, tmp);
}

int spong_level_normal_reference(
        const spong_rational_input *A, size_t na,
        const spong_rational_input *B, size_t nb,
        const double *points, size_t count, int32_t direction,
        spong_level_normal_measurement *out) {
    if (A == NULL || B == NULL || na == 0 || nb == 0 || points == NULL
            || out == NULL || count < 2 || count > SIZE_MAX/2
            || (direction != -1 && direction != 1)) return -1;
    mpq_t *qa = read_poly(A, na), *qb = read_poly(B, nb);
    if (qa == NULL || qb == NULL) {
        clear_poly(qa, na); clear_poly(qb, nb); return -1;
    }
    mpq_t a,b,da,db,av,ap,bv,bp,ga,gb,cross,dot,nd,ng,tmp,ratio,ll,lr,al,bl,ar,br;
    mpq_inits(a,b,da,db,av,ap,bv,bp,ga,gb,cross,dot,nd,ng,tmp,ratio,ll,lr,al,bl,ar,br,NULL);
    for (size_t i = 0; i < count-1; ++i) {
        spong_level_normal_measurement *r = &out[i];
        *r = (spong_level_normal_measurement){0, NAN, 0, 0, 0};
        const double *p = &points[2*i];
        if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]) || !isfinite(p[3])) {
            r->status = 1; continue;
        }
        mpq_set_d(al,p[0]); mpq_set_d(bl,p[1]);
        mpq_set_d(ar,p[2]); mpq_set_d(br,p[3]);
        mpq_sub(da,ar,al); mpq_sub(db,br,bl);
        if (mpq_sgn(da) == 0 && mpq_sgn(db) == 0) {
            r->status = 2; continue;
        }
        loss(ll,al,bl,qa,na,qb,nb,av,ap,tmp);
        loss(lr,ar,br,qa,na,qb,nb,av,ap,tmp);
        mpq_sub(tmp,lr,ll);
        r->loss_direction = direction * mpq_sgn(tmp);
        mpq_add(a,al,ar); mpq_div_2exp(a,a,1);
        mpq_add(b,bl,br); mpq_div_2exp(b,b,1);
        eval(av,ap,qa,na,b); eval(bv,bp,qb,nb,b);
        mpq_mul(ga,a,av); mpq_sub(ga,ga,bv); mpq_mul_2exp(ga,ga,1);
        mpq_mul(gb,a,a); mpq_mul(gb,gb,ap);
        mpq_mul(tmp,a,bp); mpq_mul_2exp(tmp,tmp,1); mpq_sub(gb,gb,tmp);
        if (mpq_sgn(ga) == 0 && mpq_sgn(gb) == 0) {
            r->status = 3; continue;
        }
        mpq_mul(cross,da,gb); mpq_mul(tmp,db,ga); mpq_sub(cross,cross,tmp);
        mpq_mul(dot,da,ga); mpq_mul(tmp,db,gb); mpq_add(dot,dot,tmp);
        r->cross_nonzero = mpq_sgn(cross) != 0;
        r->flow_alignment = direction * mpq_sgn(dot);
        mpq_mul(nd,da,da); mpq_mul(tmp,db,db); mpq_add(nd,nd,tmp);
        mpq_mul(ng,ga,ga); mpq_mul(tmp,gb,gb); mpq_add(ng,ng,tmp);
        mpq_mul(ratio,nd,ng); mpq_mul(tmp,cross,cross); mpq_div(ratio,tmp,ratio);
        r->sin_squared = mpq_get_d(ratio);
    }
    mpq_clears(a,b,da,db,av,ap,bv,bp,ga,gb,cross,dot,nd,ng,tmp,ratio,ll,lr,al,bl,ar,br,NULL);
    clear_poly(qa,na); clear_poly(qb,nb);
    return 0;
}
