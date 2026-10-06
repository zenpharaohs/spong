/* Poincare / slow-Frobenius minimum entry charts. See the public header.
 * GMP floating arithmetic carries guard bits; rational inputs are translated
 * before any binary64 rounding. The independent checker owns admission. */
#include "spong/spong_analytic_arrival.h"
#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SIDE 13
#define COUNT (SIDE*SIDE)
#define IX(i,j) ((i)*SIDE+(j))
typedef struct { mpf_t c[COUNT]; } poly;
struct spong_arrival_chart {
    int order, digits, slow;
    mp_bitcnt_t bits;
    poly K[2];
    mpf_t rates[2], frame[4], margin;
    mpf_t condition, backward_error, forward_error;
};
static void initv(mpf_t *v, size_t n, mp_bitcnt_t bits) {
    for (size_t i=0;i<n;i++) mpf_init2(v[i],bits);
}
static void clearv(mpf_t *v, size_t n) {
    for (size_t i=0;i<n;i++) mpf_clear(v[i]);
}
static void zero(poly *p) { for(int k=0;k<COUNT;k++) mpf_set_ui(p->c[k],0); }
static void mul(poly *out,const poly *a,const poly *b,int n,mpf_t tmp) {
    zero(out);
    for(int i=0;i<=n;i++) for(int j=0;j<=n-i;j++) {
        if(!mpf_sgn(a->c[IX(i,j)])) continue;
        for(int k=0;k<=n-i-j;k++) for(int l=0;l<=n-i-j-k;l++) {
            if(!mpf_sgn(b->c[IX(k,l)])) continue;
            mpf_mul(tmp,a->c[IX(i,j)],b->c[IX(k,l)]);
            mpf_add(out->c[IX(i+k,j+l)],out->c[IX(i+k,j+l)],tmp);
        }
    }
}
static int number(mpf_t v,const char *s,int rational) {
    if(!s || !*s || strlen(s)>100000) return -1;
    if(rational) {
        mpq_t q;mpq_init(q);
        int ok=mpq_set_str(q,s,10)==0 && mpz_sgn(mpq_denref(q));
        if(ok) { mpq_canonicalize(q);mpf_set_q(v,q); }
        mpq_clear(q);return ok?0:-1;
    }
    char *t=malloc(strlen(s)+1);if(!t) return -1;
    strcpy(t,s);for(char *p=t;*p;p++) if(*p=='e'||*p=='E') *p='@';
    int rc=mpf_set_str(v,t,10);free(t);return rc;
}
static char *decimal(const mpf_t x,int digits) {
    /* Caller-owned malloc string; GMP writes into our bounded buffer. */
    char mant[512];mp_exp_t exponent;
    mpf_get_str(mant,&exponent,10,(size_t)digits,x);
    char *out=malloc(600);if(!out) return NULL;
    if(!mpf_sgn(x)) strcpy(out,"0");
    else snprintf(out,600,"%s0.%se%ld",mant[0]=='-'?"-":"",mant+(mant[0]=='-'),(long)exponent);
    return out;
}
const char *spong_analytic_arrival_error(int s) {
    switch(s) {
    case SPONG_ARRIVAL_DIVISOR:return "arrival_resonance_or_small_divisor";
    case SPONG_ARRIVAL_MATCHING:return "arrival_matching_not_converged";
    case SPONG_ARRIVAL_RANGE:return "arrival_numeric_range";
    case SPONG_ARRIVAL_MEMORY:return "arrival_allocation";
    case SPONG_ARRIVAL_PRECISION:return "arrival_divisor_precision_budget";
    case SPONG_ARRIVAL_SOLVE:return "arrival_homological_solve_unresolved";
    default:return "invalid_analytic_arrival_request";
    }
}
void spong_arrival_chart_destroy(spong_arrival_chart *c) {
    if(!c) return;
    for(int d=0;d<2;d++) clearv(c->K[d].c,COUNT);
    clearv(c->rates,2);clearv(c->frame,4);mpf_clear(c->margin);
    mpf_clear(c->condition);mpf_clear(c->backward_error);mpf_clear(c->forward_error);free(c);
}
/* Screen cancellation in EACH modal subtraction, not against the other
 * eigenvalue. Reserve >=20 decimal digits relative to the requested input
 * precision. This is a numerical policy, not a series convergence claim.
 * Estimate the worst spectral condition and budget extra working bits so
 * physical-coordinate elimination can resolve the slow equation too. */
static int precision_plan(const spong_arrival_chart_request *r,mp_bitcnt_t *bits) {
    mpf_t w[12];mp_bitcnt_t base=(mp_bitcnt_t)ceil(r->decimal_digits*3.321928094887363)+64;
    initv(w,12,base);int rc=SPONG_ARRIVAL_INVALID;mp_exp_t extra=0;
    if(number(w[0],r->rates[0],0)||number(w[1],r->rates[1],0)) goto done;
    if(mpf_sgn(w[0])<=0||mpf_cmp(w[0],w[1])>0) goto done;
    mpf_div(w[2],w[1],w[0]);int slow=mpf_cmp_ui(w[2],10000)>=0;
    mpf_set_ui(w[11],10);mpf_pow_ui(w[11],w[11],r->decimal_digits-20);mpf_ui_div(w[11],1,w[11]);
    for(int degree=2;degree<=r->order;degree++) for(int i=0;i<=degree;i++) {
        int j=degree-i;if(slow&&j) continue;
        mpf_mul_ui(w[2],w[0],i);mpf_mul_ui(w[3],w[1],j);mpf_add(w[2],w[2],w[3]);
        for(int k=0;k<2;k++) {
            mpf_sub(w[4+k],w[2],w[k]);mpf_abs(w[4+k],w[4+k]);
            if(!mpf_sgn(w[4+k])) {rc=SPONG_ARRIVAL_DIVISOR;goto done;}
            mpf_add(w[6],w[2],w[k]);mpf_div(w[6],w[4+k],w[6]);
            if(mpf_cmp(w[6],w[11])<=0) {rc=SPONG_ARRIVAL_PRECISION;goto done;}
        }
        mpf_div(w[7],mpf_cmp(w[4],w[5])>0?w[4]:w[5],mpf_cmp(w[4],w[5])<0?w[4]:w[5]);
        mp_exp_t e;mpf_get_d_2exp(&e,w[7]);if(e>extra) extra=e;
    }
    /* The infinity-norm condition is <=2 times the spectral condition for
     * a symmetric 2x2 system. Cap memory/work rather than silently truncate. */
    if(extra>4096 || base+(mp_bitcnt_t)extra+2>4096) {rc=SPONG_ARRIVAL_PRECISION;goto done;}
    *bits=base+(mp_bitcnt_t)extra+2;rc=0;
 done:clearv(w,12);return rc;
}

/* Row equilibration + partial pivoting, with residual against ORIGINAL A.
 * Condition times componentwise backward error is retained as a conservative
 * numerical forward-error indicator. It is not an interval proof. */
static int homological_solve(spong_arrival_chart *c, const mpf_t a,const mpf_t b,
    const mpf_t d,const mpf_t f,const mpf_t g,mpf_t x,mpf_t y) {
    mpf_t w[24];initv(w,24,c->bits);int rc=SPONG_ARRIVAL_SOLVE;
    mpf_abs(w[0],a);mpf_abs(w[1],b);mpf_abs(w[2],d);
    mpf_set(w[3],mpf_cmp(w[0],w[1])>0?w[0]:w[1]);
    mpf_set(w[4],mpf_cmp(w[2],w[1])>0?w[2]:w[1]);
    if(!mpf_sgn(w[3])||!mpf_sgn(w[4])) goto done;
    mpf_div(w[5],a,w[3]);mpf_div(w[6],b,w[3]);mpf_div(w[7],f,w[3]);
    mpf_div(w[8],b,w[4]);mpf_div(w[9],d,w[4]);mpf_div(w[10],g,w[4]);
    mpf_abs(w[11],w[5]);mpf_abs(w[12],w[8]);
    if(mpf_cmp(w[11],w[12])<0) {mpf_swap(w[5],w[8]);mpf_swap(w[6],w[9]);mpf_swap(w[7],w[10]);}
    if(!mpf_sgn(w[5])) goto done;
    mpf_div(w[11],w[8],w[5]);mpf_mul(w[12],w[11],w[6]);mpf_sub(w[12],w[9],w[12]);
    if(!mpf_sgn(w[12])) goto done;
    mpf_mul(w[13],w[11],w[7]);mpf_sub(w[13],w[10],w[13]);mpf_div(y,w[13],w[12]);
    mpf_mul(w[13],w[6],y);mpf_sub(w[13],w[7],w[13]);mpf_div(x,w[13],w[5]);
    mpf_mul(w[14],a,d);mpf_mul(w[15],b,b);mpf_sub(w[14],w[14],w[15]);mpf_abs(w[14],w[14]);
    if(!mpf_sgn(w[14])) goto done;
    mpf_add(w[15],w[0],w[1]);mpf_add(w[16],w[2],w[1]);
    if(mpf_cmp(w[16],w[15])>0) mpf_set(w[15],w[16]);
    mpf_mul(w[15],w[15],w[15]);mpf_div(w[15],w[15],w[14]);
    if(mpf_cmp(w[15],c->condition)>0) mpf_set(c->condition,w[15]);
    mpf_set_ui(w[16],0);
    for(int row=0;row<2;row++) {
        mpf_mul(w[17],row?b:a,x);mpf_mul(w[18],row?d:b,y);
        mpf_add(w[19],w[17],w[18]);mpf_sub(w[19],w[19],row?g:f);mpf_abs(w[19],w[19]);
        mpf_abs(w[17],w[17]);mpf_abs(w[18],w[18]);mpf_abs(w[20],row?g:f);
        mpf_add(w[17],w[17],w[18]);mpf_add(w[17],w[17],w[20]);
        if(mpf_sgn(w[17])) mpf_div(w[19],w[19],w[17]);
        else if(mpf_sgn(w[19])) goto done;
        if(mpf_cmp(w[19],w[16])>0) mpf_set(w[16],w[19]);
    }
    /* Include a working-arithmetic floor even when the evaluated residual
     * rounds to zero. This is an error indicator, not directed rounding. */
    mpf_set_ui(w[18],1);mpf_div_2exp(w[18],w[18],c->bits-8);
    if(mpf_cmp(w[16],w[18])<0) mpf_set(w[16],w[18]);
    if(mpf_cmp(w[16],c->backward_error)>0) mpf_set(c->backward_error,w[16]);
    mpf_mul(w[17],w[15],w[16]);
    if(mpf_cmp_ui(w[17],1)>=0) goto done;
    mpf_ui_sub(w[18],1,w[17]);mpf_mul_ui(w[17],w[17],2);mpf_div(w[17],w[17],w[18]);
    if(mpf_cmp(w[17],c->forward_error)>0) mpf_set(c->forward_error,w[17]);
    if(number(w[18],"1e-20",0)) goto done;
    if(mpf_cmp(w[17],w[18])>0) goto done;
    rc=0;
 done:clearv(w,24);return rc;
}

void spong_arrival_conditioning_destroy(spong_arrival_conditioning *r) {
    if(!r) return;
    free(r->relative_divisor);free(r->matrix_condition);free(r->backward_error);free(r->forward_error);
    memset(r,0,sizeof(*r));
}
int spong_arrival_chart_inspect(const spong_arrival_chart *c,spong_arrival_conditioning *r) {
    if(!r) return SPONG_ARRIVAL_INVALID;
    memset(r,0,sizeof(*r));if(!c) return SPONG_ARRIVAL_INVALID;
    r->working_bits=(unsigned long)c->bits;
    r->relative_divisor=decimal(c->margin,c->digits);r->matrix_condition=decimal(c->condition,c->digits);
    r->backward_error=decimal(c->backward_error,c->digits);r->forward_error=decimal(c->forward_error,c->digits);
    if(!r->relative_divisor||!r->matrix_condition||!r->backward_error||!r->forward_error) {
        spong_arrival_conditioning_destroy(r);return SPONG_ARRIVAL_MEMORY;
    }
    return 0;
}
int spong_arrival_chart_create(const spong_arrival_chart_request *r,spong_arrival_chart **out) {
    if(!out) return SPONG_ARRIVAL_INVALID;
    *out=NULL;
    if(!r || r->order<2 || r->order>12 || r->decimal_digits<50 || r->decimal_digits>100
       || !r->gradient || !r->hessian || !r->frame || !r->rates) return SPONG_ARRIVAL_INVALID;
    mp_bitcnt_t working_bits;int plan=precision_plan(r,&working_bits);if(plan) return plan;
    spong_arrival_chart *c=calloc(1,sizeof(*c));if(!c) return SPONG_ARRIVAL_MEMORY;
    c->order=r->order;c->digits=r->decimal_digits;c->bits=working_bits;
    for(int d=0;d<2;d++) initv(c->K[d].c,COUNT,c->bits);
    initv(c->rates,2,c->bits);initv(c->frame,4,c->bits);mpf_init2(c->margin,c->bits);mpf_set_ui(c->margin,1);
    mpf_init2(c->condition,c->bits);mpf_set_ui(c->condition,1);
    mpf_init2(c->backward_error,c->bits);mpf_init2(c->forward_error,c->bits);
    poly *work=calloc(2*SIDE+5,sizeof(poly));
    if(!work) { spong_arrival_chart_destroy(c);return SPONG_ARRIVAL_MEMORY; }
    for(int i=0;i<2*SIDE+5;i++) initv(work[i].c,COUNT,c->bits);
    poly *pa=work,*pb=work+SIDE,*product=work+2*SIDE,*nonlinear=product+1,*field=nonlinear+2;
    mpf_t t[15];initv(t,15,c->bits);
    int rc=SPONG_ARRIVAL_INVALID,n=c->order,stride=n+1;
    for(int i=0;i<2;i++) if(number(c->rates[i],r->rates[i],0)) goto done;
    if(mpf_sgn(c->rates[0])<=0 || mpf_cmp(c->rates[0],c->rates[1])>0) goto done;
    for(int i=0;i<4;i++) if(number(c->frame[i],r->frame[i],0)) goto done;
    for(int i=0;i<4;i++) if(number(t[i],r->hessian[i],1)) goto done;
    for(int d=0;d<2;d++) for(int i=0;i<=n;i++) for(int j=0;j<=n-i;j++)
        if(number(field[d].c[IX(i,j)],r->gradient[(d*stride+i)*stride+j],1)) goto done;
    mpf_div(t[4],c->rates[1],c->rates[0]);c->slow=mpf_cmp_ui(t[4],10000)>=0;
    for(int d=0;d<2;d++) {
        mpf_set(c->K[d].c[IX(1,0)],c->frame[2*d]);
        mpf_set(c->K[d].c[IX(0,1)],c->frame[2*d+1]);
    }
    for(int degree=2;degree<=n;degree++) {
        zero(pa);zero(pb);mpf_set_ui(pa[0].c[0],1);mpf_set_ui(pb[0].c[0],1);
        for(int k=1;k<=degree;k++) {
            mul(pa+k,pa+k-1,c->K,degree,t[13]);
            mul(pb+k,pb+k-1,c->K+1,degree,t[13]);
        }
        zero(nonlinear);zero(nonlinear+1);
        for(int i=0;i<=degree;i++) for(int j=0;j<=degree-i;j++) {
            if(i+j<2 || (!mpf_sgn(field[0].c[IX(i,j)]) && !mpf_sgn(field[1].c[IX(i,j)]))) continue;
            mul(product,pa+i,pb+j,degree,t[13]);
            for(int d=0;d<2;d++) for(int k=0;k<=degree;k++) {
                int l=degree-k;
                mpf_mul(t[13],product->c[IX(k,l)],field[d].c[IX(i,j)]);
                mpf_add(nonlinear[d].c[IX(k,l)],nonlinear[d].c[IX(k,l)],t[13]);
            }
        }
        for(int i=0;i<=degree;i++) {
            int j=degree-i;if(c->slow && j) continue;
            mpf_mul_ui(t[4],c->rates[0],i);mpf_mul_ui(t[5],c->rates[1],j);mpf_add(t[4],t[4],t[5]);
            for(int k=0;k<2;k++) {
                mpf_sub(t[5],t[4],c->rates[k]);mpf_abs(t[5],t[5]);
                mpf_add(t[6],t[4],c->rates[k]);mpf_div(t[5],t[5],t[6]);
                if(mpf_cmp(t[5],c->margin)<0) mpf_set(c->margin,t[5]);
            }
            mpf_sub(t[5],t[0],t[4]);mpf_sub(t[6],t[3],t[4]);
            mpf_neg(t[8],nonlinear[0].c[IX(i,j)]);mpf_neg(t[9],nonlinear[1].c[IX(i,j)]);
            rc=homological_solve(c,t[5],t[1],t[6],t[8],t[9],c->K[0].c[IX(i,j)],c->K[1].c[IX(i,j)]);
            if(rc) goto done;
        }
    }
    rc=0;*out=c;
 done:
    clearv(t,15);
    for(int i=0;i<2*SIDE+5;i++) clearv(work[i].c,COUNT);
    free(work);if(rc) spong_arrival_chart_destroy(c);return rc;
}

/* Evaluate the polynomial and its Jacobian using precomputed powers. */
static void evaluate(const spong_arrival_chart *c,const mpf_t u,const mpf_t v,
                     mpf_t *out,mpf_t *pu,mpf_t *pv,mpf_t tmp) {
    for(int k=0;k<6;k++) mpf_set_ui(out[k],0);
    mpf_set_ui(pu[0],1);mpf_set_ui(pv[0],1);
    for(int i=1;i<=c->order;i++) { mpf_mul(pu[i],pu[i-1],u);mpf_mul(pv[i],pv[i-1],v); }
    for(int d=0;d<2;d++) for(int i=0;i<=c->order;i++) for(int j=0;j<=c->order-i;j++) {
        const mpf_t *a=&c->K[d].c[IX(i,j)];if(!mpf_sgn(*a)) continue;
        mpf_mul(tmp,*a,pu[i]);mpf_mul(tmp,tmp,pv[j]);mpf_add(out[d],out[d],tmp);
        if(i) { mpf_mul(tmp,*a,pu[i-1]);mpf_mul(tmp,tmp,pv[j]);mpf_mul_ui(tmp,tmp,i);mpf_add(out[2+2*d],out[2+2*d],tmp); }
        if(j) { mpf_mul(tmp,*a,pu[i]);mpf_mul(tmp,tmp,pv[j-1]);mpf_mul_ui(tmp,tmp,j);mpf_add(out[3+2*d],out[3+2*d],tmp); }
    }
}
/* exp(-x), x>=0, for display only. Taylor on [0,1/8] then squaring.
 * x>10000 is negligible under the explicit coefficient range below.
 * The retained analytic representation never truncates its exponential tail. */
static void exp_negative(mpf_t out,const mpf_t x,mpf_t *w,mp_bitcnt_t bits) {
    if(mpf_cmp_ui(x,10000)>0) { mpf_set_ui(out,0);return; }
    mpf_set(w[0],x);unsigned k=0;
    while(mpf_cmp_d(w[0],.125)>0) { mpf_div_2exp(w[0],w[0],1);k++; }
    mpf_neg(w[0],w[0]);mpf_set_ui(out,1);mpf_set_ui(w[1],1);
    mpf_set_ui(w[2],1);mpf_div_2exp(w[2],w[2],bits);
    for(unsigned j=1;j<4096;j++) {
        mpf_mul(w[1],w[1],w[0]);mpf_div_ui(w[1],w[1],j);mpf_add(out,out,w[1]);
        mpf_abs(w[3],w[1]);if(mpf_cmp(w[3],w[2])<0) break;
    }
    while(k--) mpf_mul(out,out,out);
}
void spong_analytic_arrival_destroy(spong_analytic_arrival *r) {
    if(!r) return;
    if(r->terms) for(size_t i=0;i<3*r->term_count;i++) free(r->terms[i]);
    if(r->parameters) for(size_t i=0;i+1<r->point_count;i++) free(r->parameters[i]);
    free(r->terms);free(r->parameters);free(r->points);
    free(r->amplitudes[0]);free(r->amplitudes[1]);free(r->spectral_ratio);free(r->divisor_margin);
    memset(r,0,sizeof(*r));
}
/* Per-call diagnostics; chart remains immutable under concurrent matching. */
typedef struct {
    unsigned iterations, backtracks, stages;
    int continuation;
    mpf_t condition, forward, residual;
} match_stats;

/* General 2x2 Jacobian: row equilibration and partial pivoting. The
 * condition/residual test concerns this solve only, not chart convergence. */
static int match_solve(const spong_arrival_chart *c,mpf_t *J,mpf_t *f,
                       mpf_t *step,match_stats *stats) {
    mpf_t w[24];initv(w,24,c->bits);int ok=0;
    for(int k=0;k<4;k++) mpf_abs(w[k],J[k]);
    for(int row=0;row<2;row++) {
        mpf_set(w[4],mpf_cmp(w[2*row],w[2*row+1])>0?w[2*row]:w[2*row+1]);
        if(!mpf_sgn(w[4])) goto done;
        mpf_div(w[5+3*row],J[2*row],w[4]);
        mpf_div(w[6+3*row],J[2*row+1],w[4]);mpf_div(w[7+3*row],f[row],w[4]);
    }
    mpf_abs(w[11],w[5]);mpf_abs(w[12],w[8]);
    if(mpf_cmp(w[11],w[12])<0) for(int k=0;k<3;k++) mpf_swap(w[5+k],w[8+k]);
    if(!mpf_sgn(w[5])) goto done;
    mpf_div(w[11],w[8],w[5]);mpf_mul(w[12],w[11],w[6]);mpf_sub(w[12],w[9],w[12]);
    if(!mpf_sgn(w[12])) goto done;
    mpf_mul(w[13],w[11],w[7]);mpf_sub(w[13],w[10],w[13]);mpf_div(step[1],w[13],w[12]);
    mpf_mul(w[13],w[6],step[1]);mpf_sub(w[13],w[7],w[13]);mpf_div(step[0],w[13],w[5]);
    mpf_mul(w[14],J[0],J[3]);mpf_mul(w[15],J[1],J[2]);mpf_sub(w[14],w[14],w[15]);mpf_abs(w[14],w[14]);
    if(!mpf_sgn(w[14])) goto done;
    mpf_add(w[15],w[0],w[1]);mpf_add(w[16],w[2],w[3]);
    if(mpf_cmp(w[16],w[15])>0) mpf_set(w[15],w[16]);
    mpf_add(w[16],w[3],w[1]);mpf_add(w[17],w[2],w[0]);
    if(mpf_cmp(w[17],w[16])>0) mpf_set(w[16],w[17]);
    mpf_mul(w[15],w[15],w[16]);mpf_div(w[15],w[15],w[14]);
    if(mpf_cmp(w[15],stats->condition)>0) mpf_set(stats->condition,w[15]);
    mpf_set_ui(w[16],1);mpf_div_2exp(w[16],w[16],c->bits-8);
    for(int row=0;row<2;row++) {
        mpf_mul(w[17],J[2*row],step[0]);mpf_mul(w[18],J[2*row+1],step[1]);
        mpf_add(w[19],w[17],w[18]);mpf_sub(w[19],w[19],f[row]);mpf_abs(w[19],w[19]);
        mpf_abs(w[17],w[17]);mpf_abs(w[18],w[18]);mpf_abs(w[20],f[row]);
        mpf_add(w[17],w[17],w[18]);mpf_add(w[17],w[17],w[20]);
        if(mpf_sgn(w[17])) mpf_div(w[19],w[19],w[17]);
        if(mpf_cmp(w[19],w[16])>0) mpf_set(w[16],w[19]);
    }
    mpf_mul(w[17],w[15],w[16]);if(mpf_cmp_ui(w[17],1)>=0) goto done;
    mpf_ui_sub(w[18],1,w[17]);mpf_mul_ui(w[17],w[17],2);mpf_div(w[17],w[17],w[18]);
    if(mpf_cmp(w[17],stats->forward)>0) mpf_set(stats->forward,w[17]);
    if(number(w[18],"1e-20",0)) goto done;ok=mpf_cmp(w[17],w[18])<=0;
 done:clearv(w,24);return ok;
}

static void residual_norm(mpf_t norm,mpf_t *f,mpf_t tmp) {
    mpf_abs(norm,f[0]);mpf_abs(tmp,f[1]);if(mpf_cmp(tmp,norm)>0) mpf_set(norm,tmp);
}

/* Fast Newton path preserves the established root choice. A failed path
 * restarts from the origin and follows K(u,v)=t*offset with damped Newton.
 * Intermediate t values are inverse-chart work, NEVER returned geometry. */
static int match_newton(const spong_arrival_chart *c,mpf_t *offset,mpf_t *uv,
                        int damped,match_stats *stats) {
    mpf_t w[16],values[6],pu[SIDE],pv[SIDE];
    initv(w,16,c->bits);initv(values,6,c->bits);initv(pu,SIDE,c->bits);initv(pv,SIDE,c->bits);
    int ok=0;unsigned limit=damped?48:24;
    residual_norm(w[0],offset,w[1]);mpf_add_ui(w[0],w[0],1);
    mpf_set_ui(w[1],10);mpf_pow_ui(w[1],w[1],c->digits-20);mpf_div(w[0],w[0],w[1]);
    for(unsigned iteration=0;iteration<=limit && stats->iterations<2048;iteration++) {
        evaluate(c,uv[0],uv[1],values,pu,pv,w[15]);
        mpf_sub(w[2],values[0],offset[0]);mpf_sub(w[3],values[1],offset[1]);
        residual_norm(w[4],w+2,w[15]);mpf_set(stats->residual,w[4]);
        /* Test Jacobian resolution even at an apparently converged point. */
        if(!match_solve(c,values+2,w+2,w+5,stats)) break;
        if(mpf_cmp(w[4],w[0])<0) {
            /* A tiny residual alone can hide a poorly determined inverse.
             * Require a small resolved correction in amplitude coordinates. */
            residual_norm(w[7],w+5,w[15]);residual_norm(w[8],uv,w[15]);mpf_add_ui(w[8],w[8],1);
            if(number(w[9],"1e-20",0)) break;
            mpf_mul(w[8],w[8],w[9]);
            if(mpf_cmp(w[7],w[8])<=0) {ok=1;break;}
        }
        if(iteration==limit) break;
        stats->iterations++;mpf_set_ui(w[7],1);int accepted=0;
        for(unsigned backtrack=0;backtrack<(damped?32u:1u);backtrack++) {
            for(int d=0;d<2;d++) {mpf_mul(w[8+d],w[7],w[5+d]);mpf_sub(w[8+d],uv[d],w[8+d]);}
            int range=0;for(int d=0;d<2;d++) {mp_exp_t e;mpf_get_d_2exp(&e,w[8+d]);if(e>4096) range=1;}
            if(!range) {
                if(!damped) accepted=1;
                else {
                    evaluate(c,w[8],w[9],values,pu,pv,w[15]);
                    mpf_sub(w[10],values[0],offset[0]);mpf_sub(w[11],values[1],offset[1]);
                    residual_norm(w[12],w+10,w[15]);
                    /* Armijo decrease, infinity-norm residual. */
                    mpf_div_ui(w[13],w[7],4);mpf_ui_sub(w[13],1,w[13]);mpf_mul(w[13],w[13],w[4]);
                    accepted=mpf_cmp(w[12],w[13])<=0;
                }
            }
            if(accepted) {mpf_set(uv[0],w[8]);mpf_set(uv[1],w[9]);break;}
            mpf_div_2exp(w[7],w[7],1);stats->backtracks++;
        }
        if(!accepted) break;
    }
    clearv(w,16);clearv(values,6);clearv(pu,SIDE);clearv(pv,SIDE);return ok;
}

static int match_amplitudes(const spong_arrival_chart *c,mpf_t *offset,mpf_t *uv,
                            match_stats *stats) {
    if(match_newton(c,offset,uv,0,stats)) return 1;
    stats->continuation=1;
    mpf_t w[9];initv(w,9,c->bits);mpf_set_ui(uv[0],0);mpf_set_ui(uv[1],0);
    /* w: t, dt, next_t, trial offset[2], trial uv[2], minimum dt, scratch. */
    mpf_set_ui(w[1],1);mpf_div_ui(w[1],w[1],8);
    mpf_set_ui(w[7],1);mpf_div_2exp(w[7],w[7],20);int ok=0;
    for(unsigned attempt=0;attempt<128 && stats->iterations<2048;attempt++) {
        mpf_add(w[2],w[0],w[1]);if(mpf_cmp_ui(w[2],1)>0) mpf_set_ui(w[2],1);
        for(int d=0;d<2;d++) {mpf_mul(w[3+d],offset[d],w[2]);mpf_set(w[5+d],uv[d]);}
        if(match_newton(c,w+3,w+5,1,stats)) {
            stats->stages++;mpf_set(w[0],w[2]);for(int d=0;d<2;d++) mpf_set(uv[d],w[5+d]);
            if(mpf_cmp_ui(w[0],1)==0) {ok=1;break;}
            mpf_mul_2exp(w[1],w[1],1);
        } else {mpf_div_2exp(w[1],w[1],1);if(mpf_cmp(w[1],w[7])<0) break;}
    }
    clearv(w,9);return ok;
}

void spong_arrival_matching_destroy(spong_arrival_matching *r) {
    if(!r) return;
    free(r->max_condition);free(r->max_forward_error);free(r->residual);memset(r,0,sizeof(*r));
}
int spong_arrival_chart_match(const spong_arrival_chart *c,const char *const center[2],
    const double start[2],const double target[2],spong_analytic_arrival *r) {
    return spong_arrival_chart_match_inspect(c,center,start,target,r,NULL);
}
int spong_arrival_chart_match_inspect(const spong_arrival_chart *c,const char *const center[2],
    const double start[2],const double target[2],spong_analytic_arrival *r,spong_arrival_matching *info) {
    return spong_arrival_chart_match_exact_inspect(c,center,start,target,NULL,r,info);
}
int spong_arrival_chart_match_exact_inspect(const spong_arrival_chart *c,const char *const center[2],
    const double start[2],const double target[2],const char *const exact_start[2],
    spong_analytic_arrival *r,spong_arrival_matching *info) {
    if(info) memset(info,0,sizeof(*info));
    if(!r) return SPONG_ARRIVAL_INVALID;
    memset(r,0,sizeof(*r));
    if(!c || !center || !start || !target) return SPONG_ARRIVAL_INVALID;
    for(int d=0;d<2;d++) if(!isfinite(start[d]) || !isfinite(target[d])) return SPONG_ARRIVAL_INVALID;
    mpf_t w[32],pu[SIDE],pv[SIDE],vals[6];
    initv(w,32,c->bits);initv(pu,SIDE,c->bits);initv(pv,SIDE,c->bits);initv(vals,6,c->bits);
    match_stats stats={0};mpf_init2(stats.condition,c->bits);mpf_init2(stats.forward,c->bits);mpf_init2(stats.residual,c->bits);
    /* w: center[0:2], offset[2:4], amplitudes[4:6], residual[6:8], scratch. */
    int rc=SPONG_ARRIVAL_INVALID;
    for(int d=0;d<2;d++) {
        if(number(w[d],center[d],1)) goto done;
        if(exact_start) {if(number(w[2+d],exact_start[d],1)) goto done;}
        else mpf_set_d(w[2+d],start[d]);
        mpf_sub(w[2+d],w[2+d],w[d]);
    }
    for(int d=0;d<2;d++) {
        mpf_mul(w[4+d],c->frame[d],w[2]);mpf_mul(w[8],c->frame[2+d],w[3]);mpf_add(w[4+d],w[4+d],w[8]);
    }
    if(!match_amplitudes(c,w+2,w+4,&stats)) {rc=SPONG_ARRIVAL_MATCHING;goto done;}
    /* Populate final powers independently of the inverse solver's scratch. */
    evaluate(c,w[4],w[5],vals,pu,pv,w[8]);
    mpf_div(w[14],c->rates[1],c->rates[0]);
    r->terms=calloc(3*COUNT,sizeof(char*));r->parameters=calloc(514,sizeof(char*));r->points=calloc(2*515,sizeof(double));
    if(!r->terms || !r->parameters || !r->points) { rc=SPONG_ARRIVAL_MEMORY;goto done; }
    r->slow_frobenius=c->slow;
    r->amplitudes[0]=decimal(w[4],c->digits);r->amplitudes[1]=decimal(w[5],c->digits);
    r->spectral_ratio=decimal(w[14],c->digits);r->divisor_margin=decimal(c->margin,c->digits);
    if(!r->amplitudes[0]||!r->amplitudes[1]||!r->spectral_ratio||!r->divisor_margin) { rc=SPONG_ARRIVAL_MEMORY;goto done; }
    /* Powers from the final matching evaluation. */
    for(int i=0;i<=c->order;i++) for(int j=0;j<=c->order-i;j++) {
        mpf_mul(w[8],pu[i],pv[j]);
        mpf_mul(w[10],c->K[0].c[IX(i,j)],w[8]);mpf_mul(w[11],c->K[1].c[IX(i,j)],w[8]);
        if(!mpf_sgn(w[10]) && !mpf_sgn(w[11])) continue;
        for(int d=10;d<=11;d++) { mp_exp_t e;mpf_get_d_2exp(&e,w[d]);if(e>3320) {rc=SPONG_ARRIVAL_RANGE;goto done;} }
        mpf_mul_ui(w[12],w[14],j);mpf_add_ui(w[12],w[12],i);
        size_t k=3*r->term_count++;
        r->terms[k]=decimal(w[12],c->digits);r->terms[k+1]=decimal(w[10],c->digits);r->terms[k+2]=decimal(w[11],c->digits);
        if(!r->terms[k]||!r->terms[k+1]||!r->terms[k+2]) {rc=SPONG_ARRIVAL_MEMORY;goto done;}
    }
    if(!r->term_count) {rc=SPONG_ARRIVAL_MATCHING;goto done;}
    /* Merge sorted slow/fast parameter grids, deduplicating ties. */
    unsigned si=0,fi=0;
    while(si<=256 || fi<=256) {
        mpf_set_ui(w[15],si);mpf_div_ui(w[15],w[15],16);
        mpf_set_ui(w[16],fi);mpf_div_ui(w[16],w[16],16);mpf_div(w[16],w[16],w[14]);
        int cmp=si>256?1:fi>256?-1:mpf_cmp(w[15],w[16]);
        mpf_set(w[17],cmp<=0?w[15]:w[16]);if(cmp<=0) si++;if(cmp>=0) fi++;
        size_t k=r->point_count++;
        r->parameters[k]=decimal(w[17],c->digits);
        if(!r->parameters[k]) {rc=SPONG_ARRIVAL_MEMORY;goto done;}
        exp_negative(w[18],w[17],w+23,c->bits);mpf_mul(w[18],w[18],w[4]);
        mpf_mul(w[19],w[17],w[14]);exp_negative(w[20],w[19],w+23,c->bits);mpf_mul(w[20],w[20],w[5]);
        evaluate(c,w[18],w[20],vals,pu,pv,w[8]);
        for(int d=0;d<2;d++) { mpf_add(w[21],w[d],vals[d]);r->points[2*k+d]=mpf_get_d(w[21]);if(!isfinite(r->points[2*k+d])) {rc=SPONG_ARRIVAL_RANGE;goto done;} }
    }
    for(int d=0;d<2;d++) { r->points[d]=start[d];r->points[2*r->point_count+d]=target[d]; }
    r->point_count++;rc=0;
 done:
    if(info) {
        info->iterations=stats.iterations;info->backtracks=stats.backtracks;
        info->stages=stats.stages;info->continuation=stats.continuation;
        info->max_condition=decimal(stats.condition,c->digits);
        info->max_forward_error=decimal(stats.forward,c->digits);info->residual=decimal(stats.residual,c->digits);
        if(!info->max_condition||!info->max_forward_error||!info->residual) {spong_arrival_matching_destroy(info);rc=SPONG_ARRIVAL_MEMORY;}
    }
    mpf_clear(stats.condition);mpf_clear(stats.forward);mpf_clear(stats.residual);
    clearv(w,32);clearv(pu,SIDE);clearv(pv,SIDE);clearv(vals,6);
    /* On partial failure parameters count equals point_count (no endpoint). */
    if(rc) { if(r->parameters && r->point_count) free(r->parameters[r->point_count-1]);spong_analytic_arrival_destroy(r); }
    return rc;
}
