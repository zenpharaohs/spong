/* Native arbitrary-precision, L-stable 3-stage Radau IIA construction.
 * No Python or global precision; exported cubics retain working precision. */
#include "spong/spong_precision_bridge.h"
#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 6
#define LIMIT 64

typedef struct {
    unsigned rows,cols; mp_bitcnt_t bits;
    mpf_t *g;
    mpf_t A[9], basis[9], nodes[3], tolerance, max_condition;
} core;
static void init(mpf_t *p,size_t n,mp_bitcnt_t bits) {for(size_t i=0;i<n;i++) mpf_init2(p[i],bits);}
static void clear(mpf_t *p,size_t n) {for(size_t i=0;i<n;i++) mpf_clear(p[i]);}
static int parse(mpf_t x,const char *s,int rational) {
    if(!s||strlen(s)>100000) return 0;
    if(rational) {
        mpq_t q;mpq_init(q);int ok=!mpq_set_str(q,s,10)&&mpz_sgn(mpq_denref(q));
        if(ok) {mpq_canonicalize(q);mpf_set_q(x,q);}mpq_clear(q);return ok;
    }
    char *p=malloc(strlen(s)+1);if(!p)return 0;strcpy(p,s);
    for(char *c=p;*c;c++)if(*c=='e'||*c=='E')*c='@';
    int ok=!mpf_set_str(x,p,10);free(p);return ok;
}
static char *decimal(const mpf_t x,unsigned digits) {
    size_t n=(size_t)digits+100;char *p=malloc(n);if(!p)return NULL;
    int used=gmp_snprintf(p,n,"%.*Fe",(int)digits,x);
    if(used<0||(size_t)used>=n){free(p);return NULL;}return p;
}
static void normmax(mpf_t dst,const mpf_t x,mpf_t tmp) {mpf_abs(tmp,x);if(mpf_cmp(tmp,dst)>0)mpf_set(dst,tmp);}
static void tableau(core *c) {
    mpf_t w[5];init(w,5,c->bits);mpf_sqrt_ui(w[0],6);
    mpf_ui_sub(c->nodes[0],4,w[0]);mpf_div_ui(c->nodes[0],c->nodes[0],10);
    mpf_add_ui(c->nodes[1],w[0],4);mpf_div_ui(c->nodes[1],c->nodes[1],10);mpf_set_ui(c->nodes[2],1);
    /* Integrate each Lagrange polynomial to construct the tableau itself. */
    for(int i=0;i<3;i++) {
        int j=(i+1)%3,k=(i+2)%3;
        mpf_sub(w[1],c->nodes[i],c->nodes[j]);mpf_sub(w[2],c->nodes[i],c->nodes[k]);mpf_mul(w[1],w[1],w[2]);
        mpf_mul(c->basis[3*i],c->nodes[j],c->nodes[k]);mpf_div(c->basis[3*i],c->basis[3*i],w[1]);
        mpf_add(c->basis[3*i+1],c->nodes[j],c->nodes[k]);mpf_neg(c->basis[3*i+1],c->basis[3*i+1]);mpf_div(c->basis[3*i+1],c->basis[3*i+1],w[1]);
        mpf_ui_div(c->basis[3*i+2],1,w[1]);
        for(int stage=0;stage<3;stage++) {
            mpf_set_ui(c->A[stage*3+i],0);mpf_set(w[2],c->nodes[stage]);
            for(int d=0;d<3;d++) {
                mpf_mul(w[3],c->basis[3*i+d],w[2]);mpf_div_ui(w[3],w[3],(unsigned)d+1);
                mpf_add(c->A[stage*3+i],c->A[stage*3+i],w[3]);mpf_mul(w[2],w[2],c->nodes[stage]);
            }
        }
    }
    clear(w,5);
}
static void field(core *c,const mpf_t *z,mpf_t *f,mpf_t *J) {
    mpf_t ap[LIMIT],bp[LIMIT],w[2];init(ap,c->rows,c->bits);init(bp,c->cols,c->bits);init(w,2,c->bits);
    mpf_set_ui(ap[0],1);mpf_set_ui(bp[0],1);
    for(unsigned i=1;i<c->rows;i++)mpf_mul(ap[i],ap[i-1],z[0]);
    for(unsigned j=1;j<c->cols;j++)mpf_mul(bp[j],bp[j-1],z[1]);
    for(int d=0;d<2;d++) {
        mpf_set_ui(f[d],0);if(J){mpf_set_ui(J[2*d],0);mpf_set_ui(J[2*d+1],0);}
        for(unsigned i=0;i<c->rows;i++)for(unsigned j=0;j<c->cols;j++) {
            mpf_srcptr a=c->g[(d*c->rows+i)*c->cols+j];if(!mpf_sgn(a))continue;
            mpf_mul(w[0],a,ap[i]);mpf_mul(w[0],w[0],bp[j]);mpf_sub(f[d],f[d],w[0]);
            if(J&&i){mpf_mul(w[0],a,ap[i-1]);mpf_mul(w[0],w[0],bp[j]);mpf_mul_ui(w[0],w[0],i);mpf_sub(J[2*d],J[2*d],w[0]);}
            if(J&&j){mpf_mul(w[0],a,ap[i]);mpf_mul(w[0],w[0],bp[j-1]);mpf_mul_ui(w[0],w[0],j);mpf_sub(J[2*d+1],J[2*d+1],w[0]);}
        }
    }
    clear(ap,c->rows);clear(bp,c->cols);clear(w,2);
}
/* Invert the 6x6 Newton matrix with pivoting. An explicit condition indicator
 * charges lost bits before the solve is used. This is not an interval bound. */
static int solve(core *c,mpf_t *matrix,mpf_t *rhs,mpf_t *delta) {
    mpf_t a[N*(2*N)],w[7];init(a,N*2*N,c->bits);init(w,7,c->bits);
    for(int i=0;i<N;i++) {
        mpf_set_ui(w[0],0);
        for(int j=0;j<N;j++){mpf_set(a[i*2*N+j],matrix[i*N+j]);mpf_abs(w[1],matrix[i*N+j]);mpf_add(w[0],w[0],w[1]);}
        if(mpf_cmp(w[0],w[2])>0)mpf_set(w[2],w[0]);mpf_set_ui(a[i*2*N+N+i],1);
    }
    int ok=0;
    for(int k=0;k<N;k++) {
        int pivot=k;mpf_abs(w[0],a[k*2*N+k]);
        for(int i=k+1;i<N;i++){mpf_abs(w[1],a[i*2*N+k]);if(mpf_cmp(w[1],w[0])>0){pivot=i;mpf_set(w[0],w[1]);}}
        if(!mpf_sgn(w[0]))goto done;
        if(pivot!=k)for(int j=0;j<2*N;j++)mpf_swap(a[k*2*N+j],a[pivot*2*N+j]);
        mpf_set(w[0],a[k*2*N+k]);for(int j=0;j<2*N;j++)mpf_div(a[k*2*N+j],a[k*2*N+j],w[0]);
        for(int i=0;i<N;i++)if(i!=k){mpf_set(w[0],a[i*2*N+k]);for(int j=0;j<2*N;j++){mpf_mul(w[1],w[0],a[k*2*N+j]);mpf_sub(a[i*2*N+j],a[i*2*N+j],w[1]);}}
    }
    for(int i=0;i<N;i++) {
        mpf_set_ui(w[0],0);mpf_set_ui(delta[i],0);
        for(int j=0;j<N;j++){mpf_abs(w[1],a[i*2*N+N+j]);mpf_add(w[0],w[0],w[1]);mpf_mul(w[1],a[i*2*N+N+j],rhs[j]);mpf_add(delta[i],delta[i],w[1]);}
        if(mpf_cmp(w[0],w[3])>0)mpf_set(w[3],w[0]);
    }
    mpf_mul(w[4],w[2],w[3]);if(mpf_cmp(w[4],c->max_condition)>0)mpf_set(c->max_condition,w[4]);
    mpf_div_2exp(w[5],w[4],c->bits-32);
    if(mpf_cmp(w[5],c->tolerance)>0){ok=-1;goto done;}
    /* Recompute backward error in the original matrix. */
    mpf_set_ui(w[4],0);mpf_set_ui(w[5],0);
    for(int i=0;i<N;i++) {
        mpf_neg(w[0],rhs[i]);mpf_abs(w[1],rhs[i]);
        for(int j=0;j<N;j++){mpf_mul(w[6],matrix[i*N+j],delta[j]);mpf_add(w[0],w[0],w[6]);mpf_abs(w[6],w[6]);mpf_add(w[1],w[1],w[6]);}
        normmax(w[4],w[0],w[6]);normmax(w[5],w[1],w[6]);
    }
    mpf_mul(w[5],w[5],c->tolerance);ok=mpf_cmp(w[4],w[5])<=0;
 done:clear(a,N*2*N);clear(w,7);return ok;
}
static int step(core *c,const mpf_t *z,const mpf_t h,mpf_t *polynomial) {
    mpf_t K[N],matrix[N*N],rhs[N],delta[N],stage[2],f[2],J[4],w[5];
    init(K,N,c->bits);init(matrix,N*N,c->bits);init(rhs,N,c->bits);init(delta,N,c->bits);init(stage,2,c->bits);init(f,2,c->bits);init(J,4,c->bits);init(w,5,c->bits);
    field(c,z,f,NULL);for(int s=0;s<3;s++)for(int d=0;d<2;d++)mpf_mul(K[2*s+d],h,f[d]);
    int ok=0;
    for(int iteration=0;iteration<24;iteration++) {
        for(int s=0;s<3;s++) {
            for(int d=0;d<2;d++){mpf_set(stage[d],z[d]);for(int j=0;j<3;j++){mpf_mul(w[0],c->A[3*s+j],K[2*j+d]);mpf_add(stage[d],stage[d],w[0]);}}
            field(c,stage,f,J);
            for(int d=0;d<2;d++) {
                int row=2*s+d;mpf_mul(rhs[row],h,f[d]);mpf_sub(rhs[row],rhs[row],K[row]);
                for(int j=0;j<3;j++)for(int e=0;e<2;e++) {
                    int col=2*j+e;mpf_mul(w[0],h,c->A[3*s+j]);mpf_mul(w[0],w[0],J[2*d+e]);mpf_neg(matrix[row*N+col],w[0]);
                    if(row==col)mpf_add_ui(matrix[row*N+col],matrix[row*N+col],1);
                }
            }
        }
        int solved=solve(c,matrix,rhs,delta);if(solved<=0){ok=solved;goto done;}
        int converged=1;
        for(int i=0;i<N;i++) {
            mpf_add(K[i],K[i],delta[i]);mpf_abs(w[1],delta[i]);mpf_abs(w[2],K[i]);
            mpf_mul(w[2],w[2],c->tolerance);
            if(mpf_cmp(w[1],w[2])>0)converged=0;
        }
        if(converged){ok=1;break;}
    }
    if(ok)for(int d=0;d<2;d++) {
        mpf_set(polynomial[4*d],z[d]);
        for(int k=0;k<3;k++) {
            mpf_set_ui(polynomial[4*d+k+1],0);
            for(int s=0;s<3;s++){mpf_mul(w[0],K[2*s+d],c->basis[3*s+k]);mpf_div_ui(w[0],w[0],(unsigned)k+1);mpf_add(polynomial[4*d+k+1],polynomial[4*d+k+1],w[0]);}
        }
    }
 done:clear(K,N);clear(matrix,N*N);clear(rhs,N);clear(delta,N);clear(stage,2);clear(f,2);clear(J,4);clear(w,5);return ok;
}
static int geometry(core *c,mpf_t *p,double *angle) {
    mpf_t x[2],v[2],f[2],w[8];init(x,2,c->bits);init(v,2,c->bits);init(f,2,c->bits);init(w,8,c->bits);
    int ok=1;*angle=0;
    for(int sample=0;sample<=4;sample++) {
        mpf_set_ui(w[0],(unsigned)sample);mpf_div_ui(w[0],w[0],4);
        for(int d=0;d<2;d++) {
            mpf_set(x[d],p[4*d+3]);mpf_mul_ui(v[d],p[4*d+3],3);
            for(int k=2;k>=0;k--){mpf_mul(x[d],x[d],w[0]);mpf_add(x[d],x[d],p[4*d+k]);}
            for(int k=2;k>=1;k--){mpf_mul(v[d],v[d],w[0]);mpf_mul_ui(w[1],p[4*d+k],(unsigned)k);mpf_add(v[d],v[d],w[1]);}
        }
        field(c,x,f,NULL);mpf_set_ui(w[1],0);mpf_set_ui(w[2],0);mpf_set_ui(w[3],0);
        for(int d=0;d<2;d++){mpf_mul(w[4],v[d],f[d]);mpf_add(w[1],w[1],w[4]);mpf_mul(w[4],v[d],v[d]);mpf_add(w[2],w[2],w[4]);mpf_mul(w[4],f[d],f[d]);mpf_add(w[3],w[3],w[4]);}
        if(mpf_sgn(w[1])<=0||!mpf_sgn(w[2])||!mpf_sgn(w[3])){ok=0;break;}
        mpf_mul(w[4],v[0],f[1]);mpf_mul(w[5],v[1],f[0]);mpf_sub(w[4],w[4],w[5]);mpf_mul(w[4],w[4],w[4]);mpf_mul(w[5],w[2],w[3]);mpf_div(w[4],w[4],w[5]);
        double a=mpf_get_d(w[4]);if(a>*angle)*angle=a;if(a>1e-8){ok=0;break;}
    }
    clear(x,2);clear(v,2);clear(f,2);clear(w,8);return ok;
}
void spong_precision_bridge_destroy(spong_precision_bridge_result *r) {
    if(!r)return;if(r->coefficients)for(size_t i=0;i<r->steps*8;i++)free(r->coefficients[i]);
    free(r->coefficients);free(r->max_solve_condition);memset(r,0,sizeof(*r));
}
int spong_precision_bridge(const spong_precision_bridge_request *r,spong_precision_bridge_result *out) {
    if(!out)return -1;memset(out,0,sizeof(*out));
    if(!r||!r->gradient||!r->rates||r->rows<1||r->rows>LIMIT||r->columns<1||r->columns>LIMIT||r->decimal_digits<50||r->decimal_digits>200||!r->max_steps||r->max_steps>2048)return -1;
    mpf_t plan[3];init(plan,3,256);int valid=parse(plan[0],r->rates[0],0)&&parse(plan[1],r->rates[1],0)&&mpf_sgn(plan[0])>0&&mpf_cmp(plan[0],plan[1])<=0;
    if(!valid){clear(plan,3);return -1;}mpf_div(plan[2],plan[1],plan[0]);mp_exp_t exp;mpf_get_d_2exp(&exp,plan[2]);
    if(exp>4096){clear(plan,3);return -3;}
    long bits=(long)ceil(r->decimal_digits*3.321928094887363)+128+(exp>0?exp:0);clear(plan,3);if(bits>4096)return -3;
    core c={0};c.rows=r->rows;c.cols=r->columns;c.bits=(mp_bitcnt_t)bits;out->working_bits=(unsigned long)bits;
    size_t count=2*(size_t)c.rows*c.cols;c.g=malloc(count*sizeof(mpf_t));if(!c.g)return -2;
    init(c.g,count,c.bits);init(c.A,9,c.bits);init(c.basis,9,c.bits);init(c.nodes,3,c.bits);mpf_init2(c.tolerance,c.bits);mpf_init2(c.max_condition,c.bits);
    mpf_set_ui(c.tolerance,10);mpf_pow_ui(c.tolerance,c.tolerance,r->decimal_digits-20);mpf_ui_div(c.tolerance,1,c.tolerance);
    mpf_t z[2],h,cap,slow,fast,p[8];init(z,2,c.bits);mpf_init2(h,c.bits);mpf_init2(cap,c.bits);mpf_init2(slow,c.bits);mpf_init2(fast,c.bits);init(p,8,c.bits);
    int rc=-1;for(size_t i=0;i<count;i++)if(!parse(c.g[i],r->gradient[i],1))goto done;
    if(!parse(slow,r->rates[0],0)||!parse(fast,r->rates[1],0))goto done;
    mpf_ui_div(h,1,fast);mpf_div_ui(h,h,4);mpf_ui_div(cap,4,slow);tableau(&c);
    out->coefficients=calloc(8*(size_t)r->max_steps,sizeof(char*));if(!out->coefficients){rc=-2;goto done;}
    rc=0;unsigned digits=(unsigned)ceil(bits*0.3010299956639812)+4;
    for(unsigned i=0;i<r->max_steps;i++) {
        int accepted=0;double angle=0;
        for(unsigned retry=0;retry<32;retry++) {
            int st=step(&c,z,h,p);
            if(st<0){out->precision_refusals++;out->termination=1;goto finished;}
            if(!st)out->solve_failures++;
            if(st&&geometry(&c,p,&angle)){accepted=1;break;}
            out->rejected++;mpf_div_2exp(h,h,1);
        }
        if(!accepted){out->termination=1;break;}
        out->steps++;
        for(int k=0;k<8;k++){out->coefficients[(out->steps-1)*8+k]=decimal(p[k],digits);if(!out->coefficients[(out->steps-1)*8+k]){rc=-2;goto done;}}
        for(int d=0;d<2;d++){mpf_set_ui(z[d],0);for(int k=0;k<4;k++)mpf_add(z[d],z[d],p[4*d+k]);}
        mpf_mul_ui(h,h,angle<1e-10?9:6);mpf_div_ui(h,h,5);if(mpf_cmp(h,cap)>0)mpf_set(h,cap);
    }
 finished:out->max_solve_condition=decimal(c.max_condition,40);if(!out->max_solve_condition)rc=-2;
 done:clear(z,2);mpf_clear(h);mpf_clear(cap);mpf_clear(slow);mpf_clear(fast);clear(p,8);
    clear(c.g,count);free(c.g);clear(c.A,9);clear(c.basis,9);clear(c.nodes,3);mpf_clear(c.tolerance);mpf_clear(c.max_condition);
    if(rc)spong_precision_bridge_destroy(out);return rc;
}
