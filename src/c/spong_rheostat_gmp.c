/* C99/GMP rheostat proposal engine.  All arithmetic and numerical decisions
 * are here; adapters marshal exact inputs and report numerical evidence.
 * mpf is deliberately NOT presented as outward-rounded interval arithmetic. */
#include "spong/spong_rheostat.h"
#include <gmp.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define CAP 262144
#define DEG 65
#define ORD 21
#define DIM 8
typedef struct { mpf_ptr v,d; } dual;
struct spong_rheostat {
    mpf_t *pool;
    size_t top, initialized, base;
    mp_bitcnt_t bits;
    jmp_buf abort;
    int status;
    char reason[128];
    spong_rheostat_policy policy;
    dual zero,one,eps,tol,A[DEG],B[DEG],C,radius,fraction;
    size_t na,nb;
    dual saddle[2][2],losses[2],level,nodes[4],table[4][4],weights[4];
    dual anchor[2],normal[2];
    int chart_ready,direction[2];
    mpq_t qa[DEG],qb[DEG];
    size_t qa_init,qb_init;
    uint64_t solves,halvings,evaluations;
    double backward;
};
static void fail(spong_rheostat *c,int code,const char *why) {
    c->status=code; snprintf(c->reason,sizeof(c->reason),"%s",why); longjmp(c->abort,1);
}
static mpf_ptr slot(spong_rheostat *c) {
    if(c->top==CAP) fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"temporary arithmetic budget");
    if(c->top==c->initialized) mpf_init2(c->pool[c->initialized++],c->bits);
    mpf_ptr x=c->pool[c->top++];mpf_set_ui(x,0);return x;
}
static dual fresh(spong_rheostat *c) {dual x={slot(c),slot(c)};return x;}
static dual number(spong_rheostat *c,long n) {dual x=fresh(c);mpf_set_si(x.v,n);return x;}
static dual plus(spong_rheostat *c,dual a,dual b) {
    dual x=fresh(c);mpf_add(x.v,a.v,b.v);mpf_add(x.d,a.d,b.d);return x;
}
static dual minus(spong_rheostat *c,dual a,dual b) {
    dual x=fresh(c);mpf_sub(x.v,a.v,b.v);mpf_sub(x.d,a.d,b.d);return x;
}
static dual times(spong_rheostat *c,dual a,dual b) {
    dual x=fresh(c);mpf_ptr t=slot(c);mpf_mul(x.v,a.v,b.v);
    mpf_mul(x.d,a.d,b.v);mpf_mul(t,a.v,b.d);mpf_add(x.d,x.d,t);return x;
}
static dual divide(spong_rheostat *c,dual a,dual b) {
    if(!mpf_sgn(b.v))fail(c,SPONG_RHEOSTAT_CONDITIONING,"zero denominator");
    dual x=fresh(c);mpf_div(x.v,a.v,b.v);mpf_mul(x.d,x.v,b.d);
    mpf_sub(x.d,a.d,x.d);mpf_div(x.d,x.d,b.v);return x;
}
static dual square_root(spong_rheostat *c,dual a) {
    if(mpf_sgn(a.v)<=0)fail(c,SPONG_RHEOSTAT_CONDITIONING,"nonpositive square root");
    dual x=fresh(c);mpf_sqrt(x.v,a.v);mpf_div(x.d,a.d,x.v);mpf_div_2exp(x.d,x.d,1);return x;
}
static dual scalar(spong_rheostat *c,dual a) {dual x=fresh(c);mpf_set(x.v,a.v);return x;}
static dual absolute(spong_rheostat *c,dual a) {dual x=fresh(c);mpf_abs(x.v,a.v);return x;}
static void put(dual a,dual b) {mpf_set(a.v,b.v);mpf_set(a.d,b.d);}
#define num(n) number(c,(n))
#define add(a,b) plus(c,(a),(b))
#define sub(a,b) minus(c,(a),(b))
#define mul(a,b) times(c,(a),(b))
#define divv(a,b) divide(c,(a),(b))
#define root(a) square_root(c,(a))
#define val(a) scalar(c,(a))
#define ab(a) absolute(c,(a))
#define cmp(a,b) mpf_cmp((a).v,(b).v)
#define sgn(a) mpf_sgn((a).v)
static dual maximum(dual a,dual b) {return cmp(a,b)>0?a:b;}
static dual parse(spong_rheostat *c,const char *s) {
    if(!s || strlen(s)>4096)fail(c,SPONG_RHEOSTAT_INVALID,"invalid numeric input length");
    dual x=fresh(c);
    if(strchr(s,'/')) {
        mpq_t q;mpq_init(q);int bad=mpq_set_str(q,s,10);
        if(!bad && mpz_sgn(mpq_denref(q))>0){mpq_canonicalize(q);mpf_set_q(x.v,q);}
        else bad=1;
        mpq_clear(q);if(bad)fail(c,SPONG_RHEOSTAT_INVALID,"invalid rational input");
    } else if(mpf_set_str(x.v,s,10))fail(c,SPONG_RHEOSTAT_INVALID,"invalid decimal input");
    signed long exponent=0;mpf_get_d_2exp(&exponent,x.v);
    if(mpf_sgn(x.v) && (exponent>262144 || exponent< -262144))
        fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"numeric exponent budget");
    return x;
}
/* Bounded elementary functions: sqrt reduction puts ln's atanh series at
 * |z|<1/8. exp uses a Taylor series on |x|<=1/8 and repeated squaring. */
static dual logarithm(spong_rheostat *c,dual x) {
    if(sgn(x)<=0)fail(c,SPONG_RHEOSTAT_CONDITIONING,"log domain");
    dual original=x,y=x;int k=0;
    while(cmp(y,divv(num(5),num(4)))>0 || cmp(y,divv(num(4),num(5)))<0) {
        if(++k>32)fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"log range reduction");y=root(y);
    }
    dual z=divv(sub(y,c->one),add(y,c->one)),z2=mul(z,z),term=z,sum=z;
    for(int n=1;n<2048;n++) {
        term=mul(term,z2);dual delta=divv(term,num(2*n+1));sum=add(sum,delta);
        if(cmp(ab(delta),c->eps)<0) {
            dual out=fresh(c);mpf_mul_2exp(out.v,sum.v,k+1);
            mpf_div(out.d,original.d,original.v);return out;
        }
    }
    fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"log series");return c->zero;
}
static dual exponential(spong_rheostat *c,dual x) {
    dual y=val(x);int k=0;
    while(cmp(ab(y),divv(c->one,num(8)))>0) {
        if(++k>32)fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"exp range reduction");y=divv(y,num(2));
    }
    dual sum=c->one,term=c->one;
    for(int n=1;n<2048;n++) {
        term=divv(mul(term,y),num(n));sum=add(sum,term);
        if(cmp(ab(term),c->eps)<0) {
            for(int j=0;j<k;j++)sum=mul(sum,sum);
            dual out=val(sum);mpf_mul(out.d,out.v,x.d);return out;
        }
    }
    fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"exp series");return c->zero;
}
static dual poly(spong_rheostat *c,const dual *p,size_t n,dual x,int derivative) {
    dual y=c->zero;
    for(int i=(int)n-1;i>=derivative;i--) {
        long scale=1;for(int j=0;j<derivative;j++)scale*=i-j;
        y=add(mul(y,x),mul(p[i],num(scale)));
    }return y;
}
static dual loss(spong_rheostat *c,dual *z) {
    return add(sub(c->C,mul(num(2),mul(z[0],poly(c,c->B,c->nb,z[1],0)))),
               mul(mul(z[0],z[0]),poly(c,c->A,c->na,z[1],0)));
}
static void gradient(spong_rheostat *c,dual *z,dual *g) {
    g[0]=mul(num(2),sub(mul(z[0],poly(c,c->A,c->na,z[1],0)),poly(c,c->B,c->nb,z[1],0)));
    g[1]=sub(mul(mul(z[0],z[0]),poly(c,c->A,c->na,z[1],1)),
              mul(num(2),mul(z[0],poly(c,c->B,c->nb,z[1],1))));
}
static void hessian(spong_rheostat *c,dual *z,dual h[2][2]) {
    h[0][0]=mul(num(2),poly(c,c->A,c->na,z[1],0));
    h[0][1]=h[1][0]=mul(num(2),sub(mul(z[0],poly(c,c->A,c->na,z[1],1)),poly(c,c->B,c->nb,z[1],1)));
    h[1][1]=sub(mul(mul(z[0],z[0]),poly(c,c->A,c->na,z[1],2)),mul(num(2),mul(z[0],poly(c,c->B,c->nb,z[1],2))));
}
/* Row equilibrated partial pivoting. Check BOTH RHS columns against the
 * original unequilibrated system, independently of the factorization. */
static int solve(spong_rheostat *c,int n,dual M[DIM][DIM],dual *rhs,dual *x) {
    dual a[DIM][DIM],b[DIM];
    dual guard=mul(num(64),c->eps);
    for(int i=0;i<n;i++) {
        dual scale=c->zero;for(int j=0;j<n;j++)scale=maximum(scale,ab(M[i][j]));
        if(!sgn(scale))return 0;
        for(int j=0;j<n;j++)a[i][j]=divv(M[i][j],scale);
        b[i]=divv(rhs[i],scale);
    }
    for(int k=0;k<n;k++) {
        int p=k;for(int i=k+1;i<n;i++)if(cmp(ab(a[i][k]),ab(a[p][k]))>0)p=i;
        if(cmp(ab(a[p][k]),guard)<=0)return 0;
        for(int j=0;j<n;j++){dual t=a[k][j];a[k][j]=a[p][j];a[p][j]=t;}
        dual t=b[k];b[k]=b[p];b[p]=t;
        for(int i=k+1;i<n;i++) {
            dual factor=divv(a[i][k],a[k][k]);
            for(int j=k+1;j<n;j++)a[i][j]=sub(a[i][j],mul(factor,a[k][j]));
            b[i]=sub(b[i],mul(factor,b[k]));
        }
    }
    for(int i=n-1;i>=0;i--) {
        dual sum=c->zero;for(int j=i+1;j<n;j++)sum=add(sum,mul(a[i][j],x[j]));
        x[i]=divv(sub(b[i],sum),a[i][i]);
    }
    for(int col=0;col<2;col++) {
        dual err=c->zero,scale=c->zero;
        for(int i=0;i<n;i++) {
            dual sum=c->zero,den=c->zero;
            for(int j=0;j<n;j++) {
                dual xx=val(x[j]);if(col)mpf_set(xx.v,x[j].d);
                sum=add(sum,mul(M[i][j],xx));den=add(den,mul(ab(M[i][j]),ab(xx)));
            }
            dual rr=val(rhs[i]);if(col)mpf_set(rr.v,rhs[i].d);
            err=maximum(err,ab(sub(sum,rr)));scale=maximum(scale,add(den,ab(rr)));
        }
        if(sgn(scale)) {
            dual ratio=divv(err,scale);double d=mpf_get_d(ratio.v);
            if(d>c->backward)c->backward=d;
            if(cmp(ratio,mul(num(256*n),c->eps))>0)return 0;
        } else if(sgn(err))return 0;
    }
    c->solves++;return 1;
}
static void tableau(spong_rheostat *c) {
    int s=c->policy.stages;dual half=divv(c->one,num(2));
    if(s==2) {
        dual r=divv(root(num(3)),num(6));put(c->nodes[0],sub(half,r));put(c->nodes[1],add(half,r));
    } else if(s==3) {
        dual r=divv(root(num(15)),num(10));put(c->nodes[0],sub(half,r));put(c->nodes[1],half);put(c->nodes[2],add(half,r));
    } else {
        dual r=root(divv(num(6),num(5)));
        dual outer=divv(root(divv(add(num(3),mul(num(2),r)),num(7))),num(2));
        dual inner=divv(root(divv(sub(num(3),mul(num(2),r)),num(7))),num(2));
        put(c->nodes[0],sub(half,outer));put(c->nodes[1],sub(half,inner));put(c->nodes[2],add(half,inner));put(c->nodes[3],add(half,outer));
    }
    for(int j=0;j<s;j++) {
        dual p[5]={c->one},den=c->one;int n=1;
        for(int k=0;k<s;k++)if(k!=j) {
            dual q[5];for(int i=0;i<=n;i++)q[i]=c->zero;
            for(int i=0;i<n;i++){q[i]=sub(q[i],mul(c->nodes[k],p[i]));q[i+1]=add(q[i+1],p[i]);}
            n++;for(int i=0;i<n;i++)p[i]=q[i];den=mul(den,sub(c->nodes[j],c->nodes[k]));
        }
        dual primitive[5];primitive[0]=c->zero;
        for(int i=0;i<n;i++)primitive[i+1]=divv(p[i],mul(num(i+1),den));
        for(int i=0;i<s;i++)put(c->table[i][j],poly(c,primitive,n+1,c->nodes[i],0));
        put(c->weights[j],poly(c,primitive,n+1,c->one,0));
    }
}
static void series_mul(spong_rheostat *c,dual *out,dual *a,dual *b,int n) {
    for(int k=0;k<=n;k++){out[k]=c->zero;for(int j=0;j<=k;j++)out[k]=add(out[k],mul(a[j],b[k-j]));}
}
static void compose(spong_rheostat *c,dual *out,const dual *p,int len,dual *b,int n,int deriv) {
    for(int j=0;j<=n;j++)out[j]=c->zero;
    for(int k=len-1;k>=deriv;k--) {
        dual temp[ORD];series_mul(c,temp,out,b,n);
        for(int j=0;j<=n;j++)out[j]=temp[j];
        out[0]=add(out[0],mul(p[k],num(deriv?k:1)));
    }
}
static void launch(spong_rheostat *c,dual lam,int index,dual *out,dual residual) {
    dual h[2][2];hessian(c,c->saddle[index],h);dual j[2][2],l2=mul(lam,lam);
    for(int i=0;i<2;i++)for(int k=0;k<2;k++)j[i][k]=sub(c->zero,mul(i?c->one:l2,h[i][k]));
    dual tr=add(j[0][0],j[1][1]);
    dual det=sub(mul(j[0][0],j[1][1]),mul(j[0][1],j[1][0]));
    dual det_scale=add(ab(mul(j[0][0],j[1][1])),ab(mul(j[0][1],j[1][0])));
    if(sgn(det)>=0 || cmp(ab(det),mul(mul(num(64),c->eps),det_scale))<=0)
        fail(c,SPONG_RHEOSTAT_CONDITIONING,"saddle inertia not numerically resolved");
    dual disc=root(sub(mul(tr,tr),mul(num(4),det)));
    dual large=divv(sgn(tr)>=0?add(tr,disc):sub(tr,disc),num(2));
    dual small=divv(det,large),rho=index?(sgn(large)<0?large:small):(sgn(large)>0?large:small);
    if(cmp(ab(rho),mul(mul(num(64),c->eps),maximum(ab(large),ab(small))))<=0)
        fail(c,SPONG_RHEOSTAT_CONDITIONING,"unresolved small eigenvalue");
    dual v[2]={j[0][1],sub(rho,j[0][0])},w[2]={sub(rho,j[1][1]),j[1][0]};
    dual vn=add(mul(v[0],v[0]),mul(v[1],v[1])),wn=add(mul(w[0],w[0]),mul(w[1],w[1]));
    if(cmp(wn,vn)>0){v[0]=w[0];v[1]=w[1];vn=wn;}
    dual norm=root(vn);for(int i=0;i<2;i++)v[i]=divv(v[i],norm);
    if(cmp(ab(v[1]),mul(num(64),c->eps))<=0)fail(c,SPONG_RHEOSTAT_CONDITIONING,"horizontal departure cannot use b direction");
    if(sgn(v[1])<0)for(int i=0;i<2;i++)v[i]=sub(c->zero,v[i]);
    dual a[ORD],b[ORD];
    for(int i=0;i<ORD;i++){a[i]=fresh(c);b[i]=fresh(c);}
    put(a[0],c->saddle[index][0]);put(b[0],c->saddle[index][1]);put(a[1],v[0]);put(b[1],v[1]);
    size_t mark=c->top;
    for(int n=2;n<=(int)c->policy.launch_order;n++) {
        c->top=mark;
        dual A[ORD],B[ORD],Ap[ORD],Bp[ORD],aA[ORD],aa[ORD],aaAp[ORD],aBp[ORD];
        compose(c,A,c->A,c->na,b,n,0);compose(c,B,c->B,c->nb,b,n,0);
        compose(c,Ap,c->A,c->na,b,n,1);compose(c,Bp,c->B,c->nb,b,n,1);
        series_mul(c,aA,a,A,n);series_mul(c,aa,a,a,n);series_mul(c,aaAp,aa,Ap,n);series_mul(c,aBp,a,Bp,n);
        dual rhs[2]={mul(mul(num(-2),l2),sub(aA[n],B[n])),sub(mul(num(2),aBp[n]),aaAp[n])};
        dual mat[DIM][DIM],x[DIM];
        /* Dual matrix sensitivity: solve primal and then M x' = f'-M' x. */
        dual dm[2][2];
        for(int i=0;i<2;i++)for(int k=0;k<2;k++){dm[i][k]=sub(i==k?mul(num(n),rho):c->zero,j[i][k]);mat[i][k]=val(dm[i][k]);}
        dual rr[2]={val(rhs[0]),val(rhs[1])};
        if(!solve(c,2,mat,rr,x))fail(c,SPONG_RHEOSTAT_CONDITIONING,"launch recurrence solve");
        dual primal[2]={x[0],x[1]};
        for(int i=0;i<2;i++) {
            rr[i]=fresh(c);mpf_set(rr[i].v,rhs[i].d);
            for(int k=0;k<2;k++){dual d=val(dm[i][k]);mpf_set(d.v,dm[i][k].d);rr[i]=sub(rr[i],mul(d,primal[k]));}
        }
        if(!solve(c,2,mat,rr,x))fail(c,SPONG_RHEOSTAT_CONDITIONING,"launch sensitivity solve");
        put(a[n],primal[0]);mpf_set(a[n].d,x[0].v);put(b[n],primal[1]);mpf_set(b[n].d,x[1].v);
    }
    c->top=mark;dual t=mul(c->radius,num(c->direction[index]));int n=c->policy.launch_order+1;
    put(out[0],poly(c,a,n,t,0));put(out[1],poly(c,b,n,t,0));dual g[2];gradient(c,out,g);
    dual r0=sub(mul(sub(c->zero,l2),g[0]),mul(mul(rho,t),poly(c,a,n,t,1)));
    dual r1=sub(sub(c->zero,g[1]),mul(mul(rho,t),poly(c,b,n,t,1)));
    put(residual,maximum(ab(r0),ab(r1)));
}
static dual poly_magnitude(spong_rheostat *c,const dual *p,size_t n,dual x,int derivative) {
    dual bound=c->zero; x=ab(x);
    for(int i=(int)n-1;i>=derivative;i--)
        bound=add(mul(bound,x),mul(ab(p[i]),num(derivative?i:1)));
    return bound;
}
static void field(spong_rheostat *c,dual *z,dual lam,dual factor,dual *out,dual jac[2][2]) {
    dual g[2];gradient(c,z,g);dual l2=mul(lam,lam),w[2]={mul(l2,g[0]),g[1]};
    dual aa=ab(z[0]);
    dual ga_bound=mul(num(2),add(mul(aa,poly_magnitude(c,c->A,c->na,z[1],0)),poly_magnitude(c,c->B,c->nb,z[1],0)));
    dual gb_bound=add(mul(mul(aa,aa),poly_magnitude(c,c->A,c->na,z[1],1)),mul(mul(num(2),aa),poly_magnitude(c,c->B,c->nb,z[1],1)));
    if(cmp(maximum(ab(g[0]),ab(g[1])),mul(mul(num(1024),c->eps),maximum(ga_bound,gb_bound)))<=0)
        fail(c,SPONG_RHEOSTAT_CONDITIONING,"gradient below polynomial evaluation resolution floor");
    dual q=add(mul(g[0],w[0]),mul(g[1],g[1]));
    if(sgn(q)<=0)fail(c,SPONG_RHEOSTAT_CONDITIONING,"loss parametrization reached a critical point");
    for(int i=0;i<2;i++)out[i]=divv(mul(factor,w[i]),q);
    if(jac) {
        dual h[2][2];hessian(c,z,h);
        dual hw[2];for(int i=0;i<2;i++)hw[i]=add(mul(h[i][0],w[0]),mul(h[i][1],w[1]));
        for(int i=0;i<2;i++)for(int j=0;j<2;j++)
            jac[i][j]=mul(factor,sub(divv(mul(i?c->one:l2,h[i][j]),q),divv(mul(num(2),mul(w[i],hw[j])),mul(q,q))));
    }
}
/* The stage nonlinear and linear tolerances are separate. Derivative scale
 * must never relax the primal convergence criterion (or conversely). */
static int advance(spong_rheostat *c,dual *z,dual lam,dual initial,dual rate,
                   dual x0,dual step,int depth) {
    size_t entry=c->top;
    int s=c->policy.stages,n=2*s,ok=0;
    dual slopes[DIM],factors[4];
    for(int i=0;i<n;i++)slopes[i]=fresh(c);
    for(int i=0;i<s;i++) {
        factors[i]=mul(mul(initial,exponential(c,mul(rate,add(x0,mul(c->nodes[i],step))))),rate);
        dual f[2];field(c,z,lam,factors[i],f,NULL);put(slopes[2*i],f[0]);put(slopes[2*i+1],f[1]);
    }
    size_t mark=c->top;
    for(unsigned it=0;it<c->policy.max_newton;it++) {
        c->top=mark;dual states[4][2],jac[4][2][2],res[DIM],mat[DIM][DIM],correction[DIM];
        for(int i=0;i<s;i++) {
            for(int d=0;d<2;d++) {
                dual sum=c->zero;for(int j=0;j<s;j++)sum=add(sum,mul(c->table[i][j],slopes[2*j+d]));
                states[i][d]=add(z[d],mul(step,sum));
            }
            dual f[2];field(c,states[i],lam,factors[i],f,jac[i]);
            for(int d=0;d<2;d++)res[2*i+d]=sub(slopes[2*i+d],f[d]);
        }
        int converged=1;
        for(int col=0;col<2;col++) {
            dual err=c->zero,scale=c->one;
            for(int i=0;i<n;i++) {
                dual r=val(res[i]),v=val(slopes[i]);if(col){mpf_set(r.v,res[i].d);mpf_set(v.v,slopes[i].d);}
                err=maximum(err,ab(r));scale=maximum(scale,ab(v));
            }
            if(cmp(err,mul(c->tol,scale))>0)converged=0;
        }
        if(converged){ok=1;break;}
        for(int i=0;i<s;i++)for(int d=0;d<2;d++)for(int j=0;j<s;j++)for(int e=0;e<2;e++)
            mat[2*i+d][2*j+e]=sub(num(i==j && d==e),mul(mul(step,c->table[i][j]),val(jac[i][d][e])));
        dual rhs[DIM];for(int i=0;i<n;i++)rhs[i]=sub(c->zero,res[i]);
        if(!solve(c,n,mat,rhs,correction))break;
        for(int i=0;i<n;i++)put(slopes[i],add(slopes[i],correction[i]));
    }
    if(ok) {
        for(int d=0;d<2;d++) {
            dual sum=c->zero;for(int j=0;j<s;j++)sum=add(sum,mul(c->weights[j],slopes[2*j+d]));
            put(z[d],add(z[d],mul(step,sum)));
        }
        c->top=entry;return 1;
    }
    c->top=entry;
    if(depth>=(int)c->policy.max_halvings)return 0;
    c->halvings++;dual half=divv(step,num(2)),next=add(x0,half);
    if(!advance(c,z,lam,initial,rate,x0,half,depth+1))return 0;
    if(!advance(c,z,lam,initial,rate,next,half,depth+1))return 0;
    c->top=entry;return 1;
}
static void shot(spong_rheostat *c,dual lam,int index,dual *z,dual drift,dual residual) {
    size_t mark=c->top;launch(c,lam,index,z,residual);c->top=mark;
    dual initial=sub(loss(c,z),c->losses[index]),target=sub(c->level,c->losses[index]);
    if(sgn(mul(initial,target))<=0 || cmp(ab(initial),ab(target))>=0)
        fail(c,SPONG_RHEOSTAT_CONDITIONING,"launch lies outside saddle-to-section loss interval");
    dual rate=logarithm(c,divv(target,initial)),step=divv(c->one,num(c->policy.steps));
    size_t fixed=c->top;
    for(unsigned k=0;k<c->policy.steps;k++) {
        c->top=fixed;dual x0=divv(num(k),num(c->policy.steps));
        if(!advance(c,z,lam,initial,rate,x0,step,0))fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"Gauss solve failed after step halving budget");
    }
    c->top=fixed;put(drift,sub(loss(c,z),c->level));c->top=mark;
}
/* Exact rational interval normal-derivative test. It guards the chart for
 * the computed points, not enclosure of the true invariant manifolds. */
typedef struct {mpq_t lo,hi;} interval;
static void ii(interval *x){mpq_init(x->lo);mpq_init(x->hi);}
static void ic(interval *x){mpq_clear(x->lo);mpq_clear(x->hi);}
static void ia(interval *z,const interval *a,const interval *b){mpq_add(z->lo,a->lo,b->lo);mpq_add(z->hi,a->hi,b->hi);}
static void im(interval *z,const interval *a,const interval *b) {
    mpq_t p[4];for(int i=0;i<4;i++)mpq_init(p[i]);
    mpq_mul(p[0],a->lo,b->lo);mpq_mul(p[1],a->lo,b->hi);mpq_mul(p[2],a->hi,b->lo);mpq_mul(p[3],a->hi,b->hi);
    mpq_set(z->lo,p[0]);mpq_set(z->hi,p[0]);for(int i=1;i<4;i++){if(mpq_cmp(p[i],z->lo)<0)mpq_set(z->lo,p[i]);if(mpq_cmp(p[i],z->hi)>0)mpq_set(z->hi,p[i]);}
    for(int i=0;i<4;i++)mpq_clear(p[i]);
}
static void ipoly(interval *out,mpq_t *p,int n,const interval *x,int derivative) {
    interval a,t,co;ii(&a);ii(&t);ii(&co);
    for(int k=n-1;k>=derivative;k--) {
        mpq_set(co.lo,p[k]);if(derivative){mpz_mul_ui(mpq_numref(co.lo),mpq_numref(co.lo),k);mpq_canonicalize(co.lo);}
        mpq_set(co.hi,co.lo);im(&t,&a,x);ia(&a,&t,&co);
    }
    mpq_set(out->lo,a.lo);mpq_set(out->hi,a.hi);ic(&a);ic(&t);ic(&co);
}
static int chart_check(spong_rheostat *c,dual points[5][2]) {
    interval a,b,A,B,Ap,Bp,t,u,ga,gb,n,q,sum;interval *all[]={&a,&b,&A,&B,&Ap,&Bp,&t,&u,&ga,&gb,&n,&q,&sum};
    for(int i=0;i<13;i++)ii(all[i]);
    mpq_t x;mpq_init(x);
    for(int d=0;d<2;d++) {
        interval *v=d?&b:&a;mpq_set_f(v->lo,points[0][d].v);mpq_set(v->hi,v->lo);
        for(int k=1;k<5;k++){mpq_set_f(x,points[k][d].v);if(mpq_cmp(x,v->lo)<0)mpq_set(v->lo,x);if(mpq_cmp(x,v->hi)>0)mpq_set(v->hi,x);}
    }
    ipoly(&A,c->qa,c->na,&b,0);ipoly(&B,c->qb,c->nb,&b,0);ipoly(&Ap,c->qa,c->na,&b,1);ipoly(&Bp,c->qb,c->nb,&b,1);
    im(&t,&a,&A);mpq_sub(ga.lo,t.lo,B.hi);mpq_sub(ga.hi,t.hi,B.lo);
    mpq_set_si(n.lo,2,1);mpq_set(n.hi,n.lo);im(&ga,&ga,&n);
    im(&t,&a,&a);im(&u,&t,&Ap);im(&t,&a,&Bp);im(&t,&t,&n);
    mpq_sub(gb.lo,u.lo,t.hi);mpq_sub(gb.hi,u.hi,t.lo);
    mpq_set_f(n.lo,c->normal[0].v);mpq_set(n.hi,n.lo);im(&q,&ga,&n);
    mpq_set_f(n.lo,c->normal[1].v);mpq_set(n.hi,n.lo);im(&t,&gb,&n);ia(&sum,&q,&t);
    int ok=mpq_sgn(sum.lo)>0;
    mpq_clear(x);for(int i=0;i<13;i++)ic(all[i]);return ok;
}
static void project(spong_rheostat *c,dual *z,dual distance) {
    dual offset=fresh(c);size_t mark=c->top;
    for(unsigned k=0;k<c->policy.max_newton;k++) {
        c->top=mark;dual g[2];gradient(c,z,g);dual residual=sub(loss(c,z),c->level);
        dual den=add(mul(g[0],c->normal[0]),mul(g[1],c->normal[1]));
        dual den_scale=add(ab(mul(g[0],c->normal[0])),ab(mul(g[1],c->normal[1])));
        if(sgn(den)<=0 || cmp(ab(den),mul(mul(num(64),c->eps),den_scale))<=0)
            fail(c,SPONG_RHEOSTAT_CHART,"unresolved normal projection denominator");
        dual delta=divv(residual,den);put(offset,add(offset,val(delta)));
        for(int i=0;i<2;i++)put(z[i],sub(z[i],mul(delta,c->normal[i])));
        dual d=val(delta);mpf_set(d.v,delta.d);
        if(cmp(ab(delta),c->tol)<0 && cmp(ab(d),c->tol)<0){put(distance,ab(offset));c->top=mark;return;}
    }
    fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"normal projection iteration budget");
}
static void evaluate(spong_rheostat *c,dual lam,dual *out) {
    dual z[2][2],drift[2],residual[2],raw[2][2];
    for(int k=0;k<2;k++) {
        for(int d=0;d<2;d++){z[k][d]=fresh(c);raw[k][d]=fresh(c);}
        drift[k]=fresh(c);residual[k]=fresh(c);
    }
    size_t mark=c->top;
    mpf_set_ui(lam.d,1);
    if(sgn(lam)<=0)fail(c,SPONG_RHEOSTAT_INVALID,"Lambda must be positive");
    for(int k=0;k<2;k++){shot(c,lam,k,z[k],drift[k],residual[k]);for(int d=0;d<2;d++)put(raw[k][d],z[k][d]);}
    if(!c->chart_ready) {
        for(int d=0;d<2;d++)put(c->anchor[d],val(divv(add(z[0][d],z[1][d]),num(2))));
        dual g[2];gradient(c,c->anchor,g);dual norm=root(add(mul(g[0],g[0]),mul(g[1],g[1])));
        for(int d=0;d<2;d++)put(c->normal[d],divv(g[d],norm));
    }
    for(int k=0;k<2;k++)project(c,z[k],out[17+k]);
    dual points[5][2];for(int d=0;d<2;d++){points[0][d]=c->anchor[d];points[1][d]=raw[0][d];points[2][d]=raw[1][d];points[3][d]=z[0][d];points[4][d]=z[1][d];}
    if(!chart_check(c,points))fail(c,SPONG_RHEOSTAT_CHART,"exact normal monotonicity test failed");
    c->chart_ready=1;
    dual gap=sub(mul(c->normal[1],sub(z[0][0],z[1][0])),mul(c->normal[0],sub(z[0][1],z[1][1])));
    put(out[0],lam);put(out[1],gap);mpf_set(out[2].v,gap.d);put(out[3],c->level);
    for(int k=0;k<2;k++) {
        for(int d=0;d<2;d++){put(out[4+4*k+d],z[k][d]);mpf_set(out[6+4*k+d].v,z[k][d].d);}
        put(out[12+k],drift[k]);put(out[14+k],residual[k]);
    }
    c->evaluations++;c->top=mark;
}
/* Exact refinement of a checked root interval. The polynomial and all
 * bisection signs stay rational; Newton is not used for saddle identity. */
static void qeval(mpq_t out,mpq_t *p,int n,const mpq_t x) {
    mpq_set_ui(out,0,1);for(int k=n-1;k>=0;k--){mpq_mul(out,out,x);mpq_add(out,out,p[k]);}
}
static int saddle_root(spong_rheostat *c,int source,const char *lower,const char *upper,dual out) {
    mpq_t p[2*DEG],lo,hi,mid,v,t,w,eps;
    for(int i=0;i<2*DEG;i++)mpq_init(p[i]);
    mpq_inits(lo,hi,mid,v,t,w,eps,NULL);
    int status=SPONG_RHEOSTAT_INVALID,n=0;char *texts[2*DEG]={0};spong_sturm_plan *plan=NULL;
    mpz_t lcm,z;mpz_inits(lcm,z,NULL);mpz_set_ui(lcm,1);
    if(!lower||!upper||strlen(lower)>4096||strlen(upper)>4096)goto done;
    if(mpq_set_str(lo,lower,10)||mpq_set_str(hi,upper,10)||mpz_sgn(mpq_denref(lo))<=0||mpz_sgn(mpq_denref(hi))<=0)goto done;
    mpq_canonicalize(lo);mpq_canonicalize(hi);if(mpq_cmp(lo,hi)>0)goto done;
    if(source==0){n=c->nb;for(int i=0;i<n;i++)mpq_set(p[i],c->qb[i]);}
    else if(source==1) {
        n=c->na+c->nb-2;
        for(int i=0;i<(int)c->na;i++)for(int j=0;j<(int)c->nb;j++)if(i+j) {
            mpq_mul(t,c->qa[i],c->qb[j]);mpq_set_si(w,2*j-i,1);mpq_mul(t,t,w);mpq_add(p[i+j-1],p[i+j-1],t);
        }
    } else goto done;
    while(n>0&&!mpq_sgn(p[n-1]))n--;if(n<2)goto done;
    for(int i=0;i<n;i++)mpz_lcm(lcm,lcm,mpq_denref(p[i]));
    for(int i=0;i<n;i++) {
        mpz_divexact(z,lcm,mpq_denref(p[i]));mpz_mul(z,z,mpq_numref(p[i]));
        if(mpz_sizeinbase(z,2)>32768){status=SPONG_RHEOSTAT_WORK_LIMIT;goto done;}
        texts[i]=malloc(mpz_sizeinbase(z,10)+3);if(!texts[i]){status=SPONG_RHEOSTAT_ALLOCATION;goto done;}mpz_get_str(texts[i],10,z);
    }
    /* Parameterized exact models contain the working-precision rational eta.
     * Bound PRS coefficient growth proportionally to that precision; never
     * remove the exact work budget or substitute a floating root identity. */
    spong_exact_policy policy={512*c->bits,100000,4096};spong_sturm_analysis analysis;
    if(spong_sturm_plan_create_decimal((const char*const*)texts,n,&policy,&plan,&analysis)){status=SPONG_RHEOSTAT_WORK_LIMIT;goto done;}
    char ln[8192],ld[8192],hn[8192],hd[8192];
    mpz_get_str(ln,10,mpq_numref(lo));mpz_get_str(ld,10,mpq_denref(lo));mpz_get_str(hn,10,mpq_numref(hi));mpz_get_str(hd,10,mpq_denref(hi));
    uint32_t count=0;if(spong_sturm_plan_count(plan,ln,ld,hn,hd,&count))goto done;
    qeval(v,p,n,lo);int sign=mpq_sgn(v);
    if(!sign){if(count!=0)goto done;mpf_set_q(out.v,lo);status=0;goto done;}
    if(count!=1)goto done;
    qeval(v,p,n,hi);if(!mpq_sgn(v)){mpf_set_q(out.v,hi);status=0;goto done;}
    if(mpq_sgn(v)==sign)goto done;
    mpq_set_ui(eps,1,1);mpz_mul_2exp(mpq_denref(eps),mpq_denref(eps),c->bits+16);
    for(unsigned k=0;k<c->bits+32768;k++) {
        mpq_add(mid,lo,hi);mpq_div_2exp(mid,mid,1);qeval(v,p,n,mid);
        if(!mpq_sgn(v)){mpf_set_q(out.v,mid);status=0;goto done;}
        mpq_sub(t,hi,lo);mpq_abs(w,mid);mpq_add(w,w,eps); /* absolute target below 1; relative above 1 */
        if(mpq_cmp_ui(w,1,1)<0)mpq_set_ui(w,1,1);mpq_mul(w,w,eps);
        if(mpq_cmp(t,w)<=0){mpf_set_q(out.v,mid);status=0;goto done;}
        if(mpq_sgn(v)==sign)mpq_set(lo,mid);else mpq_set(hi,mid);
    }
    status=SPONG_RHEOSTAT_WORK_LIMIT;
done:
    spong_sturm_plan_destroy(plan);for(int i=0;i<2*DEG;i++){free(texts[i]);mpq_clear(p[i]);}
    mpq_clears(lo,hi,mid,v,t,w,eps,NULL);mpz_clears(lcm,z,NULL);return status;
}
static void report(spong_rheostat *c,spong_rheostat_result *r,dual *out) {
    r->status=c->status;snprintf(r->reason,sizeof(r->reason),"%s",c->reason);
    r->stage_solves=c->solves;r->step_halvings=c->halvings;r->evaluations=c->evaluations;r->max_backward_error=c->backward;
    if(!c->status&&out)for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)
        gmp_snprintf(r->values[i],SPONG_RHEOSTAT_TEXT,"%.*Fe",(int)(c->policy.precision_bits*.30103)+3,out[i].v);
}
void spong_rheostat_destroy(spong_rheostat *c) {
    if(!c)return;for(size_t i=0;i<c->initialized;i++)mpf_clear(c->pool[i]);free(c->pool);
    for(size_t i=0;i<c->qa_init;i++)mpq_clear(c->qa[i]);for(size_t i=0;i<c->qb_init;i++)mpq_clear(c->qb[i]);free(c);
}
int spong_rheostat_create(const char *const *A,size_t na,const char *const *B,size_t nb,
    const char *C,const char *const bounds[4],const int32_t sources[2],const int32_t directions[2],
    const char *radius,const char *fraction,const spong_rheostat_policy *policy,
    spong_rheostat **context,spong_rheostat_result *result) {
    if(!result)return SPONG_RHEOSTAT_INVALID;
    memset(result,0,sizeof(*result));
    if(!context){result->status=SPONG_RHEOSTAT_INVALID;return result->status;}
    *context=NULL;
    if(!policy||!A||!B||!C||!bounds||!sources||!directions||!na||!nb||na>DEG||nb>DEG||
       policy->precision_bits<128||policy->precision_bits>512||policy->stages<2||policy->stages>4||
       !policy->steps||policy->steps>65536||policy->launch_order<2||policy->launch_order>=ORD||
       !policy->max_newton||policy->max_newton>64||policy->max_halvings>8||
       (directions[0]!=-1&&directions[0]!=1)||(directions[1]!=-1&&directions[1]!=1)) {
        result->status=SPONG_RHEOSTAT_INVALID;snprintf(result->reason,sizeof(result->reason),"invalid rheostat request");return result->status;
    }
    spong_rheostat *c=calloc(1,sizeof(*c));if(!c){result->status=SPONG_RHEOSTAT_ALLOCATION;return result->status;}
    c->policy=*policy;c->bits=policy->precision_bits+64;c->na=na;c->nb=nb;
    c->pool=calloc(CAP,sizeof(mpf_t));if(!c->pool){free(c);result->status=SPONG_RHEOSTAT_ALLOCATION;return result->status;}
    if(setjmp(c->abort)){report(c,result,NULL);spong_rheostat_destroy(c);return result->status;}
    c->zero=num(0);c->one=num(1);c->eps=num(1);mpf_div_2exp(c->eps.v,c->eps.v,policy->precision_bits);
    c->tol=mul(num(1024),c->eps);
    for(size_t i=0;i<na;i++) {
        mpq_init(c->qa[i]);c->qa_init++;
        if(!A[i]||strlen(A[i])>4096||mpq_set_str(c->qa[i],A[i],10)||mpz_sgn(mpq_denref(c->qa[i]))<=0)fail(c,1,"invalid exact A coefficient");
        mpq_canonicalize(c->qa[i]);c->A[i]=fresh(c);mpf_set_q(c->A[i].v,c->qa[i]);
    }
    for(size_t i=0;i<nb;i++) {
        mpq_init(c->qb[i]);c->qb_init++;
        if(!B[i]||strlen(B[i])>4096||mpq_set_str(c->qb[i],B[i],10)||mpz_sgn(mpq_denref(c->qb[i]))<=0)fail(c,1,"invalid exact B coefficient");
        mpq_canonicalize(c->qb[i]);c->B[i]=fresh(c);mpf_set_q(c->B[i].v,c->qb[i]);
    }
    c->C=parse(c,C);c->radius=parse(c,radius);c->fraction=parse(c,fraction);
    if(sgn(c->radius)<=0||sgn(c->fraction)<=0||cmp(c->fraction,c->one)>=0)fail(c,1,"invalid radius or level fraction");
    for(int k=0;k<2;k++) {
        c->direction[k]=directions[k];c->losses[k]=fresh(c);c->anchor[k]=fresh(c);c->normal[k]=fresh(c);
        for(int d=0;d<2;d++)c->saddle[k][d]=fresh(c);
    }
    c->level=fresh(c);
    for(int i=0;i<4;i++){c->nodes[i]=fresh(c);c->weights[i]=fresh(c);for(int j=0;j<4;j++)c->table[i][j]=fresh(c);}
    c->base=c->top;
    for(int k=0;k<2;k++) {
        int status=saddle_root(c,sources[k],bounds[2*k],bounds[2*k+1],c->saddle[k][1]);
        if(status)fail(c,status,"saddle interval did not validate/refine");
        dual aa=poly(c,c->A,c->na,c->saddle[k][1],0);
        if(sgn(aa)<=0)fail(c,SPONG_RHEOSTAT_CONDITIONING,"nonpositive transverse curvature");
        put(c->saddle[k][0],divv(poly(c,c->B,c->nb,c->saddle[k][1],0),aa));
        put(c->losses[k],loss(c,c->saddle[k]));c->top=c->base;
    }
    if(cmp(c->losses[0],c->losses[1])<=0)fail(c,1,"source loss must exceed target loss");
    put(c->level,add(c->losses[0],mul(c->fraction,sub(c->losses[1],c->losses[0]))));
    tableau(c);c->top=c->base;*context=c;report(c,result,NULL);return 0;
}
/* Expose the same native germs used by loss continuation for display shots.
 * Points/sensitivities occupy the crossing slots; all gap/projection slots
 * are zero because these are launch points, not section crossings. */
int spong_rheostat_launch(spong_rheostat *c,const char *lambda,spong_rheostat_result *r) {
    if(!r)return SPONG_RHEOSTAT_INVALID;
    memset(r,0,sizeof(*r));if(!c){r->status=SPONG_RHEOSTAT_INVALID;return r->status;}
    c->status=0;c->reason[0]=0;c->top=c->base;
    c->solves=c->halvings=c->evaluations=0;c->backward=0;
    if(setjmp(c->abort)){report(c,r,NULL);return r->status;}
    dual out[SPONG_RHEOSTAT_FIELDS];for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)out[i]=fresh(c);
    dual lam=parse(c,lambda);
    if(sgn(lam)<=0)fail(c,SPONG_RHEOSTAT_INVALID,"lambda must be positive");
    mpf_set_ui(lam.d,1);put(out[0],lam);put(out[3],c->level);
    size_t mark=c->top;
    for(int k=0;k<2;k++) {
        dual z[2]={fresh(c),fresh(c)};launch(c,lam,k,z,out[14+k]);
        for(int d=0;d<2;d++) {
            put(out[4+4*k+d],z[d]);mpf_set(out[6+4*k+d].v,z[d].d);
        }
        c->top=mark;
    }
    report(c,r,out);return 0;
}
int spong_rheostat_evaluate(spong_rheostat *c,const char *lambda,spong_rheostat_result *r) {
    if(!r)return SPONG_RHEOSTAT_INVALID;
    memset(r,0,sizeof(*r));if(!c){r->status=SPONG_RHEOSTAT_INVALID;return r->status;}
    c->status=0;c->reason[0]=0;c->top=c->base;
    c->solves=c->halvings=c->evaluations=0;c->backward=0;
    if(setjmp(c->abort)){report(c,r,NULL);return r->status;}
    dual out[SPONG_RHEOSTAT_FIELDS];for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)out[i]=fresh(c);
    dual lam=parse(c,lambda);evaluate(c,lam,out);report(c,r,out);return 0;
}
static int locate_impl(spong_rheostat *c,const char *lower,const char *upper,const char *tolerance,
    uint32_t iterations,const char *window_lower,const char *window_upper,spong_rheostat_result *r) {
    if(!r)return SPONG_RHEOSTAT_INVALID;
    memset(r,0,sizeof(*r));if(!c){r->status=SPONG_RHEOSTAT_INVALID;return r->status;}
    c->status=0;c->reason[0]=0;c->top=c->base;
    c->solves=c->halvings=c->evaluations=0;c->backward=0;
    if(setjmp(c->abort)){report(c,r,NULL);return r->status;}
    dual lo=parse(c,lower),hi=parse(c,upper),tol=parse(c,tolerance),out[SPONG_RHEOSTAT_FIELDS],left[SPONG_RHEOSTAT_FIELDS],right[SPONG_RHEOSTAT_FIELDS];
    if(sgn(lo)<=0||cmp(lo,hi)>=0||sgn(tol)<=0||!iterations||iterations>256)fail(c,1,"invalid root bracket or budget");
    if(cmp(tol,mul(mul(num(64),c->eps),add(c->one,ab(hi))))<0)
        fail(c,SPONG_RHEOSTAT_CONDITIONING,"requested root tolerance below working precision floor");
    dual floor=window_lower?parse(c,window_lower):val(lo),ceiling=window_upper?parse(c,window_upper):val(hi);
    if(sgn(floor)<=0||cmp(floor,lo)>0||cmp(hi,ceiling)>0)fail(c,1,"seed bracket outside positive search bounds");
    if(window_lower) {
        dual pad=divv(sub(hi,lo),num(2));
        dual a=sub(lo,pad),b=add(hi,pad);
        put(lo,cmp(a,floor)>0?a:floor);put(hi,cmp(b,ceiling)<0?b:ceiling);
    }
    for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++){out[i]=fresh(c);left[i]=fresh(c);right[i]=fresh(c);}
    size_t mark=c->top;
    evaluate(c,lo,left);c->top=mark;evaluate(c,hi,right);c->top=mark;
    if(window_lower)for(int repair=0;repair<4 && sgn(left[1])*sgn(right[1])>0;repair++) {
        c->top=mark;
        dual *best=cmp(ab(left[1]),ab(right[1]))<=0?left:right;
        dual pad=sub(hi,lo);
        if(sgn(best[2]))pad=maximum(pad,mul(num(2),ab(divv(best[1],best[2]))));
        dual a=sub(lo,pad),b=add(hi,pad);
        a=cmp(a,floor)>0?a:floor;b=cmp(b,ceiling)<0?b:ceiling;
        if(cmp(a,lo)==0&&cmp(b,hi)==0)break;
        put(lo,a);put(hi,b);evaluate(c,lo,left);c->top=mark;evaluate(c,hi,right);c->top=mark;
    }
    if(sgn(left[1])*sgn(right[1])>0)fail(c,SPONG_RHEOSTAT_NO_BRACKET,"numerical section gap does not change sign");
    for(unsigned it=0;it<iterations;it++) {
        c->top=mark;
        dual *best=cmp(ab(left[1]),ab(right[1]))<=0?left:right;
        dual correction=sgn(best[2])?divv(best[1],best[2]):sub(hi,lo);
        if(!sgn(best[1])||cmp(ab(correction),tol)<=0||cmp(sub(hi,lo),tol)<=0) {
            for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)put(out[i],best[i]);put(out[16],sub(hi,lo));report(c,r,out);return 0;
        }
        dual proposal=sub(best[0],correction);
        if(cmp(proposal,lo)<=0||cmp(proposal,hi)>=0)proposal=divv(add(lo,hi),num(2));
        evaluate(c,proposal,out);
        if(sgn(left[1])*sgn(out[1])<=0){put(hi,proposal);for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)put(right[i],out[i]);}
        else {put(lo,proposal);for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++)put(left[i],out[i]);}
    }
    fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"root iteration budget exhausted");return 0;
}

int spong_rheostat_locate(spong_rheostat *c,const char *lower,const char *upper,const char *tol,
    uint32_t iterations,spong_rheostat_result *r) {
    return locate_impl(c,lower,upper,tol,iterations,NULL,NULL,r);
}
int spong_rheostat_locate_seeded(spong_rheostat *c,const char *lower,const char *upper,
    const char *window_lower,const char *window_upper,const char *tol,
    uint32_t iterations,spong_rheostat_result *r) {
    if(!window_lower||!window_upper) {
        if(r){memset(r,0,sizeof(*r));r->status=SPONG_RHEOSTAT_INVALID;}
        return SPONG_RHEOSTAT_INVALID;
    }
    return locate_impl(c,lower,upper,tol,iterations,window_lower,window_upper,r);
}

#include "spong_rheostat_pair.inc"
