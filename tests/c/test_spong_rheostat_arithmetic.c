/* Private arithmetic tests compile the kernel into this test translation
 * unit. The linked core supplies the independent exact/Sturm routines. */
#include "../../src/c/spong_rheostat_gmp.c"
#include <assert.h>
int main(void) {
    spong_rheostat *c=calloc(1,sizeof(*c));assert(c);
    c->bits=256;c->policy.precision_bits=192;c->pool=calloc(CAP,sizeof(mpf_t));assert(c->pool);
    assert(setjmp(c->abort)==0);
    c->zero=num(0);c->one=num(1);c->eps=num(1);mpf_div_2exp(c->eps.v,c->eps.v,192);
    dual M[DIM][DIM],rhs[DIM],x[DIM];
    M[0][0]=parse(c,"1e-100");M[0][1]=parse(c,"2e-100");
    M[1][0]=parse(c,"3e100");M[1][1]=parse(c,"5e100");
    rhs[0]=parse(c,"5e-100");rhs[1]=parse(c,"13e100");
    mpf_set(rhs[0].d,rhs[0].v);mpf_set(rhs[1].d,rhs[1].v);
    assert(solve(c,2,M,rhs,x));
    assert(cmp(ab(sub(x[0],num(1))),parse(c,"1e-60"))<0);
    assert(cmp(ab(sub(x[1],num(2))),parse(c,"1e-60"))<0);
    assert(mpf_cmp(x[0].v,x[0].d)==0 && mpf_cmp(x[1].v,x[1].d)==0);
    M[0][0]=M[0][1]=M[1][0]=c->one;M[1][1]=add(c->one,parse(c,"1e-65"));
    assert(!solve(c,2,M,rhs,x));
    M[0][0]=c->zero;M[0][1]=c->one;M[1][0]=c->one;M[1][1]=c->zero;
    rhs[0]=num(2);rhs[1]=num(1);assert(solve(c,2,M,rhs,x));
    assert(cmp(x[0],num(1))==0 && cmp(x[1],num(2))==0);
    for(int k=-50;k<=50;k+=10) {
        dual a=divv(num(k),num(7));mpf_set_ui(a.d,1);
        dual b=logarithm(c,exponential(c,a));
        assert(cmp(ab(sub(a,b)),parse(c,"1e-55"))<0);
        assert(mpf_cmp_ui(b.d,1)==0 || fabs(mpf_get_d(b.d)-1)<1e-15);
    }
    /* Direct subtraction would erase the small positive eigenvalue at 256 bits;
     * det/large retains it. The policy would refuse it as unresolved. */
    dual tr=parse(c,"-1e100"),det=num(-1);
    dual large=divv(sub(tr,root(sub(mul(tr,tr),mul(num(4),det)))),num(2));
    dual small=divv(det,large);
    assert(cmp(ab(sub(small,parse(c,"1e-100"))),parse(c,"1e-160"))<0);
    /* A level curve crossing a b-turn: a is not a function of b here.
     * L=2-2a+(1+b^2)a^2, level=3/2, backbone point (1/2,1). */
    c->na=3;c->nb=1;c->C=num(2);c->level=divv(num(3),num(2));
    c->A[0]=c->one;c->A[1]=c->zero;c->A[2]=c->one;c->B[0]=c->one;
    for(int i=0;i<3;i++){mpq_init(c->qa[i]);c->qa_init++;mpq_set_ui(c->qa[i],i==1?0:1,1);}
    mpq_init(c->qb[0]);c->qb_init++;mpq_set_ui(c->qb[0],1,1);
    c->normal[0]=c->zero;c->normal[1]=c->one;c->tol=mul(num(1024),c->eps);c->policy.max_newton=24;
    dual points[5][2],z[2],distance=fresh(c);
    for(int i=0;i<5;i++){points[i][0]=divv(num(50),num(100));points[i][1]=c->one;}
    for(int k=0;k<2;k++) {
        z[0]=divv(num(k?51:49),num(100));z[1]=num(1);mpf_set_ui(z[0].d,1);
        points[k+1][0]=val(z[0]);points[k+1][1]=val(z[1]);
        project(c,z,distance);assert(sgn(distance)>0);assert(cmp(z[1],c->one)<0);
        assert(cmp(ab(sub(loss(c,z),c->level)),parse(c,"1e-55"))<0);
        dual g[2];gradient(c,z,g);dual dot=c->zero;
        for(int i=0;i<2;i++){dual tangent=val(z[i]);mpf_set(tangent.v,z[i].d);dot=add(dot,mul(val(g[i]),tangent));}
        assert(cmp(ab(dot),parse(c,"1e-55"))<0);
        points[k+3][0]=val(z[0]);points[k+3][1]=val(z[1]);
    }
    assert(chart_check(c,points));c->normal[1]=num(-1);assert(!chart_check(c,points));
    spong_rheostat_destroy(c);return 0;
}
