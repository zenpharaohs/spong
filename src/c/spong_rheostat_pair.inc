/* Included in the GMP engine to reuse its checked arithmetic and linear solve.
 * The two-parameter driver never treats numerical mpf results as enclosures. */
#include "spong/spong_rheostat_pair.h"
typedef struct {
    mpq_t A[3][DEG],B[2][DEG],C;
    const spong_rheostat_pair_family *input;
    spong_rheostat *math;
    dual anchor[2][2],normal[2][2];
    int charts_ready;
    spong_rheostat_pair_result *result;
} pair_workspace;
static int pair_qread(mpq_t q,const char *s) {
    if(!s||strlen(s)>4096||mpq_set_str(q,s,10)||mpz_sgn(mpq_denref(q))<=0)return 0;
    mpq_canonicalize(q);return 1;
}
static char *pair_qtext(mpq_t q) {
    size_t n=mpz_sizeinbase(mpq_numref(q),10)+mpz_sizeinbase(mpq_denref(q),10)+4;
    char *s=malloc(n);if(s)mpq_get_str(s,10,q);return s;
}
/* Return failures normally so a rejected trust-region trial can be backtracked
 * and all temporary exact storage is freed before the caller may longjmp. */
static int pair_evaluate(pair_workspace *w,dual lam,dual eta,dual gap[2],dual slope[2],
                         char reason[128]) {
    spong_rheostat *c=w->math;const spong_rheostat_pair_family *f=w->input;
    char *A[DEG]={0},*B[DEG]={0},*C=NULL;
    mpq_t t,q,e;mpq_inits(t,q,e,NULL);mpq_set_f(e,eta.v);
    int status=SPONG_RHEOSTAT_ALLOCATION;
    for(size_t i=0;i<f->na;i++) {
        mpq_mul(q,w->A[2][i],e);mpq_add(q,q,w->A[1][i]);
        mpq_mul(q,q,e);mpq_add(q,q,w->A[0][i]);A[i]=pair_qtext(q);if(!A[i])goto done;
    }
    for(size_t i=0;i<f->nb;i++) {
        mpq_mul(q,w->B[1][i],e);mpq_add(q,q,w->B[0][i]);B[i]=pair_qtext(q);if(!B[i])goto done;
    }
    C=pair_qtext(w->C);if(!C)goto done;
    char lambda[SPONG_RHEOSTAT_TEXT];gmp_snprintf(lambda,sizeof(lambda),"%.*Fg",(int)(c->bits*.30103)+3,lam.v);
    for(int k=0;k<2;k++) {
        spong_rheostat *shot_context=NULL;spong_rheostat_result r;
        status=spong_rheostat_create((const char*const*)A,f->na,(const char*const*)B,f->nb,C,
            f->bounds+4*k,f->sources+2*k,f->directions+2*k,f->launch_radius,f->level_fraction,
            &c->policy,&shot_context,&r);
        if(status){snprintf(reason,128,"pair %d initialization: %.85s",k,r.reason);goto done;}
        if(w->charts_ready) {
            for(int d=0;d<2;d++){put(shot_context->anchor[d],w->anchor[k][d]);put(shot_context->normal[d],w->normal[k][d]);}
            shot_context->chart_ready=1;
        }
        status=spong_rheostat_evaluate(shot_context,lambda,&r);
        w->result->evaluations++;
        w->result->max_backward_error=fmax(w->result->max_backward_error,r.max_backward_error);
        if(!status) {
            if(mpf_set_str(gap[k].v,r.values[1],10)||mpf_set_str(slope[k].v,r.values[2],10))status=SPONG_RHEOSTAT_INVALID;
            if(!w->charts_ready)for(int d=0;d<2;d++){put(w->anchor[k][d],shot_context->anchor[d]);put(w->normal[k][d],shot_context->normal[d]);}
        }
        if(status)snprintf(reason,128,"pair %d evaluation: %.85s",k,r.reason);
        spong_rheostat_destroy(shot_context);if(status)goto done;
    }
    w->charts_ready=1;
done:
    for(size_t i=0;i<DEG;i++){free(A[i]);free(B[i]);}free(C);mpq_clears(t,q,e,NULL);
    if(status==SPONG_RHEOSTAT_ALLOCATION)snprintf(reason,128,"pair exact-input allocation failed");
    return status;
}
static dual pair_norm(spong_rheostat *c,dual *gap,dual *scale) {
    return maximum(ab(divv(gap[0],scale[0])),ab(divv(gap[1],scale[1])));
}
static void pair_run(pair_workspace *w,const char *const initial[2],const char *const lower[2],
                     const char *const upper[2],const char *step_text,const char *tol_text,unsigned budget) {
    spong_rheostat *c=w->math;spong_rheostat_pair_result *r=w->result;
    dual x[2],lo[2],hi[2],span[2],gap[2],slope[2],eta_slope[2],error[2],scale[2],corr[2];
    dual trial[2],trialgap[2],trialslope[2],out[SPONG_PAIR_FIELDS];
    for(int k=0;k<2;k++) {
        x[k]=parse(c,initial[k]);lo[k]=parse(c,lower[k]);hi[k]=parse(c,upper[k]);span[k]=sub(hi[k],lo[k]);
        if(sgn(span[k])<=0||cmp(x[k],lo[k])<=0||cmp(x[k],hi[k])>=0)fail(c,1,"initial pair parameters must lie strictly inside bounds");
        gap[k]=fresh(c);slope[k]=fresh(c);eta_slope[k]=fresh(c);error[k]=fresh(c);scale[k]=fresh(c);corr[k]=fresh(c);
        trial[k]=fresh(c);trialgap[k]=fresh(c);trialslope[k]=fresh(c);
    }
    if(sgn(lo[0])<=0)fail(c,1,"Lambda search bounds must be positive");
    dual fd=parse(c,step_text),tol=parse(c,tol_text);
    if(sgn(fd)<=0||sgn(tol)<=0||!budget||budget>64)fail(c,1,"invalid pair difference step, tolerance, or iteration budget");
    if(cmp(tol,mul(num(1024),c->eps))<0)fail(c,3,"pair tolerance below working precision floor");
    for(int i=0;i<SPONG_PAIR_FIELDS;i++)out[i]=fresh(c);
    size_t mark=c->top;char reason[128]={0};
    int status=pair_evaluate(w,x[0],x[1],gap,slope,reason);if(status)fail(c,status,reason);
    for(unsigned iteration=0;iteration<budget;iteration++) {
        r->iterations=iteration+1;c->top=mark;
        dual margin=maximum(c->zero,cmp(sub(x[1],lo[1]),sub(hi[1],x[1]))<0?sub(x[1],lo[1]):sub(hi[1],x[1]));
        dual h=val(cmp(fd,divv(margin,num(4)))<0?fd:divv(margin,num(4)));
        dual plusgap[2],minusgap[2],dummy[2],coarse[2],fine[2];
        for(int k=0;k<2;k++){plusgap[k]=fresh(c);minusgap[k]=fresh(c);dummy[k]=fresh(c);coarse[k]=fresh(c);fine[k]=fresh(c);}
        size_t derivative_mark=c->top;int resolved=0;
        for(int refinement=0;refinement<16;refinement++) {
            c->top=derivative_mark;r->derivative_refinements++;
            if(cmp(h,mul(mul(num(1024),c->eps),maximum(c->one,ab(x[1]))))<0)fail(c,3,"eta difference step below arithmetic resolution");
            for(int pass=0;pass<2;pass++) {
                dual hh=pass?divv(h,num(2)):h;
                status=pair_evaluate(w,x[0],add(x[1],hh),plusgap,dummy,reason);
                if(!status)status=pair_evaluate(w,x[0],sub(x[1],hh),minusgap,dummy,reason);
                if(status)break;
                for(int k=0;k<2;k++)put(pass?fine[k]:coarse[k],divv(sub(plusgap[k],minusgap[k]),mul(num(2),hh)));
            }
            if(!status) {
                resolved=1;
                for(int k=0;k<2;k++) {
                    put(error[k],divv(ab(sub(fine[k],coarse[k])),num(3)));
                    put(eta_slope[k],divv(sub(mul(num(4),fine[k]),coarse[k]),num(3)));
                    dual reference=maximum(ab(eta_slope[k]),ab(divv(mul(slope[k],span[0]),span[1])));
                    if(cmp(error[k],mul(parse(c,"1e-8"),reference))>0)resolved=0;
                }
                if(resolved)break;
            }
            put(h,divv(h,num(2)));
        }
        if(!resolved)fail(c,status?status:SPONG_RHEOSTAT_CONDITIONING,status?reason:"eta derivative refinement did not converge");
        dual M[DIM][DIM],rhs[2];
        for(int k=0;k<2;k++) {
            M[k][0]=mul(slope[k],span[0]);M[k][1]=mul(eta_slope[k],span[1]);
            put(scale[k],maximum(ab(M[k][0]),ab(M[k][1])));
            if(!sgn(scale[k]))fail(c,3,"zero row in two-gap Jacobian");rhs[k]=sub(c->zero,gap[k]);
        }
        dual determinant=divv(sub(mul(M[0][0],M[1][1]),mul(M[0][1],M[1][0])),mul(scale[0],scale[1]));
        dual derivative_error=maximum(divv(mul(error[0],span[1]),scale[0]),divv(mul(error[1],span[1]),scale[1]));
        if(cmp(ab(determinant),add(mul(num(128),c->eps),mul(num(16),derivative_error)))<=0)
            fail(c,3,"two-gap Jacobian rank unresolved against derivative error");
        if(!solve(c,2,M,rhs,corr))fail(c,3,"two-gap correction solve failed backward-error check");
        dual correction_norm=maximum(ab(corr[0]),ab(corr[1]));dual residual_norm=pair_norm(c,gap,scale);
        if(cmp(correction_norm,tol)<=0&&cmp(residual_norm,tol)<=0) {
            put(out[0],x[0]);put(out[1],x[1]);put(out[2],gap[0]);put(out[3],gap[1]);
            put(out[4],corr[0]);put(out[5],corr[1]);put(out[6],determinant);
            put(out[7],error[0]);put(out[8],error[1]);put(out[9],correction_norm);put(out[10],residual_norm);
            for(int k=0;k<SPONG_PAIR_FIELDS;k++)gmp_snprintf(r->values[k],SPONG_RHEOSTAT_TEXT,"%.*Fg",(int)(c->policy.precision_bits*.30103)+3,out[k].v);
            r->max_backward_error=fmax(r->max_backward_error,c->backward);return;
        }
        dual alpha=num(1);size_t trial_mark=c->top;int accepted=0;
        for(int attempt=0;attempt<20;attempt++) {
            c->top=trial_mark;
            for(int k=0;k<2;k++)put(trial[k],add(x[k],mul(mul(alpha,corr[k]),span[k])));
            if(cmp(trial[0],lo[0])>0&&cmp(trial[0],hi[0])<0&&cmp(trial[1],lo[1])>0&&cmp(trial[1],hi[1])<0) {
                status=pair_evaluate(w,trial[0],trial[1],trialgap,trialslope,reason);
                if(!status&&cmp(pair_norm(c,trialgap,scale),mul(sub(c->one,mul(parse(c,"0.0001"),alpha)),residual_norm))<0) {
                    for(int k=0;k<2;k++){put(x[k],trial[k]);put(gap[k],trialgap[k]);put(slope[k],trialslope[k]);}accepted=1;break;
                }
            }
            put(alpha,divv(alpha,num(2)));r->backtracks++;
        }
        if(!accepted)fail(c,SPONG_RHEOSTAT_CONDITIONING,"bounded pair line search could not reduce the residual");
    }
    fail(c,SPONG_RHEOSTAT_WORK_LIMIT,"two-parameter iteration budget exhausted");
}
int spong_rheostat_pair_locate(const spong_rheostat_pair_family *f,
    const char *const initial[2],const char *const lower[2],const char *const upper[2],
    const char *difference_step,const char *tolerance,const spong_rheostat_policy *policy,
    unsigned iterations,spong_rheostat_pair_result *r) {
    if(!r)return SPONG_RHEOSTAT_INVALID;memset(r,0,sizeof(*r));
    if(!f||!policy||!initial||!lower||!upper||!f->na||!f->nb||f->na>DEG||f->nb>DEG||
       policy->precision_bits<128||policy->precision_bits>512){r->status=1;snprintf(r->reason,128,"invalid pair family or precision");return 1;}
    for(int i=0;i<3;i++)if(!f->A[i]){r->status=1;return 1;}
    for(int i=0;i<2;i++)if(!f->B[i]){r->status=1;return 1;}
    pair_workspace *w=calloc(1,sizeof(*w));spong_rheostat *c=calloc(1,sizeof(*c));
    if(!w||!c){free(w);free(c);r->status=5;return 5;}
    w->math=c;w->input=f;w->result=r;c->policy=*policy;c->bits=policy->precision_bits+64;
    c->pool=calloc(CAP,sizeof(mpf_t));
    for(int i=0;i<3;i++)for(int j=0;j<DEG;j++)mpq_init(w->A[i][j]);
    for(int i=0;i<2;i++)for(int j=0;j<DEG;j++)mpq_init(w->B[i][j]);mpq_init(w->C);
    if(setjmp(c->abort)){r->status=c->status;snprintf(r->reason,128,"%s",c->reason);goto done;}
    if(!c->pool)fail(c,5,"pair workspace allocation failed");
    c->zero=num(0);c->one=num(1);c->eps=num(1);mpf_div_2exp(c->eps.v,c->eps.v,policy->precision_bits);
    for(int i=0;i<3;i++)for(size_t j=0;j<f->na;j++)if(!pair_qread(w->A[i][j],f->A[i][j]))fail(c,1,"invalid exact family A coefficient");
    for(int i=0;i<2;i++)for(size_t j=0;j<f->nb;j++)if(!pair_qread(w->B[i][j],f->B[i][j]))fail(c,1,"invalid exact family B coefficient");
    if(!pair_qread(w->C,f->C))fail(c,1,"invalid exact family loss constant");
    for(int k=0;k<2;k++)for(int d=0;d<2;d++){w->anchor[k][d]=fresh(c);w->normal[k][d]=fresh(c);}
    pair_run(w,initial,lower,upper,difference_step,tolerance,iterations);
done:
    for(int i=0;i<3;i++)for(int j=0;j<DEG;j++)mpq_clear(w->A[i][j]);
    for(int i=0;i<2;i++)for(int j=0;j<DEG;j++)mpq_clear(w->B[i][j]);mpq_clear(w->C);
    for(size_t i=0;i<c->initialized;i++)mpf_clear(c->pool[i]);free(c->pool);free(c);free(w);return r->status;
}
