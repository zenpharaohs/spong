#include "spong/spong_section.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
static int refuse(spong_section_result *r,int status,const char *reason) {
    r->status=status;snprintf(r->reason,sizeof(r->reason),"%s",reason);return status;
}
spong_section_policy spong_section_default_policy(double step,unsigned max_steps) {
    spong_section_policy p={step,1e-13,1e-12,1e4,max_steps,12,64};return p;
}
static double abs_poly(const double *p,size_t n,double b) {
    double x=0;for(size_t i=n;i>0;i--)x=x*fabs(b)+fabs(p[i-1]);return x;
}
static int value(const spong_field *f,const double z[2],double *loss,double *floor) {
    if(!isfinite(z[0])||!isfinite(z[1]))return 0;
    *loss=spong_field_loss(f,z[0],z[1]);
    double scale=fabs(f->C)+fabs(z[0])*(2*abs_poly(f->B,f->nB,z[1])+
                      fabs(z[0])*abs_poly(f->A,f->nA,z[1]));
    *floor=8*DBL_EPSILON*fmax(1,scale);
    return isfinite(*loss)&&isfinite(*floor);
}
/* Independent full/two-half-step comparison; the more accurate halves own
 * the returned point. A failed stage solve is never replaced by a chord. */
static int advance(const spong_field *f,const double z[2],double h,
                   const spong_section_policy *p,double out[2],double *error) {
    double whole[2],half[2];
    if(!spong_normalized_step(f,z,h,8,whole)||
       !spong_normalized_step(f,z,h/2,8,half)||
       !spong_normalized_step(f,half,h/2,8,out))return 0;
    *error=hypot(out[0]-whole[0],out[1]-whole[1])/255.;
    return isfinite(*error)&&isfinite(out[0])&&isfinite(out[1])&&
        *error<=p->atol+p->rtol*fmax(hypot(z[0],z[1]),hypot(out[0],out[1]));
}
int spong_section_trace(const spong_field *f,const double start[2],int direction,
    double level,const spong_section_policy *p,double *vertices,size_t capacity,
    spong_section_result *r) {
    if(!r)return SPONG_SECTION_INVALID;memset(r,0,sizeof(*r));
    if(!f||!start||!p||!vertices||!f->A||!f->B||!f->nA||!f->nB||
       (f->nAp&&!f->Ap)||(f->nApp&&!f->App)||(f->nBp&&!f->Bp)||
       (f->nBpp&&!f->Bpp)||(f->nN&&!f->N)||(f->nNp&&!f->Np)||
       (direction!=1&&direction!=-1)||!isfinite(level)||!isfinite(p->step)||p->step<=0||
       !isfinite(p->atol)||p->atol<=0||!isfinite(p->rtol)||p->rtol<0||
       !isfinite(p->max_radius)||p->max_radius<=0||!p->max_steps||
       p->max_steps>4000000||p->max_halvings>32||!p->max_refinements||
       p->max_refinements>256||capacity<(size_t)p->max_steps+1)
        return refuse(r,1,"invalid field, policy, or vertex capacity");
    double z[2]={start[0],start[1]},L,floor;
    if(!value(f,z,&L,&floor))return refuse(r,3,"nonfinite initial loss");
    vertices[0]=z[0];vertices[1]=z[1];r->count=1;
    if(direction*(level-L)<-floor)return refuse(r,1,"section behind launch direction");
    if(hypot(z[0],z[1])>p->max_radius)return refuse(r,5,"launch outside radius budget");
    double unit[2];if(!spong_normalized_fj((void*)f,z,unit,NULL))
        return refuse(r,3,"initial gradient direction unresolved");
    if(L==level){r->loss_roundoff_scale=floor;return 0;}
    for(unsigned n=0;n<p->max_steps;n++) {
        double h=direction*p->step,next[2],Ln=0,fn=0,error=0;int accepted=0;
        for(unsigned k=0;k<=p->max_halvings;k++) {
            if(advance(f,z,h,p,next,&error)&&value(f,next,&Ln,&fn)&&
               direction*(Ln-L)>0){accepted=1;break;}
            r->rejected_steps++;h/=2;
        }
        if(!accepted)return refuse(r,4,"stage solve, monotonicity, or step error refused");
        r->max_step_error=fmax(r->max_step_error,error);
        if(direction*(Ln-level)>=0) {
            double lo=0,hi=1,flo=L-level,fhi=Ln-level,best[2]={next[0],next[1]};
            double residual=fhi,bestfloor=fn;
            for(unsigned j=0;j<p->max_refinements;j++) {
                r->refinements++;
                /* Safeguarded secant. Bisection guarantees bracket progress. */
                double t=lo-flo*(hi-lo)/(fhi-flo);
                if(!isfinite(t)||t<=lo+.05*(hi-lo)||t>=hi-.05*(hi-lo))t=(lo+hi)/2;
                double trial[2],Lt,ft,err;
                if(!advance(f,z,h*t,p,trial,&err)||!value(f,trial,&Lt,&ft))
                    return refuse(r,4,"fractional GL8 event step refused");
                double v=Lt-level;
                if(fabs(v)<fabs(residual)){residual=v;bestfloor=ft;best[0]=trial[0];best[1]=trial[1];}
                double coordinate_tol=p->atol+p->rtol*hypot(trial[0],trial[1]);
                if(v==0 || (fabs(residual)<=bestfloor && fabs(h)*(hi-lo)<=coordinate_tol)) {
                    double grad[2];spong_field_gradient(f,best[0],best[1],grad);
                    double speed=hypot(grad[0],grad[1]);
                    if(!isfinite(speed)||speed*coordinate_tol<bestfloor)
                        return refuse(r,3,"loss event ill-conditioned at coordinate tolerance");
                    vertices[2*r->count]=best[0];vertices[2*r->count+1]=best[1];r->count++;
                    r->loss_residual=residual;r->loss_roundoff_scale=bestfloor;return 0;
                }
                if(direction*v<0){lo=t;flo=v;}else{hi=t;fhi=v;}
                if(hi-lo<=8*DBL_EPSILON)break;
            }
            return refuse(r,3,"loss-section event did not resolve within its budget");
        }
        if(hypot(next[0],next[1])>p->max_radius)return refuse(r,5,"section trace left radius budget");
        if(hypot(next[0]-z[0],next[1]-z[1])<=8*DBL_EPSILON*fmax(1,hypot(z[0],z[1])))
            return refuse(r,3,"section trace stagnated");
        z[0]=next[0];z[1]=next[1];L=Ln;floor=fn;
        vertices[2*r->count]=z[0];vertices[2*r->count+1]=z[1];r->count++;
    }
    return refuse(r,2,"section trace step budget exhausted");
}
