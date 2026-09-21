#include "spong/spong_section.h"
#include <assert.h>
#include <math.h>
int main(void) {
    const double A[]={1,0,1},Ap[]={0,2},App[]={2},B[]={1},Z[]={0};
    spong_field f={A,Ap,App,B,Z,Z,Z,Z,3,2,1,1,1,1,1,1,1};
    double z[]={2,0},v[202];spong_section_result r;
    spong_section_policy p=spong_section_default_policy(.17,100);
    assert(spong_section_trace(&f,z,-1,.25,&p,v,101,&r)==0);
    assert(fabs(v[2*(r.count-1)]-1.5)<2e-12);
    assert(spong_section_trace(&f,z,1,.25,&p,v,101,&r)==SPONG_SECTION_INVALID);
    assert(spong_section_trace(&f,z,-1,.25,&p,v,1,&r)==SPONG_SECTION_INVALID);
    assert(spong_section_trace(NULL,z,-1,.25,&p,v,101,&r)==SPONG_SECTION_INVALID);
    assert(spong_section_trace(&f,z,-1,.25,&p,v,101,NULL)==SPONG_SECTION_INVALID);
    p.max_steps=1;p.step=.01;
    assert(spong_section_trace(&f,z,-1,.25,&p,v,101,&r)==SPONG_SECTION_WORK_LIMIT);
    p=spong_section_default_policy(.17,100);p.max_refinements=1;
    assert(spong_section_trace(&f,z,-1,.25,&p,v,101,&r)==SPONG_SECTION_CONDITIONING);
    p=spong_section_default_policy(.17,100);f.C=1001;
    assert(spong_section_trace(&f,z,-1,1000.25,&p,v,101,&r)==SPONG_SECTION_CONDITIONING);
    return 0;
}
