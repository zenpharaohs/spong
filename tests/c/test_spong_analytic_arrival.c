#include "spong/spong_analytic_arrival.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
int main(void) {
    const char *g[162];for(int i=0;i<162;i++) g[i]="0";
    g[9]="1";g[81+1]="5/2";
    const char *h[]={"1","0","0","5/2"};
    const char *frame[]={"1","0","0","1"};
    const char *rates[]={"1","2.5"};
    spong_arrival_chart_request req={8,80,g,h,frame,rates};
    spong_arrival_chart *chart=NULL;
    assert(spong_arrival_chart_create(&req,&chart)==0);
    spong_arrival_conditioning info;
    assert(spong_arrival_chart_inspect(chart,&info)==0);
    assert(info.working_bits>330);
    assert(strtod(info.forward_error,NULL)<1e-20);
    spong_arrival_conditioning_destroy(&info);
    const char *center[]={"0","0"};double start[]={.25,-.125},target[]={0,0};
    spong_analytic_arrival result;
    for(int repeat=0;repeat<3;repeat++) {
        assert(spong_arrival_chart_match(chart,center,start,target,&result)==0);
        assert(result.term_count==2 && result.point_count>250);
        for(size_t i=0;i+1<result.point_count;i++) {
            double s=strtod(result.parameters[i],NULL);
            assert(fabs(result.points[2*i]-.25*exp(-s))<1e-15);
            assert(fabs(result.points[2*i+1]+.125*exp(-2.5*s))<1e-15);
        }
        assert(result.points[2*result.point_count-1]==0);
        spong_analytic_arrival_destroy(&result);
    }
    spong_arrival_matching mi;
    assert(spong_arrival_chart_match_inspect(chart,center,start,target,&result,&mi)==0);
    assert(mi.continuation==0 && mi.max_condition && mi.residual);
    assert(strtod(mi.max_forward_error,NULL)<1e-20);
    spong_arrival_matching_destroy(&mi);spong_analytic_arrival_destroy(&result);
    double invalid[]={NAN,0};
    assert(spong_arrival_chart_match(chart,center,invalid,target,&result)==SPONG_ARRIVAL_INVALID);
    const char *bad_center[]={"1/0","0"};
    assert(spong_arrival_chart_match(chart,bad_center,start,target,&result)==SPONG_ARRIVAL_INVALID);
    assert(spong_arrival_chart_match(chart,center,target,target,&result)==SPONG_ARRIVAL_MATCHING);
    spong_arrival_chart_destroy(chart);
    g[18]="1";
    assert(spong_arrival_chart_create(&req,&chart)==0);
    double far_start[]={100,.1};
    assert(spong_arrival_chart_match_inspect(chart,center,far_start,target,&result,&mi)==0);
    assert(mi.continuation && mi.stages>0 && mi.iterations>24);
    assert(fabs(result.points[0]-100)<1e-15);
    spong_arrival_matching_destroy(&mi);spong_analytic_arrival_destroy(&result);
    far_start[0]=-1;
    assert(spong_arrival_chart_match_inspect(chart,center,far_start,target,&result,&mi)==SPONG_ARRIVAL_MATCHING);
    assert(mi.continuation && mi.iterations<=2048 && !result.points);
    spong_arrival_matching_destroy(&mi);spong_arrival_chart_destroy(chart);g[18]="0";
    rates[1]="1e30";h[3]="1000000000000000000000000000000";g[82]=h[3];
    assert(spong_arrival_chart_create(&req,&chart)==0);
    assert(spong_arrival_chart_inspect(chart,&info)==0);
    assert(strtod(info.relative_divisor,NULL)>.1);
    assert(strtod(info.matrix_condition,NULL)>1e29);
    assert(strtod(info.forward_error,NULL)<1e-20);
    assert(info.working_bits>430);
    spong_arrival_conditioning_destroy(&info);spong_arrival_chart_destroy(chart);
    rates[1]="2";h[3]="2";g[82]="2";
    assert(spong_arrival_chart_create(&req,&chart)==SPONG_ARRIVAL_DIVISOR && !chart);
    req.order=13;
    assert(spong_arrival_chart_create(&req,&chart)==SPONG_ARRIVAL_INVALID);
    spong_analytic_arrival_destroy(NULL);spong_arrival_chart_destroy(NULL);
    puts("native analytic arrival C API passed");return 0;
}
