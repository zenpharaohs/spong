/* Resolution regressions for the opt-in checked-arrival bridge. */
#include "spong/spong_arrival.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;
static void check(int ok,const char *message) {
    if(!ok) {fprintf(stderr,"FAIL: %s\n",message); ++failures;}
}
static void trial(double fast,double b0,int expect_growth) {
    const double zero[]={0},slow_row[]={1},fast_row[]={0,fast};
    const double *a[]={zero,slow_row},*b[]={fast_row};
    const size_t na[]={1,1},nb[]={2};
    spong_jet jet={{a,b},{na,nb},{2,1}};
    spong_arrival_request req={0};
    req.a0=1;req.b0=b0;req.slow=1;req.fast=fast;
    req.cap_r=expect_growth ? 1e-3 : 1e-8;req.max_steps=1;req.turn_reject=0.9;req.primary_order=8;
    double old_points[16],points[16];
    spong_arrival_result old,result;
    spong_trace trace={0};spong_arrival_bridge_diagnostics diag;
    spong_centered_arrival(&jet,&req,old_points,8,&old);
    int term=spong_centered_arrival_bridge_recorded(&jet,&req,points,8,&result,&trace,&diag);
    check(old.accepted==0,"legacy reproduces binary64 loss stall");
    check(term==SPONG_ARR_BUDGET && result.accepted==1,"bridge resolves one descent step");
    check(points[0]==req.a0 && points[1]==req.b0,"incoming anchor retained exactly");
    check(result.n_points>=2 && !trace.failed,"retained movement has trace evidence");
    if(expect_growth) {
        check(diag.time_step_grows>0 && diag.zero_progress>0,"zero movement triggers bounded growth");
        check(result.a_end<req.a0,"grown step moves down slow eigendirection");
        check(fabs(result.a_end-exp(-0.25/fast*pow(4,(double)diag.time_step_grows)))<2e-15,
              "slow step agrees with analytic linear flow");
    } else {
        check(diag.exact_loss_decreases>0,"exact loss resolves masked fast descent");
        check(result.b_end<req.b0 && result.b_end>0,"fast coordinate descends without crossing");
        check(fabs(result.b_end/req.b0-exp(-0.25))<1e-10,"fast step agrees with analytic linear flow");
    }
    free(trace.records);
    if(expect_growth) {
        /* The ordinary tiny finish radius gives a Richardson tolerance
         * below the observed full/two-half roundoff. Growth must refuse,
         * rather than silently weakening that independent admission test. */
        trace=(spong_trace){0}; req.cap_r=1e-8;
        term=spong_centered_arrival_bridge_recorded(&jet,&req,points,8,&result,&trace,&diag);
        check(term==SPONG_ARR_STEP_FAILURE && result.accepted==0
              && diag.richardson_rejections>0,"growth preserves Richardson threshold");
        free(trace.records);
    }
    req.clock=1;
    check(spong_centered_arrival_bridge_recorded(&jet,&req,points,8,&result,&trace,&diag)==-1,
          "bridge rejects unsupported clock policy");
}
int main(void) {
    trial(1e16,1e-17,0);
    trial(1e18,0,1);
    return failures?1:0;
}
