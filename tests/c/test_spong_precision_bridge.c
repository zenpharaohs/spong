#include "spong/spong_precision_bridge.h"
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static int failures;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);failures++;}}while(0)
int main(void) {
    const char *gradient[]={"2","2","0","0"},*rates[]={"2","2"};
    spong_precision_bridge_request req={2,1,80,4,gradient,rates};spong_precision_bridge_result out;
    CHECK(spong_precision_bridge(&req,&out)==0);CHECK(out.steps==4);CHECK(out.working_bits>256);
    mpf_t sum,tmp,expected,error;mpf_inits(sum,tmp,expected,error,NULL);mpf_set_prec(sum,512);mpf_set_prec(tmp,512);mpf_set_prec(expected,512);mpf_set_prec(error,512);
    mpf_set_ui(sum,1);
    for(int k=0;k<4;k++) {
        char *s=malloc(strlen(out.coefficients[k])+1);strcpy(s,out.coefficients[k]);for(char *c=s;*c;c++)if(*c=='e')*c='@';
        CHECK(mpf_set_str(tmp,s,10)==0);free(s);mpf_add(sum,sum,tmp);
    }
    /* R(-1/4) = 3468/4453 for three-stage Radau IIA. */
    mpf_set_ui(expected,3468);mpf_div_ui(expected,expected,4453);
    mpf_sub(error,sum,expected);mpf_abs(error,error);mpf_set_ui(tmp,10);mpf_pow_ui(tmp,tmp,90);mpf_mul(error,error,tmp);CHECK(mpf_cmp_ui(error,1)<0);
    mpf_clears(sum,tmp,expected,error,NULL);spong_precision_bridge_destroy(&out);spong_precision_bridge_destroy(&out);
    rates[1]="1e5000";CHECK(spong_precision_bridge(&req,&out)==-3);spong_precision_bridge_destroy(&out);
    rates[1]="2";gradient[0]="bad";CHECK(spong_precision_bridge(&req,&out)==-1);spong_precision_bridge_destroy(&out);
    req.rows=0;CHECK(spong_precision_bridge(&req,&out)==-1);
    return failures?1:0;
}
