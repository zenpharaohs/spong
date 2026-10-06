/* Thin ownership/serialization adapter; all numerical construction is in C. */
#include "spong/spong_analytic_arrival.h"
static void arrival_capsule_destroy(PyObject *capsule) {
    spong_arrival_chart *c=PyCapsule_GetPointer(capsule,"spong.arrival_chart");
    if(c) spong_arrival_chart_destroy(c); else PyErr_Clear();
}
static PyObject *native_arrival_chart(PyObject *self,PyObject *args) {
    (void)self;
    PyObject *input,*seq=NULL;int order,digits;
    if(!PyArg_ParseTuple(args,"Oii",&input,&order,&digits)) return NULL;
    if(order<2 || order>12 || digits<50 || digits>100) {PyErr_SetString(PyExc_ValueError,"arrival_order_or_precision");return NULL;}
    seq=PySequence_Fast(input,"arrival chart requires strings");if(!seq) return NULL;
    size_t n=2*(size_t)(order+1)*(order+1),count=n+10;
    if(PySequence_Fast_GET_SIZE(seq)!=(Py_ssize_t)count) {Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"arrival_chart_shape");return NULL;}
    const char **strings=calloc(count,sizeof(char*));if(!strings) {Py_DECREF(seq);return PyErr_NoMemory();}
    for(size_t i=0;i<count;i++) {
        strings[i]=PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seq,(Py_ssize_t)i));
        if(!strings[i]) {free(strings);Py_DECREF(seq);return NULL;}
    }
    spong_arrival_chart_request r={order,digits,strings,strings+n,strings+n+4,strings+n+8};
    spong_arrival_chart *chart=NULL;int rc;
    Py_BEGIN_ALLOW_THREADS
    rc=spong_arrival_chart_create(&r,&chart);
    Py_END_ALLOW_THREADS
    free(strings);Py_DECREF(seq);
    if(rc) {PyErr_SetString(rc==SPONG_ARRIVAL_MEMORY?PyExc_MemoryError:PyExc_ValueError,spong_analytic_arrival_error(rc));return NULL;}
    PyObject *capsule=PyCapsule_New(chart,"spong.arrival_chart",arrival_capsule_destroy);
    if(!capsule) spong_arrival_chart_destroy(chart);
    return capsule;
}
static PyObject *native_arrival_match(PyObject *self,PyObject *args) {
    (void)self;
    PyObject *capsule;const char *center[2],*exact[2]={NULL,NULL};double start[2],target[2];
    if(!PyArg_ParseTuple(args,"Ossdddd|ss",&capsule,&center[0],&center[1],&start[0],&start[1],&target[0],&target[1],&exact[0],&exact[1])) return NULL;
    if((exact[0]!=NULL)!=(exact[1]!=NULL)){PyErr_SetString(PyExc_ValueError,"exact_entry_requires_two_coordinates");return NULL;}
    spong_arrival_chart *chart=PyCapsule_GetPointer(capsule,"spong.arrival_chart");if(!chart) return NULL;
    spong_analytic_arrival r;spong_arrival_matching mi;int rc;
    Py_BEGIN_ALLOW_THREADS
    rc=spong_arrival_chart_match_exact_inspect(chart,center,start,target,exact[0]?exact:NULL,&r,&mi);
    Py_END_ALLOW_THREADS
    PyObject *matching=Py_BuildValue("{s:s,s:I,s:I,s:I,s:O,s:s,s:s,s:s}",
        "policy","conditioned-continuation-v1","iterations",mi.iterations,
        "backtracks",mi.backtracks,"stages",mi.stages,"continuation",mi.continuation?Py_True:Py_False,
        "max_jacobian_condition",mi.max_condition?mi.max_condition:"0",
        "max_solve_forward_error_indicator",mi.max_forward_error?mi.max_forward_error:"0",
        "last_residual",mi.residual?mi.residual:"0");
    spong_arrival_matching_destroy(&mi);
    if(!matching) {spong_analytic_arrival_destroy(&r);return NULL;}
    if(rc) {
        PyObject *kind=rc==SPONG_ARRIVAL_MEMORY?PyExc_MemoryError:PyExc_ValueError;
        PyObject *exc=PyObject_CallFunction(kind,"s",spong_analytic_arrival_error(rc));
        if(exc) {if(PyObject_SetAttrString(exc,"matching",matching)==0) PyErr_SetObject(kind,exc);Py_DECREF(exc);}
        Py_DECREF(matching);return NULL;
    }
    PyObject *points=PyList_New((Py_ssize_t)r.point_count),*terms=PyList_New((Py_ssize_t)r.term_count),
             *params=PyList_New((Py_ssize_t)r.point_count),*result=NULL,*conditioning=NULL;
    if(!points||!terms||!params) goto done;
    for(size_t i=0;i<r.point_count;i++) {
        PyObject *p=Py_BuildValue("(dd)",r.points[2*i],r.points[2*i+1]);
        PyObject *s=i+1==r.point_count?Py_NewRef(Py_None):PyUnicode_FromString(r.parameters[i]);
        if(!p||!s) {Py_XDECREF(p);Py_XDECREF(s);goto done;}
        PyList_SET_ITEM(points,(Py_ssize_t)i,p);PyList_SET_ITEM(params,(Py_ssize_t)i,s);
    }
    for(size_t i=0;i<r.term_count;i++) {
        PyObject *t=Py_BuildValue("(sss)",r.terms[3*i],r.terms[3*i+1],r.terms[3*i+2]);
        if(!t) goto done;
        PyList_SET_ITEM(terms,(Py_ssize_t)i,t);
    }
    spong_arrival_conditioning info;
    if(spong_arrival_chart_inspect(chart,&info)) {PyErr_NoMemory();goto done;}
    conditioning=Py_BuildValue("{s:s,s:k,s:s,s:s,s:s,s:s}",
        "policy","componentwise-v1","working_bits",info.working_bits,
        "min_relative_divisor",info.relative_divisor,"max_matrix_condition",info.matrix_condition,
        "max_backward_error",info.backward_error,"max_forward_error_indicator",info.forward_error);
    spong_arrival_conditioning_destroy(&info);
    if(!conditioning) goto done;
    result=Py_BuildValue("{s:O,s:O,s:O,s:(ss),s:s,s:s,s:s,s:O,s:O}",
        "points",points,"terms",terms,"parameters",params,
        "amplitudes",r.amplitudes[0],r.amplitudes[1],"spectral_ratio",r.spectral_ratio,
        "divisor_margin",r.divisor_margin,"mode",r.slow_frobenius?"slow_frobenius":"full_poincare",
        "conditioning",conditioning,"matching",matching);
 done:
    Py_DECREF(matching);Py_XDECREF(conditioning);Py_XDECREF(points);Py_XDECREF(terms);Py_XDECREF(params);spong_analytic_arrival_destroy(&r);return result;
}
