#include "spong/spong_precision_bridge.h"
static PyObject *native_precision_bridge(PyObject *self,PyObject *args) {
    (void)self;PyObject *input,*seq;unsigned rows,cols,digits,steps;const char *rates[2];
    if(!PyArg_ParseTuple(args,"OIIssII",&input,&rows,&cols,&rates[0],&rates[1],&digits,&steps))return NULL;
    if(rows<1||rows>64||cols<1||cols>64){PyErr_SetString(PyExc_ValueError,"precision_bridge_shape");return NULL;}
    seq=PySequence_Fast(input,"gradient strings required");if(!seq)return NULL;
    size_t n=2*(size_t)rows*cols;
    if(PySequence_Fast_GET_SIZE(seq)!=(Py_ssize_t)n){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"precision_bridge_shape");return NULL;}
    const char **strings=calloc(n,sizeof(char*));if(!strings){Py_DECREF(seq);return PyErr_NoMemory();}
    for(size_t i=0;i<n;i++){strings[i]=PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seq,(Py_ssize_t)i));if(!strings[i]){free(strings);Py_DECREF(seq);return NULL;}}
    spong_precision_bridge_request request={rows,cols,digits,steps,strings,rates};spong_precision_bridge_result r;int rc;
    Py_BEGIN_ALLOW_THREADS
    rc=spong_precision_bridge(&request,&r);
    Py_END_ALLOW_THREADS
    free(strings);Py_DECREF(seq);
    if(rc){PyErr_SetString(rc==-2?PyExc_MemoryError:PyExc_ValueError,rc==-3?"precision_bridge_bit_budget":"precision_bridge_request");return NULL;}
    PyObject *polys=PyList_New((Py_ssize_t)r.steps),*result=NULL;
    if(!polys)goto done;
    for(size_t i=0;i<r.steps;i++) {
        char **p=r.coefficients+8*i;
        PyObject *poly=Py_BuildValue("((ssss)(ssss))",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
        if(!poly)goto done;PyList_SET_ITEM(polys,(Py_ssize_t)i,poly);
    }
    result=Py_BuildValue("{s:O,s:k,s:I,s:I,s:I,s:i,s:s}","polynomials",polys,"working_bits",r.working_bits,
        "rejected",r.rejected,"solve_failures",r.solve_failures,"precision_refusals",r.precision_refusals,
        "termination",r.termination,"max_solve_condition",r.max_solve_condition);
 done:Py_XDECREF(polys);spong_precision_bridge_destroy(&r);return result;
}
