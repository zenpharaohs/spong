/* Marshaling only: no numerical thresholds or terminal decisions. */
#include "spong/spong_rheostat.h"
static void rheostat_capsule_destroy(PyObject *capsule) {
    spong_rheostat *c=PyCapsule_GetPointer(capsule,"spong.rheostat");
    if(c)spong_rheostat_destroy(c);else PyErr_Clear();
}
static PyObject *rheostat_result_dict(const spong_rheostat_result *r) {
    PyObject *v=PyTuple_New(SPONG_RHEOSTAT_FIELDS);if(!v)return NULL;
    for(int i=0;i<SPONG_RHEOSTAT_FIELDS;i++) {
        PyObject *s=PyUnicode_FromString(r->values[i]);if(!s){Py_DECREF(v);return NULL;}PyTuple_SET_ITEM(v,i,s);
    }
    return Py_BuildValue("{s:i,s:s,s:N,s:K,s:K,s:K,s:d}","status",r->status,"reason",r->reason,
        "values",v,"stage_solves",(unsigned long long)r->stage_solves,
        "step_halvings",(unsigned long long)r->step_halvings,
        "evaluations",(unsigned long long)r->evaluations,"max_backward_error",r->max_backward_error);
}
static PyObject *native_rheostat_create(PyObject *self,PyObject *args) {
    (void)self;PyObject *A,*B,*bounds;const char *C,*radius,*fraction;
    int s0,s1,d0,d1;unsigned bits,stages,steps,order,newton,halvings;
    if(!PyArg_ParseTuple(args,"OOsOiiiiIIIIIIss",&A,&B,&C,&bounds,&s0,&s1,&d0,&d1,
        &bits,&stages,&steps,&order,&newton,&halvings,&radius,&fraction))return NULL;
    PyObject *seqs[3]={NULL,NULL,NULL};const char **texts[3]={NULL,NULL,NULL};PyObject *inputs[3]={A,B,bounds};
    PyObject *answer=NULL;Py_ssize_t counts[3];
    for(int k=0;k<3;k++) {
        seqs[k]=PySequence_Fast(inputs[k],"expected string sequence");if(!seqs[k])goto done;
        counts[k]=PySequence_Fast_GET_SIZE(seqs[k]);
        if(counts[k]<1||counts[k]>(k==2?4:65)||(k==2&&counts[k]!=4)){PyErr_SetString(PyExc_ValueError,"invalid input count");goto done;}
        texts[k]=PyMem_Calloc(counts[k],sizeof(char*));if(!texts[k]){PyErr_NoMemory();goto done;}
        for(Py_ssize_t j=0;j<counts[k];j++){texts[k][j]=PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seqs[k],j));if(!texts[k][j])goto done;}
    }
    int32_t sources[2]={s0,s1},directions[2]={d0,d1};
    spong_rheostat_policy policy={bits,stages,steps,order,newton,halvings};spong_rheostat_result r;spong_rheostat *c=NULL;
    int status=spong_rheostat_create(texts[0],counts[0],texts[1],counts[1],C,texts[2],sources,directions,radius,fraction,&policy,&c,&r);
    if(status){PyErr_Format(PyExc_ValueError,"native rheostat initialization (%d): %s",status,r.reason);goto done;}
    answer=PyCapsule_New(c,"spong.rheostat",rheostat_capsule_destroy);if(!answer)spong_rheostat_destroy(c);
done:
    for(int k=0;k<3;k++){Py_XDECREF(seqs[k]);PyMem_Free(texts[k]);}return answer;
}
static PyObject *native_rheostat_launch(PyObject *self,PyObject *args) {
    (void)self;PyObject *capsule;const char *lam;
    if(!PyArg_ParseTuple(args,"Os",&capsule,&lam))return NULL;
    spong_rheostat *c=PyCapsule_GetPointer(capsule,"spong.rheostat");if(!c)return NULL;
    spong_rheostat_result r;spong_rheostat_launch(c,lam,&r);return rheostat_result_dict(&r);
}
static PyObject *native_rheostat_evaluate(PyObject *self,PyObject *args) {
    (void)self;PyObject *capsule;const char *lam;
    if(!PyArg_ParseTuple(args,"Os",&capsule,&lam))return NULL;
    spong_rheostat *c=PyCapsule_GetPointer(capsule,"spong.rheostat");if(!c)return NULL;
    spong_rheostat_result r;spong_rheostat_evaluate(c,lam,&r);return rheostat_result_dict(&r);
}
static PyObject *native_rheostat_locate(PyObject *self,PyObject *args) {
    (void)self;PyObject *capsule;const char *lo,*hi,*tol;unsigned iterations;
    if(!PyArg_ParseTuple(args,"OsssI",&capsule,&lo,&hi,&tol,&iterations))return NULL;
    spong_rheostat *c=PyCapsule_GetPointer(capsule,"spong.rheostat");if(!c)return NULL;
    spong_rheostat_result r;spong_rheostat_locate(c,lo,hi,tol,iterations,&r);return rheostat_result_dict(&r);
}

static PyObject *native_rheostat_locate_seeded(PyObject *self,PyObject *args) {
    (void)self;PyObject *capsule;const char *lo,*hi,*floor,*ceiling,*tol;unsigned iterations;
    if(!PyArg_ParseTuple(args,"OsssssI",&capsule,&lo,&hi,&floor,&ceiling,&tol,&iterations))return NULL;
    spong_rheostat *c=PyCapsule_GetPointer(capsule,"spong.rheostat");if(!c)return NULL;
    spong_rheostat_result r;spong_rheostat_locate_seeded(c,lo,hi,floor,ceiling,tol,iterations,&r);return rheostat_result_dict(&r);
}

#include "spong/spong_rheostat_pair.h"
static PyObject *native_rheostat_pair_locate(PyObject *self,PyObject *args) {
    (void)self;PyObject *arrays,*bounds,*sources,*directions,*initial,*lower,*upper,*policy_object;
    const char *C,*radius,*fraction,*difference,*tolerance;unsigned iterations;
    if(!PyArg_ParseTuple(args,"OsOOOOOOssssOI",&arrays,&C,&bounds,&sources,&directions,
                         &initial,&lower,&upper,&radius,&fraction,&difference,&tolerance,&policy_object,&iterations))return NULL;
    PyObject *owned[13]={0};const char **texts[9]={0};PyObject *answer=NULL;
    owned[0]=PySequence_Fast(arrays,"need five family coefficient arrays");
    if(!owned[0])goto done;
    if(PySequence_Fast_GET_SIZE(owned[0])!=5){PyErr_SetString(PyExc_ValueError,"need A0,A1,A2,B0,B1");goto done;}
    PyObject *inputs[9];for(int i=0;i<5;i++)inputs[i]=PySequence_Fast_GET_ITEM(owned[0],i);
    inputs[5]=bounds;inputs[6]=initial;inputs[7]=lower;inputs[8]=upper;Py_ssize_t counts[9];
    for(int i=0;i<9;i++) {
        owned[i+1]=PySequence_Fast(inputs[i],"expected numeric string sequence");if(!owned[i+1])goto done;
        counts[i]=PySequence_Fast_GET_SIZE(owned[i+1]);
        if(counts[i]<1||counts[i]>(i<5?65:i==5?8:2)|| (i==5&&counts[i]!=8)||(i>5&&counts[i]!=2)){
            PyErr_SetString(PyExc_ValueError,"invalid pair input count");goto done;}
        texts[i]=PyMem_Calloc(counts[i],sizeof(char*));if(!texts[i]){PyErr_NoMemory();goto done;}
        for(Py_ssize_t j=0;j<counts[i];j++){texts[i][j]=PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(owned[i+1],j));if(!texts[i][j])goto done;}
    }
    if(counts[0]!=counts[1]||counts[0]!=counts[2]||counts[3]!=counts[4]){PyErr_SetString(PyExc_ValueError,"family arrays need matching lengths");goto done;}
    spong_rheostat_pair_family f={0};for(int i=0;i<3;i++)f.A[i]=texts[i];f.na=counts[0];
    for(int i=0;i<2;i++)f.B[i]=texts[i+3];f.nb=counts[3];f.C=C;
    for(int i=0;i<8;i++)f.bounds[i]=texts[5][i];f.launch_radius=radius;f.level_fraction=fraction;
    PyObject *integer_inputs[3]={sources,directions,policy_object};unsigned policy_values[6];
    for(int k=0;k<3;k++) {
        owned[10+k]=PySequence_Fast(integer_inputs[k],"expected integer tuple");if(!owned[10+k])goto done;
        int n=k==2?6:4;if(PySequence_Fast_GET_SIZE(owned[10+k])!=n){PyErr_SetString(PyExc_ValueError,"invalid policy or saddle identifiers");goto done;}
        for(int j=0;j<n;j++){
            long v=PyLong_AsLong(PySequence_Fast_GET_ITEM(owned[10+k],j));if(PyErr_Occurred())goto done;
            if(k==2){if(v<0||v>65536){PyErr_SetString(PyExc_ValueError,"invalid policy integer");goto done;}policy_values[j]=(unsigned)v;}
            else{if(v< -1||v>1){PyErr_SetString(PyExc_ValueError,"invalid saddle identifier");goto done;}if(k)f.directions[j]=(int32_t)v;else f.sources[j]=(int32_t)v;}
        }
    }
    spong_rheostat_policy p={policy_values[0],policy_values[1],policy_values[2],policy_values[3],policy_values[4],policy_values[5]};
    spong_rheostat_pair_result r;
    spong_rheostat_pair_locate(&f,texts[6],texts[7],texts[8],difference,tolerance,&p,iterations,&r);
    PyObject *values=PyTuple_New(SPONG_PAIR_FIELDS);if(!values)goto done;
    for(int i=0;i<SPONG_PAIR_FIELDS;i++){PyObject *v=PyUnicode_FromString(r.values[i]);if(!v){Py_DECREF(values);goto done;}PyTuple_SET_ITEM(values,i,v);}
    answer=Py_BuildValue("{s:i,s:s,s:N,s:I,s:I,s:I,s:I,s:d}","status",r.status,"reason",r.reason,"values",values,
        "iterations",r.iterations,"evaluations",r.evaluations,"backtracks",r.backtracks,
        "derivative_refinements",r.derivative_refinements,"max_backward_error",r.max_backward_error);
done:
    for(int i=0;i<13;i++)Py_XDECREF(owned[i]);for(int i=0;i<9;i++)PyMem_Free(texts[i]);return answer;
}
