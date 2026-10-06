#ifndef SPONG_TRACE_H
#define SPONG_TRACE_H
/* Optional observational event stream. No solver decisions consume it.
 * Records are 24 binary64 values (schema 1); see docs/native_trace.md.
 * Caller owns storage and must free(trace.records). Allocation failure is
 * sticky and must be reported, never interpreted as missing/passing evidence.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef struct { double v[24]; } spong_dense;
typedef struct {
    spong_dense *records;
    size_t count, capacity;
    int failed;
} spong_trace;
static inline size_t spong_trace_push(spong_trace *t, const spong_dense *r) {
    if (!t || t->failed) return SIZE_MAX;
    if (t->count == t->capacity) {
        size_t n = t->capacity ? t->capacity*2 : 256;
        if (n < t->capacity || n > SIZE_MAX/sizeof(*t->records)) {
            t->failed = 1; return SIZE_MAX;
        }
        void *p = realloc(t->records, n*sizeof(*t->records));
        if (!p) { t->failed = 1; return SIZE_MAX; }
        t->records = p; t->capacity = n;
    }
    t->records[t->count] = *r;
    return t->count++;
}
static inline void spong_trace_vertex(spong_trace *t, size_t vertex,
        double a, double b, size_t source, double parameter) {
    spong_dense r = {{0}};
    r.v[0] = 3; r.v[4] = a; r.v[5] = b;
    r.v[20] = (double)vertex; r.v[22] = parameter;
    r.v[23] = source == SIZE_MAX ? -1.0 : (double)source;
    spong_trace_push(t, &r);
}
static inline void spong_trace_trim(spong_trace *t, size_t count) {
    spong_dense r = {{0}}; r.v[0] = 4; r.v[20] = (double)count;
    spong_trace_push(t, &r);
}
#endif
