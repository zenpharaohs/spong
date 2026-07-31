#include "spong/spong_topology.h"

#include <math.h>
#include <stdlib.h>

enum { SPONG_CONTACT_LEAF_SEGMENTS = 32 };

typedef struct {
    size_t lo, hi;             /* inclusive point indices */
    double amin, amax, bmin, bmax;
    size_t left, right;
    int leaf;
} contact_node;

typedef enum { TASK_SELF = 1, TASK_PAIR = 2 } task_kind;

typedef struct {
    task_kind kind;
    size_t first, second;
} contact_task;

struct spong_contact_scan {
    const double *points1;
    const double *points2;
    double tolerance;
    int self_scan;
    contact_node *nodes1, *nodes2;
    size_t nodes_used1, nodes_used2;
    size_t root1, root2;
    contact_task *stack;
    size_t stack_size, stack_capacity;
    int leaf_active;
    size_t ilo, ihi, jlo, jhi, i, j;
};

static const double *point_at(const double *points, size_t i) {
    return points + 2*i;
}

static size_t build_node(contact_node *nodes, size_t *used,
                         const double *points, size_t lo, size_t hi) {
    size_t index = (*used)++;
    contact_node *node = &nodes[index];
    const double *p = point_at(points, lo);
    node->lo = lo;
    node->hi = hi;
    node->amin = node->amax = p[0];
    node->bmin = node->bmax = p[1];
    for (size_t i = lo+1; i <= hi; ++i) {
        p = point_at(points, i);
        if (p[0] < node->amin) node->amin = p[0];
        if (p[0] > node->amax) node->amax = p[0];
        if (p[1] < node->bmin) node->bmin = p[1];
        if (p[1] > node->bmax) node->bmax = p[1];
    }
    node->leaf = hi-lo <= SPONG_CONTACT_LEAF_SEGMENTS;
    node->left = node->right = 0;
    if (!node->leaf) {
        size_t mid = (lo+hi)/2;
        node->left = build_node(nodes, used, points, lo, mid);
        node->right = build_node(nodes, used, points, mid, hi);
    }
    return index;
}

static int boxes_overlap(const contact_node *x, const contact_node *y,
                         double tolerance) {
    return !(x->amax+tolerance < y->amin
             || y->amax+tolerance < x->amin
             || x->bmax+tolerance < y->bmin
             || y->bmax+tolerance < x->bmin);
}

static double orient(const double *a, const double *b, const double *c) {
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
}

static double point_segment_distance(const double *p, const double *a,
                                     const double *b) {
    double dx = b[0]-a[0], dy = b[1]-a[1];
    double px = p[0]-a[0], py = p[1]-a[1];
    double q = dx*dx+dy*dy;
    if (q == 0.0) return hypot(px, py);
    double t = (px*dx+py*dy)/q;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return hypot(px-t*dx, py-t*dy);
}

static int segment_event(const double *a, const double *b,
                         const double *c, const double *d,
                         double tolerance, spong_contact_event *event) {
    double scale = hypot(b[0]-a[0], b[1]-a[1]);
    double other = hypot(d[0]-c[0], d[1]-c[1]);
    if (other > scale) scale = other;
    if (scale < 1.0) scale = 1.0;
    double floor = tolerance*scale;
    double o1 = orient(a, b, c), o2 = orient(a, b, d);
    double o3 = orient(c, d, a), o4 = orient(c, d, b);
    if (o1*o2 < 0.0 && o3*o4 < 0.0) {
        double den = ((b[0]-a[0])*(d[1]-c[1])
                      -(b[1]-a[1])*(d[0]-c[0]));
        double t = ((c[0]-a[0])*(d[1]-c[1])
                    -(c[1]-a[1])*(d[0]-c[0]))/den;
        event->kind = SPONG_CONTACT_CROSS;
        event->x = a[0]+t*(b[0]-a[0]);
        event->y = a[1]+t*(b[1]-a[1]);
        return 1;
    }
    double distance = point_segment_distance(a, c, d);
    other = point_segment_distance(b, c, d);
    if (other < distance) distance = other;
    other = point_segment_distance(c, a, b);
    if (other < distance) distance = other;
    other = point_segment_distance(d, a, b);
    if (other < distance) distance = other;
    double min_orientation = fabs(o1);
    if (fabs(o2) < min_orientation) min_orientation = fabs(o2);
    if (fabs(o3) < min_orientation) min_orientation = fabs(o3);
    if (fabs(o4) < min_orientation) min_orientation = fabs(o4);
    if (min_orientation <= floor && distance <= tolerance) {
        event->kind = SPONG_CONTACT_AMBIGUOUS;
        event->x = 0.25*(a[0]+b[0]+c[0]+d[0]);
        event->y = 0.25*(a[1]+b[1]+c[1]+d[1]);
        return 1;
    }
    return 0;
}

static int push_task(spong_contact_scan *scan, task_kind kind,
                     size_t first, size_t second) {
    if (scan->stack_size == scan->stack_capacity) {
        size_t capacity = scan->stack_capacity ? 2*scan->stack_capacity : 64;
        contact_task *grown = (contact_task *)realloc(
            scan->stack, capacity*sizeof(contact_task));
        if (grown == NULL) return -1;
        scan->stack = grown;
        scan->stack_capacity = capacity;
    }
    scan->stack[scan->stack_size++] = (contact_task){kind, first, second};
    return 0;
}

static void activate_leaf(spong_contact_scan *scan,
                          const contact_node *first,
                          const contact_node *second) {
    scan->leaf_active = 1;
    scan->ilo = first->lo;
    scan->ihi = first->hi;
    scan->jlo = second->lo;
    scan->jhi = second->hi;
    scan->i = scan->ilo;
    scan->j = scan->jlo;
}

spong_contact_scan *spong_contact_scan_create(
        const double *points1, size_t point_count1,
        const double *points2, size_t point_count2,
        double tolerance, int32_t self_scan) {
    if (points1 == NULL || point_count1 < 2 || tolerance < 0.0
            || !isfinite(tolerance)) return NULL;
    if (!self_scan && (points2 == NULL || point_count2 < 2)) return NULL;
    spong_contact_scan *scan = (spong_contact_scan *)calloc(1, sizeof(*scan));
    if (scan == NULL) return NULL;
    scan->points1 = points1;
    scan->points2 = self_scan ? points1 : points2;
    scan->tolerance = tolerance;
    scan->self_scan = !!self_scan;
    scan->nodes1 = (contact_node *)calloc(2*point_count1, sizeof(contact_node));
    if (scan->nodes1 == NULL) goto fail;
    scan->root1 = build_node(
        scan->nodes1, &scan->nodes_used1, points1, 0, point_count1-1);
    if (scan->self_scan) {
        scan->nodes2 = scan->nodes1;
        scan->root2 = scan->root1;
        if (push_task(scan, TASK_SELF, scan->root1, scan->root1) < 0)
            goto fail;
    } else {
        scan->nodes2 = (contact_node *)calloc(
            2*point_count2, sizeof(contact_node));
        if (scan->nodes2 == NULL) goto fail;
        scan->root2 = build_node(
            scan->nodes2, &scan->nodes_used2, points2, 0, point_count2-1);
        if (push_task(scan, TASK_PAIR, scan->root1, scan->root2) < 0)
            goto fail;
    }
    return scan;
fail:
    spong_contact_scan_destroy(scan);
    return NULL;
}

static int next_leaf_event(spong_contact_scan *scan,
                           spong_contact_event *event) {
    while (scan->i < scan->ihi) {
        while (scan->j < scan->jhi) {
            size_t i = scan->i, j = scan->j++;
            if (scan->self_scan && (!(i < j) || j-i <= 1)) continue;
            const double *a = point_at(scan->points1, i);
            const double *b = point_at(scan->points1, i+1);
            const double *c = point_at(scan->points2, j);
            const double *d = point_at(scan->points2, j+1);
            double amin = a[0] < b[0] ? a[0] : b[0];
            double amax = a[0] > b[0] ? a[0] : b[0];
            double bmin = a[1] < b[1] ? a[1] : b[1];
            double bmax = a[1] > b[1] ? a[1] : b[1];
            double cmin = c[0] < d[0] ? c[0] : d[0];
            double cmax = c[0] > d[0] ? c[0] : d[0];
            double dmin = c[1] < d[1] ? c[1] : d[1];
            double dmax = c[1] > d[1] ? c[1] : d[1];
            double tol = scan->tolerance;
            if (amax+tol < cmin || cmax+tol < amin
                    || bmax+tol < dmin || dmax+tol < bmin) continue;
            if (segment_event(a, b, c, d, tol, event)) {
                event->first_segment = (uint64_t)i;
                event->second_segment = (uint64_t)j;
                return 1;
            }
        }
        ++scan->i;
        scan->j = scan->jlo;
    }
    scan->leaf_active = 0;
    return 0;
}

int spong_contact_scan_next(spong_contact_scan *scan,
                            spong_contact_event *event) {
    if (scan == NULL || event == NULL) return -1;
    for (;;) {
        if (scan->leaf_active && next_leaf_event(scan, event)) return 1;
        if (scan->stack_size == 0) return 0;
        contact_task task = scan->stack[--scan->stack_size];
        contact_node *a = &scan->nodes1[task.first];
        contact_node *b = &scan->nodes2[task.second];
        if (task.kind == TASK_SELF) {
            if (a->leaf) {
                activate_leaf(scan, a, a);
                continue;
            }
            /* LIFO reverse of self(left), self(right), pair(left,right). */
            if (push_task(scan, TASK_PAIR, a->left, a->right) < 0
                    || push_task(scan, TASK_SELF, a->right, a->right) < 0
                    || push_task(scan, TASK_SELF, a->left, a->left) < 0)
                return -1;
            continue;
        }
        if (!boxes_overlap(a, b, scan->tolerance)) continue;
        if (a->leaf && b->leaf) {
            activate_leaf(scan, a, b);
            continue;
        }
        if (b->leaf || (!a->leaf && a->hi-a->lo >= b->hi-b->lo)) {
            if (push_task(scan, TASK_PAIR, a->left, task.second) < 0
                    || push_task(scan, TASK_PAIR, a->right, task.second) < 0)
                return -1;
        } else {
            if (push_task(scan, TASK_PAIR, task.first, b->left) < 0
                    || push_task(scan, TASK_PAIR, task.first, b->right) < 0)
                return -1;
        }
    }
}

void spong_contact_scan_destroy(spong_contact_scan *scan) {
    if (scan == NULL) return;
    free(scan->nodes1);
    if (scan->nodes2 != scan->nodes1) free(scan->nodes2);
    free(scan->stack);
    free(scan);
}
