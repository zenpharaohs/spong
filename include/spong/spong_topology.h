#ifndef SPONG_TOPOLOGY_H
#define SPONG_TOPOLOGY_H

/* Streaming C99 contact predicates for piecewise-linear Morse skeletons. */

#include <stddef.h>
#include <stdint.h>

#include "spong/spong_resolution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPONG_CONTACT_CROSS = 1,
    SPONG_CONTACT_AMBIGUOUS = 2
} spong_contact_kind;

typedef struct {
    uint64_t first_segment;
    uint64_t second_segment;
    int32_t kind;              /* spong_contact_kind */
    double x;
    double y;
} spong_contact_event;

typedef struct spong_contact_scan spong_contact_scan;

/*
 * The coordinate arrays are packed (a,b) pairs and remain owned by the
 * caller. They must stay alive until spong_contact_scan_destroy(). A self
 * scan visits every non-adjacent segment pair of points1 exactly once.
 */
SPONG_API spong_contact_scan *spong_contact_scan_create(
    const double *points1, size_t point_count1,
    const double *points2, size_t point_count2,
    double tolerance, int32_t self_scan);

/* Return 1 for an event, 0 at end of stream, and -1 on error. */
SPONG_API int spong_contact_scan_next(
    spong_contact_scan *scan, spong_contact_event *event);

SPONG_API void spong_contact_scan_destroy(spong_contact_scan *scan);

#ifdef __cplusplus
}
#endif

#endif
