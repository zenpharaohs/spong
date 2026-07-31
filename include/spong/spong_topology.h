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

typedef enum {
    SPONG_TOPOLOGY_REASON_NONE = 0,
    SPONG_TOPOLOGY_REASON_BRANCH_ABORT = 1,
    SPONG_TOPOLOGY_REASON_SEGMENT_BUDGET = 2,
    SPONG_TOPOLOGY_REASON_EVENT_BUDGET = 3,
    SPONG_TOPOLOGY_REASON_BRANCH_INVENTORY = 4,
    SPONG_TOPOLOGY_REASON_CONTACT = 5,
    SPONG_TOPOLOGY_REASON_UNSTABLE_END = 6,
    SPONG_TOPOLOGY_REASON_STABLE_TAIL = 7
} spong_topology_reason;

typedef struct {
    uint64_t saddle_count;
    uint64_t branch_count;
    uint64_t stable_count;
    uint64_t unstable_count;
    uint64_t segment_count;
    uint64_t segment_budget;
    uint64_t raw_event_count;
    uint64_t raw_event_budget;
    uint64_t forbidden_count;
    uint64_t ambiguous_count;
    uint64_t uncertified_unstable_ends;
    uint64_t uncertified_stable_tails;
    int32_t branch_aborted;
} spong_topology_analysis;

typedef struct {
    int32_t certified;
    int32_t audit_complete;
    int32_t branch_inventory_certified;
    int32_t primary_reason;    /* spong_topology_reason */
    uint64_t expected_stable;
    uint64_t expected_unstable;
} spong_topology_result;

/* Reduce measured topology evidence to the frontend-independent outcome. */
SPONG_API int spong_topology_decide(
    const spong_topology_analysis *analysis,
    spong_topology_result *result);

SPONG_API const char *spong_topology_reason_name(int32_t reason);

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
