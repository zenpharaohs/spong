"""Order-preserving work scheduling, independent of trajectory construction."""
import os
from concurrent.futures import ThreadPoolExecutor

def workers() -> int:
    """How many branches to trace concurrently.

    SPONG_WORKERS, default 1.  Threads only help under the native engine,
    which releases the GIL for a whole segment; under the Python engine the
    trace loop holds it and a pool costs contention for no speed (measured at
    1.00x on eight threads).

    Measured ceiling on the zoo: total branch time over longest branch is
    about 4.5x, so beyond six workers there is nothing left to win.
    """
    try:
        n = int(os.environ.get("SPONG_WORKERS", "1"))
    except ValueError:
        return 1
    return max(1, n)


def map_ordered(fn, tasks, workers: int = 1):
    """Apply fn to tasks, returning results in SUBMISSION order.

    The order is the submission order -- not a completion order, not a sort
    key -- so raising the worker count cannot move a ledger entry or change a
    golden.  That is the whole of the parallelisation contract.

    Longest-first dispatch would reduce the tail, but reordering submission
    would reorder the ledger; the imbalance measured on the zoo (about 4.5x
    total over longest) is mild enough that a plain dynamic queue is close to
    the ceiling anyway.
    """
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, tasks))
