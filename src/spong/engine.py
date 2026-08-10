"""Selection of the continuation engine, and order-preserving assembly.

MILESTONE 1 OF THE NATIVE MIGRATION
-----------------------------------
The switch exists and ``native`` routes straight back to the Python
implementation.  Nothing here is faster.  The point is to prove the plumbing
-- selection, the differential harness, the assembly contract -- against the
frozen goldens BEFORE any C exists, so that every later commit replaces one
piece of the engine behind a switch already known to work.

    SPONG_ENGINE=python|native          process default
    engine.use("native")                programmatic
    with engine.using("native"): ...    scoped

WHY THE DEFAULT STAYS PYTHON
----------------------------
``charts`` and ``gauss`` are the reference implementation.  They are not
scaffolding to be deleted once C works: they are what a skeptic runs to
check that the fast path did not change a portrait.  A rewrite of the
geometry engine is only safe while both are runnable side by side.

THE ORDERING CONTRACT
---------------------
Branch results are assembled in SUBMISSION order and never sorted afterwards.
The serial order is therefore also the parallel order, which is what makes
parallelising the trace loops incapable of moving a ledger entry or changing
a golden.  ``map_ordered`` is the only sanctioned way to run them
concurrently; it preserves order whatever the workers finish in.

Parallelism is deliberately NOT enabled here.  While the engine is Python it
is GIL-bound -- measured at 1.00x on eight threads -- so a worker pool would
add contention and no speed.  ``map_ordered`` exists now so that the
assembly contract is established and tested before it can matter.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from . import charts


# ------------------------------------------------------------------ #
# native core availability
# ------------------------------------------------------------------ #
# The C core is imported lazily by charts, which falls back to the
# reference implementation if the import fails (interpreter or ABI skew).
# That fallback must never be silent -- it costs far more than speed --
# so the probe is done once here, the reason is kept for banners, and a
# request for the native engine that cannot be honoured says so on stderr.

def _probe_native():
    try:
        from . import _native
    except ImportError as exc:
        return f"import failed: {exc}"
    if not hasattr(_native, "continue_curve"):
        return ("loaded, but continue_curve is missing -- stale build; "
                "run pip install -e . --no-build-isolation")
    return None


_NATIVE_ERROR = _probe_native()


def native_error():
    """None if the C core is importable and complete, else the reason."""
    return _NATIVE_ERROR


def _warn_if_unhonoured(name):
    if name == "native" and _NATIVE_ERROR is not None:
        import sys
        print("spong.engine: native engine requested but the C core is "
              f"unavailable ({_NATIVE_ERROR}); traces will run on the "
              "python reference implementation", file=sys.stderr)


class PythonEngine:
    """The reference continuation engine: charts + gauss, pure Python."""

    name = "python"
    is_native = False

    @staticmethod
    def trace_stable(*args, **kwargs):
        return charts.trace_stable(*args, **kwargs)

    @staticmethod
    def trace_unstable(*args, **kwargs):
        return charts.trace_unstable(*args, **kwargs)


class NativeEngine(PythonEngine):
    """The C dispatcher.

    At milestone 1 this inherits every method from PythonEngine, so selecting
    it changes nothing.  Each method is overridden as the corresponding piece
    moves into the C core, and ``is_native`` becomes true only when the whole
    dispatcher is across -- so a partially migrated engine cannot claim to be
    the fast path.
    """

    name = "native"
    is_native = False


_ENGINES = {e.name: e for e in (PythonEngine, NativeEngine)}
_active = _ENGINES.get(os.environ.get("SPONG_ENGINE", "python"), PythonEngine)
_warn_if_unhonoured(_active.name)


def current():
    """The active engine class."""
    return _active


def active_name() -> str:
    return _active.name


def available() -> tuple[str, ...]:
    return tuple(sorted(_ENGINES))


def use(name: str):
    """Select an engine for the rest of the process."""
    global _active
    if name not in _ENGINES:
        raise ValueError(
            f"unknown engine {name!r}; have {', '.join(available())}")
    _warn_if_unhonoured(name)
    _active = _ENGINES[name]
    return _active


@contextmanager
def using(name: str):
    """Select an engine for the duration of a block."""
    previous = _active.name
    try:
        yield use(name)
    finally:
        use(previous)


# ------------------------------------------------------------------ #
# dispatch
# ------------------------------------------------------------------ #

def trace_stable(*args, **kwargs):
    return _active.trace_stable(*args, **kwargs)


def trace_unstable(*args, **kwargs):
    return _active.trace_unstable(*args, **kwargs)


# ------------------------------------------------------------------ #
# assembly
# ------------------------------------------------------------------ #

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
