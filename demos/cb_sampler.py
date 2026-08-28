"""Thin ctypes access to the exact continuous-Bernoulli stream sampler.

SPONG does not copy or reimplement the posterior arithmetic.  An explicit
shared library (or ``CB_CORE_LIBRARY``) remains the normal standalone
contract.  Interactive demos may additionally ask this module to discover a
sibling ``continuous-bernoulli`` checkout and build its ``cb_core.c`` into a
content-addressed temporary cache.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np


_BUILD_LOCK = threading.Lock()


def _source_candidates():
    configured = os.environ.get("CB_CORE_SOURCE")
    if configured:
        yield Path(configured).expanduser()
    # spong/demos/cb_sampler.py and continuous-bernoulli are sibling repos in
    # the development layout used by the interactive explorer.
    yield (Path(__file__).resolve().parents[2]
           / "continuous-bernoulli" / "src" / "c" / "cb_core.c")


def _built_candidates(source):
    suffixes = (".dylib", ".so", ".dll")
    roots = (source.parent, source.parents[2] / "build",
             source.parents[2] / "lib")
    for root in roots:
        for suffix in suffixes:
            yield root / f"libcb_core{suffix}"


def _build_shared(source):
    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform.startswith("linux"):
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]
    else:
        raise ValueError(
            "automatic cb_core build is supported on macOS and Linux; "
            "set CB_CORE_LIBRARY to a prebuilt library")

    fingerprint = hashlib.sha256(source.read_bytes())
    include_dir = source.parent / "include"
    if include_dir.is_dir():
        for header in sorted(include_dir.glob("*.h")):
            fingerprint.update(header.name.encode("utf-8"))
            fingerprint.update(header.read_bytes())
    digest = fingerprint.hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "spong-cb-core"
    target = cache / f"libcb_core-{digest}{suffix}"
    with _BUILD_LOCK:
        if target.is_file():
            return target.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / f".{target.name}.{os.getpid()}"
        command = [os.environ.get("CC", "cc"), "-O3", "-std=c99",
                   *shared_flags, str(source), "-o", str(temporary), "-lm"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temporary, target)
        except (OSError, subprocess.CalledProcessError) as exc:
            temporary.unlink(missing_ok=True)
            detail = (exc.stderr.strip()
                      if isinstance(exc, subprocess.CalledProcessError)
                      else str(exc))
            raise ValueError(
                f"could not build continuous-Bernoulli cb_core: {detail}") \
                from exc
    return target.resolve()


def resolve_library(library=None, *, auto_build=False):
    """Resolve the exact sampler library, optionally building a sibling core.

    Explicit configuration always wins.  Automatic building is opt-in so a
    batch experiment never invokes a compiler unexpectedly; the local viewer
    opts in because it is otherwise needlessly awkward to launch.
    """
    configured = library or os.environ.get("CB_CORE_LIBRARY")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"continuous-Bernoulli shared library does not exist: {path}")
        return path

    sources = [path.resolve() for path in _source_candidates()
               if path.is_file()]
    if auto_build and sources:
        return _build_shared(sources[0])
    for source in sources:
        for candidate in _built_candidates(source):
            if candidate.is_file():
                return candidate.resolve()

    extra = ("; no cb_core.c was found in the sibling "
             "continuous-bernoulli checkout" if auto_build else "")
    raise ValueError(
        "continuous-Bernoulli shared library required; pass library=... or "
        f"set CB_CORE_LIBRARY{extra}")


class ContinuousBernoulliBank:
    """One independent exact posterior stream per bandit arm."""

    def __init__(self, n_arms, seed=1729, library=None, buffer_capacity=64):
        self.library_path = resolve_library(library)
        self._lib = ctypes.CDLL(str(self.library_path))
        pointer = ctypes.c_void_p
        self._lib.cb_stream_create.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_int]
        self._lib.cb_stream_create.restype = pointer
        self._lib.cb_stream_destroy.argtypes = [pointer]
        self._lib.cb_stream_destroy.restype = None
        self._lib.cb_stream_update.argtypes = [pointer, ctypes.c_double]
        self._lib.cb_stream_update.restype = ctypes.c_int
        try:
            set_stats = self._lib.cb_stream_set_stats
        except AttributeError as exc:
            raise ValueError(
                "continuous-Bernoulli library lacks cb_stream_set_stats; "
                "rebuild it from the current continuous-bernoulli source") \
                from exc
        set_stats.argtypes = [pointer, ctypes.c_double, ctypes.c_double]
        set_stats.restype = ctypes.c_int
        self._lib.cb_stream_draw.argtypes = [
            pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self._lib.cb_stream_draw.restype = ctypes.c_int
        self._streams = []
        for index in range(int(n_arms)):
            stream = self._lib.cb_stream_create(
                0.0, 0.0, int(seed), index, int(buffer_capacity))
            if not stream:
                self.close()
                raise MemoryError(
                    f"could not create continuous-Bernoulli stream {index}")
            self._streams.append(stream)
        self._draw_buffer = (ctypes.c_double * 1)()
        self._active = np.ones(len(self._streams), dtype=bool)

    def draw_all(self):
        draws = np.full(len(self._streams), np.inf, dtype=float)
        for index, stream in enumerate(self._streams):
            if not self._active[index]:
                continue
            if self._lib.cb_stream_draw(stream, 1, self._draw_buffer) != 1:
                raise RuntimeError(
                    f"continuous-Bernoulli draw failed for arm {index}")
            draws[index] = self._draw_buffer[0]
        return draws

    def deactivate(self, arm):
        """Remove a terminal arm from subsequent posterior draws."""
        self._active[int(arm)] = False

    def update(self, arm, observation):
        """Accumulate one ordinary posterior observation.

        This remains available for experiments whose sufficient statistic is
        a history.  Rested descent allocation uses :meth:`set_observation`
        instead: its arm state is the current held loss, not an average of
        all losses seen along the trajectory.
        """
        value = float(observation)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "continuous-Bernoulli observations must be finite values "
                "in the closed interval [0,1]")
        if self._lib.cb_stream_update(self._streams[int(arm)], value) != 1:
            raise RuntimeError(
                f"continuous-Bernoulli update failed for arm {int(arm)}")

    def set_observation(self, arm, observation, selections):
        """Set the descent-bandit state to ``(N*Z, N)`` exactly.

        ``Z`` is the arm's current held loss statistic and ``N`` is its
        number of allocated chunks.  Replacing the sufficient statistics is
        what makes a rested optimizer arm sample-and-hold; accumulating all
        past losses would describe a different bandit.
        """
        value = float(observation)
        count = int(selections)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "continuous-Bernoulli observations must be finite values "
                "in the closed interval [0,1]")
        if count != selections or count <= 0:
            raise ValueError("selections must be a positive integer")
        if self._lib.cb_stream_set_stats(
                self._streams[int(arm)], count*value, float(count)) != 1:
            raise RuntimeError(
                f"continuous-Bernoulli set_stats failed for arm {int(arm)}")

    def close(self):
        streams = getattr(self, "_streams", ())
        for stream in streams:
            self._lib.cb_stream_destroy(stream)
        self._streams = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        self.close()
