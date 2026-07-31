"""Thin ctypes access to the exact continuous-Bernoulli stream sampler.

SPONG does not copy or reimplement the posterior arithmetic.  Build
``cb_core.c`` from the continuous-bernoulli package as a shared library and
pass its path explicitly (or through ``CB_CORE_LIBRARY``).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class ContinuousBernoulliBank:
    """One independent exact posterior stream per bandit arm."""

    def __init__(self, n_arms, seed=1729, library=None, buffer_capacity=64):
        path = library or os.environ.get("CB_CORE_LIBRARY")
        if not path:
            raise ValueError(
                "continuous-Bernoulli shared library required; pass "
                "library=... or set CB_CORE_LIBRARY")
        self.library_path = Path(path).expanduser().resolve()
        self._lib = ctypes.CDLL(str(self.library_path))
        pointer = ctypes.c_void_p
        self._lib.cb_stream_create.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_int]
        self._lib.cb_stream_create.restype = pointer
        self._lib.cb_stream_destroy.argtypes = [pointer]
        self._lib.cb_stream_destroy.restype = None
        self._lib.cb_stream_update.argtypes = [pointer, ctypes.c_double]
        self._lib.cb_stream_update.restype = None
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

    def draw_all(self):
        draws = np.empty(len(self._streams), dtype=float)
        for index, stream in enumerate(self._streams):
            if self._lib.cb_stream_draw(stream, 1, self._draw_buffer) != 1:
                raise RuntimeError(
                    f"continuous-Bernoulli draw failed for arm {index}")
            draws[index] = self._draw_buffer[0]
        return draws

    def update(self, arm, observation):
        value = float(observation)
        if not 0.0 <= value <= 1.0:
            raise ValueError("continuous-Bernoulli observations must be in [0,1]")
        self._lib.cb_stream_update(self._streams[int(arm)], value)

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
