"""ctypes bridge for the fused, resumable Adam optimizer demo kernel."""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np

from demos import optimizers as opt


_SOURCE = Path(__file__).with_suffix(".c")
_LOCK = threading.Lock()
_LIB = None
_UNAVAILABLE = False
_DOUBLE = ct.c_double
_INT = ct.c_int
_U64 = ct.c_uint64
_POINTER = ct.POINTER(_DOUBLE)
_REASONS = {1: "nonfinite_state", 2: "nonfinite_gradient",
            3: "nonfinite_step"}
SAMPLES_PER_BLOCK = 65536


def _load():
    global _LIB, _UNAVAILABLE
    if os.environ.get("SPONG_DEMO_ADAM_NATIVE") == "0":
        return None
    if _LIB is not None or _UNAVAILABLE:
        return _LIB
    with _LOCK:
        if _LIB is not None or _UNAVAILABLE:
            return _LIB
        if sys.platform == "darwin":
            suffix, flags = ".dylib", ["-dynamiclib"]
        elif sys.platform.startswith("linux"):
            suffix, flags = ".so", ["-shared", "-fPIC"]
        else:
            _UNAVAILABLE = True
            return None
        digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / "spong-native-adam" / (
            f"libspong_adam-{digest}{suffix}")
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.{threading.get_ident()}")
            command = [os.environ.get("CC", "cc"), "-O3", "-std=c99",
                       "-ffp-contract=off", *flags, str(_SOURCE),
                       "-o", str(temporary), "-lm"]
            try:
                subprocess.run(command, check=True, capture_output=True)
                os.replace(temporary, target)
            except (OSError, subprocess.CalledProcessError):
                temporary.unlink(missing_ok=True)
                _UNAVAILABLE = True
                return None
        try:
            lib = ct.CDLL(str(target))
        except OSError:
            _UNAVAILABLE = True
            return None
        function = lib.spong_adam_chunk
        function.argtypes = [
            _POINTER, _INT, _POINTER, _INT, _POINTER, _INT,
            _POINTER, _INT, _INT, _POINTER, _POINTER, _POINTER,
            ct.POINTER(_U64), _INT, _DOUBLE, _U64, _U64,
            _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE, _DOUBLE,
            _POINTER, ct.POINTER(_INT)]
        function.restype = _INT
        _LIB = lib
    return _LIB


def available():
    """Whether the local C compiler/library can supply the Adam kernel."""
    return _load() is not None


def _schedule(lr):
    if isinstance(lr, opt.InverseSqrtSchedule):
        return 1, lr.base_lr, lr.warmup_steps, 0, 0.0
    if isinstance(lr, opt.CosineSchedule):
        warmup = max(1, int(round(lr.total_steps * lr.warmup_fraction)))
        return 2, lr.base_lr, warmup, lr.total_steps, lr.final_fraction
    if np.isscalar(lr):
        return 0, float(lr), 1, 0, 0.0
    return None


def run_chunk(state, gradient, steps, should_stop=None):
    """Return (successful points, failure reason), or None for fallback.

    NumPy generates the same per-arm sample stream in one matrix draw.  The C
    kernel consumes that matrix and updates the Adam moments and iterates in
    order.  Loss and posterior observations stay with the Python allocator.
    """
    if not isinstance(state, opt.AdamState) or not isinstance(
            gradient, opt.BatchGradient):
        return None
    if any(np.shape(value) != (2,) for value in
           (state.z, state.first, state.second)):
        return None
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    schedule = _schedule(state.lr)
    lib = _load()
    if schedule is None or lib is None:
        return None
    if not np.all(np.isfinite(state.z)):
        return np.empty((0, 2), dtype=float), "nonfinite_state"
    kind, base, warmup, horizon, final_fraction = schedule
    f = np.ascontiguousarray(gradient.f, dtype=np.float64)
    g = np.ascontiguousarray(gradient.g, dtype=np.float64)
    gp = np.ascontiguousarray(gradient.gp, dtype=np.float64)
    z = np.ascontiguousarray(state.z, dtype=np.float64)
    first = np.ascontiguousarray(state.first, dtype=np.float64)
    second = np.ascontiguousarray(state.second, dtype=np.float64)
    t = _U64(state.t)

    def ptr(array):
        return array.ctypes.data_as(_POINTER)

    outputs = []
    remaining = int(steps)
    # Limit the sample matrix to roughly 512 KiB.  The interactive controls
    # allow batches of 100,000 and chunks of 1,000; materializing their full
    # Cartesian product would otherwise allocate 800 MiB for one arm.
    block_limit = max(1, SAMPLES_PER_BLOCK // gradient.n)
    while remaining:
        if should_stop is not None and should_stop():
            raise TimeoutError("allocation stopped by interactive time limit")
        count = min(remaining, block_limit)
        samples = np.ascontiguousarray(
            gradient._draw_many(count), dtype=np.float64)
        points = np.empty((count, 2), dtype=np.float64)
        done = _INT()
        code = lib.spong_adam_chunk(
            ptr(f), len(f), ptr(g), len(g), ptr(gp), len(gp), ptr(samples),
            gradient.n, count, ptr(z), ptr(first), ptr(second), ct.byref(t),
            kind, base, warmup, horizon, final_fraction, state.b1, state.b2,
            state.eps, state.weight_decay, ptr(points), ct.byref(done))
        if code < 0 or code not in (0, 1, 2, 3):
            raise RuntimeError(f"native Adam kernel failed with code {code}")
        outputs.append(points[:done.value].copy())
        if code:
            break
        remaining -= count
    state.z, state.first, state.second, state.t = z, first, second, t.value
    return np.concatenate(outputs), _REASONS.get(code)
