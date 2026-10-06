"""spong: certified phase portraits for the single polynomial neuron.

See SPONG_FOUNDING.md for the mathematics, the certificate semantics
(EXACT / VALIDATED / RESIDUAL / EMPIRICAL), and the module contracts.
Phase 0: scaffold only.
"""

__version__ = "0.0.1"

__all__ = [
    "Resolution", "ResolutionPolicy", "ResolutionReason", "ResolutionStatus",
    "resolve",
]


def __getattr__(name):
    # Importing an independent checker must not load the portrait constructor.
    # Keep the existing public API, loading resolution only when requested.
    if name in __all__:
        from . import resolution
        value = getattr(resolution, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
