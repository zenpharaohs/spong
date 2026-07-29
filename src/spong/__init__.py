"""spong: certified phase portraits for the single polynomial neuron.

See SPONG_FOUNDING.md for the mathematics, the certificate semantics
(EXACT / VALIDATED / RESIDUAL / EMPIRICAL), and the module contracts.
Phase 0: scaffold only.
"""

__version__ = "0.0.1"

from .resolution import (Resolution, ResolutionPolicy, ResolutionReason,
                         ResolutionStatus, resolve)

__all__ = [
    "Resolution", "ResolutionPolicy", "ResolutionReason", "ResolutionStatus",
    "resolve",
]
