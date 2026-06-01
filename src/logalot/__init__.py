"""LogALot — local-first ham-radio logger with a local-LLM capture front-end.

The implemented, stdlib-only core is re-exported here. The AI pipeline modules
(``capture``, ``asr``, ``parse``) are imported lazily by callers that need them,
so the core stays dependency-free.
"""
from __future__ import annotations

from .export_adif import to_adif, write_adif
from .models import QSO, band_for_freq
from .store import Store
from .validate import (
    confidence_flag,
    expand_phonetics,
    is_valid_call,
    normalise_call,
)

__version__ = "0.0.1"

__all__ = [
    "QSO",
    "band_for_freq",
    "Store",
    "to_adif",
    "write_adif",
    "expand_phonetics",
    "normalise_call",
    "is_valid_call",
    "confidence_flag",
    "__version__",
]
