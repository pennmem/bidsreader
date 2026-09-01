"""The readers.

Every dataset-specific reader lives in this folder, one module each. To support
a new BIDS archive, add a module here that subclasses :class:`BaseReader` and
re-export it below — see the "Creating a New Reader" section of the README.

- :class:`BaseReader` — shared BIDS path construction and metadata queries
- :class:`CMLBIDSReader` — CML (Computational Memory Lab) datasets
- :class:`NiaBIDSReader` — Nia Therapeutics BIDS conversions
"""

from .basereader import BaseReader
from .cmlbidsreader import CMLBIDSReader, CML_ROOT
from .niabidsreader import NiaBIDSReader, NIA_ROOT

__all__ = [
    "BaseReader",
    "CMLBIDSReader",
    "CML_ROOT",
    "NiaBIDSReader",
    "NIA_ROOT",
]
