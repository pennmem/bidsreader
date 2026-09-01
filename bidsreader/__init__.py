"""bidsreader — readers for BIDS-formatted electrophysiology archives.

The package is split in two:

- :mod:`bidsreader.readers` — one module per dataset-specific reader. Start here.
- :mod:`bidsreader.src` — the shared machinery the readers are built from.

Everything below is re-exported so the common cases stay a single import::

    from bidsreader import CMLBIDSReader, NiaBIDSReader, FileNotFoundBIDSError
"""

from .readers.basereader import BaseReader
from .readers.cmlbidsreader import CMLBIDSReader
from .readers.niabidsreader import NiaBIDSReader
from .src.filtering import (
    filter_events_df_by_trial_types,
    filter_raw_events_by_trial_types,
    filter_epochs_by_trial_types,
    filter_by_trial_types,
)
from .src.convert import mne_epochs_to_ptsa, mne_raw_to_ptsa
from .src.units import detect_unit, get_scale_factor, convert_unit
from .src.exc import (
    BIDSReaderError,
    InvalidOptionError,
    MissingRequiredFieldError,
    FileNotFoundBIDSError,
    AmbiguousMatchError,
    DataParseError,
    DependencyError,
    ExternalLibraryError,
)
from collections import namedtuple

__version__ = "0.4.0"
version_info = namedtuple("VersionInfo", "major,minor,patch")(
    *__version__.split('.'))
