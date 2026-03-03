from .basereader import BaseReader
from .cmlbidsreader import CMLBIDSReader
from .filtering import (
    filter_events_df_by_trial_types,
    filter_raw_events_by_trial_types,
    filter_epochs_by_trial_types,
    filter_by_trial_types,
)
from .convert import mne_epochs_to_ptsa, mne_raw_to_ptsa
from .units import detect_unit, get_scale_factor, convert_unit
from collections import namedtuple

__version__ = "0.1.0"
version_info = namedtuple("VersionInfo", "major,minor,patch")(
    *__version__.split('.'))
