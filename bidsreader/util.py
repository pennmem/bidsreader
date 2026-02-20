import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import mne
from ptsa.data.timeseries import TimeSeries
from pathlib import Path
from typing import Iterable, Any, Tuple, Sequence, Optional, Union, Dict
import re
import numpy as np
import pandas as pd
import warnings
import json
from .constants import CML_ROOT, INTRACRANIAL_FIELDS, SCALP_FIELDS, VALID_EEG_TYPES, VALID_ACQ
from ._errorwrap import public_api

def validate_option(name: str, value: Any, allowed: Iterable[Any]) -> Any:
    if value is None:
        return None
    if value not in allowed:
        raise InvalidOptionError(f"{name} must be one of: {allowed}. Got {value!r}")
    return value