from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from typing import Iterable, Optional, TYPE_CHECKING
from ._errorwrap import public_api
from .helpers import merge_duplicate_sample_events

if TYPE_CHECKING:
    from ptsa.data.timeseries import TimeSeries


@public_api
def mne_epochs_to_ptsa(epochs: mne.Epochs, events: pd.DataFrame) -> TimeSeries:
    from ptsa.data.timeseries import TimeSeries
    events = merge_duplicate_sample_events(events)
    return TimeSeries.from_mne_epochs(epochs, events)


@public_api
def mne_raw_to_ptsa(raw: mne.io.BaseRaw, picks: Optional[Iterable[str]] = None, tmin: float = None, tmax: float = None) -> TimeSeries:
    from ptsa.data.timeseries import TimeSeries

    inst = raw.copy()
    if tmin is not None or tmax is not None:
        inst.crop(tmin=tmin, tmax=tmax)

    if picks is not None:
        if all(isinstance(p, str) for p in picks):
            pick_idx = [inst.ch_names.index(ch) for ch in picks]
        else:
            pick_idx = list(picks)

        data = inst.get_data(picks=pick_idx)
        ch_names = [inst.ch_names[i] for i in pick_idx]
    else:
        data = inst.get_data()
        ch_names = inst.ch_names

    sfreq = float(inst.info["sfreq"])
    times = inst.times

    ts = TimeSeries.create(
        data,
        samplerate=sfreq,
        dims=("channel", "time"),
        coords={
            "channel": np.asarray(ch_names, dtype=object),
            "time": np.asarray(times, dtype=float),
        },
        attrs={
            "mne_meas_date": str(inst.info.get("meas_date")),
            "mne_first_samp": int(inst.first_samp),
        },
    )
    return ts
