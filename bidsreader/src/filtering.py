import mne
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Dict
from ._errorwrap import public_api


def _label_has_trial_type(label: str, trial_types: list[str]) -> bool:
    # exact token match within merged labels like "WORD/STIM"
    tokens = label.split("/")
    return any(t in tokens for t in trial_types)


def _ensure_list(trial_types: Iterable[str] | str) -> list[str]:
    return [trial_types] if isinstance(trial_types, str) else list(trial_types)


@public_api
def filter_events_df_by_trial_types(
    events_df: pd.DataFrame,
    trial_types: Iterable[str] | str,
) -> tuple[pd.DataFrame, np.ndarray]:
    tt = _ensure_list(trial_types)

    mask = events_df["trial_type"].isin(tt).to_numpy()
    filtered_df = events_df.loc[mask].copy()

    # integer positions (0..n-1) into the *original* events_df rows
    df_idx = np.flatnonzero(mask)

    return filtered_df, df_idx


@public_api
def filter_raw_events_by_trial_types(
    raw: mne.io.BaseRaw,
    trial_types: Iterable[str] | str,
) -> tuple[np.ndarray, Dict[str, int], np.ndarray]:
    tt = _ensure_list(trial_types)

    events_raw, event_id = mne.events_from_annotations(raw)

    filtered_event_id = {
        k: v for k, v in event_id.items()
        if _label_has_trial_type(k, tt)
    }

    if filtered_event_id:
        codes = np.fromiter(filtered_event_id.values(), dtype=int)
        mask = np.isin(events_raw[:, 2], codes)
        filtered_events = events_raw[mask]
        raw_idx = np.flatnonzero(mask)  # indices into events_raw
    else:
        filtered_events = events_raw[:0].copy()
        filtered_event_id = {}
        raw_idx = np.array([], dtype=int)

    return filtered_events, filtered_event_id, raw_idx


@public_api
def filter_epochs_by_trial_types(
    epochs: mne.Epochs,
    trial_types: Iterable[str] | str,
) -> tuple[mne.Epochs, Dict[str, int], np.ndarray]:
    tt = _ensure_list(trial_types)

    keys = [
        k for k in epochs.event_id.keys()
        if _label_has_trial_type(k, tt)
    ]
    filtered_event_id = {k: epochs.event_id[k] for k in keys}

    if keys:
        filtered_epochs = epochs[keys]
        codes = np.fromiter(filtered_event_id.values(), dtype=int)
        mask = np.isin(epochs.events[:, 2], codes)
        ep_idx = np.flatnonzero(mask)  # indices into original epochs
    else:
        filtered_epochs = epochs.copy()[[]]
        ep_idx = np.array([], dtype=int)

    return filtered_epochs, filtered_event_id, ep_idx


@public_api
def filter_by_trial_types(
    trial_types: Iterable[str] | str,
    *,
    events_df: Optional[pd.DataFrame] = None,
    raw: Optional[mne.io.BaseRaw] = None,
    epochs: Optional[mne.Epochs] = None,
) -> tuple[
    Optional[pd.DataFrame],
    Optional[np.ndarray],     # filtered_events (from raw)
    Optional[mne.Epochs],
    Dict[str, int],
    np.ndarray,               # filtered_event_idx (0..n-1)
]:
    tt = _ensure_list(trial_types)

    filtered_df: Optional[pd.DataFrame] = None
    filtered_events: Optional[np.ndarray] = None
    filtered_epochs: Optional[mne.Epochs] = None

    df_idx: Optional[np.ndarray] = None
    raw_idx: Optional[np.ndarray] = None
    ep_idx: Optional[np.ndarray] = None

    event_id_raw: Optional[Dict[str, int]] = None
    event_id_epochs: Optional[Dict[str, int]] = None

    n_df = None
    n_raw = None
    n_ep = None

    # ---- DF ----
    if events_df is not None:
        filtered_df, df_idx = filter_events_df_by_trial_types(events_df, tt)
        n_df = len(filtered_df)

    # ---- RAW ----
    raw_onsets = None
    if raw is not None:
        filtered_events, event_id_raw, raw_idx = filter_raw_events_by_trial_types(raw, tt)
        n_raw = int(filtered_events.shape[0])
        raw_onsets = filtered_events[:, 0].astype(int)

    # ---- EPOCHS ----
    ep_onsets = None
    if epochs is not None:
        filtered_epochs, event_id_epochs, ep_idx = filter_epochs_by_trial_types(epochs, tt)
        n_ep = len(filtered_epochs)

        if event_id_epochs:
            codes = np.fromiter(event_id_epochs.values(), dtype=int)
            mask = np.isin(epochs.events[:, 2], codes)
            ep_onsets = epochs.events[mask, 0].astype(int)
        else:
            ep_onsets = np.array([], dtype=int)

    # ---- Check event_id consistency (keys) ----
    if event_id_raw is not None and event_id_epochs is not None:
        if set(event_id_raw.keys()) != set(event_id_epochs.keys()):
            raise ValueError(
                "filtered_event_id key mismatch between raw and epochs.\n"
                f"raw keys={sorted(event_id_raw.keys())}\n"
                f"epochs keys={sorted(event_id_epochs.keys())}"
            )

        # Strong alignment check: same event onsets
        if raw_onsets is not None and ep_onsets is not None:
            if raw_onsets.shape != ep_onsets.shape or not np.array_equal(raw_onsets, ep_onsets):
                raise ValueError(
                    "raw/epochs trial alignment mismatch: filtered event sample onsets differ.\n"
                    f"n_raw={len(raw_onsets)} n_epochs={len(ep_onsets)}"
                )

        filtered_event_id = event_id_raw  # choose one
    elif event_id_raw is not None:
        filtered_event_id = event_id_raw
    elif event_id_epochs is not None:
        filtered_event_id = event_id_epochs
    else:
        filtered_event_id = {}

    # ---- Trial count consistency (for whichever inputs are provided) ----
    ns = [n for n in (n_df, n_raw, n_ep) if n is not None]
    n = ns[0] if ns else 0

    if any(x != n for x in ns):
        raise ValueError(
            "Trial count mismatch across provided inputs.\n"
            f"n_df={n_df} n_raw={n_raw} n_epochs={n_ep}"
        )

    filtered_event_idx = np.arange(n, dtype=int)
    return filtered_df, filtered_events, filtered_epochs, filtered_event_id, filtered_event_idx
