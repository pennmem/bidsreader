import mne
import numpy as np
from typing import Optional, Union
from ptsa.data.timeseries import TimeSeries
from ._errorwrap import public_api

# ---------- unit constants ----------
_UNIT_EXPONENTS = {
    "V": 0, "mV": -3, "uV": -6, "nV": -9,
    "T": 0, "mT": -3, "uT": -6, "nT": -9, "fT": -15,
}

_FIFF_UNIT_TO_BASE = {107: "V", 201: "T", 0: None}

_FIFF_MUL_TO_EXP = {
    0: 0, -3: -3, -6: -6, -9: -9, -12: -12, -15: -15, 3: 3, 6: 6,
}

_EXP_TO_PREFIX = {
    0: "", -3: "m", -6: "u", -9: "n", -12: "p", -15: "f", 3: "k", 6: "M",
}

# ---------- internal helpers ----------

def _normalize_unit(unit: str) -> str:
    return unit.replace("µ", "u")


def _detect_unit_mne(inst: Union[mne.io.BaseRaw, mne.Epochs]) -> str:
    """Detect unit string from an MNE Raw or Epochs object."""
    eeg_types = {"eeg", "seeg", "ecog", "ieeg", "dbs"}

    for ch_info in inst.info["chs"]:
        ch_kind = mne.io.pick.channel_type(
            inst.info, inst.ch_names.index(ch_info["ch_name"]),
        )
        if ch_kind not in eeg_types:
            continue

        fiff_unit = ch_info.get("unit", 0)
        fiff_mul = ch_info.get("unit_mul", 0)

        base = _FIFF_UNIT_TO_BASE.get(fiff_unit)
        if base is None:
            raise ValueError(
                f"Unknown FIFF unit code {fiff_unit} on channel "
                f"'{ch_info['ch_name']}'. Pass current_unit= explicitly."
            )

        exp = _FIFF_MUL_TO_EXP.get(fiff_mul, 0)
        prefix = _EXP_TO_PREFIX.get(exp, "")
        return f"{prefix}{base}"

    raise ValueError(
        "No EEG/iEEG/SEEG/ECoG channel found. Cannot detect unit."
    )


def _detect_unit_ptsa(ts: TimeSeries) -> str:
    """Detect unit string from a PTSA TimeSeries."""
    for key in ("units", "unit"):
        val = ts.attrs.get(key)
        if val is not None and str(val).strip():
            unit_str = _normalize_unit(str(val).strip())
            if unit_str in _UNIT_EXPONENTS:
                return unit_str
            raise ValueError(
                f"TimeSeries has unit '{val}' which is not recognized. "
                f"Known: {sorted(_UNIT_EXPONENTS.keys())}"
            )

    raise ValueError(
        "TimeSeries has no 'units' or 'unit' attribute. "
        "Pass current_unit= explicitly."
    )


def _convert_mne(
    inst: Union[mne.io.BaseRaw, mne.Epochs],
    factor: float,
    target_unit: str,
    copy: bool,
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Scale MNE data and update FIFF unit metadata."""
    if copy:
        inst = inst.copy()

    inst.apply_function(lambda x: x * factor, picks="all", channel_wise=False)

    base_char = target_unit[-1]
    target_exp = _UNIT_EXPONENTS[target_unit]
    fiff_unit_code = {"V": 107, "T": 201}.get(base_char, 0)
    fiff_mul = min(
        _FIFF_MUL_TO_EXP.keys(),
        key=lambda k: abs(_FIFF_MUL_TO_EXP[k] - target_exp),
    )

    # Update unit metadata on all EEG/SEEG/ECoG channels
    eeg_kinds = {2, 302, 802, 803}  # EEG, EEG_REF, SEEG, ECOG
    for ch in inst.info["chs"]:
        if ch.get("kind", 0) in eeg_kinds or ch.get("unit", 0) in (107, 201):
            ch["unit"] = fiff_unit_code
            ch["unit_mul"] = fiff_mul

    return inst


def _convert_ptsa(
    ts: TimeSeries,
    factor: float,
    target_unit: str,
    copy: bool,
) -> TimeSeries:
    """Scale PTSA TimeSeries data and update attrs."""
    if copy:
        result = ts * factor
    else:
        ts.values[:] *= factor
        result = ts

    # Update all unit-related attrs so users know the current unit
    result.attrs["units"] = target_unit
    result.attrs["unit"] = target_unit

    return result

# ---------- public API ----------

@public_api
def detect_unit(
    data: Union[mne.io.BaseRaw, mne.Epochs, TimeSeries],
    current_unit: Optional[str] = None,
) -> str:
    """Detect or validate the unit of EEG data.

    Parameters
    ----------
    data : mne.io.BaseRaw, mne.Epochs, or PTSA TimeSeries
        The data object to inspect.
    current_unit : str, optional
        If provided, overrides auto-detection. Validated against
        known units and returned directly.

    Returns
    -------
    str
        Unit string like "V", "mV", "uV", "nV", "T", etc.

    Raises
    ------
    ValueError
        If unit cannot be detected and current_unit is not provided.
    """
    if current_unit is not None:
        normalized = _normalize_unit(current_unit)
        if normalized not in _UNIT_EXPONENTS:
            raise ValueError(
                f"Unknown unit '{current_unit}'. "
                f"Known: {sorted(_UNIT_EXPONENTS.keys())}"
            )
        return normalized

    if isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        return _detect_unit_mne(data)

    if isinstance(data, TimeSeries):
        return _detect_unit_ptsa(data)

    raise TypeError(
        f"Cannot detect unit from {type(data).__name__}. "
        f"Expected mne.io.BaseRaw, mne.Epochs, or TimeSeries."
    )


@public_api
def get_scale_factor(from_unit: str, to_unit: str) -> float:
    """Compute multiplicative factor to convert between units.

    Parameters
    ----------
    from_unit : str
        Current unit (e.g. "V").
    to_unit : str
        Target unit (e.g. "uV").

    Returns
    -------
    float
        Multiply data by this value to convert.

    Examples
    --------
    >>> get_scale_factor("V", "uV")
    1000000.0
    >>> get_scale_factor("uV", "V")
    1e-06
    """
    from_u = _normalize_unit(from_unit)
    to_u = _normalize_unit(to_unit)

    if from_u not in _UNIT_EXPONENTS:
        raise ValueError(f"Unknown source unit '{from_unit}'")
    if to_u not in _UNIT_EXPONENTS:
        raise ValueError(f"Unknown target unit '{to_unit}'")

    from_base = from_u[-1]
    to_base = to_u[-1]
    if from_base != to_base:
        raise ValueError(
            f"Cannot convert between different base units: "
            f"'{from_unit}' ({from_base}) -> '{to_unit}' ({to_base})"
        )

    from_exp = _UNIT_EXPONENTS[from_u]
    to_exp = _UNIT_EXPONENTS[to_u]
    return 10.0 ** (from_exp - to_exp)


@public_api
def convert_unit(
    data: Union[mne.io.BaseRaw, mne.Epochs, TimeSeries],
    target: str,
    *,
    current_unit: Optional[str] = None,
    copy: bool = True,
) -> Union[mne.io.BaseRaw, mne.Epochs, TimeSeries]:
    """Convert EEG data to a target unit.

    Parameters
    ----------
    data : mne.io.BaseRaw, mne.Epochs, or PTSA TimeSeries
        The data to convert.
    target : str
        Target unit string (e.g. "uV", "mV", "V").
    current_unit : str, optional
        Override auto-detection of the current unit. Required if
        the data object doesn't store unit metadata.
    copy : bool
        If True (default), return a copy. If False, modify in place.

    Returns
    -------
    Same type as input, with data scaled to the target unit.
    """
    detected = detect_unit(data, current_unit=current_unit)
    target_normalized = _normalize_unit(target)
    factor = get_scale_factor(detected, target_normalized)

    if factor == 1.0:
        return data.copy() if copy else data

    if isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        return _convert_mne(data, factor, target_normalized, copy)

    if isinstance(data, TimeSeries):
        return _convert_ptsa(data, factor, target_normalized, copy)

    raise TypeError(f"Cannot convert type {type(data).__name__}")
