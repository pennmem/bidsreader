"""
Tests for bidsreader.units

What is tested:
  - _normalize_unit: all non-ASCII character replacements (µ, μ, Ω, Ω, °)
  - _EXP_TO_PREFIX / _UNIT_EXPONENTS: full coverage of SI prefixes
  - get_scale_factor: standard conversions, new SI prefixes (da, h, d, c, pico),
    same unit, cross-base error, unknown unit error
  - detect_unit: current_unit override, normalization, unknown unit, unsupported type
  - _detect_unit_mne: detects V from EEG channel, handles non-EEG-only inst,
    unknown FIFF unit code, unit_mul exponents
  - _convert_mne: scales data, updates FIFF unit/unit_mul metadata
  - _is_timeseries: returns False without PTSA, True with PTSA TimeSeries
  - _detect_unit_ptsa / _convert_ptsa: skipped if PTSA not installed
  - convert_unit: full end-to-end with MNE Raw and Epochs
"""
import pytest

import numpy as np
import mne

from bidsreader.units import (
    _normalize_unit,
    _UNIT_EXPONENTS,
    _EXP_TO_PREFIX,
    _FIFF_UNIT_TO_BASE,
    _detect_unit_mne,
    _convert_mne,
    _is_timeseries,
    get_scale_factor,
    detect_unit,
    convert_unit,
)
from bidsreader.convert import mne_raw_to_ptsa
from bidsreader.exc import ExternalLibraryError


# ---------------------------------------------------------------------------
# _normalize_unit
# ---------------------------------------------------------------------------
class TestNormalizeUnit:
    """Tests for _normalize_unit — all non-ASCII replacements."""

    def test_micro_sign_u00b5(self):
        assert _normalize_unit("\u00b5V") == "uV"

    def test_greek_mu_u03bc(self):
        assert _normalize_unit("\u03bcV") == "uV"

    def test_greek_omega_u03a9(self):
        assert _normalize_unit("\u03a9") == "Ohm"

    def test_ohm_sign_u2126(self):
        assert _normalize_unit("\u2126") == "Ohm"

    def test_degree_sign(self):
        assert _normalize_unit("°C") == "degC"

    def test_already_ascii(self):
        assert _normalize_unit("uV") == "uV"

    def test_no_change_needed(self):
        assert _normalize_unit("V") == "V"

    def test_multiple_replacements(self):
        assert _normalize_unit("\u00b5\u03a9") == "uOhm"


# ---------------------------------------------------------------------------
# _UNIT_EXPONENTS / _EXP_TO_PREFIX coverage
# ---------------------------------------------------------------------------
class TestUnitConstants:
    """Tests that all SI prefixes are present and consistent."""

    @pytest.mark.parametrize("unit,exp", [
        # Volts — full range
        ("PV", 15), ("TV", 12), ("GV", 9), ("MV", 6), ("kV", 3),
        ("hV", 2), ("daV", 1),
        ("V", 0),
        ("dV", -1), ("cV", -2),
        ("mV", -3), ("uV", -6), ("nV", -9), ("pV", -12), ("fV", -15),
        # Tesla — full range
        ("PT", 15), ("TT", 12), ("GT", 9), ("MT", 6), ("kT", 3),
        ("hT", 2), ("daT", 1),
        ("T", 0),
        ("dT", -1), ("cT", -2),
        ("mT", -3), ("uT", -6), ("nT", -9), ("pT", -12), ("fT", -15),
    ])
    def test_unit_exponent_present(self, unit, exp):
        assert unit in _UNIT_EXPONENTS
        assert _UNIT_EXPONENTS[unit] == exp

    @pytest.mark.parametrize("exp", range(-15, 16))
    def test_exp_to_prefix_covers_all(self, exp):
        assert exp in _EXP_TO_PREFIX


# ---------------------------------------------------------------------------
# get_scale_factor
# ---------------------------------------------------------------------------
class TestGetScaleFactor:
    """Tests for get_scale_factor."""

    def test_v_to_uv(self):
        assert get_scale_factor("V", "uV") == pytest.approx(1e6)

    def test_uv_to_v(self):
        assert get_scale_factor("uV", "V") == pytest.approx(1e-6)

    def test_same_unit(self):
        assert get_scale_factor("uV", "uV") == pytest.approx(1.0)

    def test_mv_to_uv(self):
        assert get_scale_factor("mV", "uV") == pytest.approx(1e3)

    def test_tesla_units(self):
        assert get_scale_factor("T", "fT") == pytest.approx(1e15)

    # New SI prefix conversions
    def test_v_to_dav(self):
        assert get_scale_factor("V", "daV") == pytest.approx(1e-1)

    def test_hv_to_v(self):
        assert get_scale_factor("hV", "V") == pytest.approx(1e2)

    def test_dv_to_mv(self):
        assert get_scale_factor("dV", "mV") == pytest.approx(1e2)

    def test_cv_to_uv(self):
        assert get_scale_factor("cV", "uV") == pytest.approx(1e4)

    def test_pv_to_fv(self):
        assert get_scale_factor("pV", "fV") == pytest.approx(1e3)

    def test_gv_to_kv(self):
        assert get_scale_factor("GV", "kV") == pytest.approx(1e6)

    def test_pt_to_t(self):
        assert get_scale_factor("pT", "T") == pytest.approx(1e-12)

    def test_dat_to_ct(self):
        assert get_scale_factor("daT", "cT") == pytest.approx(1e3)

    # Error cases
    def test_cross_base_raises(self):
        with pytest.raises(ExternalLibraryError, match="Cannot convert between different base units:"):
            get_scale_factor("V", "T")

    def test_cross_base_multiple_letters_raises(self):
        with pytest.raises(ExternalLibraryError, match="Cannot convert between different base units:"):
            get_scale_factor("uV", "pT")

    def test_unknown_source_empty_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown source unit"):
            get_scale_factor("", "V")
    
    def test_unknown_target_empty_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown target unit"):
            get_scale_factor("V", "")

    def test_unknown_source_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown source unit"):
            get_scale_factor("kW", "V")

    def test_unknown_both_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown source unit"):
            get_scale_factor("kW", "kW")

    def test_unknown_target_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown target unit"):
            get_scale_factor("V", "kW")


# ---------------------------------------------------------------------------
# detect_unit
# ---------------------------------------------------------------------------
class TestDetectUnit:
    """Tests for detect_unit."""

    def test_current_unit_override(self):
        result = detect_unit(None, current_unit="uV")
        assert result == "uV"

    def test_current_unit_normalizes_micro_sign(self):
        result = detect_unit(None, current_unit="\u00b5V")
        assert result == "uV"

    def test_current_unit_normalizes_greek_mu(self):
        result = detect_unit(None, current_unit="\u03bcV")
        assert result == "uV"

    def test_current_unit_new_prefixes(self):
        for unit in ("daV", "hV", "dV", "cV", "pV"):
            result = detect_unit(None, current_unit=unit)
            assert result == unit

    def test_unknown_current_unit_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown unit"):
            detect_unit(None, current_unit="kW")

    def test_unsupported_type_raises(self):
        with pytest.raises(ExternalLibraryError, match="Cannot detect unit"):
            detect_unit("not_a_data_object")


# ---------------------------------------------------------------------------
# _detect_unit_mne
# ---------------------------------------------------------------------------
class TestDetectUnitMne:
    """Tests for _detect_unit_mne — reads FIFF unit metadata from MNE objects."""

    def test_detects_v_from_eeg_raw(self):
        raw = _make_raw()
        assert _detect_unit_mne(raw) == "V"

    def test_detects_v_from_epochs(self):
        raw = _make_raw(n_channels=2, n_times=1000, sfreq=256.0)
        events = np.array([[50, 0, 1], [300, 0, 1]])
        epochs = mne.Epochs(raw, events, {"s": 1}, tmin=0.0, tmax=0.1,
                            baseline=None, preload=True)
        assert _detect_unit_mne(epochs) == "V"

    def test_detects_unit_mul_minus6(self):
        raw = _make_raw()
        for ch in raw.info["chs"]:
            ch["unit_mul"] = -6
        assert _detect_unit_mne(raw) == "uV"

    def test_detects_unit_mul_minus3(self):
        raw = _make_raw()
        for ch in raw.info["chs"]:
            ch["unit_mul"] = -3
        assert _detect_unit_mne(raw) == "mV"

    def test_detects_unit_mul_minus9(self):
        raw = _make_raw()
        for ch in raw.info["chs"]:
            ch["unit_mul"] = -9
        assert _detect_unit_mne(raw) == "nV"

    def test_unknown_fiff_unit_code_raises(self):
        raw = _make_raw()
        for ch in raw.info["chs"]:
            ch["unit"] = 999
        with pytest.raises(ValueError, match="Unknown FIFF unit code"):
            _detect_unit_mne(raw)

    def test_no_eeg_channels_raises(self):
        # Create raw with only misc channels
        data = np.ones((2, 100))
        info = mne.create_info(["MISC1", "MISC2"], sfreq=256.0, ch_types="misc")
        raw = mne.io.RawArray(data, info)
        with pytest.raises(ValueError, match="No EEG/iEEG/SEEG/ECoG channel found"):
            _detect_unit_mne(raw)

    def test_skips_non_eeg_channels(self):
        # Mix of misc + eeg: should skip misc and detect from eeg
        data = np.ones((3, 100))
        info = mne.create_info(["MISC1", "EEG001", "EEG002"], sfreq=256.0,
                               ch_types=["misc", "eeg", "eeg"])
        raw = mne.io.RawArray(data, info)
        assert _detect_unit_mne(raw) == "V"

    def test_tesla_unit_code(self):
        raw = _make_raw()
        for ch in raw.info["chs"]:
            ch["unit"] = 201  # Tesla
            ch["unit_mul"] = -15
        assert _detect_unit_mne(raw) == "fT"


# ---------------------------------------------------------------------------
# _convert_mne
# ---------------------------------------------------------------------------
class TestConvertMne:
    """Tests for _convert_mne — scales data and updates FIFF metadata."""

    def test_scales_data(self):
        raw = _make_raw()
        original = raw.get_data().copy()
        result = _convert_mne(raw, 1e6, "uV", copy=True)
        np.testing.assert_allclose(result.get_data(), original * 1e6)

    def test_copy_true_returns_new_object(self):
        raw = _make_raw()
        result = _convert_mne(raw, 2.0, "mV", copy=True)
        assert result is not raw

    def test_copy_false_modifies_in_place(self):
        raw = _make_raw()
        result = _convert_mne(raw, 2.0, "mV", copy=False)
        assert result is raw

    def test_updates_fiff_unit_code_volts(self):
        raw = _make_raw()
        result = _convert_mne(raw, 1e6, "uV", copy=True)
        for ch in result.info["chs"]:
            assert ch["unit"] == 107  # FIFF code for Volts

    def test_updates_fiff_unit_mul(self):
        raw = _make_raw()
        result = _convert_mne(raw, 1e6, "uV", copy=True)
        for ch in result.info["chs"]:
            assert ch["unit_mul"] == -6

    def test_updates_fiff_unit_mul_mv(self):
        raw = _make_raw()
        result = _convert_mne(raw, 1e3, "mV", copy=True)
        for ch in result.info["chs"]:
            assert ch["unit_mul"] == -3

    def test_updates_fiff_unit_mul_nv(self):
        raw = _make_raw()
        result = _convert_mne(raw, 1e9, "nV", copy=True)
        for ch in result.info["chs"]:
            assert ch["unit_mul"] == -9


# ---------------------------------------------------------------------------
# _is_timeseries
# ---------------------------------------------------------------------------
class TestIsTimeseries:
    """Tests for _is_timeseries — lazy PTSA isinstance check."""

    def test_returns_false_for_non_timeseries(self):
        assert _is_timeseries("hello") is False
        assert _is_timeseries(42) is False
        assert _is_timeseries(None) is False
        assert _is_timeseries(np.array([1, 2, 3])) is False

    def test_returns_false_for_mne_raw(self):
        raw = _make_raw()
        assert _is_timeseries(raw) is False


# ---------------------------------------------------------------------------
# _detect_unit_ptsa / _convert_ptsa (skipped without PTSA)
# ---------------------------------------------------------------------------
class TestPtsaFunctions:
    """Tests for PTSA-dependent internal functions. Skipped if PTSA not installed."""

    @pytest.fixture(autouse=True)
    def _skip_without_ptsa(self):
        pytest.importorskip("ptsa", reason="PTSA not installed")

    def _make_timeseries(self, unit_val="uV"):
        from ptsa.data.timeseries import TimeSeries
        ts = TimeSeries.create(
            np.ones((2, 10)),
            samplerate=256.0,
            dims=("channel", "time"),
            coords={
                "channel": np.array(["ch0", "ch1"], dtype=object),
                "time": np.arange(10, dtype=float),
            },
        )
        ts.attrs["units"] = unit_val
        ts.attrs["unit"] = unit_val
        return ts

    # _detect_unit_ptsa
    def test_detect_unit_ptsa_reads_units_attr(self):
        from bidsreader.units import _detect_unit_ptsa
        ts = self._make_timeseries("uV")
        assert _detect_unit_ptsa(ts) == "uV"

    def test_detect_unit_ptsa_normalizes_micro(self):
        from bidsreader.units import _detect_unit_ptsa
        ts = self._make_timeseries("\u00b5V")
        assert _detect_unit_ptsa(ts) == "uV"

    def test_detect_unit_ptsa_unrecognized_raises(self):
        from bidsreader.units import _detect_unit_ptsa
        ts = self._make_timeseries("kW")
        with pytest.raises(ValueError, match="not recognized"):
            _detect_unit_ptsa(ts)

    def test_detect_unit_ptsa_no_attr_raises(self):
        from bidsreader.units import _detect_unit_ptsa
        ts = self._make_timeseries("uV")
        del ts.attrs["units"]
        del ts.attrs["unit"]
        with pytest.raises(ValueError, match="no 'units' or 'unit' attribute"):
            _detect_unit_ptsa(ts)

    # _convert_ptsa
    def test_convert_ptsa_copy_true(self):
        from bidsreader.units import _convert_ptsa
        ts = self._make_timeseries("V")
        result = _convert_ptsa(ts, 1e6, "uV", copy=True)
        np.testing.assert_allclose(result.values, ts.values * 1e6)
        assert result.attrs["units"] == "uV"
        assert result.attrs["unit"] == "uV"

    def test_convert_ptsa_copy_false(self):
        from bidsreader.units import _convert_ptsa
        ts = self._make_timeseries("V")
        original_values = ts.values.copy()
        result = _convert_ptsa(ts, 1e6, "uV", copy=False)
        assert result is ts
        np.testing.assert_allclose(ts.values, original_values * 1e6)
        assert ts.attrs["units"] == "uV"

    # _is_timeseries
    def test_is_timeseries_true_for_ptsa(self):
        ts = self._make_timeseries("uV")
        assert _is_timeseries(ts) is True

    # convert_unit end-to-end with PTSA
    def test_convert_unit_ptsa(self):
        ts = self._make_timeseries("V")
        ts.attrs["units"] = "V"
        ts.attrs["unit"] = "V"
        result = convert_unit(ts, "uV", current_unit="V", copy=True)
        np.testing.assert_allclose(result.values, ts.values * 1e6)
        assert result.attrs["units"] == "uV"


# ---------------------------------------------------------------------------
# Helper: create a minimal MNE RawArray
# ---------------------------------------------------------------------------
def _make_raw(n_channels=2, n_times=100, sfreq=256.0):
    """Create a minimal MNE RawArray with EEG channels for testing."""
    data = np.ones((n_channels, n_times)) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)

# ---------------------------------------------------------------------------
# convert_unit
# ---------------------------------------------------------------------------
class TestConvertUnit:
    """Tests for convert_unit with MNE Raw objects."""

    def test_same_unit_copy_true(self):
        raw = _make_raw()
        result = convert_unit(raw, "V", current_unit="V", copy=True)
        assert result is not raw
        np.testing.assert_array_equal(result.get_data(), raw.get_data())

    def test_same_unit_copy_false(self):
        raw = _make_raw()
        result = convert_unit(raw, "V", current_unit="V", copy=False)
        assert result is raw

    def test_v_to_uv_scales_data(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        result = convert_unit(raw, "uV", current_unit="V", copy=True)
        np.testing.assert_allclose(result.get_data(), original_data * 1e6)

    def test_uv_to_mv_scales_data(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        result = convert_unit(raw, "mV", current_unit="uV", copy=True)
        np.testing.assert_allclose(result.get_data(), original_data * 1e-3)

    def test_copy_true_does_not_modify_original(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        convert_unit(raw, "uV", current_unit="V", copy=True)
        np.testing.assert_array_equal(raw.get_data(), original_data)

    def test_copy_false_modifies_in_place(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        result = convert_unit(raw, "uV", current_unit="V", copy=False)
        assert result is raw
        np.testing.assert_allclose(raw.get_data(), original_data * 1e6)

    def test_normalizes_target_micro_sign(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        result = convert_unit(raw, "\u00b5V", current_unit="V", copy=True)
        np.testing.assert_allclose(result.get_data(), original_data * 1e6)

    def test_new_prefix_hv_to_mv(self):
        raw = _make_raw()
        original_data = raw.get_data().copy()
        result = convert_unit(raw, "mV", current_unit="hV", copy=True)
        # hV (10^2) -> mV (10^-3) = factor of 10^5
        np.testing.assert_allclose(result.get_data(), original_data * 1e5)

    def test_unsupported_type_raises(self):
        with pytest.raises(ExternalLibraryError):
            convert_unit("not_data", "uV", current_unit="V")

    def test_with_mne_epochs(self):
        raw = _make_raw(n_channels=2, n_times=1000, sfreq=256.0)
        events = np.array([[50, 0, 1], [300, 0, 1]])
        event_id = {"stim": 1}
        epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.1,
                            baseline=None, preload=True)
        original_data = epochs.get_data().copy()
        result = convert_unit(epochs, "uV", current_unit="V", copy=True)
        assert isinstance(result, mne.Epochs)
        np.testing.assert_allclose(result.get_data(), original_data * 1e6)
    
    def test_wrong_data_np_type(self):
        with pytest.raises(ExternalLibraryError):
            convert_unit(np.array([1.0, 2.0]), "uV", current_unit="V")

    def test_wrong_data_list_type(self):
        with pytest.raises(ExternalLibraryError):
            convert_unit([1.0, 2.0], "uV", current_unit="V")

    def test_wrong_data_dict_type(self):
        with pytest.raises(ExternalLibraryError):
            convert_unit({"data": 1.0}, "uV", current_unit="V")

    def test_wrong_data_none_type(self):
        with pytest.raises(ExternalLibraryError):
            convert_unit(None, "uV", current_unit="V")

# ---------------------------------------------------------------------------
# convert_unit with PTSA TimeSeries (skipped without PTSA)
# ---------------------------------------------------------------------------
class TestConvertUnitTimeSeries:
    """Tests for convert_unit with PTSA TimeSeries objects."""

    @pytest.fixture(autouse=True)
    def _skip_without_ptsa(self):
        pytest.importorskip("ptsa", reason="PTSA not installed")

    def _make_ts(self, unit="V"):
        from ptsa.data.timeseries import TimeSeries
        ts = TimeSeries.create(
            np.ones((2, 100)) * 1e-6,
            samplerate=256.0,
            dims=("channel", "time"),
            coords={
                "channel": np.array(["ch0", "ch1"], dtype=object),
                "time": np.arange(100, dtype=float),
            },
        )
        ts.attrs["units"] = unit
        ts.attrs["unit"] = unit
        return ts

    def test_v_to_uv(self):
        ts = self._make_ts("V")
        original = ts.values.copy()
        result = convert_unit(ts, "uV", current_unit="V", copy=True)
        np.testing.assert_allclose(result.values, original * 1e6)
        assert result.attrs["units"] == "uV"
        assert result.attrs["unit"] == "uV"

    def test_uv_to_mv(self):
        ts = self._make_ts("uV")
        original = ts.values.copy()
        result = convert_unit(ts, "mV", current_unit="uV", copy=True)
        np.testing.assert_allclose(result.values, original * 1e-3)
        assert result.attrs["units"] == "mV"

    def test_same_unit_copy_true(self):
        ts = self._make_ts("uV")
        result = convert_unit(ts, "uV", current_unit="uV", copy=True)
        np.testing.assert_array_equal(result.values, ts.values)
        assert result is not ts

    def test_same_unit_copy_false(self):
        ts = self._make_ts("uV")
        result = convert_unit(ts, "uV", current_unit="uV", copy=False)
        assert result is ts

    def test_copy_true_does_not_modify_original(self):
        ts = self._make_ts("V")
        original = ts.values.copy()
        convert_unit(ts, "uV", current_unit="V", copy=True)
        np.testing.assert_array_equal(ts.values, original)

    def test_copy_false_modifies_in_place(self):
        ts = self._make_ts("V")
        original = ts.values.copy()
        result = convert_unit(ts, "uV", current_unit="V", copy=False)
        assert result is ts
        np.testing.assert_allclose(ts.values, original * 1e6)
        assert ts.attrs["units"] == "uV"

    def test_auto_detect_unit_from_attrs(self):
        ts = self._make_ts("mV")
        original = ts.values.copy()
        result = convert_unit(ts, "uV", copy=True)
        np.testing.assert_allclose(result.values, original * 1e3)
        assert result.attrs["units"] == "uV"

    def test_new_prefix_hv_to_cv(self):
        ts = self._make_ts("hV")
        original = ts.values.copy()
        result = convert_unit(ts, "cV", current_unit="hV", copy=True)
        # hV (10^2) -> cV (10^-2) = factor of 10^4
        np.testing.assert_allclose(result.values, original * 1e4)
        assert result.attrs["units"] == "cV"

    def test_normalizes_micro_sign_target(self):
        ts = self._make_ts("V")
        original = ts.values.copy()
        result = convert_unit(ts, "\u00b5V", current_unit="V", copy=True)
        np.testing.assert_allclose(result.values, original * 1e6)
        assert result.attrs["units"] == "uV"
