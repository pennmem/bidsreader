"""
Tests for bidsreader.convert

What is tested:
  - mne_raw_to_ptsa: basic conversion from MNE Raw to PTSA TimeSeries
  - mne_epochs_to_ptsa: basic conversion from MNE Epochs to PTSA TimeSeries

Note: These tests require both MNE and PTSA to be installed.
      Tests are skipped if PTSA is not available.
"""
import pytest
import numpy as np

mne = pytest.importorskip("mne")

try:
    from ptsa.data.timeseries import TimeSeries
    HAS_PTSA = True
except ImportError:
    HAS_PTSA = False


def _make_raw(n_channels=3, n_times=1000, sfreq=256.0):
    """Helper: create a minimal MNE RawArray for testing."""
    data = np.random.randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


@pytest.mark.skipif(not HAS_PTSA, reason="PTSA not installed")
class TestMneRawToPtsa:
    """Tests for mne_raw_to_ptsa."""

    def test_basic_conversion(self):
        from bidsreader.convert import mne_raw_to_ptsa

        raw = _make_raw(n_channels=2, n_times=500, sfreq=256.0)
        ts = mne_raw_to_ptsa(raw)

        assert isinstance(ts, TimeSeries)
        assert ts.shape == (2, 500)
        assert float(ts.samplerate) == 256.0

    def test_with_picks(self):
        from bidsreader.convert import mne_raw_to_ptsa

        raw = _make_raw(n_channels=3, n_times=500, sfreq=256.0)
        ts = mne_raw_to_ptsa(raw, picks=["EEG000", "EEG002"])

        assert ts.shape[0] == 2

    def test_with_crop(self):
        from bidsreader.convert import mne_raw_to_ptsa

        raw = _make_raw(n_channels=2, n_times=1000, sfreq=100.0)
        # 1000 samples at 100 Hz = 10 seconds; crop to [1.0, 3.0]
        ts = mne_raw_to_ptsa(raw, tmin=1.0, tmax=3.0)

        # Duration should be approximately 2 seconds
        duration = ts.coords["time"].values[-1] - ts.coords["time"].values[0]
        assert duration == pytest.approx(2.0, abs=0.02)


@pytest.mark.skipif(not HAS_PTSA, reason="PTSA not installed")
class TestMneEpochsToPtsa:
    """Tests for mne_epochs_to_ptsa."""

    def test_basic_conversion(self):
        from bidsreader.convert import mne_epochs_to_ptsa
        import pandas as pd

        raw = _make_raw(n_channels=2, n_times=2000, sfreq=256.0)
        events = np.array([[100, 0, 1], [600, 0, 1]])
        event_id = {"stim": 1}
        epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.5,
                            baseline=None, preload=True)

        events_df = pd.DataFrame({
            "sample": [100, 600],
            "trial_type": ["stim", "stim"],
        })

        ts = mne_epochs_to_ptsa(epochs, events_df)
        assert isinstance(ts, TimeSeries)
