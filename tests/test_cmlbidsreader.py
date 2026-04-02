"""
Tests for bidsreader.cmlbidsreader

What is tested:
  Unit tests:
    - __init__: default root is CML_ROOT
    - _determine_device: LTP prefix -> eeg, R prefix -> ieeg, None -> None, unknown -> None
    - _determine_space: directory not found, no coordsystem file, multiple files, valid single
                        match, unparseable filename
    - _validate_acq: scalp returns None, ieeg+None raises, ieeg+valid passes, ieeg+invalid raises
    - _get_needed_fields: ieeg -> INTRACRANIAL_FIELDS, eeg -> SCALP_FIELDS
    - is_intracranial: ieeg -> True, eeg -> False

  Integration tests (skipped if data paths do not exist):
    iEEG (FR1, root=/data/LTP_BIDS/FR1):
      - Auto-detected device and space
      - load_events, load_electrodes, load_channels, load_combined_channels
      - load_coordsystem_desc, load_raw, load_epochs
      - get_subject_sessions, get_subject_tasks
    EEG (ValueCourier, root=/data/LTP_BIDS/ValueCourier):
      - Auto-detected device and space
      - load_events, load_electrodes, load_channels
      - load_raw, load_epochs
      - get_subject_sessions, get_subject_tasks, get_dataset_subjects
"""
import pytest
import pandas as pd
import mne
from pathlib import Path

from bidsreader.cmlbidsreader import CMLBIDSReader, CML_ROOT
from bidsreader.exc import (
    InvalidOptionError,
    FileNotFoundBIDSError,
    AmbiguousMatchError,
    DataParseError,
)

# ---------------------------------------------------------------------------
# Integration test paths and skip conditions
# ---------------------------------------------------------------------------
IEEG_ROOT = Path("/data/LTP_BIDS/FR1")
EEG_ROOT = Path("/data/LTP_BIDS/ValueCourier")

skip_no_ieeg = pytest.mark.skipif(
    not IEEG_ROOT.exists(), reason=f"iEEG data not available at {IEEG_ROOT}"
)
skip_no_eeg = pytest.mark.skipif(
    not EEG_ROOT.exists(), reason=f"EEG data not available at {EEG_ROOT}"
)

IEEG_SUBJECT = "R1001P"
IEEG_SESSION = "0"
IEEG_TASK = "FR1"

EEG_SUBJECT = "LTP606"
EEG_SESSION = "0"
EEG_TASK = "valuecourier"


@pytest.fixture
def ieeg_reader():
    return CMLBIDSReader(
        root=IEEG_ROOT, subject=IEEG_SUBJECT, task=IEEG_TASK, session=IEEG_SESSION,
    )


@pytest.fixture
def eeg_reader():
    return CMLBIDSReader(
        root=EEG_ROOT, subject=EEG_SUBJECT, task=EEG_TASK, session=EEG_SESSION,
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestCMLBIDSReaderInit:
    """Tests for CMLBIDSReader.__init__."""

    def test_default_root_is_cml_root(self):
        r = CMLBIDSReader(subject="R1001P", task="FR1", session="0", device="ieeg")
        assert str(r.root) == CML_ROOT

    def test_custom_root(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, subject="LTP001", task="FR1", session="0")
        assert r.root == tmp_root

    def test_invalid_device_raises(self, tmp_root):
        with pytest.raises(InvalidOptionError):
            CMLBIDSReader(root=tmp_root, device="magnetoencephalography")

    def test_valid_devices(self, tmp_root):
        for dev in ("eeg", "ieeg"):
            r = CMLBIDSReader(root=tmp_root, device=dev)
            assert r.device == dev


# ---------------------------------------------------------------------------
# _determine_device
# ---------------------------------------------------------------------------
class TestDetermineDevice:
    """Tests for CMLBIDSReader._determine_device."""

    def test_ltp_prefix_returns_eeg(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, subject="LTP001", task="FR1", session="0")
        assert r._determine_device() == "eeg"

    def test_r_prefix_returns_ieeg(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0")
        assert r._determine_device() == "ieeg"

    def test_none_subject_returns_none(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, task="FR1", session="0")
        assert r._determine_device() is None

    def test_unknown_prefix_returns_none(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, subject="XYZ001", task="FR1", session="0")
        assert r._determine_device() is None


# ---------------------------------------------------------------------------
# _determine_space
# ---------------------------------------------------------------------------
class TestDetermineSpace:
    """Tests for CMLBIDSReader._determine_space."""

    def _make_data_dir(self, tmp_root, subject="R1001P", session="0", device="ieeg"):
        """Helper: create the expected BIDS directory structure."""
        data_dir = tmp_root / f"sub-{subject}" / f"ses-{session}" / device
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def test_directory_not_found_raises(self, tmp_root):
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")
        with pytest.raises(FileNotFoundBIDSError, match="data directory does not exist"):
            r._determine_space()

    def test_no_coordsystem_file_raises(self, tmp_root):
        self._make_data_dir(tmp_root)
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")
        with pytest.raises(FileNotFoundBIDSError, match="no.*coordsystem.json"):
            r._determine_space()

    def test_multiple_coordsystem_files_raises(self, tmp_root):
        data_dir = self._make_data_dir(tmp_root)
        (data_dir / "sub-R1001P_ses-0_space-MNI_coordsystem.json").touch()
        (data_dir / "sub-R1001P_ses-0_space-TAL_coordsystem.json").touch()
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")
        with pytest.raises(AmbiguousMatchError, match="multiple coordsystem"):
            r._determine_space()

    def test_single_valid_match(self, tmp_root):
        data_dir = self._make_data_dir(tmp_root)
        (data_dir / "sub-R1001P_ses-0_space-MNI152NLin6ASym_coordsystem.json").touch()
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")
        assert r._determine_space() == "MNI152NLin6ASym"

    def test_unparseable_filename_raises(self, tmp_root):
        data_dir = self._make_data_dir(tmp_root)
        # No _space- token in filename
        (data_dir / "sub-R1001P_ses-0_coordsystem.json").touch()
        r = CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")
        with pytest.raises(DataParseError, match="could not parse space"):
            r._determine_space()


# ---------------------------------------------------------------------------
# _validate_acq
# ---------------------------------------------------------------------------
class TestValidateAcq:
    """Tests for CMLBIDSReader._validate_acq."""

    def test_scalp_returns_none(self, cml_reader_eeg):
        assert cml_reader_eeg._validate_acq("bipolar") is None

    def test_ieeg_none_acquisition_raises(self, cml_reader_ieeg):
        with pytest.raises(InvalidOptionError, match="acquisition is set to None"):
            cml_reader_ieeg._validate_acq(None)

    def test_ieeg_valid_acquisition(self, cml_reader_ieeg):
        assert cml_reader_ieeg._validate_acq("bipolar") == "bipolar"
        assert cml_reader_ieeg._validate_acq("monopolar") == "monopolar"

    def test_ieeg_invalid_acquisition_raises(self, cml_reader_ieeg):
        with pytest.raises(InvalidOptionError):
            cml_reader_ieeg._validate_acq("referential")


# ---------------------------------------------------------------------------
# _get_needed_fields
# ---------------------------------------------------------------------------
class TestGetNeededFields:
    """Tests for CMLBIDSReader._get_needed_fields."""

    def test_ieeg_returns_intracranial(self, cml_reader_ieeg):
        assert cml_reader_ieeg._get_needed_fields() == CMLBIDSReader.INTRACRANIAL_FIELDS

    def test_eeg_returns_scalp(self, cml_reader_eeg):
        assert cml_reader_eeg._get_needed_fields() == CMLBIDSReader.SCALP_FIELDS


# ---------------------------------------------------------------------------
# is_intracranial
# ---------------------------------------------------------------------------
class TestIsIntracranial:
    """Tests for CMLBIDSReader.is_intracranial."""

    def test_ieeg_is_intracranial(self, cml_reader_ieeg):
        assert cml_reader_ieeg.is_intracranial() is True

    def test_eeg_is_not_intracranial(self, cml_reader_eeg):
        assert cml_reader_eeg.is_intracranial() is False


# ===========================================================================
# Integration tests — real data (skipped if paths don't exist)
# ===========================================================================

@skip_no_ieeg
class TestIEEGIntegration:
    """Integration tests against real iEEG (FR1) data."""

    def test_auto_device(self, ieeg_reader):
        assert ieeg_reader.device == "ieeg"

    def test_is_intracranial(self, ieeg_reader):
        assert ieeg_reader.is_intracranial() is True

    def test_space_detected(self, ieeg_reader):
        assert ieeg_reader.space == "MNI152NLin6ASym"

    def test_load_events(self, ieeg_reader):
        df = ieeg_reader.load_events()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 756
        expected_cols = {
            "onset", "sample", "trial_type", "duration", "subject", "session",
            "experiment", "list", "serialpos", "item_name", "stim_file",
            "answer", "test", "response_time",
        }
        assert expected_cols.issubset(df.columns)

    def test_load_electrodes(self, ieeg_reader):
        df = ieeg_reader.load_electrodes()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 88
        for col in ("name", "x", "y", "z", "hemisphere", "type",
                     "ind.region", "stein.region", "das.region"):
            assert col in df.columns

    def test_load_channels_monopolar(self, ieeg_reader):
        df = ieeg_reader.load_channels("monopolar")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 88
        for col in ("name", "type", "units", "sampling_frequency"):
            assert col in df.columns

    def test_load_channels_bipolar(self, ieeg_reader):
        df = ieeg_reader.load_channels("bipolar")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 72
        for col in ("name", "type", "units", "reference", "sampling_frequency"):
            assert col in df.columns

    def test_load_combined_channels_monopolar(self, ieeg_reader):
        df = ieeg_reader.load_combined_channels("monopolar")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 88
        for col in ("name", "x", "y", "z", "type", "units",
                     "ind.region", "stein.region", "das.region"):
            assert col in df.columns

    def test_load_combined_channels_bipolar(self, ieeg_reader):
        df = ieeg_reader.load_combined_channels("bipolar")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 72
        for col in ("ch1", "ch2", "name", "type", "units",
                     "x_mid", "y_mid", "z_mid",
                     "ind.region_ch1", "ind.region_ch2",
                     "stein.region_ch1", "stein.region_ch2"):
            assert col in df.columns

    def test_load_coordsystem_desc(self, ieeg_reader):
        desc = ieeg_reader.load_coordsystem_desc()
        assert isinstance(desc, dict)
        assert desc == {
            "iEEGCoordinateSystem": desc["iEEGCoordinateSystem"],
            "iEEGCoordinateUnits": desc["iEEGCoordinateUnits"],
        }
        assert set(desc.keys()) == {"iEEGCoordinateSystem", "iEEGCoordinateUnits"}

    def test_load_raw(self, ieeg_reader):
        raw = ieeg_reader.load_raw("monopolar")
        assert isinstance(raw, mne.io.BaseRaw)
        assert len(raw.ch_names) == 88
        assert raw.info["sfreq"] == 500.0

    def test_load_epochs(self, ieeg_reader):
        epochs = ieeg_reader.load_epochs(
            tmin=-0.5, tmax=1.0, acquisition="monopolar",
        )
        assert isinstance(epochs, mne.Epochs)

    def test_get_subject_sessions(self, ieeg_reader):
        sessions = ieeg_reader.get_subject_sessions()
        assert sessions == ["0", "1"]

    def test_get_subject_tasks(self, ieeg_reader):
        tasks = ieeg_reader.get_subject_tasks()
        assert tasks == ["FR1"]


@skip_no_eeg
class TestEEGIntegration:
    """Integration tests against real EEG (VCBehOnly) data."""

    def test_auto_device(self, eeg_reader):
        assert eeg_reader.device == "eeg"

    def test_is_not_intracranial(self, eeg_reader):
        assert eeg_reader.is_intracranial() is False

    def test_space_detected(self, eeg_reader):
        assert eeg_reader.space == "CapTrak"

    def test_load_events(self, eeg_reader):
        df = eeg_reader.load_events()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 242
        for col in ("trial_type", "subject", "session", "experiment",
                     "recalled", "serialpos", "item", "stim_file"):
            assert col in df.columns

    def test_load_electrodes(self, eeg_reader):
        df = eeg_reader.load_electrodes()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 136
        for col in ("name", "x", "y", "z"):
            assert col in df.columns

    def test_load_channels(self, eeg_reader):
        df = eeg_reader.load_channels()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 137
        for col in ("name", "type", "units", "sampling_frequency", "status"):
            assert col in df.columns

    def test_load_raw(self, eeg_reader):
        raw = eeg_reader.load_raw()
        assert isinstance(raw, mne.io.BaseRaw)
        assert len(raw.ch_names) == 137
        assert raw.info["sfreq"] == 2048.0

    def test_load_epochs(self, eeg_reader):
        epochs = eeg_reader.load_epochs(tmin=-0.5, tmax=1.0)
        assert isinstance(epochs, mne.Epochs)

    def test_get_subject_sessions(self, eeg_reader):
        sessions = eeg_reader.get_subject_sessions()
        assert sessions == ["0", "1", "2", "3", "4", "5"]

    def test_get_subject_tasks(self, eeg_reader):
        tasks = eeg_reader.get_subject_tasks()
        assert tasks == ["valuecourier"]

    def test_get_dataset_subjects(self, eeg_reader):
        subjects = eeg_reader.get_dataset_subjects()
        assert sorted(subjects) == ["LTP606", "LTP607"]
