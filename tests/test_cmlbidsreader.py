"""
Tests for bidsreader.cmlbidsreader

What is tested:
  - __init__: default root is CML_ROOT
  - _determine_device: LTP prefix -> eeg, R prefix -> ieeg, None -> None, unknown -> None
  - _determine_space: directory not found, no coordsystem file, multiple files, valid single
                      match, unparseable filename
  - _validate_acq: scalp returns None, ieeg+None raises, ieeg+valid passes, ieeg+invalid raises
  - _get_needed_fields: ieeg -> INTRACRANIAL_FIELDS, eeg -> SCALP_FIELDS
  - is_intracranial: ieeg -> True, eeg -> False
"""
import pytest
from pathlib import Path

from bidsreader.cmlbidsreader import CMLBIDSReader, CML_ROOT
from bidsreader.exc import (
    InvalidOptionError,
    FileNotFoundBIDSError,
    AmbiguousMatchError,
    DataParseError,
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
        with pytest.raises(InvalidOptionError, match="acquisition was not set"):
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
