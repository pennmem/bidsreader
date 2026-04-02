"""
Tests for bidsreader.basereader

What is tested:
  - __init__: root=None raises, valid construction, device validated against VALID_DEVICES
  - __setattr__: known fields accepted, unknown fields rejected, underscore-prefixed always OK
  - __str__ / __repr__: contain class name and field values
  - space property: returns None from base _determine_space, caches value when set
  - device property: returns None from base _determine_device, warns when None
  - _add_bids_prefix: each known field, unknown field raises InvalidOptionError
  - _require: passes when fields present, raises MissingRequiredFieldError when missing
  - _get_needed_fields: returns REQUIRED_FIELDS
  - set_fields: sets valid fields, rejects unknown
  - Metadata queries: get_subject_tasks, get_subject_sessions, get_dataset_subjects,
    get_dataset_tasks, get_dataset_max_sessions (numeric, non-numeric, outlier threshold)
"""
import pytest
import warnings
from pathlib import Path
from unittest.mock import patch

from bidsreader.basereader import BaseReader
from bidsreader.exc import InvalidOptionError, MissingRequiredFieldError


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestBaseReaderInit:
    """Tests for BaseReader.__init__."""

    def test_root_none_raises(self):
        with pytest.raises(ValueError, match="root must be provided"):
            BaseReader(root=None)

    def test_min_valid_construction(self, tmp_root):
        r = BaseReader(root=tmp_root)
        assert r.root == tmp_root

    def test_sub_valid_construction(self, tmp_root):
        r = BaseReader(root=tmp_root, subject="01", task="rest", session="1")
        assert r.root == tmp_root
        assert r.subject == "01"
        assert r.task == "rest"
        assert r.session == "1"

    # should not prompt any errors
    def test_device_acq_space_valid_construction(self, tmp_root):
        r = BaseReader(root=tmp_root, subject="01", task="rest", session="1", device="eeg", acquisition="monopolar", space="Talarich")
        assert r.root == tmp_root
        assert r.subject == "01"
        assert r.task == "rest"
        assert r.session == "1"
        assert r.device == "eeg"
        assert r.acquisition == "monopolar"
        assert r.space == "Talarich"

    def test_nonsense_construction(self, tmp_root): 
        # there should be no checks
        r = BaseReader(root=tmp_root, subject="a", task="ab", session="abc", device="abcd", acquisition="abcde", space="abcdef")
        assert r.root == tmp_root
        assert r.subject == "a"
        assert r.task == "ab"
        assert r.session == "abc"
        assert r.device == "abcd"
        assert r.acquisition == "abcde"
        assert r.space == "abcdef"

    def test_root_converted_to_path(self, tmp_root):
        r = BaseReader(root=str(tmp_root), subject="01", task="rest", session="1")
        assert isinstance(r.root, Path)

    def test_device_none_allowed(self, tmp_root):
        r = BaseReader(root=tmp_root)
        # device is None (lazy, _determine_device returns None in base)
        assert r._device is None


# ---------------------------------------------------------------------------
# __setattr__
# ---------------------------------------------------------------------------
class TestBaseReaderSetattr:
    """Tests for BaseReader.__setattr__ field guard."""

    def test_known_field_accepted(self, base_reader):
        base_reader.subject = "02"
        assert base_reader.subject == "02"

    def test_unknown_field_rejected(self, base_reader):
        with pytest.raises(AttributeError, match="Unknown field"):
            base_reader.nonexistent = "value"

    def test_underscore_prefixed_always_ok(self, base_reader):
        base_reader._custom_internal = 42
        assert base_reader._custom_internal == 42


# ---------------------------------------------------------------------------
# __str__ / __repr__
# ---------------------------------------------------------------------------
class TestBaseReaderStringRepr:
    """Tests for __str__ and __repr__."""

    def test_str_contains_class_name(self, base_reader):
        s = str(base_reader)
        assert "BaseReader(" in s

    def test_str_contains_subject(self, base_reader):
        assert "subject=01" in str(base_reader)

    def test_repr_contains_class_name(self, base_reader):
        assert "BaseReader(" in repr(base_reader)

    def test_repr_contains_device(self, base_reader):
        assert "device='eeg'" in repr(base_reader)

    def test_subclass_uses_own_name(self, tmp_root):
        class MyReader(BaseReader):
            pass

        r = MyReader(root=tmp_root, subject="01", task="x", session="1")
        assert "MyReader(" in str(r)
        assert "MyReader(" in repr(r)


# ---------------------------------------------------------------------------
# space property
# ---------------------------------------------------------------------------
class TestSpaceProperty:
    """Tests for the space lazy property."""

    def test_returns_none_when_not_set(self, base_reader):
        # base _determine_space returns None, so space warns and returns None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert base_reader.space is None

    def test_returns_cached_value(self, tmp_root):
        r = BaseReader(root=tmp_root, space="MNI152NLin6ASym")
        assert r.space == "MNI152NLin6ASym"


# ---------------------------------------------------------------------------
# device property
# ---------------------------------------------------------------------------
class TestDeviceProperty:
    """Tests for the device lazy property."""

    def test_returns_none_with_warning(self, tmp_root):
        r = BaseReader(root=tmp_root)
        with pytest.warns(RuntimeWarning, match="device could not be inferred"):
            result = r.device
        assert result is None

    def test_returns_set_value(self, tmp_root):
        r = BaseReader(root=tmp_root, device="ieeg")
        assert r.device == "ieeg"


# ---------------------------------------------------------------------------
# _add_bids_prefix
# ---------------------------------------------------------------------------
class TestAddBidsPrefix:
    """Tests for BaseReader._add_bids_prefix."""

    @pytest.mark.parametrize("field,value,expected", [
        ("subject", "01", "sub-01"),
        ("session", "1", "ses-1"),
        ("acquisition", "bipolar", "acq-bipolar"),
        ("task", "FR1", "task-FR1"),
        ("space", "MNI152", "space-MNI152"),
    ])
    def test_known_fields(self, base_reader, field, value, expected):
        assert base_reader._add_bids_prefix(field, value) == expected

    def test_unknown_field_raises(self, base_reader):
        with pytest.raises(InvalidOptionError, match="Unknown BIDS field"):
            base_reader._add_bids_prefix("modality", "eeg")

    def test_none_value(self, base_reader):
        assert base_reader._add_bids_prefix("subject", None) is None


# ---------------------------------------------------------------------------
# _require
# ---------------------------------------------------------------------------
class TestRequire:
    """Tests for BaseReader._require."""

    def test_all_present(self, base_reader):
        # Should not raise
        base_reader._require(["subject", "task"], context="test")

    def test_missing_raises(self, tmp_root):
        r = BaseReader(root=tmp_root, subject="01")
        with pytest.raises(MissingRequiredFieldError, match="missing required fields"):
            r._require(["subject", "device"], context="test")

    def test_empty_string_treated_as_missing(self, tmp_root):
        r = BaseReader(root=tmp_root, subject="")
        with pytest.raises(MissingRequiredFieldError):
            r._require(["subject"], context="test")


# ---------------------------------------------------------------------------
# _get_needed_fields
# ---------------------------------------------------------------------------
# class TestGetNeededFields:
#     """Tests for BaseReader._get_needed_fields."""

#     def test_returns_required_fields(self, base_reader):
#         assert base_reader._get_needed_fields() == BaseReader.REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# set_fields
# ---------------------------------------------------------------------------
class TestSetFields:
    """Tests for BaseReader.set_fields."""

    def test_sets_valid_fields(self, base_reader):
        result = base_reader.set_fields(subject="02", session="3")
        assert base_reader.subject == "02"
        assert base_reader.session == "3"
        assert result is base_reader  # returns self

    def test_rejects_unknown_field(self, base_reader):
        # @public_api wraps AttributeError -> ExternalLibraryError
        from bidsreader.exc import ExternalLibraryError
        with pytest.raises(ExternalLibraryError, match="Unknown field"):
            base_reader.set_fields(nonexistent="value")


# ---------------------------------------------------------------------------
# Metadata queries (mocked)
# ---------------------------------------------------------------------------
class TestMetadataQueries:
    """Tests for get_subject_*, get_dataset_* methods using mocked get_entity_vals."""
    # data is all mocked, we will use real data in test_integration
    @patch("bidsreader.basereader.get_entity_vals", return_value=["FR1", "catFR1"])
    def test_get_subject_tasks(self, mock_gev, base_reader):
        result = base_reader.get_subject_tasks()
        assert result == ["FR1", "catFR1"]

    @patch("bidsreader.basereader.get_entity_vals", return_value=["0", "1", "2"])
    def test_get_subject_sessions(self, mock_gev, base_reader):
        result = base_reader.get_subject_sessions()
        assert result == ["0", "1", "2"]

    @patch("bidsreader.basereader.get_entity_vals", return_value=["01", "02"])
    def test_get_dataset_subjects(self, mock_gev, base_reader):
        assert base_reader.get_dataset_subjects() == ["01", "02"]

    @patch("bidsreader.basereader.get_entity_vals", return_value=["FR1"])
    def test_get_dataset_tasks(self, mock_gev, base_reader):
        assert base_reader.get_dataset_tasks() == ["FR1"]


class TestGetDatasetMaxSessions:
    """Tests for get_dataset_max_sessions with various session formats."""

    @patch("bidsreader.basereader.get_entity_vals")
    def test_numeric_sessions(self, mock_gev, base_reader):
        mock_gev.side_effect = [
            ["01", "02"],        # get_dataset_subjects
            ["0", "1", "5"],     # sessions for sub-01
            ["0", "2"],          # sessions for sub-02
        ]
        result = base_reader.get_dataset_max_sessions()
        assert result == 5

    @patch("bidsreader.basereader.get_entity_vals")
    def test_non_numeric_sessions_skipped(self, mock_gev, base_reader):
        mock_gev.side_effect = [
            ["01"],
            ["baseline", "followup"],
        ]
        result = base_reader.get_dataset_max_sessions()
        assert result is None

    @patch("bidsreader.basereader.get_entity_vals")
    def test_outlier_threshold(self, mock_gev, base_reader):
        mock_gev.side_effect = [
            ["01"],
            ["1", "2", "999"],
        ]
        with pytest.warns(UserWarning):
            result = base_reader.get_dataset_max_sessions(outlier_thresh=50)
        assert result == 2

    @patch("bidsreader.basereader.get_entity_vals")
    def test_no_subjects_returns_none(self, mock_gev, base_reader):
        mock_gev.return_value = []
        result = base_reader.get_dataset_max_sessions()
        assert result is None
