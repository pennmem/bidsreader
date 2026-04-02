"""
Tests for bidsreader.exc

What is tested:
  - All exception classes exist
  - BIDSReaderError is the base for the entire hierarchy
  - Multi-inheritance is correct (e.g. InvalidOptionError is both BIDSReaderError and ValueError)
  - Each exception can be raised and caught by its parent
"""
import pytest

from bidsreader.exc import (
    BIDSReaderError,
    InvalidOptionError,
    MissingRequiredFieldError,
    FileNotFoundBIDSError,
    AmbiguousMatchError,
    DataParseError,
    DependencyError,
    ExternalLibraryError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    @pytest.mark.parametrize("exc_cls", [
        InvalidOptionError,
        MissingRequiredFieldError,
        FileNotFoundBIDSError,
        AmbiguousMatchError,
        DataParseError,
        DependencyError,
        ExternalLibraryError,
    ])
    def test_all_inherit_from_bids_reader_error(self, exc_cls):
        assert issubclass(exc_cls, BIDSReaderError)

    def test_bids_reader_error_is_exception(self):
        assert issubclass(BIDSReaderError, Exception)


class TestMultiInheritance:
    """Tests for dual-inheritance on specific exception classes."""

    def test_invalid_option_is_value_error(self):
        assert issubclass(InvalidOptionError, ValueError)

    def test_missing_required_field_is_value_error(self):
        assert issubclass(MissingRequiredFieldError, ValueError)

    def test_file_not_found_bids_is_file_not_found(self):
        assert issubclass(FileNotFoundBIDSError, FileNotFoundError)


class TestExceptionsCanBeRaised:
    """Tests that each exception can be raised and caught by parent."""

    @pytest.mark.parametrize("exc_cls", [
        InvalidOptionError,
        MissingRequiredFieldError,
        FileNotFoundBIDSError,
        AmbiguousMatchError,
        DataParseError,
        DependencyError,
        ExternalLibraryError,
    ])
    def test_catch_as_bids_reader_error(self, exc_cls):
        with pytest.raises(BIDSReaderError):
            raise exc_cls("test message")

    def test_invalid_option_caught_as_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidOptionError("bad option")

    def test_file_not_found_caught_as_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            raise FileNotFoundBIDSError("missing file")
