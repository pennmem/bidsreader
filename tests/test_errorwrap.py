"""
Tests for bidsreader._errorwrap

What is tested:
  - public_api decorator passes through BIDSReaderError untouched
  - public_api wraps FileNotFoundError -> FileNotFoundBIDSError
  - public_api wraps json.JSONDecodeError -> DataParseError
  - public_api wraps KeyError -> DataParseError
  - public_api wraps ValueError -> ExternalLibraryError
  - public_api wraps generic Exception -> ExternalLibraryError
  - public_api preserves return value on success
  - public_api preserves function name via functools.wraps
"""
import json
import pytest

from bidsreader._errorwrap import public_api
from bidsreader.exc import (
    BIDSReaderError,
    InvalidOptionError,
    FileNotFoundBIDSError,
    DataParseError,
    ExternalLibraryError,
)


class TestPublicApiPassthrough:
    """Tests that BIDSReaderError subclasses are not re-wrapped."""

    def test_bids_reader_error_passes_through(self):
        @public_api
        def fn():
            raise InvalidOptionError("bad")

        with pytest.raises(InvalidOptionError):
            fn()

    def test_base_bids_reader_error_passes_through(self):
        @public_api
        def fn():
            raise BIDSReaderError("generic")

        with pytest.raises(BIDSReaderError):
            fn()


class TestPublicApiWrapping:
    """Tests that external exceptions are wrapped into the BIDS hierarchy."""

    def test_file_not_found_wrapped(self):
        @public_api
        def fn():
            raise FileNotFoundError("missing")

        with pytest.raises(FileNotFoundBIDSError):
            fn()

    def test_json_decode_error_wrapped(self):
        @public_api
        def fn():
            raise json.JSONDecodeError("bad json", "", 0)

        with pytest.raises(DataParseError):
            fn()

    def test_key_error_wrapped(self):
        @public_api
        def fn():
            raise KeyError("trial_type")

        with pytest.raises(DataParseError):
            fn()

    def test_value_error_wrapped(self):
        @public_api
        def fn():
            raise ValueError("something wrong")

        with pytest.raises(ExternalLibraryError):
            fn()

    def test_generic_exception_wrapped(self):
        @public_api
        def fn():
            raise RuntimeError("unexpected")

        with pytest.raises(ExternalLibraryError):
            fn()


class TestPublicApiSuccess:
    """Tests that the decorator does not interfere with normal returns."""

    def test_return_value_preserved(self):
        @public_api
        def fn(x):
            return x * 2

        assert fn(5) == 10

    def test_function_name_preserved(self):
        @public_api
        def my_function():
            pass

        assert my_function.__name__ == "my_function"
