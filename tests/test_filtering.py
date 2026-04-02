"""
Tests for bidsreader.filtering

What is tested:
  - _label_has_trial_type: exact match, slash-separated, no match
  - _ensure_list: string input -> single-element list, list input -> same list
  - filter_events_df_by_trial_types: correct filtering and index array
  - filter_by_trial_types: df-only path works, no inputs returns empty
"""
import pytest
import numpy as np
import pandas as pd

from bidsreader.filtering import (
    _label_has_trial_type,
    _ensure_list,
    filter_events_df_by_trial_types,
    filter_by_trial_types,
)


# ---------------------------------------------------------------------------
# _label_has_trial_type
# ---------------------------------------------------------------------------
class TestLabelHasTrialType:
    """Tests for _label_has_trial_type."""

    def test_exact_match(self):
        assert _label_has_trial_type("WORD", ["WORD"]) is True

    def test_slash_separated(self):
        assert _label_has_trial_type("WORD/STIM", ["STIM"]) is True

    def test_no_match(self):
        assert _label_has_trial_type("WORD", ["STIM"]) is False

    def test_partial_token_no_match(self):
        assert _label_has_trial_type("WORD", ["WOR"]) is False

    def test_multiple_trial_types(self):
        assert _label_has_trial_type("REST", ["WORD", "REST"]) is True

    def test_empty_trial_types(self):
        assert _label_has_trial_type("WORD", []) is False


# ---------------------------------------------------------------------------
# _ensure_list
# ---------------------------------------------------------------------------
class TestEnsureList:
    """Tests for _ensure_list."""

    def test_string_becomes_list(self):
        assert _ensure_list("WORD") == ["WORD"]

    def test_list_stays_list(self):
        assert _ensure_list(["WORD", "STIM"]) == ["WORD", "STIM"]

    def test_tuple_becomes_list(self):
        assert _ensure_list(("A", "B")) == ["A", "B"]


# ---------------------------------------------------------------------------
# filter_events_df_by_trial_types
# ---------------------------------------------------------------------------
class TestFilterEventsDfByTrialTypes:
    """Tests for filter_events_df_by_trial_types."""

    def test_filters_correct_rows(self, sample_events_df):
        filtered_df, idx = filter_events_df_by_trial_types(sample_events_df, "WORD")
        assert len(filtered_df) == 3
        assert list(filtered_df["trial_type"].unique()) == ["WORD"]

    def test_returns_correct_indices(self, sample_events_df):
        filtered_df, idx = filter_events_df_by_trial_types(sample_events_df, "STIM")
        np.testing.assert_array_equal(idx, [2, 4])

    def test_multiple_trial_types(self, sample_events_df):
        filtered_df, idx = filter_events_df_by_trial_types(sample_events_df, ["WORD", "STIM"])
        assert len(filtered_df) == 5  # all rows

    def test_no_match_returns_empty(self, sample_events_df):
        filtered_df, idx = filter_events_df_by_trial_types(sample_events_df, "NONEXISTENT")
        assert len(filtered_df) == 0
        assert len(idx) == 0

    def test_string_input(self, sample_events_df):
        filtered_df, idx = filter_events_df_by_trial_types(sample_events_df, "WORD")
        assert len(filtered_df) == 3


# ---------------------------------------------------------------------------
# filter_by_trial_types
# ---------------------------------------------------------------------------
class TestFilterByTrialTypes:
    """Tests for filter_by_trial_types (combined dispatcher)."""

    def test_df_only(self, sample_events_df):
        filt_df, filt_events, filt_epochs, event_id, event_idx = filter_by_trial_types(
            "WORD", events_df=sample_events_df,
        )
        assert filt_df is not None
        assert len(filt_df) == 3
        assert filt_events is None
        assert filt_epochs is None
        np.testing.assert_array_equal(event_idx, np.arange(3))

    def test_no_inputs_returns_empty(self):
        filt_df, filt_events, filt_epochs, event_id, event_idx = filter_by_trial_types("WORD")
        assert filt_df is None
        assert filt_events is None
        assert filt_epochs is None
        assert event_id == {}
        assert len(event_idx) == 0
