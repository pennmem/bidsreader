"""
Tests for bidsreader.helpers

What is tested:
  - validate_option: None passthrough, valid value, invalid raises InvalidOptionError
  - space_from_coordsystem_fname: with space token, without space token
  - add_prefix: None input, already-prefixed value, value needing prefix
  - merge_duplicate_sample_events: no duplicates, duplicate trial_type merge, all-NaN
  - find_coord_triplets: bare xyz, prefixed xyz, mixed columns, no triplets
  - combine_bipolar_electrodes: pair splitting, midpoint computation, no synthetic pair region
  - normalize_trial_types: converts iterable of strings to set
  - match_event_label: exact match, slash-separated label, no match
"""
import pytest
import numpy as np
import pandas as pd

from bidsreader.helpers import (
    validate_option,
    space_from_coordsystem_fname,
    add_prefix,
    merge_duplicate_sample_events,
    find_coord_triplets,
    combine_bipolar_electrodes,
    normalize_trial_types,
    match_event_label,
)
from bidsreader.exc import InvalidOptionError


# ---------------------------------------------------------------------------
# validate_option
# ---------------------------------------------------------------------------
class TestValidateOption:
    """Tests for validate_option."""

    def test_none_passthrough(self):
        assert validate_option("field", None, ["a", "b"]) is None

    def test_valid_value(self):
        assert validate_option("field", "a", ["a", "b"]) == "a"

    def test_invalid_raises(self):
        with pytest.raises(InvalidOptionError):
            validate_option("field", "c", ["a", "b"])


# ---------------------------------------------------------------------------
# space_from_coordsystem_fname
# ---------------------------------------------------------------------------
class TestSpaceFromCoordSystemFname:
    """Tests for space_from_coordsystem_fname."""

    def test_with_space(self):
        fname = "sub-01_ses-01_space-MNI152NLin6ASym_coordsystem.json"
        assert space_from_coordsystem_fname(fname) == "MNI152NLin6ASym"

    def test_without_space(self):
        fname = "sub-01_ses-01_coordsystem.json"
        assert space_from_coordsystem_fname(fname) is None


# ---------------------------------------------------------------------------
# add_prefix
# ---------------------------------------------------------------------------
class TestAddPrefix:
    """Tests for add_prefix."""

    def test_none_returns_none(self):
        assert add_prefix(None, "sub-") is None

    def test_already_prefixed(self):
        assert add_prefix("sub-01", "sub-") == "sub-01"

    def test_adds_prefix(self):
        assert add_prefix("01", "sub-") == "sub-01"

    def test_int_value_converted(self):
        assert add_prefix(1, "ses-") == "ses-1"


# ---------------------------------------------------------------------------
# merge_duplicate_sample_events
# ---------------------------------------------------------------------------
class TestMergeDuplicateSampleEvents:
    """Tests for merge_duplicate_sample_events."""

    def test_no_duplicates(self):
        df = pd.DataFrame({"sample": [1, 2, 3], "trial_type": ["A", "B", "C"]})
        result = merge_duplicate_sample_events(df)
        assert len(result) == 3
        assert list(result["trial_type"]) == ["A", "B", "C"]

    def test_duplicate_samples_merge_trial_type(self):
        df = pd.DataFrame({
            "sample": [1, 1, 2],
            "trial_type": ["WORD", "STIM", "REST"],
        })
        result = merge_duplicate_sample_events(df)
        assert len(result) == 2
        merged_row = result[result["sample"] == 1].iloc[0]
        assert merged_row["trial_type"] == "WORD/STIM"

    def test_duplicate_samples_empty_merge_trial_type(self):
        df = pd.DataFrame({
            "sample": [1, 1, 2],
            "trial_type": ["", "STIM", "REST"],
        })
        result = merge_duplicate_sample_events(df)
        assert len(result) == 2
        merged_row = result[result["sample"] == 1].iloc[0]
        assert merged_row["trial_type"] == "/STIM"

    def test_duplicate_samples_merge_mutliple_trial_type(self):
        df = pd.DataFrame({
            "sample": [1, 1, 2, 2],
            "trial_type": ["WORD", "STIM", "REST", "STIM"],
        })
        result = merge_duplicate_sample_events(df)
        assert len(result) == 2
        merged_row = result[result["sample"] == 1].iloc[0]
        assert merged_row["trial_type"] == "WORD/STIM"
        merged_row2 = result[result["sample"] == 2].iloc[0]
        assert merged_row2["trial_type"] == "REST/STIM"

    def test_duplicate_trial_types_deduplicated(self):
        df = pd.DataFrame({
            "sample": [1, 1],
            "trial_type": ["WORD", "WORD"],
        })
        result = merge_duplicate_sample_events(df)
        assert result.iloc[0]["trial_type"] == "WORD"

    def test_all_nan_trial_type(self):
        df = pd.DataFrame({
            "sample": [1, 1],
            "trial_type": [np.nan, np.nan],
        })
        result = merge_duplicate_sample_events(df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["trial_type"])

    def test_preserves_column_order(self):
        df = pd.DataFrame({"sample": [1], "onset": [0.5], "trial_type": ["A"]})
        result = merge_duplicate_sample_events(df)
        assert list(result.columns) == ["sample", "onset", "trial_type"]


# ---------------------------------------------------------------------------
# find_coord_triplets
# ---------------------------------------------------------------------------
class TestFindCoordTriplets:
    """Tests for find_coord_triplets."""

    def test_bare_xyz(self):
        result = find_coord_triplets(["name", "x", "y", "z"])
        assert "" in result
        assert result[""] == ("x", "y", "z")

    def test_prefixed_xyz(self):
        result = find_coord_triplets(["name", "tal.x", "tal.y", "tal.z"])
        assert "tal" in result
        assert result["tal"] == ("tal.x", "tal.y", "tal.z")

    def test_mixed(self):
        cols = ["name", "x", "y", "z", "tal.x", "tal.y", "tal.z"]
        result = find_coord_triplets(cols)
        assert "" in result
        assert "tal" in result

    def test_no_triplets(self):
        result = find_coord_triplets(["name", "x", "y"])
        assert result == {}

    def test_incomplete_prefix(self):
        result = find_coord_triplets(["tal.x", "tal.y"])
        assert "tal" not in result


# ---------------------------------------------------------------------------
# combine_bipolar_electrodes
# ---------------------------------------------------------------------------
class TestCombineBipolarElectrodes:
    """Tests for combine_bipolar_electrodes."""

    def test_basic_pair_split(self, sample_bipolar_channels_df, sample_electrodes_df):
        result = combine_bipolar_electrodes(sample_bipolar_channels_df, sample_electrodes_df)
        assert "ch1" in result.columns
        assert "ch2" in result.columns
        assert result.iloc[0]["ch1"] == "A1"
        assert result.iloc[0]["ch2"] == "A2"

    def test_midpoint_computation(self, sample_bipolar_channels_df, sample_electrodes_df):
        result = combine_bipolar_electrodes(sample_bipolar_channels_df, sample_electrodes_df)
        # A1=(1,5,9), A2=(2,6,10) -> midpoint=(1.5, 5.5, 9.5)
        assert result.iloc[0]["x_mid"] == pytest.approx(1.5)
        assert result.iloc[0]["y_mid"] == pytest.approx(5.5)
        assert result.iloc[0]["z_mid"] == pytest.approx(9.5)

    def test_no_synthetic_pair_region_column(self):
        """Neurorad assigns pair-region labels via atlas lookup at the
        midpoint voxel, not by contact-level agreement. The combiner
        must therefore NOT synthesize a ``{col}_pair`` column. Per-contact
        region columns are still present as ``{col}_ch1`` / ``{col}_ch2``.
        """
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0],
            "wb.region": ["hippocampus", "hippocampus"],
        })
        result = combine_bipolar_electrodes(pairs, elecs)
        assert "wb.region_pair" not in result.columns
        assert result.iloc[0]["wb.region_ch1"] == "hippocampus"
        assert result.iloc[0]["wb.region_ch2"] == "hippocampus"

    def test_pixels_space_floors_midpoint(self):
        """In Pixels (CT voxel) space the midpoint is floored to an integer
        voxel index, matching the neurorad pipeline's voxel handling."""
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 2.0], "y": [5.0, 8.0], "z": [9.0, 10.0],
        })
        # raw midpoint = (1.5, 6.5, 9.5) -> floored = (1.0, 6.0, 9.0)
        result = combine_bipolar_electrodes(pairs, elecs, space="Pixels")
        assert result.iloc[0]["x_mid"] == pytest.approx(1.0)
        assert result.iloc[0]["y_mid"] == pytest.approx(6.0)
        assert result.iloc[0]["z_mid"] == pytest.approx(9.0)

    def test_pixels_space_floors_multiple_pairs(self):
        """Regression: flooring must work across many pairs at once. A scalar
        math.floor here raised 'only length-1 arrays can be converted to
        Python scalars' for any subject with >1 bipolar pair."""
        pairs = pd.DataFrame({"name": ["A1-A2", "B1-B2"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2", "B1", "B2"],
            "x": [1.2, 2.9, 3.1, 4.4],
            "y": [5.0, 6.0, 7.0, 8.0],
            "z": [9.0, 10.0, 11.0, 12.0],
        })
        result = combine_bipolar_electrodes(pairs, elecs, space="Pixels")
        # A1-A2 raw mid x = 2.05 -> 2.0 ; B1-B2 raw mid x = 3.75 -> 3.0
        assert list(result["x_mid"]) == pytest.approx([2.0, 3.0])

    def test_non_pixels_space_does_not_floor(self):
        """Default / non-Pixels spaces keep the fractional midpoint."""
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 2.0], "y": [5.0, 8.0], "z": [9.0, 10.0],
        })
        result = combine_bipolar_electrodes(pairs, elecs, space="MNI152NLin6ASym")
        assert result.iloc[0]["x_mid"] == pytest.approx(1.5)
        assert result.iloc[0]["y_mid"] == pytest.approx(6.5)

    def test_missing_contact_coords_yield_nan_midpoint(self):
        """If a pair references a contact missing from elec_df, its midpoint
        columns are NaN rather than raising."""
        pairs = pd.DataFrame({"name": ["A1-A2", "A1-GHOST"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 3.0], "y": [5.0, 7.0], "z": [9.0, 11.0],
        })
        result = combine_bipolar_electrodes(pairs, elecs)
        assert result.iloc[0]["x_mid"] == pytest.approx(2.0)
        assert pd.isna(result.iloc[1]["x_mid"])

    def test_prefixed_coord_triplet_midpoint(self):
        """Prefixed coordinate triplets (e.g. tal.x/tal.y/tal.z) also get a
        {prefix}.{axis}_mid column."""
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        elecs = pd.DataFrame({
            "name": ["A1", "A2"],
            "tal.x": [0.0, 4.0], "tal.y": [0.0, 2.0], "tal.z": [0.0, 6.0],
        })
        result = combine_bipolar_electrodes(pairs, elecs)
        assert result.iloc[0]["tal.x_mid"] == pytest.approx(2.0)
        assert result.iloc[0]["tal.y_mid"] == pytest.approx(1.0)
        assert result.iloc[0]["tal.z_mid"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# normalize_trial_types
# ---------------------------------------------------------------------------
class TestNormalizeTrialTypes:
    """Tests for normalize_trial_types."""

    def test_strings_to_set(self):
        assert normalize_trial_types(["WORD", "STIM"]) == {"WORD", "STIM"}

    def test_deduplication(self):
        assert normalize_trial_types(["WORD", "WORD"]) == {"WORD"}


# ---------------------------------------------------------------------------
# match_event_label
# ---------------------------------------------------------------------------
class TestMatchEventLabel:
    """Tests for match_event_label."""

    def test_exact_match(self):
        assert match_event_label("WORD", ["WORD"]) is True

    def test_slash_separated_match(self):
        assert match_event_label("WORD/STIM", ["STIM"]) is True

    def test_no_match(self):
        assert match_event_label("WORD", ["STIM"]) is False

    def test_partial_string_no_match(self):
        # "WOR" is not an exact token in "WORD"
        assert match_event_label("WORD", ["WOR"]) is False
