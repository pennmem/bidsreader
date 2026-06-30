"""
Tests for bidsreader._neurorad_algo

These are upstream-free unit tests for the copied neurorad pair-location
algorithm. The rhino-only parity test in test_pair_midpoint_parity.py pins
this implementation against the real neurorad_pipeline; here we exercise the
copy's behavior directly so it is covered even off rhino.

What is tested:
  - pair_coordinate: midpoint, None-in/None-out short-circuit, return type
  - pair_coordinate_axis: vectorized midpoint, NaN propagation, scalar input
  - constant maps: shape/content sanity for the mirrored neurorad constants
"""
import numpy as np
import pytest

from bidsreader._neurorad_algo import (
    pair_coordinate,
    pair_coordinate_axis,
    VALID_COORDINATE_SPACES,
    VALID_COORDINATE_TYPES,
    VALID_ATLASES,
    BIDS_SPACE_TO_NEURORAD,
)


# ---------------------------------------------------------------------------
# pair_coordinate
# ---------------------------------------------------------------------------
class TestPairCoordinate:
    """Tests for pair_coordinate (scalar / single-triplet midpoint)."""

    def test_midpoint(self):
        result = pair_coordinate([0.0, 0.0, 0.0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_negative_and_fractional(self):
        result = pair_coordinate([-1.5, 0.0, 10.25], [1.5, 0.0, -10.25])
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_none_first_returns_none(self):
        assert pair_coordinate(None, [1.0, 2.0, 3.0]) is None

    def test_none_second_returns_none(self):
        assert pair_coordinate([1.0, 2.0, 3.0], None) is None

    def test_returns_ndarray(self):
        result = pair_coordinate([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])
        assert isinstance(result, np.ndarray)

    def test_accepts_ndarray_input(self):
        result = pair_coordinate(np.array([0.0, 0.0, 0.0]), np.array([4.0, 8.0, 2.0]))
        np.testing.assert_allclose(result, [2.0, 4.0, 1.0])


# ---------------------------------------------------------------------------
# pair_coordinate_axis
# ---------------------------------------------------------------------------
class TestPairCoordinateAxis:
    """Tests for pair_coordinate_axis (vectorized single-axis midpoint)."""

    def test_vectorized_midpoint(self):
        a = [0.0, 10.0, -4.0]
        b = [2.0, 20.0, 4.0]
        np.testing.assert_allclose(pair_coordinate_axis(a, b), [1.0, 15.0, 0.0])

    def test_nan_propagates(self):
        a = [1.0, np.nan, 3.0]
        b = [3.0, 5.0, np.nan]
        result = pair_coordinate_axis(a, b)
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_scalar_input(self):
        # Single-pair case: must not crash and must return the midpoint.
        result = pair_coordinate_axis([2.0], [5.0])
        np.testing.assert_allclose(result, [3.5])

    def test_returns_float_dtype(self):
        # Integer input must be promoted so /2 does not truncate.
        result = pair_coordinate_axis([1, 2], [2, 3])
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [1.5, 2.5])


# ---------------------------------------------------------------------------
# Mirrored neurorad constants
# ---------------------------------------------------------------------------
class TestNeuroradConstants:
    """Light sanity checks on the constants copied from localization.py."""

    def test_valid_spaces(self):
        assert VALID_COORDINATE_SPACES == ("ct_voxel", "fs", "t1_mri", "t2_mri", "mni")

    def test_valid_types(self):
        assert VALID_COORDINATE_TYPES == ("raw", "corrected")

    def test_valid_atlases(self):
        assert VALID_ATLASES == ("dk", "whole_brain", "mtl")

    def test_bids_space_map_values_are_pairs(self):
        # Every value is a (coordinate_space, coordinate_type) tuple, and the
        # coordinate_type is one of the valid neurorad types.
        for space, mapped in BIDS_SPACE_TO_NEURORAD.items():
            assert len(mapped) == 2
            assert mapped[1] in VALID_COORDINATE_TYPES

    def test_pixels_maps_to_ct_voxel(self):
        assert BIDS_SPACE_TO_NEURORAD["Pixels"] == ("ct_voxel", "raw")

    def test_brainshift_maps_to_corrected(self):
        assert BIDS_SPACE_TO_NEURORAD["fsnativeBrainshift"][1] == "corrected"
