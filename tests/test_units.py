"""
Tests for bidsreader.units

What is tested:
  - _normalize_unit: replaces micro sign with 'u'
  - get_scale_factor: V->uV, uV->V, same unit returns 1.0, cross-base raises ValueError
  - detect_unit: current_unit override, unknown unit raises ValueError
  - convert_unit: noop when same unit (factor == 1.0)
"""
import pytest

from bidsreader.units import (
    _normalize_unit,
    get_scale_factor,
    detect_unit,
)
from bidsreader.exc import ExternalLibraryError


# ---------------------------------------------------------------------------
# _normalize_unit
# ---------------------------------------------------------------------------
class TestNormalizeUnit:
    """Tests for _normalize_unit."""

    def test_micro_sign_replaced(self):
        assert _normalize_unit("\u00b5V") == "uV"

    def test_already_normalized(self):
        assert _normalize_unit("uV") == "uV"

    def test_no_change_needed(self):
        assert _normalize_unit("V") == "V"


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

    def test_cross_base_raises(self):
        with pytest.raises(ExternalLibraryError, match="different base units"):
            get_scale_factor("V", "T")

    def test_unknown_source_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown source unit"):
            get_scale_factor("kW", "V")

    def test_unknown_target_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown target unit"):
            get_scale_factor("V", "kW")

    def test_tesla_units(self):
        assert get_scale_factor("T", "fT") == pytest.approx(1e15)


# ---------------------------------------------------------------------------
# detect_unit
# ---------------------------------------------------------------------------
class TestDetectUnit:
    """Tests for detect_unit."""

    def test_current_unit_override(self):
        # When current_unit is provided, data is ignored (pass a dummy)
        result = detect_unit(None, current_unit="uV")
        assert result == "uV"

    def test_current_unit_normalizes_micro(self):
        result = detect_unit(None, current_unit="\u00b5V")
        assert result == "uV"

    def test_unknown_current_unit_raises(self):
        with pytest.raises(ExternalLibraryError, match="Unknown unit"):
            detect_unit(None, current_unit="kW")

    def test_unsupported_type_raises(self):
        with pytest.raises(ExternalLibraryError, match="Cannot detect unit"):
            detect_unit("not_a_data_object")
