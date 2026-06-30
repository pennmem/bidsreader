"""
Tests for bidsreader.region_enrichment.enrich_pairs_with_cml_regions

enrich_pairs_with_cml_regions joins authoritative pair-level region labels
from the upstream CML pairs.json onto a bidsreader pairs DataFrame. It loads
those labels via a cmlreaders.CMLReader, but accepts a ready reader object so
the join logic can be tested without cmlreaders installed or rhino access.

What is tested:
  - region columns from pairs.json are joined onto matching pair names
  - whitespace on both join keys is stripped before merging
  - only the requested / available region columns are pulled
  - left-join semantics: unmatched pairs survive with NaN region labels
  - missing 'label' column in cml_pairs raises KeyError
  - reader=None with incomplete (subject, experiment, session) raises ValueError
"""
import importlib.util

import pandas as pd
import pytest

from bidsreader.region_enrichment import enrich_pairs_with_cml_regions

# enrich_pairs_with_cml_regions imports cmlreaders unconditionally (even when a
# reader is injected), so skip the whole module where cmlreaders is unavailable.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cmlreaders") is None,
    reason="cmlreaders not installed",
)


class FakeReader:
    """Stand-in for cmlreaders.CMLReader.load('pairs')."""

    def __init__(self, pairs_df):
        self._pairs_df = pairs_df

    def load(self, what):
        assert what == "pairs"
        return self._pairs_df


def _cml_pairs():
    return pd.DataFrame({
        "label": ["A1-A2", "B1-B2"],
        "ind.region": ["hippocampus", "amygdala"],
        "stein.region": ["CA1", "BLA"],
        # a column not in the default wanted-set, to confirm filtering
        "contact_1": [1, 3],
    })


class TestEnrichPairsWithCmlRegions:
    """Tests for enrich_pairs_with_cml_regions."""

    def test_joins_region_columns(self):
        pairs = pd.DataFrame({"name": ["A1-A2", "B1-B2"]})
        result = enrich_pairs_with_cml_regions(pairs, reader=FakeReader(_cml_pairs()))
        assert list(result["ind.region"]) == ["hippocampus", "amygdala"]
        assert list(result["stein.region"]) == ["CA1", "BLA"]

    def test_only_requested_region_cols_pulled(self):
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        result = enrich_pairs_with_cml_regions(
            pairs, reader=FakeReader(_cml_pairs()), region_cols=["ind.region"]
        )
        assert "ind.region" in result.columns
        assert "stein.region" not in result.columns

    def test_non_region_columns_not_pulled(self):
        """contact_1 exists in cml_pairs but is not a region col -> dropped."""
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        result = enrich_pairs_with_cml_regions(pairs, reader=FakeReader(_cml_pairs()))
        assert "contact_1" not in result.columns

    def test_strips_whitespace_on_join_keys(self):
        pairs = pd.DataFrame({"name": ["  A1-A2  "]})
        cml = _cml_pairs()
        cml.loc[0, "label"] = " A1-A2 "
        result = enrich_pairs_with_cml_regions(pairs, reader=FakeReader(cml))
        assert result.iloc[0]["ind.region"] == "hippocampus"

    def test_left_join_keeps_unmatched_pairs(self):
        pairs = pd.DataFrame({"name": ["A1-A2", "Z9-Z10"]})
        result = enrich_pairs_with_cml_regions(pairs, reader=FakeReader(_cml_pairs()))
        assert len(result) == 2
        assert result.iloc[0]["ind.region"] == "hippocampus"
        assert pd.isna(result.iloc[1]["ind.region"])

    def test_does_not_mutate_input(self):
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        enrich_pairs_with_cml_regions(pairs, reader=FakeReader(_cml_pairs()))
        assert "ind.region" not in pairs.columns

    def test_custom_label_col(self):
        pairs = pd.DataFrame({"pair": ["A1-A2"]})
        result = enrich_pairs_with_cml_regions(
            pairs, reader=FakeReader(_cml_pairs()), label_col="pair"
        )
        assert result.iloc[0]["ind.region"] == "hippocampus"

    def test_missing_label_column_raises(self):
        bad = _cml_pairs().drop(columns=["label"])
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        with pytest.raises(KeyError):
            enrich_pairs_with_cml_regions(pairs, reader=FakeReader(bad))

    def test_no_reader_and_incomplete_args_raises(self):
        pairs = pd.DataFrame({"name": ["A1-A2"]})
        with pytest.raises(ValueError):
            enrich_pairs_with_cml_regions(pairs, subject="R1001P")  # no exp/session
