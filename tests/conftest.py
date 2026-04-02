"""
Shared fixtures for bidsreader test suite.

Provides reusable BaseReader/CMLBIDSReader instances and sample DataFrames
so individual test files stay focused on behavior, not setup boilerplate.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Reader fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_root(tmp_path):
    """A temporary directory that acts as a BIDS root."""
    return tmp_path


@pytest.fixture
def base_reader(tmp_root):
    """A BaseReader with minimal valid arguments."""
    from bidsreader.basereader import BaseReader
    return BaseReader(root=tmp_root, subject="01", task="rest", session="1", device="eeg")


@pytest.fixture
def cml_reader_eeg(tmp_root):
    """A CMLBIDSReader configured for scalp EEG."""
    from bidsreader.cmlbidsreader import CMLBIDSReader
    return CMLBIDSReader(root=tmp_root, subject="LTP001", task="FR1", session="0", device="eeg")


@pytest.fixture
def cml_reader_ieeg(tmp_root):
    """A CMLBIDSReader configured for intracranial EEG."""
    from bidsreader.cmlbidsreader import CMLBIDSReader
    return CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events_df():
    """A minimal events DataFrame with sample and trial_type columns."""
    return pd.DataFrame({
        "sample": [100, 200, 300, 400, 500],
        "trial_type": ["WORD", "WORD", "STIM", "WORD", "STIM"],
        "onset": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def sample_electrodes_df():
    """A minimal electrodes DataFrame with name and xyz coordinates."""
    return pd.DataFrame({
        "name": ["A1", "A2", "B1", "B2"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [5.0, 6.0, 7.0, 8.0],
        "z": [9.0, 10.0, 11.0, 12.0],
    })


@pytest.fixture
def sample_channels_df():
    """A minimal channels DataFrame."""
    return pd.DataFrame({
        "name": ["A1", "A2", "B1", "B2"],
        "type": ["EEG", "EEG", "EEG", "EEG"],
        "units": ["uV", "uV", "uV", "uV"],
    })


@pytest.fixture
def sample_bipolar_channels_df():
    """A channels DataFrame with bipolar pair names (e.g. 'A1-A2')."""
    return pd.DataFrame({
        "name": ["A1-A2", "B1-B2"],
        "type": ["EEG", "EEG"],
        "units": ["uV", "uV"],
    })
