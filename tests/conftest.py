"""
Shared fixtures for bidsreader test suite.

Provides reusable BaseReader/CMLBIDSReader/NiaBIDSReader instances and sample
DataFrames so individual test files stay focused on behavior, not setup
boilerplate.
"""
import gzip
import json
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
    from bidsreader.readers.basereader import BaseReader
    return BaseReader(root=tmp_root, subject="01", task="rest", session="1", device="eeg")


@pytest.fixture
def cml_reader_eeg(tmp_root):
    """A CMLBIDSReader configured for scalp EEG."""
    from bidsreader.readers.cmlbidsreader import CMLBIDSReader
    return CMLBIDSReader(root=tmp_root, subject="LTP001", task="FR1", session="0", device="eeg")


@pytest.fixture
def cml_reader_ieeg(tmp_root):
    """A CMLBIDSReader configured for intracranial EEG."""
    from bidsreader.readers.cmlbidsreader import CMLBIDSReader
    return CMLBIDSReader(root=tmp_root, subject="R1001P", task="FR1", session="0", device="ieeg")


@pytest.fixture
def nia_reader(nia_dataset):
    """A NiaBIDSReader pointed at the synthetic Nia dataset."""
    from bidsreader.readers.niabidsreader import NiaBIDSReader
    return NiaBIDSReader(root=nia_dataset, subject="S002", session=0)


@pytest.fixture
def nia_reader_early_event(nia_dataset_early_event):
    """A NiaBIDSReader over a dataset with one pre-recording event."""
    from bidsreader.readers.niabidsreader import NiaBIDSReader
    return NiaBIDSReader(root=nia_dataset_early_event, subject="S002", session=0)


# ---------------------------------------------------------------------------
# Synthetic Nia BIDS dataset
# ---------------------------------------------------------------------------

#: NiaFirstSample in the synthetic sidecar — the device-clock sample at which
#: the recording begins. Event `sample` values are indices into the recording
#: and so start at 0, NOT at this value; it is deliberately non-zero so that a
#: reader wrongly applying it as an offset shows up immediately.
NIA_FIRST_SAMPLE = 1234
NIA_SFREQ = 496.582

#: Event sample indices in the fixture, recording-relative. `onset` is derived
#: from these, as the converter does (bids_convert.py: onset = sample / sfreq).
NIA_EVENT_SAMPLES = (0, 497, 994)


@pytest.fixture
def nia_dataset(tmp_path):
    """A minimal Nia BIDS dataset: one subject, one ACL session, montage 0.

    Mirrors what ``niadatsci.niashare.bids_convert.NiaBIDSConverter`` writes,
    minus the BrainVision recording itself — everything here is TSV/JSON, so
    ``load_raw`` is not exercised by tests using this fixture.

    Returns the dataset root, i.e. ``<tmp>/preclinical/ACL``.
    """
    return _write_nia_dataset(tmp_path)


@pytest.fixture
def nia_dataset_early_event(tmp_path):
    """A Nia dataset whose first event predates the recording.

    The converter blanks `sample` and `onset` to `n/a` for an event logged
    before the first stored sample, keeping the rest of the row — including the
    absolute `eegoffset_cmd`. Mirrors `TestEventsBeforeTheRecording` in
    niadatsci's own test suite.
    """
    return _write_nia_dataset(tmp_path, early_event=True)


def _write_nia_dataset(tmp_path, early_event=False):
    root = tmp_path / "preclinical" / "ACL"
    eeg = root / "sub-S002" / "ses-00" / "eeg"
    eeg.mkdir(parents=True)
    pfx = "sub-S002_ses-00_task-acl"

    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "Nia ACL (preclinical)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "niadatsci", "Version": "1.0.0"}],
    }))

    # NIA sets only the nia_* columns; the rest are mne-bids leftovers.
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\thand\tnia_protocol\tnia_subject\tnia_montage\n"
        "sub-S002\tn/a\tn/a\tn/a\tpreclinical\tS002\t0\n"
    )

    (root / "sub-S002" / "ses-00" / "sub-S002_ses-00_scans.tsv").write_text(
        f"filename\tacq_time\neeg/{pfx}_eeg.vhdr\t2025-03-01T12:00:00.000000\n"
    )

    (eeg / f"{pfx}_eeg.json").write_text(json.dumps({
        "TaskName": "acl",
        "SamplingFrequency": NIA_SFREQ,
        "EEGChannelCount": 2,
        "RecordingDuration": 10.0688,
        "RecordingType": "continuous",
        "PowerLineFrequency": 60.0,
        "Manufacturer": "Nia Therapeutics",
        "NiaProtocol": "preclinical",
        "NiaExperiment": "ACL",
        "NiaSession": 0,
        "NiaMontage": 0,
        "NiaFirstSample": NIA_FIRST_SAMPLE,
        "NiaLastSample": 6233,
        "NiaNGaps": 3,
        "NiaPropGaps": 0.0006,
        "NiaNCorrupt": 11,
        "NiaPropCorrupt": 0.0022,
        "NiaStimulation": True,
        "NiaClosedLoop": True,
        "NiaUnitsStatus": "uncalibrated",
        "NiaMicrovoltsPerLSB": "n/a",
        "NiaMeasDate": "2025-03-01 12:00:00.000000",
    }))

    # `units` is the literal string "arbitrary" when the conversion ran without
    # --microvolts-per-lsb; `group` is montage<N>.
    (eeg / f"{pfx}_channels.tsv").write_text(
        "name\ttype\tunits\tsampling_frequency\tlow_cutoff\thigh_cutoff\t"
        "notch\tstatus\tstatus_description\tgroup\n"
        f"LA1\tEEG\tarbitrary\t{NIA_SFREQ}\tn/a\tn/a\tn/a\tgood\tn/a\tmontage0\n"
        f"LA2\tEEG\tarbitrary\t{NIA_SFREQ}\tn/a\tn/a\tn/a\tgood\tn/a\tmontage0\n"
    )

    # `sample` is an index into the recording, from 0 at the first stored
    # sample; `onset` is derived from it as sample / sfreq, so the two agree by
    # construction exactly as they do in real output. `duration` stays n/a (the
    # device duration is renamed to stim_duration).
    # `eegoffset_cmd` keeps the absolute device counter: sample + NiaFirstSample.
    rows = []
    trial_types = ("STIM", "CONTROL_THERAPY", "STIM")
    blocks, conditions, amplitudes = (0, 0, 1), ("A", "B", "A"), (1000, 0, 1000)
    for i, sample in enumerate(NIA_EVENT_SAMPLES):
        eegoffset = sample + NIA_FIRST_SAMPLE
        if early_event and i == 0:
            # Logged 50 samples before the recording started: position unknown,
            # everything else intact.
            onset_s, sample_s = "n/a", "n/a"
            eegoffset = NIA_FIRST_SAMPLE - 50
        else:
            onset_s, sample_s = f"{sample / NIA_SFREQ:.6f}", str(sample)
        rows.append(
            f"{onset_s}\tn/a\t{trial_types[i]}\t{sample_s}\t{blocks[i]}\t"
            f"{conditions[i]}\t500\t{amplitudes[i]}\t0.012\t{eegoffset}\n"
        )
    (eeg / f"{pfx}_events.tsv").write_text(
        "onset\tduration\ttrial_type\tsample\tblock\tcondition\t"
        "stim_duration\tamplitude\tlatency\teegoffset_cmd\n"
        + "".join(rows)
    )
    (eeg / f"{pfx}_events.json").write_text(json.dumps({
        "onset": {"Description": "Onset of the event", "Units": "s"},
        "trial_type": {
            "Description": "Event category",
            "Levels": {"STIM": "Stimulation delivered",
                       "CONTROL_THERAPY": "Sham stimulation"},
        },
        "stim_duration": {"Description": "Device stimulation duration"},
    }))

    # Physio TSVs are written headerless; the sidecar names the columns.
    with gzip.open(eeg / f"{pfx}_recording-classifier_physio.tsv.gz", "wt") as f:
        for i in range(5):
            f.write(f"{NIA_FIRST_SAMPLE + i}\t{0.1 * i:.4f}\n")
    (eeg / f"{pfx}_recording-classifier_physio.json").write_text(json.dumps({
        "SamplingFrequency": NIA_SFREQ,
        "StartTime": 0.0,
        "Columns": ["sample", "classifier"],
        "NiaSourceFile": "clf_vals.csv",
    }))

    return root


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
