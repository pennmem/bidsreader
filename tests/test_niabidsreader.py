"""
Tests for NiaBIDSReader.

Covers the behavior that is specific to Nia conversions and would otherwise be
silently wrong:

  - Field derivation: session zero-padding, montage -> acq- entity, the always-
    'eeg' device, and task inference from the dataset.
  - Path construction through the overridden _bp().
  - Loaders: events (literal 'n/a' preserved, numeric coercion), channels,
    sidecar, physio (headerless, columns from the sidecar), participants, scans.
  - first_sample(), the device-clock offset that BrainVision cannot store.
  - get_data_index() scanning without a *_beh.tsv, and the sidecar quality
    columns it surfaces.
  - Absent electrodes/coordsystem raising a clear error rather than an
    obscure one.
  - is_nia_dataset() catching a root pointed one level too high.
"""
import gzip
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from bidsreader.readers.niabidsreader import NiaBIDSReader, NIA_ROOT
from bidsreader.src.exc import (
    AmbiguousMatchError,
    DataParseError,
    FileNotFoundBIDSError,
    InvalidOptionError,
    MissingRequiredFieldError,
)
from .conftest import NIA_EVENT_SAMPLES, NIA_FIRST_SAMPLE, NIA_SFREQ


class TestNiaBIDSReaderInit:
    def test_minimal_init(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0)
        assert r.root == Path(nia_dataset)
        assert r.subject == "S002"
        assert r.session == 0

    def test_root_required(self):
        with pytest.raises(ValueError):
            NiaBIDSReader(subject="S002", session=0)

    def test_task_left_none_not_stringified(self, nia_dataset):
        # BaseReader used to coerce None into the string "None", which would
        # make task inference impossible.
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0)
        assert r.task is None

    def test_explicit_task_kept(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0, task="acl")
        assert r.task == "acl"

    def test_ieeg_device_rejected(self, nia_dataset):
        with pytest.raises(InvalidOptionError):
            NiaBIDSReader(root=nia_dataset, subject="S002", session=0, device="ieeg")

    def test_repr_uses_class_name(self, nia_reader):
        assert "NiaBIDSReader(" in repr(nia_reader)


class TestSessionPadding:
    @pytest.mark.parametrize("session", [0, "0", "00"])
    def test_all_forms_pad_to_two_digits(self, nia_dataset, session):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=session)
        assert r._bids_session() == "00"

    def test_double_digit_session(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=12)
        assert r._bids_session() == "12"

    def test_non_integer_label_passes_through(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session="pilot")
        assert r._bids_session() == "pilot"

    def test_none_session(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002")
        assert r._bids_session() is None


class TestMontage:
    def test_montage_zero_has_no_acq_entity(self, nia_dataset):
        # Montage 0 must produce unsuffixed filenames or every path misses.
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0)
        assert r.montage == 0
        assert r.acquisition is None

    def test_nonzero_montage_sets_acq(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0, montage=2)
        assert r.acquisition == "montage2"

    def test_montage_is_settable_and_updates_acq(self, nia_reader):
        nia_reader.montage = 1
        assert nia_reader.acquisition == "montage1"
        nia_reader.montage = 0
        assert nia_reader.acquisition is None

    def test_explicit_acquisition_wins(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0,
                          acquisition="custom")
        assert r.acquisition == "custom"

    def test_montage_in_filename(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0,
                          task="acl", montage=1)
        assert "acq-montage1" in str(r._bp(suffix="events", extension=".tsv").fpath)


class TestDetermineDevice:
    def test_always_eeg(self, nia_reader):
        assert nia_reader.device == "eeg"

    def test_eeg_regardless_of_subject_prefix(self, nia_dataset):
        # CMLBIDSReader would call an "R"-prefixed subject intracranial.
        r = NiaBIDSReader(root=nia_dataset, subject="R1001P", session=0)
        assert r.device == "eeg"


class TestDetermineSpace:
    def test_space_is_none(self, nia_reader):
        assert nia_reader.space is None

    def test_no_warning_raised(self, nia_reader):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert nia_reader.space is None


class TestDetermineTask:
    def test_inferred_from_tree(self, nia_reader):
        assert nia_reader._resolve_task() == "acl"

    def test_inference_caches_onto_task(self, nia_reader):
        nia_reader._resolve_task()
        assert nia_reader.task == "acl"

    def test_falls_back_to_dataset_description(self, tmp_path):
        # A dataset whose sessions are not converted yet has no task entity.
        root = tmp_path / "preclinical" / "PS2"
        root.mkdir(parents=True)
        (root / "dataset_description.json").write_text(json.dumps({
            "Name": "Nia PS2 (preclinical)",
            "GeneratedBy": [{"Name": "niadatsci"}],
        }))
        r = NiaBIDSReader(root=root, subject="S002", session=0)
        assert r._determine_task() == "ps2"

    def test_multiple_tasks_raises(self, nia_dataset):
        eeg = nia_dataset / "sub-S002" / "ses-00" / "eeg"
        (eeg / "sub-S002_ses-00_task-ps2_eeg.json").write_text("{}")
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0)
        with pytest.raises(DataParseError, match="expected one task"):
            r._determine_task()

    def test_unresolvable_task_raises(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        r = NiaBIDSReader(root=root, subject="S002", session=0)
        with pytest.raises(DataParseError, match="Could not infer task"):
            r._resolve_task()


class TestBidsPath:
    def test_includes_padded_session_and_eeg_datatype(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0, task="acl")
        path = str(r._bp(suffix="events", extension=".tsv").fpath)
        assert "sub-S002" in path
        assert "ses-00" in path
        assert "task-acl" in path
        assert "/eeg/" in path

    def test_session_zero_not_rendered_as_ses_0(self, nia_dataset):
        # BaseReader._bp would produce "ses-0"; the files on disk are "ses-00".
        r = NiaBIDSReader(root=nia_dataset, subject="S002", session=0, task="acl")
        assert "ses-0_" not in str(r._bp(suffix="events", extension=".tsv").fpath)


class TestIsNiaDataset:
    def test_true_for_nia_dataset(self, nia_reader):
        assert nia_reader.is_nia_dataset() is True

    def test_false_one_level_too_high(self, nia_dataset):
        # The most likely user error: root at <bids_root> or <bids_root>/<protocol>.
        r = NiaBIDSReader(root=nia_dataset.parent, subject="S002", session=0)
        assert r.is_nia_dataset() is False

    def test_false_for_non_nia_description(self, tmp_path):
        (tmp_path / "dataset_description.json").write_text(json.dumps({
            "Name": "Some other dataset",
            "GeneratedBy": [{"Name": "mne-bids"}],
        }))
        r = NiaBIDSReader(root=tmp_path, subject="S002", session=0)
        assert r.is_nia_dataset() is False


class TestLoadEvents:
    def test_columns_and_rows(self, nia_reader):
        df = nia_reader.load_events()
        assert len(df) == 3
        assert list(df.columns)[:4] == ["onset", "duration", "trial_type", "sample"]

    def test_literal_na_preserved(self, nia_reader):
        # `duration` is legitimately "n/a" throughout; turning it into NaN would
        # lose the distinction from a missing value.
        df = nia_reader.load_events()
        assert (df["duration"] == "n/a").all()

    def test_numeric_columns_coerced(self, nia_reader):
        df = nia_reader.load_events()
        assert pd.api.types.is_numeric_dtype(df["sample"])
        assert pd.api.types.is_numeric_dtype(df["onset"])

    def test_trial_type_stays_string(self, nia_reader):
        df = nia_reader.load_events()
        assert set(df["trial_type"]) == {"STIM", "CONTROL_THERAPY"}

    def test_coerce_numeric_off(self, nia_reader):
        df = nia_reader.load_events(coerce_numeric=False)
        assert df["sample"].dtype == object

    def test_stim_duration_carries_device_duration(self, nia_reader):
        # The BIDS-reserved `duration` was taken, so the device value is renamed.
        df = nia_reader.load_events()
        assert "stim_duration" in df.columns
        assert (df["stim_duration"] == 500).all()

    def test_samples_are_recording_relative(self, nia_reader):
        # `sample` indexes the recording from 0, not the device clock. If this
        # ever equals NIA_FIRST_SAMPLE again, the conversion contract changed.
        df = nia_reader.load_events()
        assert list(df["sample"]) == list(NIA_EVENT_SAMPLES)
        assert df["sample"].min() == 0

    def test_eegoffset_keeps_the_absolute_clock(self, nia_reader):
        # The absolute counter is preserved separately, and the two differ by
        # exactly first_sample().
        df = nia_reader.load_events()
        assert (df["eegoffset_cmd"] - nia_reader.first_sample()
                == df["sample"]).all()

    def test_missing_fields_raise(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, task="acl")
        with pytest.raises(MissingRequiredFieldError):
            r.load_events()

    def test_events_json(self, nia_reader):
        desc = nia_reader.load_events_json()
        assert "STIM" in desc["trial_type"]["Levels"]


class TestLoadChannels:
    def test_columns(self, nia_reader):
        df = nia_reader.load_channels()
        assert list(df.columns) == [
            "name", "type", "units", "sampling_frequency", "low_cutoff",
            "high_cutoff", "notch", "status", "status_description", "group",
        ]

    def test_arbitrary_units_preserved(self, nia_reader):
        # Uncalibrated conversions carry raw ADC counts; "arbitrary" must survive
        # rather than being coerced or rejected.
        assert (nia_reader.load_channels()["units"] == "arbitrary").all()

    def test_group_is_montage_label(self, nia_reader):
        assert (nia_reader.load_channels()["group"] == "montage0").all()


class TestSidecar:
    def test_nia_provenance_keys(self, nia_reader):
        sc = nia_reader.load_sidecar()
        assert sc["NiaExperiment"] == "ACL"
        assert sc["NiaPropGaps"] == 0.0006
        assert sc["NiaStimulation"] is True

    def test_first_sample(self, nia_reader):
        assert nia_reader.first_sample() == NIA_FIRST_SAMPLE

    def test_units_status(self, nia_reader):
        assert nia_reader.units_status() == "uncalibrated"

    def test_missing_sidecar_raises_bids_error(self, nia_dataset):
        r = NiaBIDSReader(root=nia_dataset, subject="NOPE", session=0, task="acl")
        with pytest.raises(FileNotFoundBIDSError):
            r.load_sidecar()


class TestLoadPhysio:
    def test_autodetect_single_recording(self, nia_reader):
        df = nia_reader.load_physio()
        assert list(df.columns) == ["sample", "classifier"]
        assert len(df) == 5

    def test_headerless_first_row_is_data(self, nia_reader):
        # The file has no header row; losing that would silently drop a sample.
        df = nia_reader.load_physio()
        assert df["sample"].iloc[0] == NIA_FIRST_SAMPLE

    def test_explicit_recording(self, nia_reader):
        assert len(nia_reader.load_physio(recording="classifier")) == 5

    def test_list_available(self, nia_reader):
        assert nia_reader.list_available_physio() == ["classifier"]

    def test_ambiguous_when_two_recordings(self, nia_dataset, nia_reader):
        eeg = nia_dataset / "sub-S002" / "ses-00" / "eeg"
        pfx = "sub-S002_ses-00_task-acl"
        with gzip.open(eeg / f"{pfx}_recording-imu_physio.tsv.gz", "wt") as f:
            f.write("1234\t1\t0.5\n")
        (eeg / f"{pfx}_recording-imu_physio.json").write_text(json.dumps(
            {"Columns": ["eegoffset", "motion", "magnitude"]}))
        with pytest.raises(AmbiguousMatchError):
            nia_reader.load_physio()

    def test_missing_physio_raises(self, tmp_path):
        eeg = tmp_path / "sub-S002" / "ses-00" / "eeg"
        eeg.mkdir(parents=True)
        r = NiaBIDSReader(root=tmp_path, subject="S002", session=0, task="acl")
        with pytest.raises(FileNotFoundBIDSError):
            r.load_physio()

    def test_sidecar_without_columns_raises(self, nia_dataset, nia_reader):
        eeg = nia_dataset / "sub-S002" / "ses-00" / "eeg"
        pfx = "sub-S002_ses-00_task-acl"
        (eeg / f"{pfx}_recording-classifier_physio.json").write_text(json.dumps(
            {"SamplingFrequency": NIA_SFREQ}))
        with pytest.raises(DataParseError, match="Columns"):
            nia_reader.load_physio()


class TestOptionalCoordinateFiles:
    def test_electrodes_absent_raises_with_explanation(self, nia_reader):
        with pytest.raises(FileNotFoundBIDSError, match="omit electrode coordinates"):
            nia_reader.load_electrodes()

    def test_coordsystem_absent_raises(self, nia_reader):
        with pytest.raises(FileNotFoundBIDSError):
            nia_reader.load_coordsystem_desc()

    def test_electrodes_read_when_present(self, nia_dataset, nia_reader):
        eeg = nia_dataset / "sub-S002" / "ses-00" / "eeg"
        (eeg / "sub-S002_ses-00_task-acl_electrodes.tsv").write_text(
            "name\tx\ty\tz\nLA1\t1.0\t2.0\t3.0\n"
        )
        df = nia_reader.load_electrodes()
        assert list(df.columns) == ["name", "x", "y", "z"]


class TestDatasetLevelFiles:
    def test_participants(self, nia_reader):
        df = nia_reader.load_participants()
        assert df["nia_protocol"].iloc[0] == "preclinical"
        assert df["age"].iloc[0] == "n/a"

    def test_scans(self, nia_reader):
        df = nia_reader.load_scans()
        assert "acq_time" in df.columns

    def test_dataset_description(self, nia_reader):
        assert nia_reader.load_dataset_description()["Name"] == "Nia ACL (preclinical)"


class TestGetDataIndex:
    def test_finds_session_without_beh_tsv(self, nia_reader):
        # BaseReader.get_data_index globs *_beh.tsv, which Nia never writes;
        # a super() call here would return an empty frame.
        df = nia_reader.get_data_index()
        assert len(df) == 1
        assert df["subject"].iloc[0] == "S002"
        assert df["session"].iloc[0] == "00"
        assert df["task"].iloc[0] == "acl"

    def test_path_columns(self, nia_reader):
        row = nia_reader.get_data_index().iloc[0]
        assert row["events"].endswith("_events.tsv")
        assert row["channels"].endswith("_channels.tsv")
        assert row["physio_classifier"].endswith("_recording-classifier_physio.tsv.gz")

    def test_absent_files_are_none(self, nia_reader):
        row = nia_reader.get_data_index().iloc[0]
        assert row["eeg"] is None          # no BrainVision file in the fixture
        assert row["physio_imu"] is None
        assert row["electrodes"] is None

    def test_sidecar_quality_columns(self, nia_reader):
        row = nia_reader.get_data_index().iloc[0]
        assert row["prop_gaps"] == 0.0006
        assert row["prop_corrupt"] == 0.0022
        # bool(...) rather than `is True`: pandas stores these as numpy.bool_.
        assert bool(row["stimulation"])
        assert bool(row["closed_loop"])
        assert row["units_status"] == "uncalibrated"
        assert row["montage"] == 0
        assert row["experiment"] == "ACL"

    def test_explicit_root_argument(self, nia_dataset, tmp_path):
        r = NiaBIDSReader(root=tmp_path, subject="S002", session=0)
        assert len(r.get_data_index(root=nia_dataset)) == 1

    def test_empty_dataset_returns_typed_frame(self, tmp_path):
        r = NiaBIDSReader(root=tmp_path, subject="S002", session=0)
        df = r.get_data_index()
        assert df.empty
        for col in ("subject", "session", "task", "events", "prop_gaps"):
            assert col in df.columns

    def test_malformed_sidecar_does_not_abort_scan(self, nia_dataset, nia_reader):
        eeg = nia_dataset / "sub-S002" / "ses-00" / "eeg"
        (eeg / "sub-S002_ses-00_task-acl_eeg.json").write_text("{not json")
        with pytest.warns(RuntimeWarning, match="Could not read sidecar"):
            df = nia_reader.get_data_index()
        assert len(df) == 1
        assert df["prop_gaps"].iloc[0] is None

    def test_task_filter(self, nia_reader):
        assert len(nia_reader.get_data_index(task="acl")) == 1
        assert nia_reader.get_data_index(task="ps2").empty


def _stub_raw(monkeypatch, n_samples=3000):
    """Patch load_raw to return an in-memory Raw starting at sample 0.

    Sample 0 is what read_raw_bids gives for BrainVision, so this reproduces the
    real alignment situation. Annotations supply the event_id map that
    load_epochs looks trial_type up in.
    """
    import mne
    import numpy as np

    info = mne.create_info(["LA1", "LA2"], sfreq=NIA_SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, n_samples)), info, verbose=False)
    raw.set_annotations(mne.Annotations(
        onset=[0.0, 1.0, 2.0],
        duration=[0.0, 0.0, 0.0],
        description=["STIM", "CONTROL_THERAPY", "STIM"],
    ))
    monkeypatch.setattr(NiaBIDSReader, "load_raw", lambda self, **kw: raw)
    return raw


class TestLoadEpochsSampleAlignment:
    """`sample` indexes the recording directly — load_epochs must not shift it.

    The converter writes `sample` relative to the first stored sample, so any
    offset applied here silently misaligns every epoch. NIA_FIRST_SAMPLE is
    non-zero in the fixture precisely so a returning subtraction is visible.
    """

    def test_sample_used_directly(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch)
        ep = nia_reader.load_epochs(
            tmin=0.0, tmax=0.1, events=nia_reader.load_events())
        assert ep.events[0, 0] == 0
        assert ep.events[1, 0] == 497
        assert ep.events[2, 0] == 994

    def test_no_offset_applied(self, nia_reader, monkeypatch):
        # The strong form: fails for an offset of ANY size, including one small
        # enough to keep every sample non-negative.
        _stub_raw(monkeypatch)
        events = nia_reader.load_events()
        ep = nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)
        assert (ep.events[:, 0] == events["sample"].values).all()

    def test_first_sample_not_consulted(self, nia_reader, monkeypatch):
        # Epoching must not depend on the sidecar key at all.
        _stub_raw(monkeypatch)
        def boom(self):
            raise AssertionError("load_epochs must not call first_sample()")
        monkeypatch.setattr(NiaBIDSReader, "first_sample", boom)
        ep = nia_reader.load_epochs(
            tmin=0.0, tmax=0.1, events=nia_reader.load_events())
        assert len(ep.events) == 3

    def test_annotations_used_when_events_none(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch)
        ep = nia_reader.load_epochs(tmin=0.0, tmax=0.1)
        assert len(ep.events) == 3

    def test_negative_sample_raises(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch)
        events = nia_reader.load_events()
        events.loc[0, "sample"] = -1
        with pytest.raises(DataParseError, match="negative sample index"):
            nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)

    def test_missing_sample_column_raises(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch)
        with pytest.raises(DataParseError, match="'sample' column"):
            nia_reader.load_epochs(
                tmin=0.0, tmax=0.1, events=pd.DataFrame({"onset": [0.0]}))

    def test_unknown_trial_type_raises(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch)
        events = nia_reader.load_events()
        events.loc[0, "trial_type"] = "NOT_A_REAL_TYPE"
        with pytest.raises(DataParseError, match="not found in raw annotations"):
            nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)


class TestSamplesBeyondRecording:
    """Out-of-range samples must fail loudly, not misalign.

    This is what a tree converted before the switch to recording-relative
    samples looks like to the reader.
    """

    def test_sample_at_n_times_raises(self, nia_reader, monkeypatch):
        raw = _stub_raw(monkeypatch)
        events = nia_reader.load_events()
        events.loc[0, "sample"] = raw.n_times
        with pytest.raises(DataParseError, match="beyond the end of the recording"):
            nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)

    def test_old_convention_events_are_rejected(self, nia_reader, monkeypatch):
        # Exactly what the fixture used to contain: sample + NiaFirstSample.
        _stub_raw(monkeypatch, n_samples=1200)
        events = nia_reader.load_events()
        events["sample"] = events["sample"] + NIA_FIRST_SAMPLE
        with pytest.raises(DataParseError, match="predating the switch"):
            nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)

    def test_message_names_the_remedy(self, nia_reader, monkeypatch):
        _stub_raw(monkeypatch, n_samples=1200)
        events = nia_reader.load_events()
        events["sample"] = events["sample"] + NIA_FIRST_SAMPLE
        with pytest.raises(DataParseError, match="re-run the niadatsci 'share' step"):
            nia_reader.load_epochs(tmin=0.0, tmax=0.1, events=events)


class TestPreRecordingEvents:
    """Events blanked to `n/a` because they predate the first stored sample."""

    def test_load_events_warns(self, nia_reader_early_event):
        with pytest.warns(RuntimeWarning, match=r"'sample' = 'n/a'"):
            nia_reader_early_event.load_events()

    def test_load_events_returns_all_rows(self, nia_reader_early_event):
        # The event happened; only its position in this recording is unknown.
        with pytest.warns(RuntimeWarning):
            df = nia_reader_early_event.load_events()
        assert len(df) == 3
        assert df["trial_type"].iloc[0] == "STIM"
        assert int(df["eegoffset_cmd"].iloc[0]) == NIA_FIRST_SAMPLE - 50

    def test_blanked_columns_are_na(self, nia_reader_early_event):
        with pytest.warns(RuntimeWarning):
            df = nia_reader_early_event.load_events()
        assert df["sample"].iloc[0] == "n/a"
        assert df["onset"].iloc[0] == "n/a"

    def test_load_epochs_raises(self, nia_reader_early_event, monkeypatch):
        _stub_raw(monkeypatch)
        with pytest.warns(RuntimeWarning):
            events = nia_reader_early_event.load_events()
        with pytest.raises(DataParseError, match="no sample index"):
            nia_reader_early_event.load_epochs(tmin=0.0, tmax=0.1, events=events)

    def test_error_message_gives_the_filter(self, nia_reader_early_event, monkeypatch):
        _stub_raw(monkeypatch)
        with pytest.warns(RuntimeWarning):
            events = nia_reader_early_event.load_events()
        with pytest.raises(DataParseError, match=r"errors='coerce'\)\.notna\(\)"):
            nia_reader_early_event.load_epochs(tmin=0.0, tmax=0.1, events=events)

    def test_filtered_events_epoch_successfully(self, nia_reader_early_event, monkeypatch):
        # The remedy the error message suggests must actually work.
        _stub_raw(monkeypatch)
        with pytest.warns(RuntimeWarning):
            events = nia_reader_early_event.load_events()
        kept = events[pd.to_numeric(events["sample"], errors="coerce").notna()]
        ep = nia_reader_early_event.load_epochs(tmin=0.0, tmax=0.1, events=kept)
        assert len(ep.events) == 2


class TestFixtureMatchesConverterInvariants:
    """Guard the fixture against drifting from what the converter emits.

    If these fail, the fixture no longer represents real output and every other
    test in this file is measuring the wrong thing.
    """

    def test_onset_equals_sample_over_sfreq(self, nia_reader):
        # The converter derives onset from sample, so equality holds by
        # construction (bids_convert.py: events['onset'] = sample / sfreq).
        import numpy as np
        df = nia_reader.load_events()
        assert np.allclose(df["onset"], df["sample"] / NIA_SFREQ, atol=1e-6)

    def test_samples_within_the_recording(self, nia_reader):
        # Mirrors niadatsci's own assertion that max(sample) < raw.n_times.
        sc = nia_reader.load_sidecar()
        n_times = sc["RecordingDuration"] * sc["SamplingFrequency"]
        assert nia_reader.load_events()["sample"].max() < n_times

    def test_samples_start_at_zero(self, nia_reader):
        assert min(NIA_EVENT_SAMPLES) == 0


class TestModuleSurface:
    def test_nia_root_constant(self):
        assert NIA_ROOT == "/bids"

    def test_exported_from_package_root(self):
        import bidsreader
        assert bidsreader.NiaBIDSReader is NiaBIDSReader

    def test_exported_from_readers(self):
        from bidsreader.readers import NiaBIDSReader as FromPkg
        assert FromPkg is NiaBIDSReader
