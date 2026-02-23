import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import mne
from ptsa.data.timeseries import TimeSeries
from pathlib import Path
from typing import Iterable, Tuple, Optional, Union, Dict, List
import warnings
import json
from .constants import CML_ROOT, INTRACRANIAL_FIELDS, SCALP_FIELDS, VALID_EEG_TYPES, VALID_ACQ
from ._errorwrap import public_api
from .helpers import validate_option, space_from_coordsystem_fname, add_prefix, merge_duplicate_sample_events, combine_bipolar_electrodes
from .exc import BIDSReaderError, InvalidOptionError, MissingRequiredFieldError, FileNotFoundBIDSError, AmbiguousMatchError, DataParseError, DependencyError, ExternalLibraryError

class BIDSReader:
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        eeg_type: Optional[str] = None,
    ):
        self.root = Path(root) if root is not None else Path(DEFAULT_ROOT)
        self.subject = subject
        self.session = session
        self.task = task
        
        self.acquisition = acquisition
        
        self.eeg_type = validate_option(
            "eeg_type", eeg_type, VALID_EEG_TYPES
        )
        
        self._space = space

    # ---------- magic functions ----------
    def __str__(self) -> str:
        parts = [
            f"root={self.root}",
            f"subject={self.subject}",
        ]

        if self.session:
            parts.append(f"session={self.session}")
        if self.task:
            parts.append(f"task={self.task}")
        if self.eeg_type:
            parts.append(f"type={self.eeg_type}")
        if self.space:
            parts.append(f"space={self._space}")

        return f"BIDSReader({', '.join(parts)})"
    
    def __repr__(self) -> str:
        return (
            f"BIDSReader(root={self.root!r}, subject={self.subject!r}, "
            f"session={self.session!r}, task={self.task!r}, "
            f"eeg_type={self.eeg_type!r}, space={self._space!r})"
        )

    # ---------- property ----------
    @property
    def space(self) -> str:
        if self._space is None:
            self._space = self._determine_space()
        return self._space
    
    # ---------- internal helpers ----------

    def _bp(self, **kwargs) -> BIDSPath:
        bp = BIDSPath(
            root=self.root,
            subject=self.subject,
            session=str(self.session) if self.session is not None else None,
            task=self.task,
            datatype=self.eeg_type,
        )
        bp.update(**kwargs)
        return bp

    def _subject_root(self) -> Path:
        p = self.root / self._add_bids_prefix("subject", self.subject)
        return p
    
    def _add_bids_prefix(self, field: str, value: Optional[str]) -> Optional[str]:
        prefix_map = {
            "subject": "sub-",
            "session": "ses-",
            "acquisition": "acq-",
            "task": "task-",
            "space": "space-",
        }

        if field not in prefix_map:
            raise InvalidOptionError(f"Unknown BIDS field: {field}")

        return add_prefix(value, prefix_map[field])

    
    def _validate_acq(self, acquisition: Optional[str]) -> Optional[str]:
        if not self.is_intracranial():
            # scalp EEG: ignore acquisition entirely
            return None

        if acquisition is None:
            raise InvalidOptionError("acquisition was not set to bipolar, monopolar")

        return validate_option("acquisition", acquisition, VALID_ACQ)
    
    def _require(self, fields: Iterable[str], context: str = "") -> None:
        missing = [f for f in fields if getattr(self, f, None) in (None, "")]
        if missing:
            raise MissingRequiredFieldError(
                f"{context}: missing required fields: {', '.join(missing)}"
            )
            
    def _get_needed_fields(self):
        return INTRACRANIAL_FIELDS if self.is_intracranial() else SCALP_FIELDS
    
    def _attach_bipolar_midpoint_montage(self, raw: mne.io.BaseRaw) -> None:
        pairs_df = self.load_channels("bipolar")
        elec_df = self.load_electrodes()
        combo = combine_bipolar_electrodes(pairs_df, elec_df)

        if not {"name", "x_mid", "y_mid", "z_mid"}.issubset(combo.columns):
            return

        ch_pos = {
            str(r["name"]): (float(r["x_mid"]), float(r["y_mid"]), float(r["z_mid"]))
            for _, r in combo.iterrows()
            if np.isfinite(r["x_mid"]) and np.isfinite(r["y_mid"]) and np.isfinite(r["z_mid"])
        }
        if not ch_pos:
            return
        
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mni_tal")
        raw.set_montage(montage, on_missing="ignore")

    def _determine_space(self) -> str:
        subject_root = self._subject_root()
        data_dir = subject_root / self._add_bids_prefix("session", self.session) / self.eeg_type

        if not data_dir.exists():
            raise FileNotFoundBIDSError(
                f"determine_space: data directory does not exist.\n"
                f"subject_root={subject_root}\n"
                f"data_dir={data_dir}"
            )

        matches = list(data_dir.glob("*_coordsystem.json"))
        if not matches:
            raise FileNotFoundBIDSError(
                f"determine_space: no *_coordsystem.json file found.\n"
                f"data_dir={data_dir}"
            )

        if len(matches) > 1:
            raise AmbiguousMatchError(
                f"determine_space: multiple coordsystem files found.\n"
                f"files={[m.name for m in matches]}"
            )

        fname = matches[0].name
        space = space_from_coordsystem_fname(fname)

        if space is None:
            raise DataParseError(
                f"determine_space: could not parse space from filename.\n"
                f"filename={fname}"
            )

        return space

    # ---------- public API ----------
    @public_api
    def is_intracranial(self) -> bool:
        return self.eeg_type == "ieeg"
    
    # ---------- loaders ----------
    @public_api
    def load_events(self, event_type: str) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_events")
        allowed = ["beh", *VALID_EEG_TYPES] 
        event_type = validate_option("event_type", event_type, allowed)
        suffix = "beh" if event_type == "beh" else "events"

        bp = self._bp(
            datatype=event_type,
            suffix=suffix,
            extension=".tsv",
        )

        matches = bp.match()
        if not matches:
            raise FileNotFoundBIDSError(f"load_events: no file matched for {bp}")

        return pd.read_csv(matches[0].fpath, sep="\t")

    @public_api
    def load_electrodes(self) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_electrodes")

        # if space wasn’t set and cannot be determined, raise one clean message
        self._space = self._determine_space() if self._space is None else self._space
        _task = _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.eeg_type, suffix="electrodes", space=self._space, task=_task, extension=".tsv")
        return pd.read_csv(bp.fpath, sep="\t")

    @public_api
    def load_channels(self, acquisition: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_channels")

        acq = self._validate_acq(acquisition)
        bp = self._bp(datatype=self.eeg_type, suffix="channels", acquisition=acq, extension=".tsv")
        return pd.read_csv(bp.fpath, sep="\t")

    @public_api
    def load_combined_channels(self, acquisition: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_combined_channels")
        acq = self._validate_acq(acquisition)
        
        channel_df = self.load_channels(acquisition)
        elec_df = self.load_electrodes()
        if acquisition == "monopolar" or acquisition is None:
            return channel_df.merge( elec_df, on="name", how="left", suffixes=("", "_elec"), )
        if acquisition == "bipolar":
            return combine_bipolar_electrodes(channel_df, elec_df)

    @public_api
    def load_coordsystem_desc(self) -> Dict:
        self._require(self._get_needed_fields(), context="load_coordsystem")
        self._space = self._determine_space() if self._space is None else self._space

        _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.eeg_type, suffix="coordsystem", space=self._space, task=_task, extension=".json",
        )

        with open(bp.fpath, "r") as f:
            return json.load(f)

    @public_api
    def load_raw(self, acquisition: Optional[str] = None) -> mne.io.BaseRaw:
        self._require(self._get_needed_fields(), context="load_raw")

        acq = self._validate_acq(acquisition)  # None for scalp EEG; validated for iEEG

        bp_kwargs = {"datatype": self.eeg_type}
        if acq is not None:
            bp_kwargs["acquisition"] = acq
        bp = self._bp(**bp_kwargs)

        # Mute only the specific montage warning emitted during read_raw_bids
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"DigMontage is only a subset of info\.",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*is not an MNE-Python coordinate frame.*",
                category=RuntimeWarning,
            )
            raw = read_raw_bids(bp)
            
        # If bipolar iEEG: compute midpoints and attach montage so channel positions exist
        if self.is_intracranial() and acq == "bipolar":
            self._attach_bipolar_midpoint_montage(raw)

        return raw

    @public_api
    def load_epochs(self, tmin: float, tmax : float, trial_types: Optional[Iterable[str]] = None, baseline: Optional[Tuple[float | None, float | None]] = None, acquisition: Optional[str] =None, event_repeated: str = "merge", channels: Optional[Iterable[str]] = None, preload: bool = False) -> mne.Epochs:
        self._require(self._get_needed_fields(), context="load_epochs")
        raw = self.load_raw(acquisition = acquisition)
        events_raw, event_id = mne.events_from_annotations(raw)
        
        if trial_types is not None:
            _, events_raw, event_id = self.filter_events_by_trial_types(trial_types, events_df=None, raw=raw)
            
        picks = list(channels) if channels is not None else None

        return mne.Epochs(
            raw,
            events=events_raw,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=preload,
            event_repeated=event_repeated,
            picks=picks
        )
    
    # ---- simple metadata queries ----
    @public_api
    def get_subject_tasks(self) -> List[str]:
        # subject root = root/sub-XX (not session-specific)
        subject_root = self._subject_root()
        return get_entity_vals(subject_root, "task")

    @public_api
    def get_subject_sessions(self) -> List[str]:
        subject_root = self._subject_root()
        return get_entity_vals(subject_root, "session")
    
    @public_api
    def get_dataset_subjects(self) -> List[str]:
        return get_entity_vals(self.root, "subject")

    @public_api
    def get_dataset_tasks(self) -> List[str]:
        return get_entity_vals(self.root, "task")

    @public_api
    def get_dataset_max_sessions(self, outlier_thresh: Optional[int] = None) -> Optional[int]:
        subs = self.get_dataset_subjects()
        max_ses: Optional[int] = None

        for sub in subs:
            subject_root = self.root / f"sub-{str(sub).replace('sub-', '')}"
            sessions = get_entity_vals(subject_root, "session") or []

            for s in sessions:
                try:
                    si = int(str(s).replace("ses-", ""))
                except ValueError:
                    continue

                if outlier_thresh is not None and si > outlier_thresh:
                    warnings.warn("Session number is over 50. Double check dataset.")
                else:
                    max_ses = si if max_ses is None else max(max_ses, si)

        return max_ses
    
    # ----static methods ----
    @staticmethod
    @public_api
    def mne_to_ptsa(epochs: mne.Epochs, events_df: pd.DataFrame) -> TimeSeries:
        merged_events_df = merge_duplicate_sample_events(events_df)
        return TimeSeries.from_mne_epochs(epochs, merged_events_df)
    
    @staticmethod
    @public_api
    def raw_to_ptsa(raw : mne.io.BaseRaw, picks: Optional[Iterable[str]] = None, tmin: float = None, tmax: float = None) -> TimeSeries:
        inst = raw.copy()
        if tmin is not None or tmax is not None:
            # MNE crop uses absolute times in seconds within the recording
            inst.crop(tmin=tmin, tmax=tmax)

        # Resolve picks
        if picks is not None:
            if all(isinstance(p, str) for p in picks):
                pick_idx = [inst.ch_names.index(ch) for ch in picks]
            else:
                pick_idx = list(picks)

            data = inst.get_data(picks=pick_idx)         # shape (n_ch, n_times)
            ch_names = [inst.ch_names[i] for i in pick_idx]
        else:
            data = inst.get_data()                       # shape (n_ch, n_times)
            ch_names = inst.ch_names

        sfreq = float(inst.info["sfreq"])
        times = inst.times                               # seconds, shape (n_times,)

        ts = TimeSeries.create(
            data,
            samplerate=sfreq,
            dims=("channel", "time"),
            coords={
                "channel": np.asarray(ch_names, dtype=object),
                "time": np.asarray(times, dtype=float),
            },
            attrs={
                "mne_meas_date": str(inst.info.get("meas_date")),
                "mne_first_samp": int(inst.first_samp),
            },
        )
        return ts

    @staticmethod
    @public_api
    def filter_events_by_trial_types(trial_types: Iterable[str], events_df: pd.DataFrame = None, raw: mne.io.BaseRaw = None) -> Tuple[pd.DataFrame | None, np.ndarray | None, Dict[str, int] | None]:
        trial_types = set(trial_types)

        filtered_events_df = None
        if events_df is not None:
            filtered_events_df = events_df.loc[events_df["trial_type"].isin(trial_types)]

        filtered_events_raw = None
        filtered_event_id = None
        if raw is not None:
            events_raw, event_id = mne.events_from_annotations(raw)

            filtered_event_id = {k: v for k, v in event_id.items() if k in trial_types}
            if filtered_event_id:
                codes = np.fromiter(filtered_event_id.values(), dtype=int)
                filtered_events_raw = events_raw[np.isin(events_raw[:, 2], codes)]
            else:
                # No matching annotation labels
                filtered_events_raw = events_raw[:0].copy()  # empty (n_events=0, 3)
                filtered_event_id = {}

        return filtered_events_df, filtered_events_raw, filtered_event_id