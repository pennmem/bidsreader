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
from .helpers import validate_option, space_from_coordsystem_fname, add_prefix, merge_duplicate_sample_events, combine_bipolar_electrodes, match_event_label
from .exc import BIDSReaderError, InvalidOptionError, MissingRequiredFieldError, FileNotFoundBIDSError, AmbiguousMatchError, DataParseError, DependencyError, ExternalLibraryError

class BIDSReader:
    _FIELDS = {"root", "subject", "session", "task", "acquisition", "eeg_type", "_space"}
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = CML_ROOT,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str | int] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        eeg_type: Optional[str] = None,
    ):
        self.root = Path(root)
        self.subject = subject
        self.session = session
        self.task = str(task)
        
        self.acquisition = acquisition
        
        self._eeg_type = validate_option(
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
            parts.append(f"space={self.space}")

        return f"BIDSReader({', '.join(parts)})"
    
    def __repr__(self) -> str:
        return (
            f"BIDSReader(root={self.root!r}, subject={self.subject!r}, "
            f"session={self.session!r}, task={self.task!r}, "
            f"eeg_type={self.eeg_type!r}, space={self.space!r})"
        )
    
    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._FIELDS:
            raise AttributeError(f"Unknown field: {name}")
        object.__setattr__(self, name, value)

    # ---------- property ----------
    @property
    def space(self) -> Optional[str]:
        if self._space is not None:
            return self._space

        try:
            self._space = self._determine_space()
        except Exception as e:
            warnings.warn(
                f"Could not determine space automatically: {e}",
                RuntimeWarning,
            )
            return None

        return self._space
    
    @property
    def eeg_type(self) -> Optional[str]:
        if self._eeg_type is not None:
            return self._eeg_type

        try:
            self._eeg_type = self._determine_eeg_type()
        except Exception as e:
            warnings.warn(
                f"Could not determine eeg_type automatically: {e}",
                RuntimeWarning,
            )
            return None

        if self._eeg_type is None:
            warnings.warn(
                "eeg_type could not be inferred from subject.",
                RuntimeWarning,
            )

        return self._eeg_type
    
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
    
    def _determine_eeg_type(self):
        if self.subject is None:
            return None
        if self.subject.startswith("LTP"):
            return "eeg"
        if self.subject.startswith("R"):
            return "ieeg"
        return None

    # ---------- public API ----------
    @public_api
    def set_fields(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # validated by __setattr__
        return self
        
    @public_api
    def is_intracranial(self) -> bool:
        return self.eeg_type == "ieeg"
    
    # ---------- loaders ----------
    @public_api
    def load_events(self, event_type: str = "beh") -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_events")
        allowed = ["beh", self.eeg_type]
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

        _task = _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.eeg_type, suffix="electrodes", space=self.space, task=_task, extension=".tsv")
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

        _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.eeg_type, suffix="coordsystem", space=self.space, task=_task, extension=".json",
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
    def load_epochs(
        self,
        tmin: float,
        tmax: float,
        events: Optional[pd.DataFrame] = None,
        baseline: Optional[Tuple[float | None, float | None]] = None,
        acquisition: Optional[str] = None,
        event_repeated: str = "merge",
        channels: Optional[Iterable[str]] = None,
        preload: bool = False,
    ) -> mne.Epochs:
        self._require(self._get_needed_fields(), context="load_epochs")
        raw = self.load_raw(acquisition=acquisition)

        # Always derive the canonical event_id mapping from the raw annotations
        all_events_raw, all_event_id = mne.events_from_annotations(raw)

        if events is not None:
            if "sample" not in events.columns:
                raise ValueError("Events DataFrame must contain a 'sample' column")

            # Use the same integer codes that events_from_annotations would assign
            if "trial_type" in events.columns:
                codes = events["trial_type"].map(all_event_id)
                if codes.isna().any():
                    missing = set(events.loc[codes.isna(), "trial_type"].unique())
                    raise ValueError(
                        f"trial_type values not found in raw annotations: {missing}"
                    )
                codes = codes.values.astype(int)
                # Only keep event_id entries for types present in the passed events
                present_types = set(events["trial_type"].unique())
                event_id = {k: v for k, v in all_event_id.items() if k in present_types}
            else:
                codes = np.ones(len(events), dtype=int)
                event_id = {"event": 1}

            events_raw = np.column_stack([
                events["sample"].values.astype(int),
                np.zeros(len(events), dtype=int),
                codes,
            ])
        else:
            events_raw = all_events_raw
            event_id = all_event_id

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
            picks=picks,
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
    def mne_epochs_to_ptsa(epochs: mne.Epochs, events: pd.DataFrame) -> TimeSeries:
        events = merge_duplicate_sample_events(events)
        return TimeSeries.from_mne_epochs(epochs, events)
    
    @staticmethod
    @public_api
    def mne_raw_to_ptsa(raw : mne.io.BaseRaw, picks: Optional[Iterable[str]] = None, tmin: float = None, tmax: float = None) -> TimeSeries:
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
    def _label_has_trial_type(label: str, trial_types: list[str]) -> bool:
        # exact token match within merged labels like "WORD/STIM"
        tokens = label.split("/")
        return any(t in tokens for t in trial_types)

    @staticmethod
    def _ensure_list(trial_types: Iterable[str] | str) -> list[str]:
        return [trial_types] if isinstance(trial_types, str) else list(trial_types)


    @staticmethod
    @public_api
    def filter_events_df_by_trial_types(
        events_df: pd.DataFrame,
        trial_types: Iterable[str] | str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        tt = BIDSReader._ensure_list(trial_types)

        mask = events_df["trial_type"].isin(tt).to_numpy()
        filtered_df = events_df.loc[mask].copy()

        # integer positions (0..n-1) into the *original* events_df rows
        df_idx = np.flatnonzero(mask)

        return filtered_df, df_idx


    @staticmethod
    @public_api
    def filter_raw_events_by_trial_types(
        raw: mne.io.BaseRaw,
        trial_types: Iterable[str] | str,
    ) -> tuple[np.ndarray, Dict[str, int], np.ndarray]:
        tt = BIDSReader._ensure_list(trial_types)

        events_raw, event_id = mne.events_from_annotations(raw)

        filtered_event_id = {
            k: v for k, v in event_id.items()
            if BIDSReader._label_has_trial_type(k, tt)
        }

        if filtered_event_id:
            codes = np.fromiter(filtered_event_id.values(), dtype=int)
            mask = np.isin(events_raw[:, 2], codes)
            filtered_events = events_raw[mask]
            raw_idx = np.flatnonzero(mask)  # indices into events_raw
        else:
            filtered_events = events_raw[:0].copy()
            filtered_event_id = {}
            raw_idx = np.array([], dtype=int)

        return filtered_events, filtered_event_id, raw_idx


    @staticmethod
    @public_api
    def filter_epochs_by_trial_types(
        epochs: mne.Epochs,
        trial_types: Iterable[str] | str,
    ) -> tuple[mne.Epochs, Dict[str, int], np.ndarray]:
        tt = BIDSReader._ensure_list(trial_types)

        keys = [
            k for k in epochs.event_id.keys()
            if BIDSReader._label_has_trial_type(k, tt)
        ]
        filtered_event_id = {k: epochs.event_id[k] for k in keys}

        if keys:
            filtered_epochs = epochs[keys]
            codes = np.fromiter(filtered_event_id.values(), dtype=int)
            mask = np.isin(epochs.events[:, 2], codes)
            ep_idx = np.flatnonzero(mask)  # indices into original epochs
        else:
            filtered_epochs = epochs.copy()[[]]
            ep_idx = np.array([], dtype=int)

        return filtered_epochs, filtered_event_id, ep_idx
    
    @staticmethod
    @public_api
    def filter_by_trial_types(
        trial_types: Iterable[str] | str,
        *,
        events_df: Optional[pd.DataFrame] = None,
        raw: Optional[mne.io.BaseRaw] = None,
        epochs: Optional[mne.Epochs] = None,
    ) -> tuple[
        Optional[pd.DataFrame],
        Optional[np.ndarray],     # filtered_events (from raw)
        Optional[mne.Epochs],
        Dict[str, int],
        np.ndarray,               # filtered_event_idx (0..n-1)
    ]:
        tt = BIDSReader._ensure_list(trial_types)

        filtered_df: Optional[pd.DataFrame] = None
        filtered_events: Optional[np.ndarray] = None
        filtered_epochs: Optional[mne.Epochs] = None

        df_idx: Optional[np.ndarray] = None
        raw_idx: Optional[np.ndarray] = None
        ep_idx: Optional[np.ndarray] = None

        event_id_raw: Optional[Dict[str, int]] = None
        event_id_epochs: Optional[Dict[str, int]] = None

        n_df = None
        n_raw = None
        n_ep = None

        # ---- DF ----
        if events_df is not None:
            filtered_df, df_idx = BIDSReader.filter_events_df_by_trial_types(events_df, tt)
            n_df = len(filtered_df)

        # ---- RAW ----
        raw_onsets = None
        if raw is not None:
            filtered_events, event_id_raw, raw_idx = BIDSReader.filter_raw_events_by_trial_types(raw, tt)
            n_raw = int(filtered_events.shape[0])
            raw_onsets = filtered_events[:, 0].astype(int)

        # ---- EPOCHS ----
        ep_onsets = None
        if epochs is not None:
            filtered_epochs, event_id_epochs, ep_idx = BIDSReader.filter_epochs_by_trial_types(epochs, tt)
            n_ep = len(filtered_epochs)

            if event_id_epochs:
                codes = np.fromiter(event_id_epochs.values(), dtype=int)
                mask = np.isin(epochs.events[:, 2], codes)
                ep_onsets = epochs.events[mask, 0].astype(int)
            else:
                ep_onsets = np.array([], dtype=int)

        # ---- Check event_id consistency (keys) ----
        if event_id_raw is not None and event_id_epochs is not None:
            if set(event_id_raw.keys()) != set(event_id_epochs.keys()):
                raise ValueError(
                    "filtered_event_id key mismatch between raw and epochs.\n"
                    f"raw keys={sorted(event_id_raw.keys())}\n"
                    f"epochs keys={sorted(event_id_epochs.keys())}"
                )

            # Strong alignment check: same event onsets
            if raw_onsets is not None and ep_onsets is not None:
                if raw_onsets.shape != ep_onsets.shape or not np.array_equal(raw_onsets, ep_onsets):
                    raise ValueError(
                        "raw/epochs trial alignment mismatch: filtered event sample onsets differ.\n"
                        f"n_raw={len(raw_onsets)} n_epochs={len(ep_onsets)}"
                    )

            filtered_event_id = event_id_raw  # choose one
        elif event_id_raw is not None:
            filtered_event_id = event_id_raw
        elif event_id_epochs is not None:
            filtered_event_id = event_id_epochs
        else:
            filtered_event_id = {}

        # ---- Trial count consistency (for whichever inputs are provided) ----
        ns = [n for n in (n_df, n_raw, n_ep) if n is not None]
        n = ns[0] if ns else 0

        if any(x != n for x in ns):
            raise ValueError(
                "Trial count mismatch across provided inputs.\n"
                f"n_df={n_df} n_raw={n_raw} n_epochs={n_ep}"
            )

        filtered_event_idx = np.arange(n, dtype=int)
        return filtered_df, filtered_events, filtered_epochs, filtered_event_id, filtered_event_idx

    # ---------- unit constants (class-level) ----------
    _UNIT_EXPONENTS = {
        "V": 0, "mV": -3, "uV": -6, "nV": -9,
        "T": 0, "mT": -3, "uT": -6, "nT": -9, "fT": -15,
    }

    _FIFF_UNIT_TO_BASE = {107: "V", 201: "T", 0: None}

    _FIFF_MUL_TO_EXP = {
        0: 0, -3: -3, -6: -6, -9: -9, -12: -12, -15: -15, 3: 3, 6: 6,
    }

    _EXP_TO_PREFIX = {
        0: "", -3: "m", -6: "u", -9: "n", -12: "p", -15: "f", 3: "k", 6: "M",
    }

    # ---------- unit internal helpers ----------

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        return unit.replace("µ", "u")

    @staticmethod
    def _detect_unit_mne(inst: Union[mne.io.BaseRaw, mne.Epochs]) -> str:
        """Detect unit string from an MNE Raw or Epochs object."""
        eeg_types = {"eeg", "seeg", "ecog", "ieeg", "dbs"}

        for ch_info in inst.info["chs"]:
            ch_kind = mne.io.pick.channel_type(
                inst.info, inst.ch_names.index(ch_info["ch_name"]),
            )
            if ch_kind not in eeg_types:
                continue

            fiff_unit = ch_info.get("unit", 0)
            fiff_mul = ch_info.get("unit_mul", 0)

            base = BIDSReader._FIFF_UNIT_TO_BASE.get(fiff_unit)
            if base is None:
                raise ValueError(
                    f"Unknown FIFF unit code {fiff_unit} on channel "
                    f"'{ch_info['ch_name']}'. Pass current_unit= explicitly."
                )

            exp = BIDSReader._FIFF_MUL_TO_EXP.get(fiff_mul, 0)
            prefix = BIDSReader._EXP_TO_PREFIX.get(exp, "")
            return f"{prefix}{base}"

        raise ValueError(
            "No EEG/iEEG/SEEG/ECoG channel found. Cannot detect unit."
        )

    @staticmethod
    def _detect_unit_ptsa(ts: TimeSeries) -> str:
        """Detect unit string from a PTSA TimeSeries."""
        for key in ("units", "unit"):
            val = ts.attrs.get(key)
            if val is not None and str(val).strip():
                unit_str = BIDSReader._normalize_unit(str(val).strip())
                if unit_str in BIDSReader._UNIT_EXPONENTS:
                    return unit_str
                raise ValueError(
                    f"TimeSeries has unit '{val}' which is not recognized. "
                    f"Known: {sorted(BIDSReader._UNIT_EXPONENTS.keys())}"
                )

        raise ValueError(
            "TimeSeries has no 'units' or 'unit' attribute. "
            "Pass current_unit= explicitly."
        )

    @staticmethod
    def _convert_mne(
        inst: Union[mne.io.BaseRaw, mne.Epochs],
        factor: float,
        target_unit: str,
        copy: bool,
    ) -> Union[mne.io.BaseRaw, mne.Epochs]:
        """Scale MNE data and update FIFF unit metadata."""
        if copy:
            inst = inst.copy()

        inst.apply_function(lambda x: x * factor, picks="all", channel_wise=False)

        base_char = target_unit[-1]
        target_exp = BIDSReader._UNIT_EXPONENTS[target_unit]
        fiff_unit_code = {"V": 107, "T": 201}.get(base_char, 0)
        fiff_mul = min(
            BIDSReader._FIFF_MUL_TO_EXP.keys(),
            key=lambda k: abs(BIDSReader._FIFF_MUL_TO_EXP[k] - target_exp),
        )

        # Update unit metadata on all EEG/SEEG/ECoG channels
        eeg_kinds = {2, 302, 802, 803}  # EEG, EEG_REF, SEEG, ECOG
        for ch in inst.info["chs"]:
            if ch.get("kind", 0) in eeg_kinds or ch.get("unit", 0) in (107, 201):
                ch["unit"] = fiff_unit_code
                ch["unit_mul"] = fiff_mul

        return inst

    @staticmethod
    def _convert_ptsa(
        ts: TimeSeries,
        factor: float,
        target_unit: str,
        copy: bool,
    ) -> TimeSeries:
        """Scale PTSA TimeSeries data and update attrs."""
        if copy:
            result = ts * factor
        else:
            ts.values[:] *= factor
            result = ts

        # Update all unit-related attrs so users know the current unit
        result.attrs["units"] = target_unit
        result.attrs["unit"] = target_unit

        return result

    # ---------- unit public API ----------

    @staticmethod
    @public_api
    def detect_unit(
        data: Union[mne.io.BaseRaw, mne.Epochs, TimeSeries],
        current_unit: Optional[str] = None,
    ) -> str:
        """Detect or validate the unit of EEG data.

        Parameters
        ----------
        data : mne.io.BaseRaw, mne.Epochs, or PTSA TimeSeries
            The data object to inspect.
        current_unit : str, optional
            If provided, overrides auto-detection. Validated against
            known units and returned directly.

        Returns
        -------
        str
            Unit string like "V", "mV", "uV", "nV", "T", etc.

        Raises
        ------
        ValueError
            If unit cannot be detected and current_unit is not provided.
        """
        if current_unit is not None:
            normalized = BIDSReader._normalize_unit(current_unit)
            if normalized not in BIDSReader._UNIT_EXPONENTS:
                raise ValueError(
                    f"Unknown unit '{current_unit}'. "
                    f"Known: {sorted(BIDSReader._UNIT_EXPONENTS.keys())}"
                )
            return normalized

        if isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
            return BIDSReader._detect_unit_mne(data)

        if isinstance(data, TimeSeries):
            return BIDSReader._detect_unit_ptsa(data)

        raise TypeError(
            f"Cannot detect unit from {type(data).__name__}. "
            f"Expected mne.io.BaseRaw, mne.Epochs, or TimeSeries."
        )

    @staticmethod
    @public_api
    def get_scale_factor(from_unit: str, to_unit: str) -> float:
        """Compute multiplicative factor to convert between units.

        Parameters
        ----------
        from_unit : str
            Current unit (e.g. "V").
        to_unit : str
            Target unit (e.g. "uV").

        Returns
        -------
        float
            Multiply data by this value to convert.

        Examples
        --------
        >>> BIDSReader.get_scale_factor("V", "uV")
        1000000.0
        >>> BIDSReader.get_scale_factor("uV", "V")
        1e-06
        """
        from_u = BIDSReader._normalize_unit(from_unit)
        to_u = BIDSReader._normalize_unit(to_unit)

        if from_u not in BIDSReader._UNIT_EXPONENTS:
            raise ValueError(f"Unknown source unit '{from_unit}'")
        if to_u not in BIDSReader._UNIT_EXPONENTS:
            raise ValueError(f"Unknown target unit '{to_unit}'")

        from_base = from_u[-1]
        to_base = to_u[-1]
        if from_base != to_base:
            raise ValueError(
                f"Cannot convert between different base units: "
                f"'{from_unit}' ({from_base}) -> '{to_unit}' ({to_base})"
            )

        from_exp = BIDSReader._UNIT_EXPONENTS[from_u]
        to_exp = BIDSReader._UNIT_EXPONENTS[to_u]
        return 10.0 ** (from_exp - to_exp)

    @staticmethod
    @public_api
    def convert_unit(
        data: Union[mne.io.BaseRaw, mne.Epochs, TimeSeries],
        target: str,
        *,
        current_unit: Optional[str] = None,
        copy: bool = True,
    ) -> Union[mne.io.BaseRaw, mne.Epochs, TimeSeries]:
        """Convert EEG data to a target unit.

        Parameters
        ----------
        data : mne.io.BaseRaw, mne.Epochs, or PTSA TimeSeries
            The data to convert.
        target : str
            Target unit string (e.g. "uV", "mV", "V").
        current_unit : str, optional
            Override auto-detection of the current unit. Required if
            the data object doesn't store unit metadata.
        copy : bool
            If True (default), return a copy. If False, modify in place.

        Returns
        -------
        Same type as input, with data scaled to the target unit.
        """
        detected = BIDSReader.detect_unit(data, current_unit=current_unit)
        target_normalized = BIDSReader._normalize_unit(target)
        factor = BIDSReader.get_scale_factor(detected, target_normalized)

        if factor == 1.0:
            return data.copy() if copy else data

        if isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
            return BIDSReader._convert_mne(data, factor, target_normalized, copy)

        if isinstance(data, TimeSeries):
            return BIDSReader._convert_ptsa(data, factor, target_normalized, copy)

        raise TypeError(f"Cannot convert type {type(data).__name__}")