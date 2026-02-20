import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import mne
from ptsa.data.timeseries import TimeSeries
from pathlib import Path
from typing import Iterable, Any
import re
from typing import Sequence, Optional, Union
import numpy as np
import pandas as pd
import warnings

# modular level helpers
def _validate_option(name: str, value: Any, allowed: Iterable[Any]):
    if value is None:
        return None
    if value not in allowed:
        allowed_str = ", ".join(str(a) for a in allowed)
        raise ValueError(f"{name} was not set to {allowed_str}")
    return value


def _space_from_coordsystem_fname(fname: str) -> Optional[str]:
    if "_space-" not in fname:
        return None
    return fname.split("_space-")[1].split("_coordsystem.json")[0]

def _add_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    if value is None:
        return None

    value = str(value)

    # Avoid double-prefixing
    if value.startswith(prefix):
        return value

    return f"{prefix}{value}"

def _merge_duplicate_sample_events(evs: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
    df = evs.copy()

    # Ensure stable ordering so "first" is well-defined.
    df["_orig_order"] = np.arange(len(df))

    def first_non_nan(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    def merge_series(s: pd.Series):
        # General "take the first non-NaN; if only one non-NaN, that's what it is" behavior
        return first_non_nan(s)

    def merge_trial_type(s: pd.Series):
        vals = [v for v in s.tolist() if pd.notna(v)]
        # preserve order but avoid duplicates like A/A
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        if not uniq:
            return np.nan
        return "/".join(map(str, uniq))

    merged_rows = []
    for sample_val, g in df.sort_values("_orig_order").groupby(sample_col, sort=False):
        out = {}
        for col in df.columns:
            if col in ("_orig_order",):
                continue
            if col == "trial_type":
                out[col] = merge_trial_type(g[col])
            else:
                out[col] = merge_series(g[col])
        merged_rows.append(out)

    out_df = pd.DataFrame(merged_rows)

    # If you want to preserve original column order (minus helper col)
    out_df = out_df[[c for c in evs.columns if c in out_df.columns]]

    return out_df

def _find_coord_triplets(columns: Sequence[str]):
        """
        Find coordinate triplets in columns:
          - x,y,z
          - <prefix>.x, <prefix>.y, <prefix>.z  (e.g., tal.x, tal.y, tal.z)
        Returns dict: prefix -> (xcol, ycol, zcol)
        where prefix is '' for bare x/y/z.
        """
        cols = set(columns)

        triplets = {}

        # bare x/y/z
        if {"x", "y", "z"} <= cols:
            triplets[""] = ("x", "y", "z")

        # prefixed *.x/*.y/*.z
        # match things like "tal.x"
        prefixed = [c for c in cols if re.match(r"^.+\.(x|y|z)$", c)]
        prefixes = set(c.rsplit(".", 1)[0] for c in prefixed)

        for p in prefixes:
            x, y, z = f"{p}.x", f"{p}.y", f"{p}.z"
            if {x, y, z} <= cols:
                triplets[p] = (x, y, z)

        return triplets


class BIDSReader:
    INTRACRANIAL_FIELDS = ("subject", "task", "session", "eeg_type")
    SCALP_FIELDS = ("subject", "task", "session", "eeg_type")
    VALID_EEG_TYPES = ("eeg", "ieeg")
    VALID_SPACES = ("MNI152NLin6ASym", "Talarich")
    VALID_ACQ = ("bipolar", "monopolar")

    def __init__(
        self,
        root: Union[str, Path],
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str] = None,
        space: Optional[str] = None,
        eeg_type: Optional[str] = None,
    ):
        self.root = Path(root)
        self.subject = subject
        self.session = session
        self.task = task
        
        self.eeg_type = _validate_option(
            "eeg_type", eeg_type, self.VALID_EEG_TYPES
        )
        
        if not self.is_intracranial():
            self._space = None
        else:
            if space is not None:
                self._space = _validate_option(
                    "space", space, VALID_SPACES
                )
            else:
                self._space = self.determine_space()
        

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

    # ---------- property ----------
    @property
    def space(self):
        if self._space is not None:
            return self._space

        try:
            self._space = self._determine_space()
            return self._space
        except Exception:
            raise 
            
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
            raise ValueError(f"Unknown BIDS field: {field}")

        return _add_prefix(value, prefix_map[field])

    
    def _validate_acq(self, acquisition: Optional[str]) -> Optional[str]:
        if not self.is_intracranial():
            # scalp EEG: ignore acquisition entirely
            return None

        if acquisition is None:
            raise ValueError("acquisition was not set to bipolar, monopolar")

        return _validate_option("acquisition", acquisition, self.VALID_ACQ)
    
    def _require(self, fields: Iterable[str], context: str = "") -> None:
        missing = [f for f in fields if getattr(self, f, None) in (None, "")]
        if missing:
            prefix = f"{context}: " if context else ""
            raise ValueError(
                prefix + "One or more required fields were not set: " + ", ".join(missing)
            )
            
    def _get_needed_fields(self):
        return self.INTRACRANIAL_FIELDS if self.is_intracranial() else self.SCALP_FIELDS

    def _combine_bipolar_electrodes(
        self,
        pairs_df: pd.DataFrame,
        elec_df: pd.DataFrame,
        pair_col: str = "name",
        elec_name_col: str = "name",
        region_cols: Sequence[str] = ("wb.region", "ind.region", "stein.region"),
    ) -> pd.DataFrame:
        sep = "-"
        out = pairs_df.copy()

        # Split bipolar pair
        ch = out[pair_col].astype(str).str.split(sep, n=1, expand=True)
        out["ch1"] = ch[0].str.strip()
        out["ch2"] = ch[1].str.strip()

        # Detect all coordinate triplets present in electrodes df
        coord_triplets = _find_coord_triplets(elec_df.columns)

        # Keep electrode name + region cols + all coordinate columns we found
        coord_cols = [c for trip in coord_triplets.values() for c in trip]
        keep_cols = [elec_name_col, *region_cols, *coord_cols]
        keep_cols = [c for c in keep_cols if c in elec_df.columns]  # safety

        look = elec_df[keep_cols].copy()

        # Merge ch1 metadata
        look1 = look.add_suffix("_ch1").rename(columns={f"{elec_name_col}_ch1": "ch1"})
        out = out.merge(look1, on="ch1", how="left")

        # Merge ch2 metadata
        look2 = look.add_suffix("_ch2").rename(columns={f"{elec_name_col}_ch2": "ch2"})
        out = out.merge(look2, on="ch2", how="left")

        # Region agreement
        for rc in region_cols:
            if f"{rc}_ch1" in out.columns and f"{rc}_ch2" in out.columns:
                a = out[f"{rc}_ch1"]
                b = out[f"{rc}_ch2"]
                out[f"{rc}_pair"] = np.where(a.notna() & (a == b), a, np.nan)

        # Midpoints for every detected coordinate triplet
        for prefix, (xcol, ycol, zcol) in coord_triplets.items():
            for col in (xcol, ycol, zcol):
                a = out[f"{col}_ch1"]
                b = out[f"{col}_ch2"]
                mid_name = f"{col}_mid"  # e.g., "x_mid" or "tal.x_mid"
                out[mid_name] = np.where(a.notna() & b.notna(), (a + b) / 2.0, np.nan)

        return out
    
    def _attach_bipolar_midpoint_montage(self, raw) -> None:
        pairs_df = self.load_channels("bipolar")
        elec_df = self.load_electrodes()
        combo = self._combine_bipolar_electrodes(pairs_df, elec_df)

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
        if not self.is_intracranial():
            raise ValueError(
                f"determine_space: subject={self.subject}, "
                f"eeg_type={self.eeg_type} is not intracranial."
            )

        subject_root = self._subject_root()
        data_dir = subject_root / self._add_bids_prefix("session", self.session) / self.eeg_type

        if not data_dir.exists():
            raise FileNotFoundError(
                f"determine_space: data directory does not exist.\n"
                f"subject_root={subject_root}\n"
                f"data_dir={data_dir}"
            )

        matches = list(data_dir.glob("*_coordsystem.json"))
        if not matches:
            raise FileNotFoundError(
                f"determine_space: no *_coordsystem.json file found.\n"
                f"data_dir={data_dir}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"determine_space: multiple coordsystem files found.\n"
                f"files={[m.name for m in matches]}"
            )

        fname = matches[0].name
        space = _space_from_coordsystem_fname(fname)

        if space is None:
            raise ValueError(
                f"determine_space: could not parse space from filename.\n"
                f"filename={fname}"
            )

        return space

    # ---------- public API ----------
    def is_intracranial(self):
        return self.eeg_type == "ieeg"
    
    # ---------- loaders ----------

    def load_events(self, event_type: str) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_events")
        allowed = ["beh", *self.VALID_EEG_TYPES] 
        event_type = _validate_option("event_type", event_type, allowed)
        suffix = "beh" if event_type == "beh" else "events"

        bp = self._bp(
            datatype=event_type,
            suffix=suffix,
            extension=".tsv",
        )

        matches = bp.match()
        if not matches:
            raise ValueError(f"load_events: no file matched for {bp}")

        return pd.read_csv(matches[0].fpath, sep="\t")


    def load_electrodes(self) -> pd.DataFrame:
        if not self.is_intracranial():
            raise ValueError("load_electrodes: subject is not intracranial (eeg_type must be 'ieeg').")
        self._require(self._get_needed_fields(), context="load_electrodes")

        # if space wasn’t set and cannot be determined, raise one clean message
        if self.space is None:
            raise ValueError("load_electrodes: space was not set and could not be determined from *_coordsystem.json")
        bp = self._bp(datatype=self.eeg_type, suffix="electrodes", space=self.space, extension=".tsv")
        return pd.read_csv(bp, sep="\t")

    def load_channels(self, acquisition: str) -> pd.DataFrame:
        if not self.is_intracranial():
            raise ValueError("load_channels: subject is not intracranial (eeg_type must be 'ieeg').")
        self._require(self._get_needed_fields(), context="load_channels")

        acq = self._validate_acq(acquisition)

        bp = self._bp(datatype=self.eeg_type, suffix="channels", acquisition=acq, extension=".tsv")
        return pd.read_csv(bp, sep="\t")


    def load_combined_channels(self, acquisition: str) -> pd.DataFrame:
        if not self.is_intracranial():
            return ValueError("Cannot load bipolar dataframe, subject is not intracranial participant!")
        self._require(self._get_needed_fields(), context="load_combined_channels")
        acq = self._validate_acq(acquisition)  # returns None for scalp EEG, validates/raises for iEEG
        
        channel_df = self.load_channels(acquisition)
        elec_df = self.load_electrodes()
        if acquisition == "bipolar":
            return self._combine_bipolar_electrodes(channel_df, elec_df)
        if acquisition == "monopolar":
            return channel_df.merge( elec_df, on="name", how="left", suffixes=("", "_elec"), )


#     def load_raw(self, acquisition: Optional[str] = None):
#         needed_fields = self._get_needed_fields()
#         self._require(needed_fields, context="load_raw")

#         acq = self._validate_acq(acquisition)  # returns None for scalp EEG, validates/raises for iEEG

#         bp_kwargs = {"datatype": self.eeg_type}
#         if acq is not None:
#             bp_kwargs["acquisition"] = acq

#         bp = self._bp(**bp_kwargs)

#         # optional: force match before read_raw_bids so errors are consistent
#         return read_raw_bids(bp)

    def load_raw(self, acquisition: Optional[str] = None):
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


    def load_epochs(self, tmin, tmax, trial_types=None, baseline=None, acquisition=None, event_repeated="merge", channels=None, preload=False):
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
    def get_subject_tasks(self):
        # subject root = root/sub-XX (not session-specific)
        subject_root = self._subject_root()
        return get_entity_vals(subject_root, "task")

    def get_subject_sessions(self):
        subject_root = self._subject_root()
        return get_entity_vals(subject_root, "session")
    
    def get_dataset_subjects(self):
        return get_entity_vals(self.root, "subject")


    def get_dataset_tasks(self):
        return get_entity_vals(self.root, "task")


    def get_dataset_max_sessions(self, outlier_thresh=None) -> Optional[int]:
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
                    print("Warning: session number is over 50. Double check dataset.")
                else:
                    max_ses = si if max_ses is None else max(max_ses, si)

        return max_ses
    
    # ----static methods ----
    @staticmethod
    def mne_to_ptsa(epochs, events_df):
        merged_events_df = _merge_duplicate_sample_events(events_df)
        return TimeSeries.from_mne_epochs(epochs, merged_events_df)
    
    @staticmethod
    def raw_to_ptsa(raw, picks=None, tmin=None, tmax=None) -> TimeSeries:
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
    def filter_events_by_trial_types(trial_types, events_df=None, raw=None):
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