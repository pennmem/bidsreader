import numpy as np
import pandas as pd
import mne
from mne_bids import read_raw_bids
from pathlib import Path
from typing import Iterable, Tuple, Optional, Union, Dict
import warnings
import json
from .basereader import BaseReader
from ._errorwrap import public_api
from .helpers import validate_option, space_from_coordsystem_fname, combine_bipolar_electrodes
from .exc import InvalidOptionError, FileNotFoundBIDSError, AmbiguousMatchError, DataParseError

CML_ROOT = "/data/LTP_BIDS"


class CMLBIDSReader(BaseReader):
    VALID_ACQ = ("bipolar", "monopolar")
    VALID_DEVICES = ("eeg", "ieeg")
    INTRACRANIAL_FIELDS = ("subject", "task", "session", "device")
    SCALP_FIELDS = ("subject", "task", "session", "device")

    def __init__(
        self,
        root: Optional[Union[str, Path]] = CML_ROOT,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str | int] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        device: Optional[str] = None,
    ):
        device = validate_option(
            "device", device, self.VALID_DEVICES
        )
        super().__init__(
            root=root,
            subject=subject,
            task=task,
            session=session,
            space=space,
            acquisition=acquisition,
            device=device,
        )

    # ---------- internal helpers ----------

    def _determine_device(self) -> Optional[str]:
        if self.subject is None:
            return None
        if self.subject.startswith("LTP"):
            return "eeg"
        if self.subject.startswith("R"):
            return "ieeg"
        return None

    DEFAULT_SPACE = "MNI152NLin6ASym"
    SPACE_PREFERENCE = ("MNI152NLin6ASym", "Talairach" ,"fsaverage", "Pixels", "fsaverageBrainshift" "fsnative", "fsnativeBrainshift", "fsnativeDural", "t1MRI")

    def _coordsystem_dir(self) -> Path:
        subject_root = self._subject_root()
        return subject_root / self._add_bids_prefix("session", self.session) / self.device

    def list_available_spaces(self) -> list:
        """Return the sorted list of BIDS space names present as
        *_coordsystem.json files for this session."""
        data_dir = self._coordsystem_dir()
        if not data_dir.exists():
            return []
        spaces = []
        for m in data_dir.glob("*_coordsystem.json"):
            space = space_from_coordsystem_fname(m.name)
            if space is not None:
                spaces.append(space)
        return sorted(set(spaces))

    def _determine_space(self) -> str:
        data_dir = self._coordsystem_dir()

        if not data_dir.exists():
            raise FileNotFoundBIDSError(
                f"determine_space: data directory does not exist.\n"
                f"data_dir={data_dir}"
            )

        matches = list(data_dir.glob("*_coordsystem.json"))
        if not matches:
            raise FileNotFoundBIDSError(
                f"determine_space: no *_coordsystem.json file found.\n"
                f"data_dir={data_dir}"
            )

        spaces = []
        for m in matches:
            space = space_from_coordsystem_fname(m.name)
            if space is None:
                raise DataParseError(
                    f"determine_space: could not parse space from filename.\n"
                    f"filename={m.name}"
                )
            spaces.append(space)
        spaces = sorted(set(spaces))

        if len(spaces) == 1:
            return spaces[0]

        for preferred in self.SPACE_PREFERENCE:
            if preferred in spaces:
                return preferred

        raise AmbiguousMatchError(
            f"determine_space: multiple spaces found and none of the "
            f"preferred defaults {self.SPACE_PREFERENCE} present. "
            f"Available spaces: {spaces}. "
            f"Pass space=<one of these> when constructing CMLBIDSReader."
        )

    def _validate_acq(self, acquisition: Optional[str]) -> Optional[str]:
        if not self.is_intracranial():
            return None
        if acquisition is None:
            raise InvalidOptionError("acquisition is set to None")
        return validate_option("acquisition", acquisition, self.VALID_ACQ)

    def _get_needed_fields(self):
        return self.INTRACRANIAL_FIELDS if self.is_intracranial() else self.SCALP_FIELDS

    def _attach_bipolar_midpoint_montage(self, raw: mne.io.BaseRaw, space: Optional[str] = None) -> None:
        pairs_df = self.load_channels("bipolar")
        elec_df = self.load_electrodes(space=space)
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

    # ---------- public API ----------

    @public_api
    def is_intracranial(self) -> bool:
        return self.device == "ieeg"

    # ---------- loaders ----------

    @public_api
    def load_events(self, event_type: str = "beh") -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_events")
        allowed = ["beh", self.device]
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

    def _space_file(self, space: str, suffix: str, extension: str) -> Path:
        """Build a space-<label>_<suffix>.<ext> path manually, bypassing
        mne_bids.BIDSPath validation (which rejects non-standard space
        labels like 'fsnative', 'fsnativeDural', 'fsaverageBrainshift')."""
        data_dir = self._coordsystem_dir()
        task_part = f"_task-{self.task}" if self.is_intracranial() and self.task else ""
        fname = (
            f"sub-{self.subject}_"
            f"ses-{self.session}"
            f"{task_part}"
            f"_space-{space}_{suffix}{extension}"
        )
        return data_dir / fname

    @public_api
    def load_electrodes(self, space: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_electrodes")
        _space = space if space is not None else self.space
        return pd.read_csv(self._space_file(_space, "electrodes", ".tsv"), sep="\t")

    @public_api
    def load_channels(self, acquisition: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_channels")

        acq = self._validate_acq(acquisition)
        bp = self._bp(datatype=self.device, suffix="channels", acquisition=acq, extension=".tsv")
        return pd.read_csv(bp.fpath, sep="\t")

    @public_api
    def load_combined_channels(self, acquisition: Optional[str] = None, space: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_combined_channels")

        channel_df = self.load_channels(acquisition)
        elec_df = self.load_electrodes(space=space)
        if acquisition == "monopolar" or acquisition is None:
            return channel_df.merge(elec_df, on="name", how="left", suffixes=("", "_elec"))
        if acquisition == "bipolar":
            return combine_bipolar_electrodes(channel_df, elec_df)

    @public_api
    def load_coordsystem_desc(self, space: Optional[str] = None) -> Dict:
        self._require(self._get_needed_fields(), context="load_coordsystem")
        _space = space if space is not None else self.space
        with open(self._space_file(_space, "coordsystem", ".json"), "r") as f:
            return json.load(f)

    @public_api
    def load_raw(self, acquisition: Optional[str] = None, extension: Optional[str] = None) -> mne.io.BaseRaw:
        self._require(self._get_needed_fields(), context="load_raw")

        acq = self._validate_acq(acquisition)

        bp_kwargs = {"datatype": self.device}
        if acq is not None:
            bp_kwargs["acquisition"] = acq
        bp = self._bp(**bp_kwargs)

        if extension is None:
            for ext in (".bdf", ".edf", ".vhdr", ".set", ".nwb", ".fif"):
                candidate = bp.copy().update(suffix=self.device, extension=ext).fpath
                if candidate.exists():
                    bp = bp.copy().update(suffix=self.device, extension=ext)
                    break
        else :
            bp = bp.copy().update(suffix=self.device, extension=extension)

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
            warnings.filterwarnings(
                "ignore",
                message=r"Expected to find a single (electrodes\.tsv|coordsystem\.json) file.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"participants\.tsv file not found.*",
                category=RuntimeWarning,
            )
            raw = read_raw_bids(bp)

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
        extension: Optional[str] = None
    ) -> mne.Epochs:
        self._require(self._get_needed_fields(), context="load_epochs")
        raw = self.load_raw(acquisition=acquisition, extension=extension)

        all_events_raw, all_event_id = mne.events_from_annotations(raw)

        if events is not None:
            if "sample" not in events.columns:
                raise ValueError("Events DataFrame must contain a 'sample' column")

            if "trial_type" in events.columns:
                codes = events["trial_type"].map(all_event_id)
                if codes.isna().any():
                    missing = set(events.loc[codes.isna(), "trial_type"].unique())
                    raise ValueError(
                        f"trial_type values not found in raw annotations: {missing}"
                    )
                codes = codes.values.astype(int)
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

    # ---------- data index ----------
    def get_data_index(self, root: Union[str, Path] = None, task: str = None) -> pd.DataFrame:
        """Scan a BIDS root and return a session-level DataFrame with
        file-path columns for each major BIDS output.

        Parameters default to ``self.root`` and ``self.task`` when called
        on an instance, but can be overridden with explicit values.

        Extends ``BaseReader.get_data_index`` (subject / task / session)
        with columns whose values are the file path if the file exists,
        or ``None`` if it doesn't:

        - ``beh`` — behavioral events TSV
        - ``eeg`` — scalp EEG recording (edf/bdf/set/vhdr)
        - ``mono_ieeg`` — monopolar iEEG recording (edf/bdf)
        - ``bi_ieeg`` — bipolar iEEG recording (edf/bdf)
        - ``mono_channels`` — monopolar channels TSV
        - ``bi_channels`` — bipolar channels TSV
        - ``eeg_events`` / ``ieeg_events`` — device-level events TSV
        - ``electrodes`` — electrodes TSV
        - ``coordsystem`` — coordinate system JSON
        """
        root = root if root is not None else self.root
        task = task if task is not None else self.task
        df = super().get_data_index(root, task)
        if df.empty:
            return df

        root = Path(root)
        # File patterns to search for. Each entry is (column_name, subdir,
        # glob_pattern). Patterns use {pfx} for the BIDS filename prefix.
        _PATTERNS = [
            ("beh",            "beh",  "{pfx}_beh.tsv"),
            ("eeg",            "eeg",  "{pfx}_eeg.*"),
            ("mono_ieeg",      "ieeg", "{pfx}_acq-monopolar_ieeg.*"),
            ("bi_ieeg",        "ieeg", "{pfx}_acq-bipolar_ieeg.*"),
            ("mono_channels",  "ieeg", "{pfx}_acq-monopolar_channels.tsv"),
            ("bi_channels",    "ieeg", "{pfx}_acq-bipolar_channels.tsv"),
            ("eeg_channels",   "eeg",  "{pfx}_channels.tsv"),
            ("ieeg_events",    "ieeg", "{pfx}_events.tsv"),
            ("eeg_events",     "eeg",  "{pfx}_events.tsv"),
            ("electrodes",     "ieeg", "{pfx}_*electrodes.tsv"),
            ("coordsystem",    "ieeg", "{pfx}_*coordsystem.json"),
        ]
        # Data-file extensions to accept (skip sidecars).
        _DATA_EXTS = {".edf", ".bdf", ".set", ".vhdr", ".fif", ".nwb", ".mff"}

        new_cols = {col: [None] * len(df) for col, _, _ in _PATTERNS}
        for idx, row in df.iterrows():
            sub, ses = row["subject"], row["session"]
            pfx = f"sub-{sub}_ses-{ses}_task-{task}"
            sess_dir = root / f"sub-{sub}" / f"ses-{ses}"

            for col, subdir, pat in _PATTERNS:
                glob_pat = pat.format(pfx=pfx)
                matches = list((sess_dir / subdir).glob(glob_pat))
                # For recording files (eeg/ieeg), keep only data files.
                if col in ("eeg", "mono_ieeg", "bi_ieeg"):
                    matches = [m for m in matches if m.suffix in _DATA_EXTS]
                if matches:
                    new_cols[col][idx] = str(matches[0])

        for col, _, _ in _PATTERNS:
            df[col] = new_cols[col]
        return df
