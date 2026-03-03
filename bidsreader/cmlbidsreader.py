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

    def _determine_space(self) -> str:
        subject_root = self._subject_root()
        data_dir = subject_root / self._add_bids_prefix("session", self.session) / self.device

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

    def _validate_acq(self, acquisition: Optional[str]) -> Optional[str]:
        if not self.is_intracranial():
            return None
        if acquisition is None:
            raise InvalidOptionError("acquisition was not set to bipolar, monopolar")
        return validate_option("acquisition", acquisition, self.VALID_ACQ)

    def _get_needed_fields(self):
        return self.INTRACRANIAL_FIELDS if self.is_intracranial() else self.SCALP_FIELDS

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

    @public_api
    def load_electrodes(self) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_electrodes")

        _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.device, suffix="electrodes", space=self.space, task=_task, extension=".tsv")
        return pd.read_csv(bp.fpath, sep="\t")

    @public_api
    def load_channels(self, acquisition: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_channels")

        acq = self._validate_acq(acquisition)
        bp = self._bp(datatype=self.device, suffix="channels", acquisition=acq, extension=".tsv")
        return pd.read_csv(bp.fpath, sep="\t")

    @public_api
    def load_combined_channels(self, acquisition: Optional[str] = None) -> pd.DataFrame:
        self._require(self._get_needed_fields(), context="load_combined_channels")

        channel_df = self.load_channels(acquisition)
        elec_df = self.load_electrodes()
        if acquisition == "monopolar" or acquisition is None:
            return channel_df.merge(elec_df, on="name", how="left", suffixes=("", "_elec"))
        if acquisition == "bipolar":
            return combine_bipolar_electrodes(channel_df, elec_df)

    @public_api
    def load_coordsystem_desc(self) -> Dict:
        self._require(self._get_needed_fields(), context="load_coordsystem")

        _task = self.task if self.is_intracranial() else None
        bp = self._bp(datatype=self.device, suffix="coordsystem", space=self.space, task=_task, extension=".json")

        with open(bp.fpath, "r") as f:
            return json.load(f)

    @public_api
    def load_raw(self, acquisition: Optional[str] = None) -> mne.io.BaseRaw:
        self._require(self._get_needed_fields(), context="load_raw")

        acq = self._validate_acq(acquisition)

        bp_kwargs = {"datatype": self.device}
        if acq is not None:
            bp_kwargs["acquisition"] = acq
        bp = self._bp(**bp_kwargs)

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
