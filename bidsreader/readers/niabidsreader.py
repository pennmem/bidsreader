"""Reader for Nia Therapeutics BIDS conversions.

These datasets are written by ``niadatsci.niashare.bids_convert.NiaBIDSConverter``.
The layout differs from the CML archive in ways that matter to a caller:

- **One dataset per (protocol, experiment).** The real BIDS root is
  ``<bids_root>/<protocol>/<experiment>/`` — e.g. ``/bids/preclinical/ACL`` —
  because NIA session numbers restart at zero per experiment and would
  otherwise collide. ``root`` must point at that directory, not at
  ``<bids_root>``. :meth:`NiaBIDSReader.is_nia_dataset` catches the mistake.
- **``datatype`` is always ``eeg``**, never ``ieeg``.
- **No ``space-`` entity**, and normally no ``electrodes.tsv`` /
  ``coordsystem.json``.
- **``acq-`` encodes the montage**: absent for montage 0, ``acq-montage<N>``
  otherwise.
- **Sessions are zero-padded to two digits** (``ses-00``).
- **Recordings are BrainVision float32**, not EDF/BDF.
- **There is no ``_beh.tsv``**; behavior is in ``events.tsv``, and continuous
  non-EEG timeseries (classifier output, IMU motion) are ``physio`` recordings.

On sample indexing: ``events.tsv`` ``sample`` is an index into the recording,
counting from zero at the first stored sample, and ``onset`` is
``sample / SamplingFrequency``. The absolute device counter is kept separately in
the ``eegoffset`` columns; the two differ by ``NiaFirstSample`` — see
:meth:`NiaBIDSReader.first_sample`. An event logged before the recording started
has ``n/a`` for both ``sample`` and ``onset``.
"""

import gzip
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

from .basereader import BaseReader
from ..src._errorwrap import public_api
from ..src.exc import (
    AmbiguousMatchError,
    DataParseError,
    FileNotFoundBIDSError,
    InvalidOptionError,
)

# The *parent* of <protocol>/<experiment>, matching niadatsci's roots['bids'].
# A reader's `root` is one of its <protocol>/<experiment> subdirectories.
NIA_ROOT = "/bids"


class NiaBIDSReader(BaseReader):
    """Reader for a single Nia BIDS dataset.

    Parameters
    ----------
    root :
        The dataset directory, ``<bids_root>/<protocol>/<experiment>``.
    subject :
        NIA subject label, e.g. ``"S002"``.
    session :
        NIA session number. Accepts ``0``, ``"0"`` or ``"00"``; all resolve to
        ``ses-00``.
    task :
        BIDS task label. Constant within a dataset, so it is inferred from the
        dataset when omitted.
    montage :
        NIA montage number. ``0`` (the default) means no ``acq-`` entity.

    Examples
    --------
    >>> r = NiaBIDSReader(root="/bids/preclinical/ACL", subject="S002", session=0)
    >>> r.task                                          # doctest: +SKIP
    'acl'
    >>> events = r.load_events()                        # doctest: +SKIP
    """

    VALID_DEVICES = ("eeg",)
    REQUIRED_FIELDS = ("subject", "session", "task")

    #: ``GeneratedBy`` name that NiaBIDSConverter stamps into
    #: dataset_description.json, and which it uses itself to tell its own
    #: descriptions from mne-bids' stub.
    GENERATED_BY = "niadatsci"

    #: The only two ``recording-<label>`` physio streams NIA writes: ACL's
    #: closed-loop classifier output and Sensing's IMU motion frame.
    PHYSIO_LABELS = ("classifier", "imu")

    #: NIA writes BrainVision only.
    DATA_EXTS = (".vhdr",)

    #: ``montage`` is a settable public field, so it has to be in the allowlist
    #: that BaseReader.__setattr__ enforces — same arrangement as ``space``.
    _FIELDS = BaseReader._FIELDS | {"montage", "_montage"}

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str | int] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        device: Optional[str] = None,
        montage: int = 0,
    ):
        if device is not None and device not in self.VALID_DEVICES:
            raise InvalidOptionError(
                f"device must be one of: {self.VALID_DEVICES}. Got {device!r}. "
                f"Nia conversions are always datatype 'eeg'."
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
        # Setting `montage` derives `acquisition`; an explicit `acquisition=`
        # argument wins, so callers can still pin an unusual label by hand.
        if acquisition is None:
            self.montage = montage
        else:
            self._montage = montage

    # ---------- properties ----------

    @property
    def montage(self) -> int:
        return self._montage

    @montage.setter
    def montage(self, value: int) -> None:
        # Montage 0 is the default and gets clean, unsuffixed filenames. A
        # subject re-implanted onto a second montage gets `acq-montage<N>`.
        # Mirrors NiaBIDSConverter.acquisition.
        self._montage = int(value)
        self.acquisition = None if self._montage == 0 else f"montage{self._montage}"

    # ---------- internal helpers ----------

    def _determine_device(self) -> str:
        """Always ``"eeg"`` — NiaBIDSConverter never writes an ``ieeg`` datatype."""
        return "eeg"

    def _determine_space(self) -> None:
        """Always ``None`` — Nia conversions carry no ``space-`` entity."""
        return None

    def _get_needed_fields(self) -> Tuple[str, ...]:
        return self.REQUIRED_FIELDS

    def _bids_session(self) -> Optional[str]:
        """Session label, zero-padded to two digits so sessions sort lexically.

        Accepts ``0``, ``"0"`` and ``"00"`` alike. A non-integer label is passed
        through untouched rather than raising, so an unusual dataset stays
        readable.
        """
        if self.session is None:
            return None
        try:
            return f"{int(self.session):02d}"
        except (TypeError, ValueError):
            return str(self.session)

    def _bp(self, **kwargs) -> BIDSPath:
        """A BIDSPath for this session, with ``kwargs`` layered on top.

        Overrides :meth:`BaseReader._bp`, which omits ``acquisition`` and does
        not zero-pad the session. Mirrors ``NiaBIDSConverter.bids_path``.
        """
        bp = BIDSPath(
            root=self.root,
            subject=self.subject,
            session=self._bids_session(),
            task=self.task,
            acquisition=self.acquisition,
            datatype="eeg",
        )
        bp.update(**kwargs)
        return bp

    def _session_dir(self) -> Path:
        return (
            self._subject_root()
            / self._add_bids_prefix("session", self._bids_session())
            / "eeg"
        )

    @staticmethod
    def _sanitize_label(label) -> str:
        """Strip characters BIDS does not allow in an entity value.

        Verbatim from ``NiaBIDSConverter._sanitize_label`` so inferred labels
        match the written ones.
        """
        return re.sub(r"[^a-zA-Z0-9]", "", str(label))

    def _determine_task(self) -> Optional[str]:
        """Infer the task label. One experiment per dataset, so it is constant.

        Prefers the entity actually present in the tree; falls back to parsing
        ``dataset_description.json``'s ``Name`` (``"Nia <experiment> (<protocol>)"``)
        for a dataset whose sessions are not yet converted.
        """
        tasks = [t for t in (get_entity_vals(self.root, "task") or []) if t]
        if len(tasks) > 1:
            raise DataParseError(
                f"determine_task: expected one task per Nia dataset, found {sorted(tasks)}.\n"
                f"root={self.root}\n"
                f"Pass task=<one of these>, or point root at a single "
                f"<protocol>/<experiment> directory."
            )
        if tasks:
            return tasks[0]

        desc = self._read_json(Path(self.root) / "dataset_description.json", required=False)
        name = (desc or {}).get("Name", "")
        m = re.match(r"^Nia\s+(.+?)\s*\(", str(name))
        if m:
            return self._sanitize_label(m.group(1)).lower()
        return None

    def _resolve_task(self) -> str:
        """``self.task``, inferring and caching it on first use."""
        if self.task is None:
            self.task = self._determine_task()
        if self.task is None:
            raise DataParseError(
                f"Could not infer task from the dataset at {self.root}. "
                f"Pass task= explicitly."
            )
        return self.task

    def _prepare(self, context: str) -> None:
        """Resolve the task, then check the required fields are set.

        Order matters: ``task`` is one of the required fields but is normally
        inferred from the dataset rather than passed in, so it has to be
        resolved before :meth:`BaseReader._require` looks at it.
        """
        self._resolve_task()
        self._require(self._get_needed_fields(), context=context)

    @staticmethod
    def _read_json(path: Path, required: bool = True) -> Optional[Dict]:
        path = Path(path)
        if not path.exists():
            if required:
                raise FileNotFoundBIDSError(f"File not found: {path}")
            return None
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_tsv(path: Path, **kwargs) -> pd.DataFrame:
        """Read a BIDS TSV preserving the literal string ``n/a``.

        NIA writes ``n/a`` for every blank, and several columns are legitimately
        ``n/a`` throughout (``duration``, most of ``participants.tsv``). Reading
        with pandas' defaults would silently turn those into ``NaN`` and lose the
        distinction between "not applicable" and "missing". Same idiom as
        ``niadatsci.niashare.validate.BidsSessionLoader``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundBIDSError(f"File not found: {path}")
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, skip: Iterable[str] = ()) -> pd.DataFrame:
        """Convert wholly-numeric columns to numeric dtype, leaving the rest alone.

        Columns holding ``n/a`` stay object dtype, which is the point — see
        :meth:`_read_tsv`.
        """
        skip = set(skip)
        out = df.copy()
        for col in out.columns:
            if col in skip:
                continue
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().all() and len(out):
                out[col] = converted
        return out

    def _first_match(self, pattern: str) -> Optional[Path]:
        data_dir = self._session_dir()
        if not data_dir.exists():
            return None
        matches = sorted(data_dir.glob(pattern))
        return matches[0] if matches else None

    # ---------- public API ----------

    @public_api
    def is_nia_dataset(self) -> bool:
        """True if ``root`` holds a dataset written by niadatsci.

        Checks the ``GeneratedBy`` sentinel in ``dataset_description.json``.
        Worth calling first: the most likely mistake with these datasets is
        pointing ``root`` at ``<bids_root>`` instead of
        ``<bids_root>/<protocol>/<experiment>``, which otherwise fails later
        with a confusing missing-file error.
        """
        desc = self._read_json(Path(self.root) / "dataset_description.json", required=False)
        if not desc:
            return False
        return any(
            str(entry.get("Name", "")) == self.GENERATED_BY
            for entry in desc.get("GeneratedBy", [])
            if isinstance(entry, dict)
        )

    @public_api
    def first_sample(self) -> int:
        """Device-clock sample at which this recording begins.

        Provenance, not a correction. ``events.tsv`` ``sample`` is already an
        index into the recording, so epoching needs nothing from here — see
        :meth:`load_epochs`.

        What this is for is relating the two clocks. The events table also keeps
        the raw device counter in its ``eegoffset`` / ``eegoffset_cmd`` columns,
        and the two differ by exactly this value::

            eegoffset_cmd - first_sample() == sample

        Useful for matching a BIDS event back to a row in the source archive.
        Written as ``NiaFirstSample`` by ``NiaBIDSConverter.make_eeg_sidecar``.
        """
        return int(self.load_sidecar()["NiaFirstSample"])

    @public_api
    def units_status(self) -> Optional[str]:
        """``"calibrated"`` if the signal is in µV, otherwise not physical units.

        When the conversion ran without ``--microvolts-per-lsb`` the signal is
        raw 15-bit ADC counts and ``channels.tsv`` says ``arbitrary``. The
        helpers in :mod:`bidsreader.src.units` cannot scale that data, so check
        this before attempting any unit conversion.
        """
        return self.load_sidecar().get("NiaUnitsStatus")

    @public_api
    def list_available_physio(self) -> List[str]:
        """The ``recording-<label>`` physio streams present for this session."""
        data_dir = self._session_dir()
        if not data_dir.exists():
            return []
        labels = set()
        for m in data_dir.glob("*_physio.tsv.gz"):
            if "_recording-" in m.name:
                labels.add(m.name.split("_recording-")[1].split("_physio")[0])
        return sorted(labels)

    # ---------- loaders ----------

    @public_api
    def load_dataset_description(self) -> Dict:
        return self._read_json(Path(self.root) / "dataset_description.json")

    @public_api
    def load_participants(self) -> pd.DataFrame:
        """The dataset's ``participants.tsv``.

        NIA sets ``nia_protocol``, ``nia_subject`` and ``nia_montage``; the
        remaining columns are mne-bids defaults and are usually ``n/a``.
        """
        return self._read_tsv(Path(self.root) / "participants.tsv")

    @public_api
    def load_sidecar(self) -> Dict:
        """The ``_eeg.json`` sidecar, including the ``Nia*`` provenance keys.

        Those keys (``NiaPropGaps``, ``NiaPropCorrupt``, ``NiaStimulation``,
        ``NiaClosedLoop``, ``NiaUnitsStatus``, ...) are the documented way to
        screen sessions for quality.
        """
        self._prepare("load_sidecar")
        return self._read_json(self._bp(suffix="eeg", extension=".json").fpath)

    @public_api
    def load_scans(self) -> pd.DataFrame:
        """This session's ``_scans.tsv`` (``filename``, ``acq_time``)."""
        self._require(("subject", "session"), context="load_scans")
        path = (
            self._subject_root()
            / self._add_bids_prefix("session", self._bids_session())
            / f"sub-{self.subject}_ses-{self._bids_session()}_scans.tsv"
        )
        return self._read_tsv(path)

    @public_api
    def load_events(self, coerce_numeric: bool = True) -> pd.DataFrame:
        """This session's ``events.tsv``.

        Columns are ``onset, duration, trial_type, sample`` followed by every
        column of the source ``beh.csv``. Three NIA specifics:

        - ``sample`` is an index into this recording, from zero at the first
          stored sample, and ``onset`` is ``sample / SamplingFrequency``. The
          absolute device counter stays in the ``eegoffset`` columns; see
          :meth:`first_sample`.
        - An event logged before the recording started has ``n/a`` for both
          ``sample`` and ``onset``. The row is otherwise intact — only its
          position in this recording is unknown. A ``RuntimeWarning`` is emitted
          when this happens, because it leaves those columns as strings.
        - The device's stim duration is in **``stim_duration``**, not
          ``duration`` — the BIDS-reserved name was taken, so the converter
          renamed it. ``duration`` is ``n/a`` unless the conversion was run with
          ``--stim-duration-units``.

        Blank cells are the literal string ``n/a``; with ``coerce_numeric``
        (the default) columns that are wholly numeric become numeric dtype and
        the rest stay strings.
        """
        self._prepare("load_events")
        df = self._read_tsv(self._bp(suffix="events", extension=".tsv").fpath)
        self._warn_on_blanked_numeric(df)
        return self._coerce_numeric(df, skip={"trial_type"}) if coerce_numeric else df

    @staticmethod
    def _warn_on_blanked_numeric(df: pd.DataFrame) -> None:
        """Warn when a BIDS-reserved numeric column carries ``n/a``.

        Such a column stays object dtype, so ``df["sample"].max()`` silently
        compares strings instead of numbers. ``duration`` is excluded: it is
        ``n/a`` for a whole session as a matter of course, which is documented
        rather than surprising.
        """
        for col in ("sample", "onset"):
            if col not in df.columns:
                continue
            n = int((df[col] == "n/a").sum())
            if n:
                warnings.warn(
                    f"load_events: {n} row(s) have '{col}' = 'n/a' (events logged "
                    f"before the recording started), so the column is left as "
                    f"strings rather than numbers. Filter them out before doing "
                    f"arithmetic on it.",
                    RuntimeWarning,
                )

    @public_api
    def load_events_json(self) -> Dict:
        """The ``events.json`` sidecar: per-column descriptions and ``Levels``."""
        self._prepare("load_events_json")
        return self._read_json(self._bp(suffix="events", extension=".json").fpath)

    @public_api
    def load_channels(self, coerce_numeric: bool = True) -> pd.DataFrame:
        """This session's ``channels.tsv``.

        Columns are ``name, type, units, sampling_frequency, low_cutoff,
        high_cutoff, notch, status, status_description, group``. ``units`` is
        ``µV`` for a calibrated conversion and the literal ``arbitrary``
        otherwise — see :meth:`units_status`.
        """
        self._prepare("load_channels")
        df = self._read_tsv(self._bp(suffix="channels", extension=".tsv").fpath)
        return self._coerce_numeric(df, skip={"name", "type", "units", "group"}) if coerce_numeric else df

    @public_api
    def load_physio(self, recording: Optional[str] = None) -> pd.DataFrame:
        """A continuous non-EEG timeseries as a DataFrame.

        Two streams reach BIDS: ``classifier`` (ACL closed-loop output, one
        value per EEG sample) and ``imu`` (Sensing motion). Neither is a table
        of events, so neither is in ``events.tsv``.

        BIDS physio files carry **no header row** — the column names live in the
        sidecar's ``Columns`` key, which this reads first. With ``recording=None``
        the stream is auto-detected when the session has exactly one.
        """
        self._prepare("load_physio")

        if recording is None:
            available = self.list_available_physio()
            if not available:
                raise FileNotFoundBIDSError(
                    f"load_physio: no *_physio.tsv.gz found.\n"
                    f"data_dir={self._session_dir()}"
                )
            if len(available) > 1:
                raise AmbiguousMatchError(
                    f"load_physio: multiple physio recordings present: {available}. "
                    f"Pass recording=<one of these>."
                )
            recording = available[0]

        sidecar = self._read_json(
            self._bp(recording=recording, suffix="physio", extension=".json").fpath
        )
        columns = sidecar.get("Columns")
        if not columns:
            raise DataParseError(
                f"load_physio: sidecar for recording-{recording} has no 'Columns' key; "
                f"the .tsv.gz is headerless and cannot be labelled without it."
            )

        tsv_path = Path(
            self._bp(recording=recording, suffix="physio", extension=".tsv.gz").fpath
        )
        if not tsv_path.exists():
            raise FileNotFoundBIDSError(f"File not found: {tsv_path}")
        with gzip.open(tsv_path, "rt") as f:
            return pd.read_csv(f, sep="\t", header=None, names=columns, na_values=["n/a"])

    @public_api
    def load_electrodes(self) -> pd.DataFrame:
        """This session's ``electrodes.tsv`` (``name, x, y, z``).

        Usually absent: the converter writes it only when the archive's
        ``electrodes.csv`` carries coordinates, and the imaging tree is
        typically hand-populated without them.
        """
        self._prepare("load_electrodes")
        match = self._first_match("*_electrodes.tsv")
        if match is None:
            raise FileNotFoundBIDSError(
                f"load_electrodes: no *_electrodes.tsv found. Nia conversions omit "
                f"electrode coordinates unless the source electrodes.csv had x/y/z.\n"
                f"data_dir={self._session_dir()}"
            )
        return self._read_tsv(match)

    @public_api
    def load_coordsystem_desc(self) -> Dict:
        """This session's ``coordsystem.json``.

        Written only alongside ``electrodes.tsv``, with an unspecified
        ``"Other"`` coordinate system in mm. There is no ``space-`` entity.
        """
        self._prepare("load_coordsystem_desc")
        match = self._first_match("*_coordsystem.json")
        if match is None:
            raise FileNotFoundBIDSError(
                f"load_coordsystem_desc: no *_coordsystem.json found. Nia conversions "
                f"omit coordinates unless the source electrodes.csv had x/y/z.\n"
                f"data_dir={self._session_dir()}"
            )
        return self._read_json(match)

    @public_api
    def load_raw(self, extension: Optional[str] = None) -> mne.io.BaseRaw:
        """This session's recording, as an MNE ``Raw``.

        NIA writes BrainVision float32. Note that ``raw.first_samp`` is 0 —
        BrainVision cannot carry it — so relating ``events.tsv`` samples to this
        object needs :meth:`first_sample`.
        """
        self._prepare("load_raw")

        bp = self._bp(suffix="eeg")
        if extension is None:
            for ext in self.DATA_EXTS:
                if bp.copy().update(extension=ext).fpath.exists():
                    bp = bp.copy().update(extension=ext)
                    break
            else:
                raise FileNotFoundBIDSError(
                    f"load_raw: no recording found with extension in {self.DATA_EXTS}.\n"
                    f"data_dir={self._session_dir()}"
                )
        else:
            bp = bp.copy().update(extension=extension)

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
            return read_raw_bids(bp)

    @public_api
    def load_epochs(
        self,
        tmin: float,
        tmax: float,
        events: Optional[pd.DataFrame] = None,
        baseline: Optional[Tuple[float | None, float | None]] = None,
        event_repeated: str = "merge",
        channels: Optional[Iterable[str]] = None,
        preload: bool = False,
        extension: Optional[str] = None,
    ) -> mne.Epochs:
        """Epoch the recording around events.

        When ``events`` is a DataFrame from :meth:`load_events`, its ``sample``
        column is already an index into this recording, so it is used directly —
        no offset is applied. :meth:`first_sample` is *not* consulted here; it
        relates ``sample`` to the absolute device clock, which epoching does not
        need.

        Events whose ``sample`` is ``n/a`` (logged before the recording started)
        raise rather than being dropped, so a trial count can never shrink
        unnoticed. Samples outside the recording also raise — see the guards
        below.

        With ``events=None``, the raw's own annotations are used unchanged.
        """
        self._prepare("load_epochs")
        raw = self.load_raw(extension=extension)

        all_events_raw, all_event_id = mne.events_from_annotations(raw)

        if events is not None:
            if "sample" not in events.columns:
                raise DataParseError(
                    "Events DataFrame must contain a 'sample' column"
                )

            samples = pd.to_numeric(events["sample"], errors="coerce")
            if samples.isna().any():
                n = int(samples.isna().sum())
                raise DataParseError(
                    f"{n} event(s) have no sample index (n/a). The converter blanks "
                    f"'sample' and 'onset' for events logged before the recording "
                    f"started; the events are real but have no position in this "
                    f"recording. Filter them out before epoching:\n"
                    f"    events = events[pd.to_numeric(events['sample'], "
                    f"errors='coerce').notna()]"
                )
            samples = samples.astype(int).values

            if (samples < 0).any():
                raise DataParseError(
                    f"{int((samples < 0).sum())} event(s) have a negative sample "
                    f"index. 'sample' is an index into the recording, counting from "
                    f"zero at the first stored sample, so it cannot be negative."
                )
            if (samples >= raw.n_times).any():
                raise DataParseError(
                    f"{int((samples >= raw.n_times).sum())} event(s) have a sample "
                    f"index at or beyond the end of the recording "
                    f"(n_times={raw.n_times}, max sample={int(samples.max())}). "
                    f"Either these events belong to a different session, or they come "
                    f"from a conversion predating the switch to recording-relative "
                    f"samples — in which case re-run the niadatsci 'share' step for "
                    f"this session."
                )

            if "trial_type" in events.columns:
                codes = events["trial_type"].map(all_event_id)
                if codes.isna().any():
                    missing = set(events.loc[codes.isna(), "trial_type"].unique())
                    raise DataParseError(
                        f"trial_type values not found in raw annotations: {missing}"
                    )
                codes = codes.values.astype(int)
                present_types = set(events["trial_type"].unique())
                event_id = {k: v for k, v in all_event_id.items() if k in present_types}
            else:
                codes = np.ones(len(events), dtype=int)
                event_id = {"event": 1}

            events_raw = np.column_stack([
                samples,
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

    #: (column, glob) pairs, relative to the session's ``eeg/`` directory.
    #: ``{pfx}`` is the BIDS filename prefix for the session.
    _PATTERNS = [
        ("eeg", "{pfx}_eeg.vhdr"),
        ("channels", "{pfx}_channels.tsv"),
        ("events", "{pfx}_events.tsv"),
        ("sidecar", "{pfx}_eeg.json"),
        ("physio_classifier", "{pfx}_recording-classifier_physio.tsv.gz"),
        ("physio_imu", "{pfx}_recording-imu_physio.tsv.gz"),
        ("electrodes", "{pfx}_electrodes.tsv"),
    ]

    #: Sidecar key -> data-index column. These are the handles for screening
    #: sessions by quality, and are NIA's closest analogue to the CML data index.
    _SIDECAR_COLUMNS = {
        "NiaProtocol": "protocol",
        "NiaExperiment": "experiment",
        "NiaMontage": "montage",
        "NiaPropGaps": "prop_gaps",
        "NiaPropCorrupt": "prop_corrupt",
        "NiaStimulation": "stimulation",
        "NiaClosedLoop": "closed_loop",
        "NiaUnitsStatus": "units_status",
        "SamplingFrequency": "sfreq",
        "RecordingDuration": "duration",
    }

    def get_data_index(
        self, root: Union[str, Path] = None, task: str = None
    ) -> pd.DataFrame:
        """Scan a Nia dataset and return one row per session.

        Does **not** extend :meth:`BaseReader.get_data_index` — that scans for
        ``*_beh.tsv``, which Nia conversions never write.

        Returns ``subject``, ``session``, ``task``, a path column per expected
        output (``None`` where the file is absent), and the quality handles
        pulled from each session's ``_eeg.json``: ``prop_gaps``,
        ``prop_corrupt``, ``stimulation``, ``closed_loop``, ``units_status``,
        ``montage``, ``sfreq``, ``duration``, ``protocol``, ``experiment``.

        A session whose sidecar is missing or unparseable still gets a row, with
        the sidecar columns left as ``None``.
        """
        root = Path(root) if root is not None else self.root
        if root is None:
            raise ValueError(
                "root must be provided either on the instance or as an argument"
            )

        rows = []
        for sub_dir in sorted(root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            subject = sub_dir.name[len("sub-"):]
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.is_dir():
                    continue
                session = ses_dir.name[len("ses-"):]

                for pfx, row_task in self._index_prefixes(eeg_dir, subject, session, task):
                    row = {"subject": subject, "session": session, "task": row_task}
                    for col, pat in self._PATTERNS:
                        matches = sorted(eeg_dir.glob(pat.format(pfx=pfx)))
                        row[col] = str(matches[0]) if matches else None
                    row.update(self._index_sidecar_fields(row["sidecar"]))
                    rows.append(row)

        columns = (
            ["subject", "session", "task"]
            + [col for col, _ in self._PATTERNS]
            + list(self._SIDECAR_COLUMNS.values())
        )
        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(["subject", "session", "task"])
            .reset_index(drop=True)
        )

    def _index_prefixes(self, eeg_dir: Path, subject: str, session: str, task: Optional[str]):
        """Yield ``(filename_prefix, task)`` for each recording in a session dir.

        A session normally holds one recording, but ``acq-montage<N>`` allows
        more than one, so the prefix is read off the files present rather than
        assumed.
        """
        wanted = task if task is not None else self.task
        seen = set()
        for f in sorted(eeg_dir.glob("*_eeg.json")):
            pfx = f.name[: -len("_eeg.json")]
            m = re.search(r"_task-([^_]+)", pfx)
            found_task = m.group(1) if m else None
            if wanted is not None and found_task != wanted:
                continue
            if pfx not in seen:
                seen.add(pfx)
                yield pfx, found_task

    def _index_sidecar_fields(self, sidecar_path: Optional[str]) -> Dict:
        """Pull the screening columns out of one session's sidecar.

        Guarded: one malformed sidecar leaves its row's columns ``None`` rather
        than aborting a scan of the whole dataset.
        """
        fields = {col: None for col in self._SIDECAR_COLUMNS.values()}
        if not sidecar_path:
            return fields
        try:
            with open(sidecar_path, "r") as f:
                desc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"Could not read sidecar {sidecar_path}: {e}", RuntimeWarning)
            return fields
        for key, col in self._SIDECAR_COLUMNS.items():
            fields[col] = desc.get(key)
        return fields
