import re

import pandas as pd
from mne_bids import BIDSPath, get_entity_vals
from pathlib import Path
from typing import Iterable, Optional, Union, List
import warnings
from ._errorwrap import public_api
from .helpers import add_prefix
from .exc import InvalidOptionError, MissingRequiredFieldError


class BaseReader:
    _FIELDS = {"root", "subject", "session", "task", "acquisition", "_device", "_space"}
    # REQUIRED_FIELDS = ("subject", "task", "session", "device")

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str | int] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if root is None:
            raise ValueError("root must be provided")
        self.root = Path(root)
        self.subject = subject
        self.session = session
        self.task = str(task)

        self.acquisition = acquisition

        self._device = device

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
        if self.device:
            parts.append(f"type={self.device}")
        if self.space:
            parts.append(f"space={self.space}")

        cls = type(self).__name__
        return f"{cls}({', '.join(parts)})"

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(root={self.root!r}, subject={self.subject!r}, "
            f"session={self.session!r}, task={self.task!r}, "
            f"device={self.device!r}, space={self.space!r})"
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
    def device(self) -> Optional[str]:
        if self._device is not None:
            return self._device

        try:
            self._device = self._determine_device()
        except Exception as e:
            warnings.warn(
                f"Could not determine device automatically: {e}",
                RuntimeWarning,
            )
            return None

        if self._device is None:
            warnings.warn(
                "device could not be inferred from subject.",
                RuntimeWarning,
            )

        return self._device

    # ---------- internal helpers ----------

    def _bp(self, **kwargs) -> BIDSPath:
        bp = BIDSPath(
            root=self.root,
            subject=self.subject,
            session=str(self.session) if self.session is not None else None,
            task=self.task,
            datatype=self.device,
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

    def _require(self, fields: Iterable[str], context: str = "") -> None:
        missing = [f for f in fields if getattr(self, f, None) in (None, "")]
        if missing:
            raise MissingRequiredFieldError(
                f"{context}: missing required fields: {', '.join(missing)}"
            )

    # idk if this is useful for anyone, should override for proper checking
    # def _get_needed_fields(self):
    #     return self.REQUIRED_FIELDS

    def _determine_space(self) -> Optional[str]:
        """Override in subclasses to provide automatic space detection."""
        return None

    def _determine_device(self) -> Optional[str]:
        """Override in subclasses to provide automatic device detection."""
        return None

    # ---------- public API ----------
    @public_api
    def set_fields(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # validated by __setattr__
        return self

    # ---- simple metadata queries ----
    @public_api
    def get_subject_tasks(self) -> List[str]:
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
                    warnings.warn(f"Session number is over {outlier_thresh}. Double check dataset.")
                else:
                    max_ses = si if max_ses is None else max(max_ses, si)

        return max_ses

    # ---------- data index ----------
    def get_data_index(self, root: Union[str, Path] = None, task: str = None) -> pd.DataFrame:
        """Scan a BIDS root for sessions matching *task* and return a
        DataFrame with ``subject``, ``task``, ``session`` columns.

        Parameters default to ``self.root`` and ``self.task`` when called
        on an instance, but can be overridden with explicit values::

            # instance — uses reader's own root / task
            reader = BaseReader(root="/data/BIDS", task="FR1")
            df = reader.get_data_index()

            # explicit — useful without a fully configured reader
            df = reader.get_data_index(root="/other/root", task="catFR1")

        Subclasses can override to add file-path columns for each
        expected BIDS output (see ``CMLBIDSReader.get_data_index``).
        """
        root = Path(root) if root is not None else self.root
        task = task if task is not None else self.task
        if root is None:
            raise ValueError("root must be provided either on the instance or as an argument")
        if task is None:
            raise ValueError("task must be provided either on the instance or as an argument")

        pat = re.compile(
            r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-" + re.escape(task),
            re.IGNORECASE,
        )
        rows = []
        for f in root.rglob(f"*task-{task}*beh.tsv"):
            m = pat.search(f.name)
            if m:
                rows.append({
                    "subject": m.group("sub"),
                    "task": task,
                    "session": m.group("ses"),
                })
        return (
            pd.DataFrame(rows)
            .drop_duplicates()
            .sort_values(["subject", "session"])
            .reset_index(drop=True)
        )
