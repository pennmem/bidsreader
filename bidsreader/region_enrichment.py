"""Opt-in enrichment that attaches pair-level region labels from
the upstream CML ``pairs.json`` to a bidsreader pairs DataFrame.

Pure-BIDS users don't need this module. It exists only for the hybrid
case where a subject's BIDS export sits alongside the original
neurorad-pipeline output on rhino and the caller wants the pair-region
labels that neurorad computed (via independent atlas lookup at each
pair's midpoint voxel). The BIDS side can't reproduce those labels
from contact-level data, so we pull them from the upstream artifact.

See :mod:`._neurorad_algo` for why pair region labels differ from the
contact-level agreement heuristic.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


_DEFAULT_REGION_COLS: tuple[str, ...] = (
    "ind.region",
    "ind.corrected.region",
    "avg.region",
    "avg.corrected.region",
    "mni.region",
    "hcp.region",
    "stein.region",
    "das.region",
)


def enrich_pairs_with_cml_regions(
    pairs_df: pd.DataFrame,
    *,
    subject: Optional[str] = None,
    experiment: Optional[str] = None,
    session: Optional[int] = None,
    localization: Optional[int] = 0,
    montage: Optional[int] = 0,
    reader=None,
    label_col: str = "name",
    region_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy of ``pairs_df`` with CML ``*.region`` columns joined on.

    ``pairs_df`` is the output of
    :func:`bidsreader.helpers.combine_bipolar_electrodes`, which uses
    ``name`` as the pair-label column by default.
    """

    try:
        from cmlreaders import CMLReader
    except ImportError as exc:
        raise ImportError(
            "enrich_pairs_with_cml_regions requires cmlreaders. Install "
            "it (pip install cmlreaders) or load pairs.json manually."
        ) from exc

    if reader is None:
        if subject is None or experiment is None or session is None:
            raise ValueError(
                "Provide either a ready CMLReader or "
                "(subject, experiment, session)."
            )
        reader = CMLReader(
            subject=subject,
            experiment=experiment,
            session=session,
            localization=localization,
            montage=montage,
        )

    cml_pairs = reader.load("pairs")

    wanted = tuple(region_cols) if region_cols is not None else _DEFAULT_REGION_COLS
    available = [c for c in wanted if c in cml_pairs.columns]

    if "label" not in cml_pairs.columns:
        raise KeyError(
            "cml_pairs is missing the 'label' column — can't join on pair "
            "name. Got columns: %r" % (list(cml_pairs.columns),)
        )

    right = cml_pairs[["label", *available]].copy()
    right["label"] = right["label"].astype("string").str.strip()
    if label_col != "label":
        right = right.rename(columns={"label": label_col})

    out = pairs_df.copy()
    out[label_col] = out[label_col].astype("string").str.strip()

    return out.merge(right, how="left", on=label_col)
