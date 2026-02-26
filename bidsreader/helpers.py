import numpy as np
import pandas as pd
from typing import Iterable, Any, Tuple, Sequence, Optional, Dict
import re
from .exc import InvalidOptionError

def validate_option(name: str, value: Any, allowed: Iterable[Any]) -> Any:
    if value is None:
        return None
    if value not in allowed:
        raise InvalidOptionError(f"{name} must be one of: {allowed}. Got {value!r}")
    return value

def space_from_coordsystem_fname(fname: str) -> Optional[str]:
    if "_space-" not in fname:
        return None
    return fname.split("_space-")[1].split("_coordsystem.json")[0]

def add_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    if value is None:
        return None

    value = str(value)

    if value.startswith(prefix):
        return value

    return f"{prefix}{value}"

def merge_duplicate_sample_events(evs: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
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

def find_coord_triplets(columns: Sequence[str]) -> Dict[str, Tuple[str, str, str]]:
        cols = set(columns)

        triplets = {}

        if {"x", "y", "z"} <= cols:
            triplets[""] = ("x", "y", "z")

        prefixed = [c for c in cols if re.match(r"^.+\.(x|y|z)$", c)]
        prefixes = set(c.rsplit(".", 1)[0] for c in prefixed)

        for p in prefixes:
            x, y, z = f"{p}.x", f"{p}.y", f"{p}.z"
            if {x, y, z} <= cols:
                triplets[p] = (x, y, z)

        return triplets

def combine_bipolar_electrodes(
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
    coord_triplets = find_coord_triplets(elec_df.columns)

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

def normalize_trial_types(trial_types: Iterable[str]) -> set[str]:
    return {str(t) for t in trial_types}

def match_event_label(label: str, trial_types: list[str]) -> bool:
    # exact token match within merged labels like "WORD/STIM"
    tokens = label.split("/")
    return any(t in tokens for t in trial_types)