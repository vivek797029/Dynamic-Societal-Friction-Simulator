"""GDELT events → (state, week, cleavage) event intensity tensor E_k(r,t).

Also provides helpers to build the DataFrame the rest of the pipeline consumes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .india_geo import NUM_STATES, STATE_TO_IDX, iso_week_index, resolve_state

# CAMEO root codes 14–20 are material conflict / coercion; 13 = threat, 12 = reject.
# We keep 12..20 as "negative" valence candidates.
NEGATIVE_ROOT_CODES = set(range(12, 21))


def load_gdelt_events(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    frames = [pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["EventRootCode"] = pd.to_numeric(df["EventRootCode"], errors="coerce")
    df = df.dropna(subset=["GoldsteinScale", "ActionGeo_ADM1Code", "SQLDATE"])
    df["state"] = df["ActionGeo_ADM1Code"].map(resolve_state)
    df = df.dropna(subset=["state"])
    df["date"] = pd.to_datetime(df["Day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["iso_week"] = df["date"].apply(iso_week_index)
    df["is_negative"] = df["EventRootCode"].fillna(0).astype(int).isin(NEGATIVE_ROOT_CODES)
    return df


def attach_cleavage_from_actors(df: pd.DataFrame) -> pd.DataFrame:
    """Distant-supervision cleavage tag from GDELT Actor attributes.

    Produces a column `cleavage` ∈ {communal, caste, political_party, centre_state,
    economic, linguistic, other}.  When multiple attributes fire we take the
    *most specific* by a priority rule; a separate multi-label tensor is built
    in the hostility step for articles.
    """
    religion_cols = ["Actor1Religion1Code", "Actor2Religion1Code",
                     "Actor1Religion2Code", "Actor2Religion2Code"]
    ethnic_cols = ["Actor1EthnicCode", "Actor2EthnicCode"]
    type_cols = ["Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
                 "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code"]

    def row_cleavage(row: pd.Series) -> str:
        rel = {row[c] for c in religion_cols if isinstance(row[c], str) and row[c]}
        eth = {row[c] for c in ethnic_cols if isinstance(row[c], str) and row[c]}
        typ = {row[c] for c in type_cols if isinstance(row[c], str) and row[c]}

        # Communal: mixed religion pair (HIN vs MUS vs CHR etc.)
        if len({r for r in rel if r in {"HIN", "MUS", "CHR", "SIK", "BUD", "JAI"}}) >= 2:
            return "communal"
        # Caste — GDELT encodes scheduled-caste & scheduled-tribe via ethnic codes.
        if any(e in {"DAL", "SCH", "STC"} for e in eth):
            return "caste"
        # Political party
        if any(t in {"OPP", "GOV", "MIL", "COP", "JUD"} for t in typ):
            # Government vs opposition → political_party; gov vs state actor → centre_state
            if ("GOV" in typ) and ("OPP" in typ):
                return "political_party"
            if ("GOV" in typ) and ("COP" in typ):
                return "centre_state"
            return "political_party"
        # Economic (business / labour / farmer actors)
        if any(t in {"BUS", "LAB", "AGR"} for t in typ):
            return "economic"
        # Linguistic — rare in GDELT, usually requires text side; default other
        return "other"

    df = df.copy()
    df["cleavage"] = df.apply(row_cleavage, axis=1)
    return df


CLEAVAGES = ["communal", "caste", "political_party", "centre_state", "economic", "linguistic"]


def event_intensity_tensor(df: pd.DataFrame,
                           min_week: int,
                           max_week: int) -> np.ndarray:
    """Return E[S, T, K] of Goldstein-weighted negative-event intensity."""
    S, T, K = NUM_STATES, max_week - min_week + 1, len(CLEAVAGES)
    E = np.zeros((S, T, K), dtype=np.float32)
    cleavage_idx = {c: i for i, c in enumerate(CLEAVAGES)}
    df_neg = df[df["is_negative"] & df["cleavage"].isin(CLEAVAGES)]
    for _, row in df_neg.iterrows():
        si = STATE_TO_IDX.get(row["state"])
        if si is None:
            continue
        ti = int(row["iso_week"]) - min_week
        if ti < 0 or ti >= T:
            continue
        ki = cleavage_idx[row["cleavage"]]
        E[si, ti, ki] += abs(float(row["GoldsteinScale"]))
    return E
