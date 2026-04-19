"""ACLED India loader — the independent ground-truth channel.

ACLED is *not* used as a feature. It is the forecasting target against which
all friction coefficients are identified. Download CSV manually from the
ACLED export tool (free academic registration), save to data/raw/acled/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .india_geo import NUM_STATES, STATE_TO_IDX, iso_week_index


ACLED_EVENT_TYPES_OF_INTEREST = {
    "Battles",
    "Protests",
    "Riots",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Strategic developments",
}


def load_acled(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize columns (ACLED changes casing occasionally).
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "event_date" if "event_date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["country"].str.lower() == "india"]
    df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0)
    df["iso_week"] = df["date"].apply(iso_week_index)
    df["state"] = df["admin1"].astype(str).str.strip()
    return df


def state_week_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ACLED to (state, iso_week, event_type) counts + fatalities."""
    g = (
        df[df["event_type"].isin(ACLED_EVENT_TYPES_OF_INTEREST)]
        .groupby(["state", "iso_week", "event_type"])
        .agg(events=("fatalities", "size"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    return g


def build_target_tensor(df: pd.DataFrame,
                        horizons: list[int] = (1, 2, 4),
                        min_week: int | None = None,
                        max_week: int | None = None) -> dict[str, "np.ndarray"]:
    """Return {f'y_protests_h{h}': array[S,T], f'y_fatalities_h{h}': array[S,T]}.

    Target at (state r, week t) is the *future* window [t+1, t+h] sum.
    """
    import numpy as np

    agg = state_week_counts(df)
    if min_week is None:
        min_week = int(agg["iso_week"].min())
    if max_week is None:
        max_week = int(agg["iso_week"].max())
    T = max_week - min_week + 1
    S = NUM_STATES

    # Per (state, week) counts.
    protests = np.zeros((S, T), dtype=np.float32)
    fatalities = np.zeros((S, T), dtype=np.float32)
    for _, row in agg.iterrows():
        si = STATE_TO_IDX.get(row["state"])
        if si is None:
            continue
        ti = int(row["iso_week"]) - min_week
        if ti < 0 or ti >= T:
            continue
        if row["event_type"] in ("Protests", "Riots"):
            protests[si, ti] += row["events"]
        fatalities[si, ti] += row["fatalities"]

    out: dict[str, np.ndarray] = {}
    for h in horizons:
        yp = np.zeros_like(protests)
        yf = np.zeros_like(fatalities)
        for t in range(T):
            u = min(T, t + 1 + h)
            if t + 1 < T:
                yp[:, t] = protests[:, t + 1:u].sum(axis=1)
                yf[:, t] = fatalities[:, t + 1:u].sum(axis=1)
        out[f"y_protests_h{h}"] = yp
        out[f"y_fatalities_h{h}"] = yf
    out["protests_now"] = protests
    out["fatalities_now"] = fatalities
    out["min_week"] = min_week
    out["max_week"] = max_week
    return out
