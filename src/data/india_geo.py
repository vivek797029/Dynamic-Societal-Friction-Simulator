"""India geography utilities.

Maps GDELT ADM1 codes to state names, holds the canonical list of the 28
states + 8 Union Territories, and provides timezone-aware week indexing.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

# GDELT ADM1 codes for India follow the FIPS 10-4 standard. IN01..IN36 cover
# the current states + UTs. This table is curated rather than auto-generated so
# downstream code can trust the canonical labels.
ADM1_TO_STATE: dict[str, str] = {
    "IN01": "Andaman and Nicobar Islands",
    "IN02": "Andhra Pradesh",
    "IN03": "Assam",
    "IN04": "Bihar",
    "IN05": "Chandigarh",
    "IN06": "Dadra and Nagar Haveli and Daman and Diu",
    "IN07": "Delhi",
    "IN08": "Goa",
    "IN09": "Gujarat",
    "IN10": "Haryana",
    "IN11": "Himachal Pradesh",
    "IN12": "Jammu and Kashmir",
    "IN13": "Kerala",
    "IN14": "Lakshadweep",
    "IN15": "Madhya Pradesh",
    "IN16": "Maharashtra",
    "IN17": "Manipur",
    "IN18": "Meghalaya",
    "IN19": "Karnataka",
    "IN20": "Nagaland",
    "IN21": "Odisha",
    "IN22": "Puducherry",
    "IN23": "Punjab",
    "IN24": "Rajasthan",
    "IN25": "Sikkim",
    "IN26": "Tamil Nadu",
    "IN27": "Tripura",
    "IN28": "Uttar Pradesh",
    "IN29": "West Bengal",
    "IN30": "Chhattisgarh",
    "IN31": "Jharkhand",
    "IN32": "Uttarakhand",
    "IN33": "Mizoram",
    "IN34": "Arunachal Pradesh",
    "IN35": "Telangana",
    "IN36": "Ladakh",
}

STATES: list[str] = list(ADM1_TO_STATE.values())
NUM_STATES: int = len(STATES)
STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}


def resolve_state(adm1: Optional[str]) -> Optional[str]:
    """Return canonical state name for a GDELT ADM1 code, or None."""
    if adm1 is None or not isinstance(adm1, str):
        return None
    return ADM1_TO_STATE.get(adm1.strip().upper())


def iso_week_index(date: dt.datetime | dt.date, epoch: str = "2015-01-01") -> int:
    """Monotone week index since `epoch` (Monday-based, ISO 8601)."""
    if isinstance(date, dt.datetime):
        date = date.date()
    e = dt.date.fromisoformat(epoch)
    # Shift both to their ISO Monday, then subtract.
    def _monday(d: dt.date) -> dt.date:
        return d - dt.timedelta(days=d.weekday())
    return (_monday(date) - _monday(e)).days // 7


def attach_state_and_week(df: pd.DataFrame,
                          adm1_col: str = "ActionGeo_ADM1Code",
                          date_col: str = "SQLDATE") -> pd.DataFrame:
    """Add `state` and `iso_week` columns in place and return the frame."""
    df = df.copy()
    df["state"] = df[adm1_col].map(resolve_state)
    df = df.dropna(subset=["state"])
    if pd.api.types.is_integer_dtype(df[date_col]):
        # GDELT SQLDATE is YYYYMMDD int
        df["date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df["iso_week"] = df["date"].apply(iso_week_index)
    return df
