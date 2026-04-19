"""GDELT 2.0 downloader, filtered to India.

GDELT 2.0 publishes three rolling archives every 15 minutes:
  * events       — actor–action–actor records with CAMEO + Goldstein
  * mentions     — article-level mentions of events (URL + time)
  * gkg          — Global Knowledge Graph (themes, tone, entities)

Each 15-min file is a gzipped CSV addressed as:
  http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.<kind>.CSV.zip

This module walks a date range, downloads the three kinds, filters rows
to India via ActionGeo_CountryCode == 'IN' (events) or by country tag in
the GKG V2LOCATIONS field, and writes a single parquet per day per kind.

Usage:
    python -m src.data.gdelt_downloader --start 2020-01-01 --end 2020-01-07 \
        --out ./data/raw/gdelt
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests
from tqdm.auto import tqdm

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
Kind = Literal["export", "mentions", "gkg"]

# Column schemas follow the GDELT 2.0 documentation.
EVENT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

MENTIONS_COLS = [
    "GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType",
    "MentionSourceName", "MentionIdentifier", "SentenceID",
    "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset",
    "InRawText", "Confidence", "MentionDocLen", "MentionDocTone", "MentionDocTranslationInfo", "Extras",
]

# GKG columns are complex; we only keep the high-value ones for trust learning.
GKG_COLS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations", "V2Organizations",
    "V2Tone", "Dates", "GCAM", "SharingImage", "RelatedImages",
    "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations", "AllNames",
    "Amounts", "TranslationInfo", "Extras",
]

log = logging.getLogger(__name__)


def timestamps_15min(start: dt.datetime, end: dt.datetime) -> Iterable[dt.datetime]:
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        yield t
        t += dt.timedelta(minutes=15)


def fetch_zip(url: str, session: requests.Session, timeout: int = 60) -> bytes | None:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return None
        return r.content
    except requests.RequestException as e:
        log.warning("fetch failed %s: %s", url, e)
        return None


def _read_zipped_csv(blob: bytes, cols: list[str]) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            inner = z.namelist()[0]
            with z.open(inner) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=cols,
                    dtype=str,
                    quoting=3,  # QUOTE_NONE
                    on_bad_lines="skip",
                    low_memory=False,
                )
        return df
    except Exception as e:  # pragma: no cover - network/format errors
        log.warning("zip read failed: %s", e)
        return None


def _filter_india_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["ActionGeo_CountryCode"].fillna("").str.upper() == "IN"].copy()


def _filter_india_gkg(df: pd.DataFrame) -> pd.DataFrame:
    # V2Locations contains pipe-separated location-block entries where the 4th
    # semicolon-delimited field is the FIPS country code.
    def _has_india(cell: str | float) -> bool:
        if not isinstance(cell, str) or not cell:
            return False
        for loc in cell.split(";"):
            parts = loc.split("#")
            if len(parts) >= 4 and parts[3].strip().upper() == "IN":
                return True
        return False
    mask = df["V2Locations"].apply(_has_india)
    return df[mask].copy()


def _filter_india_mentions(df: pd.DataFrame, india_event_ids: set[str]) -> pd.DataFrame:
    if not india_event_ids:
        return df.iloc[0:0]
    return df[df["GlobalEventID"].isin(india_event_ids)].copy()


def download_range(start: dt.datetime,
                   end: dt.datetime,
                   out_dir: str | os.PathLike,
                   kinds: tuple[Kind, ...] = ("export", "mentions", "gkg"),
                   sleep_on_miss: float = 0.0) -> None:
    """Download GDELT 2.0 15-min files in [start, end], filter to India, write daily parquet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Bucket by day; append in-memory, flush per day per kind.
    per_day: dict[str, dict[Kind, list[pd.DataFrame]]] = {}
    india_event_ids: dict[str, set[str]] = {}

    for ts in tqdm(list(timestamps_15min(start, end)), desc="GDELT 15-min"):
        stamp = ts.strftime("%Y%m%d%H%M%S")
        day = ts.strftime("%Y-%m-%d")
        per_day.setdefault(day, {"export": [], "mentions": [], "gkg": []})
        india_event_ids.setdefault(day, set())

        if "export" in kinds:
            blob = fetch_zip(f"{BASE_URL}/{stamp}.export.CSV.zip", session)
            if blob is not None:
                df = _read_zipped_csv(blob, EVENT_COLS)
                if df is not None:
                    df = _filter_india_events(df)
                    if len(df):
                        per_day[day]["export"].append(df)
                        india_event_ids[day].update(df["GlobalEventID"].astype(str).tolist())

        if "gkg" in kinds:
            blob = fetch_zip(f"{BASE_URL}/{stamp}.gkg.csv.zip", session)
            if blob is not None:
                df = _read_zipped_csv(blob, GKG_COLS)
                if df is not None:
                    df = _filter_india_gkg(df)
                    if len(df):
                        per_day[day]["gkg"].append(df)

        if "mentions" in kinds:
            blob = fetch_zip(f"{BASE_URL}/{stamp}.mentions.CSV.zip", session)
            if blob is not None:
                df = _read_zipped_csv(blob, MENTIONS_COLS)
                if df is not None:
                    df = _filter_india_mentions(df, india_event_ids[day])
                    if len(df):
                        per_day[day]["mentions"].append(df)

    for day, kind_map in per_day.items():
        for kind, frames in kind_map.items():
            if not frames:
                continue
            out_day = out / kind
            out_day.mkdir(parents=True, exist_ok=True)
            path = out_day / f"{day}.parquet"
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            log.info("wrote %s (%d rows)", path, sum(len(f) for f in frames))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default="./data/raw/gdelt")
    p.add_argument("--kinds", nargs="+", default=["export", "mentions", "gkg"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end) + dt.timedelta(hours=23, minutes=45)
    download_range(start, end, args.out, kinds=tuple(args.kinds))


if __name__ == "__main__":
    main()
