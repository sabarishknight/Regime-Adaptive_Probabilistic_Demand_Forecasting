"""Data loading and cleaning for ENTSO-E (OPSD) hourly electricity load.

The raw file is the Open Power System Data `time_series` package which
re-publishes ENTSO-E Transparency Platform actual-load series at hourly
resolution. We extract a single country/bidding-zone series, localise the
timestamp index and interpolate the small number of missing hours.
"""
from __future__ import annotations

import pandas as pd

RAW_PATH = "data/opsd_time_series_60min.csv"
TIMESTAMP_COL = "utc_timestamp"


def load_country_load(country_col: str, raw_path: str = RAW_PATH) -> pd.Series:
    """Load a single ENTSO-E actual-load series as a clean hourly pd.Series.

    Parameters
    ----------
    country_col : str
        Column name in the OPSD file, e.g. ``FR_load_actual_entsoe_transparency``.
    """
    df = pd.read_csv(
        raw_path,
        usecols=[TIMESTAMP_COL, country_col],
        parse_dates=[TIMESTAMP_COL],
    )
    df = df.set_index(TIMESTAMP_COL).sort_index()
    s = df[country_col].astype("float64")

    # Enforce a complete, regular hourly index.
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    s = s.reindex(full_index)

    # A handful of isolated missing hours -> linear interpolation (time-based).
    n_missing = int(s.isna().sum())
    s = s.interpolate(method="time", limit_direction="both")
    s.name = "load_mw"
    s.attrs["n_interpolated"] = n_missing
    return s
