"""Feature engineering for day-ahead style load forecasting.

All predictors are constructed so they are available at prediction time for a
day-ahead horizon: calendar features (deterministic) and lagged load values at
least 24 hours old. This avoids look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lags (in hours) that are all known when producing a day-ahead forecast.
LAGS = [24, 25, 48, 72, 168, 336]
# Rolling windows are computed on a 24h-shifted series so they only use the past.
ROLL_WINDOWS = [24, 168]
TARGET = "load_mw"


def build_features(load: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of features + target indexed by timestamp."""
    df = pd.DataFrame({TARGET: load})
    idx = df.index

    # --- Calendar features ---
    hour = idx.hour
    dow = idx.dayofweek
    month = idx.month
    doy = idx.dayofyear

    df["hour"] = hour
    df["dayofweek"] = dow
    df["month"] = month
    df["is_weekend"] = (dow >= 5).astype(int)

    # Cyclical encodings for smooth periodic structure.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # --- Lagged load features (all >= 24h old) ---
    for lag in LAGS:
        df[f"lag_{lag}h"] = df[TARGET].shift(lag)

    # --- Rolling stats on a 24h-shifted series (strictly past information) ---
    shifted = df[TARGET].shift(24)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(w).std()

    df = df.dropna()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]
