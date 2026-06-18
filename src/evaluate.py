"""Evaluation metrics for point forecasts and prediction intervals."""
from __future__ import annotations

import numpy as np


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.abs(y_true)) * 100)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}


def interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     target: float = 0.90) -> dict:
    covered = (y_true >= lo) & (y_true <= hi)
    coverage = float(np.mean(covered))
    width = float(np.mean(hi - lo))
    return {
        "coverage": coverage,
        "coverage_gap": float(abs(coverage - target)),
        "mean_width": width,
    }


def rolling_coverage(y_true, lo, hi, window: int = 168) -> np.ndarray:
    """Rolling coverage (default 1-week window) to visualise stability."""
    covered = ((y_true >= lo) & (y_true <= hi)).astype(float)
    s = np.convolve(covered, np.ones(window) / window, mode="valid")
    return s
