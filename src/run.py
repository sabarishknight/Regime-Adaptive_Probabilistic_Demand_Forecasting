"""End-to-end pipeline: regime-adaptive probabilistic demand forecasting.

Pipeline
--------
1. Load ENTSO-E (OPSD) hourly load for one country.
2. Build leak-free day-ahead features.
3. Split: train (model fit) | calibration (conformal) | test (streaming eval).
4. Fit a frozen gradient-boosting point model on the TRAIN period only.
5. Build static split-conformal intervals (fixed) and adaptive (ACI) intervals.
6. Benchmark coverage / width on the full test period, a calm pre-shock window,
   and the COVID-19 shock window. The base model is never retrained.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal import AdaptiveConformal, StaticSplitConformal
from data import load_country_load
from evaluate import interval_metrics, point_metrics, rolling_coverage
from features import build_features, feature_columns

# ----------------------------- configuration ------------------------------
COUNTRY_COL = "FR_load_actual_entsoe_transparency"  # France
COUNTRY_NAME = "France (FR)"
ALPHA = 0.10            # target 90% prediction intervals
GAMMA = 0.02            # ACI adaptation rate

TRAIN_END = "2018-12-31 23:00"
CALIB_START, CALIB_END = "2019-01-01 00:00", "2019-12-31 23:00"
TEST_START, TEST_END = "2020-01-01 00:00", "2020-09-30 23:00"

# COVID-19 lockdown shock in France (national lockdown began 2020-03-17).
SHOCK_START, SHOCK_END = "2020-03-15 00:00", "2020-05-31 23:00"
# Calm pre-shock reference window inside the test set.
PRESHOCK_START, PRESHOCK_END = "2020-01-01 00:00", "2020-02-29 23:00"

OUT_DIR = "outputs"


def slice_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    return (index >= start) & (index <= end)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----------------------------- data ------------------------------------
    load = load_country_load(COUNTRY_COL)
    feat = build_features(load)
    cols = feature_columns(feat)
    idx = feat.index

    train_m = idx <= pd.Timestamp(TRAIN_END, tz="UTC")
    calib_m = slice_mask(idx, CALIB_START, CALIB_END)
    test_m = slice_mask(idx, TEST_START, TEST_END)

    X_train, y_train = feat.loc[train_m, cols], feat.loc[train_m, "load_mw"]
    X_calib, y_calib = feat.loc[calib_m, cols], feat.loc[calib_m, "load_mw"]
    X_test, y_test = feat.loc[test_m, cols], feat.loc[test_m, "load_mw"]
    test_idx = idx[test_m]

    print(f"Country: {COUNTRY_NAME}")
    print(f"Interpolated hours in raw series: {load.attrs.get('n_interpolated')}")
    print(f"Train rows:  {len(X_train):6d}  ({X_train.index.min().date()} -> {X_train.index.max().date()})")
    print(f"Calib rows:  {len(X_calib):6d}  ({X_calib.index.min().date()} -> {X_calib.index.max().date()})")
    print(f"Test rows:   {len(X_test):6d}  ({X_test.index.min().date()} -> {X_test.index.max().date()})")

    # ------------------------- base point model ----------------------------
    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)  # fit ONCE on pre-COVID data; never refit.

    pred_calib = model.predict(X_calib)
    pred_test = model.predict(X_test)
    y_test_arr = y_test.values

    pm = point_metrics(y_test_arr, pred_test)
    print("\nBase point model on 2020 test period:")
    print(f"  MAE  = {pm['mae']:.1f} MW")
    print(f"  RMSE = {pm['rmse']:.1f} MW")
    print(f"  MAPE = {pm['mape_pct']:.2f} %")

    calib_residuals = y_calib.values - pred_calib

    # --------------------- static split conformal --------------------------
    static = StaticSplitConformal(alpha=ALPHA).fit(calib_residuals)
    lo_s, hi_s = static.intervals(pred_test)

    # ------------------------ adaptive conformal ---------------------------
    aci = AdaptiveConformal(alpha=ALPHA, gamma=GAMMA).fit(calib_residuals)
    lo_a, hi_a, alpha_traj = aci.run(pred_test, y_test_arr)

    # ------------------------- benchmarking --------------------------------
    windows = {
        "full_test": slice_mask(test_idx, TEST_START, TEST_END),
        "pre_shock": slice_mask(test_idx, PRESHOCK_START, PRESHOCK_END),
        "covid_shock": slice_mask(test_idx, SHOCK_START, SHOCK_END),
    }

    results = {
        "country": COUNTRY_NAME,
        "target_coverage": 1 - ALPHA,
        "point_metrics_test": pm,
        "windows": {},
    }

    print("\n" + "=" * 74)
    print(f"{'window':14s} {'n':>6s} | {'STATIC cov':>11s} {'width':>8s} |"
          f" {'ADAPT cov':>10s} {'width':>8s}")
    print("-" * 74)
    for name, m in windows.items():
        yt = y_test_arr[m]
        sm = interval_metrics(yt, lo_s[m], hi_s[m], target=1 - ALPHA)
        am = interval_metrics(yt, lo_a[m], hi_a[m], target=1 - ALPHA)
        results["windows"][name] = {"n": int(m.sum()), "static": sm, "adaptive": am}
        print(f"{name:14s} {int(m.sum()):6d} | {sm['coverage']*100:10.1f}% "
              f"{sm['mean_width']:8.0f} | {am['coverage']*100:9.1f}% "
              f"{am['mean_width']:8.0f}")
    print("=" * 74)

    # Headline stability metric: reduction in coverage gap during the shock.
    s_gap = results["windows"]["covid_shock"]["static"]["coverage_gap"]
    a_gap = results["windows"]["covid_shock"]["adaptive"]["coverage_gap"]
    reduction = (s_gap - a_gap) / s_gap * 100 if s_gap > 0 else float("nan")
    results["shock_coverage_gap_static"] = s_gap
    results["shock_coverage_gap_adaptive"] = a_gap
    results["shock_coverage_gap_reduction_pct"] = reduction

    print(f"\nCOVID shock coverage gap |coverage-0.90|:")
    print(f"  static   = {s_gap*100:.1f} pts")
    print(f"  adaptive = {a_gap*100:.1f} pts")
    print(f"  reduction in miscoverage gap = {reduction:.1f}%")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Persist arrays for plotting.
    np.savez(
        os.path.join(OUT_DIR, "arrays.npz"),
        ts=test_idx.view("int64"),
        y=y_test_arr,
        pred=pred_test,
        lo_s=lo_s, hi_s=hi_s,
        lo_a=lo_a, hi_a=hi_a,
        alpha=alpha_traj,
    )
    print(f"\nSaved metrics.json and arrays.npz to {OUT_DIR}/")


if __name__ == "__main__":
    main()
