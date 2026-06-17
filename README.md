# Regime-Adaptive Probabilistic Demand Forecasting (Electricity, ENTSO-E)

Probabilistic day-ahead forecasting of national electricity demand with
**calibrated prediction intervals** via conformal prediction, plus a
**lightweight test-time recalibration** mechanism that keeps intervals
calibrated under distribution shift **without retraining the base model**.
The method is stress-tested on the **COVID-19 demand disruption** as a real
regime shift and benchmarked against a static conformal baseline.

## Headline results (France, 90% target intervals)

| Period | Static coverage | Adaptive coverage | Target |
|---|---|---|---|
| Full test (Jan–Sep 2020) | 90.1% | 90.1% | 90% |
| Pre-shock (Jan–Feb 2020) | 88.1% | 89.9% | 90% |
| **COVID-19 shock (15 Mar – 31 May 2020)** | **78.6%** | **90.4%** | 90% |

During the COVID-19 shock the **static** baseline's intervals decalibrated
sharply — empirical coverage fell to **78.6%** (miscoverage rose from the
nominal 10% to **21.4%**, more than double the target). The **adaptive**
method held coverage at **90.4%** (miscoverage 9.6%), **reducing the
coverage gap by 96%** (from 11.4 to 0.4 percentage points) — using the same
frozen base model.

Base point forecast on the 2020 test period: **MAE 1752 MW**, **RMSE 2800 MW**,
**MAPE 3.67%**.

### Resume bullet (final metric filled in)

> Benchmarked the adaptive model against a static baseline across the shock
> period: during the COVID-19 disruption the static model's interval coverage
> collapsed from 90% to 79% (miscoverage 21%), while the adaptive method held
> coverage at 90% — a **96% reduction in the calibration gap** with no
> retraining of the base model.

## Method

1. **Data** — ENTSO-E actual-load series (hourly), via Open Power System Data,
   2015-01-01 → 2020-09-30. France (FR) is used because it experienced a sharp,
   clean lockdown-driven demand drop in March 2020.
2. **Features** — leak-free day-ahead predictors: calendar terms (hour,
   day-of-week, month, weekend, cyclical encodings) and lagged/rolling load
   features at least 24h old.
3. **Splits** — model fit on **2015–2018**, conformal calibration on **2019**,
   streaming evaluation on **2020** (Jan–Sep). The base model is fit **once on
   pre-COVID data and never retrained**.
4. **Base model** — `HistGradientBoostingRegressor` point forecaster.
5. **Static split conformal** — a single absolute-residual quantile from the
   2019 calibration set, applied unchanged to every test point.
6. **Adaptive Conformal Inference (ACI)** — online update of the target
   miscoverage level after each observation:
   `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`. The interval width is
   re-derived each step from the calibration residual pool. Only the width
   adapts; the trained model is untouched — a lightweight test-time
   recalibration.

## Project layout

```
src/data.py        load + clean ENTSO-E/OPSD load series
src/features.py    leak-free calendar + lag/rolling features
src/conformal.py   StaticSplitConformal and AdaptiveConformal (ACI)
src/evaluate.py    point + interval metrics, rolling coverage
src/run.py         end-to-end pipeline -> outputs/metrics.json, arrays.npz
src/plots.py       figures from saved arrays
scripts/download_data.sh   fetch the raw data (~130 MB)
outputs/           metrics.json + figures
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh        # downloads the ENTSO-E/OPSD CSV
PYTHONPATH=src python src/run.py      # trains, benchmarks, writes metrics
PYTHONPATH=src python src/plots.py    # writes figures to outputs/
```

## Figures (`outputs/`)

- `rolling_coverage.png` — rolling 1-week coverage: static collapses inside the
  shock window, adaptive tracks the 90% target.
- `intervals_lockdown.png` — forecast + 90% intervals around the lockdown onset.
- `aci_alpha.png` — ACI's online `alpha_t` trajectory widening intervals as the
  regime shifts.

## Data source

Open Power System Data, *time_series* package (2020-10-06), which re-publishes
ENTSO-E Transparency Platform actual-load data. See
https://data.open-power-system-data.org/time_series/ . *Content summarized for
licensing compliance.*

## Notes & honest caveats

- Coverage is the primary calibration metric; adaptive intervals are wider
  during the shock (that is the point — they widen to stay calibrated when the
  world changes), and slightly narrower than static on average over the full
  test period (mean width 6631 vs 7775 MW).
- Results are for France; the pipeline is country-agnostic — change
  `COUNTRY_COL` in `src/run.py` to reproduce for DE, ES, IT, GB, etc.
