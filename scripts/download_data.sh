#!/usr/bin/env bash
# Download the ENTSO-E hourly load data (via Open Power System Data).
# OPSD re-publishes ENTSO-E Transparency Platform actual-load series.
# This snapshot (2020-10-06) covers 2015-01-01 .. 2020-09-30, including the
# COVID-19 demand shock used for the regime-shift validation.
set -euo pipefail

mkdir -p data
URL="https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
OUT="data/opsd_time_series_60min.csv"

if [ -f "$OUT" ]; then
  echo "Data already present at $OUT"
  exit 0
fi

echo "Downloading ENTSO-E (OPSD) hourly load data (~130 MB)..."
curl -L -o "$OUT" "$URL"
echo "Saved to $OUT"
