"""Generate figures from the saved pipeline arrays."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import rolling_coverage

OUT_DIR = "outputs"
SHOCK_START = pd.Timestamp("2020-03-15")
SHOCK_END = pd.Timestamp("2020-05-31")
TARGET = 0.90


def load_arrays():
    d = np.load(os.path.join(OUT_DIR, "arrays.npz"))
    # tz-naive UTC wall-clock timestamps (matplotlib-friendly).
    ts = pd.to_datetime(d["ts"])
    return ts, d


def plot_intervals(ts, d):
    """Forecast + intervals zoomed on the lockdown onset."""
    mask = (ts >= pd.Timestamp("2020-03-01")) & (ts <= pd.Timestamp("2020-04-15"))
    t = ts[mask]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for a, lo, hi, name, col in [
        (ax[0], d["lo_s"], d["hi_s"], "Static split-conformal", "tab:orange"),
        (ax[1], d["lo_a"], d["hi_a"], "Adaptive conformal (ACI)", "tab:green"),
    ]:
        a.fill_between(t, lo[mask], hi[mask], color=col, alpha=0.30, label="90% interval")
        a.plot(t, d["y"][mask], color="black", lw=0.9, label="Actual load")
        a.plot(t, d["pred"][mask], color="tab:blue", lw=0.8, ls="--", label="Point forecast")
        a.axvline(SHOCK_START, color="red", ls=":", lw=1.5, label="Lockdown (2020-03-17)")
        a.set_title(f"{name} - France, COVID-19 lockdown onset")
        a.set_ylabel("Load (MW)")
        a.legend(loc="upper right", fontsize=8, ncol=2)
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax[1].set_xlabel("Date (2020)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "intervals_lockdown.png"), dpi=130)
    plt.close(fig)


def plot_rolling_coverage(ts, d):
    """Rolling 1-week coverage: static collapses during shock, adaptive holds."""
    window = 168
    cov_s = rolling_coverage(d["y"], d["lo_s"], d["hi_s"], window)
    cov_a = rolling_coverage(d["y"], d["lo_a"], d["hi_a"], window)
    t = ts[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, cov_s * 100, color="tab:orange", label="Static split-conformal")
    ax.plot(t, cov_a * 100, color="tab:green", label="Adaptive conformal (ACI)")
    ax.axhline(TARGET * 100, color="black", ls="--", lw=1, label="Target 90%")
    ax.axvspan(SHOCK_START, SHOCK_END, color="red", alpha=0.12, label="COVID-19 shock window")
    ax.set_title("Rolling 1-week coverage of 90% prediction intervals - France 2020")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xlabel("Date (2020)")
    ax.set_ylim(50, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rolling_coverage.png"), dpi=130)
    plt.close(fig)


def plot_alpha(ts, d):
    """ACI's online target-miscoverage trajectory reacting to the shock."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, d["alpha"], color="tab:purple", lw=0.9)
    ax.axhline(0.10, color="black", ls="--", lw=1, label="Nominal alpha = 0.10")
    ax.axvspan(SHOCK_START, SHOCK_END, color="red", alpha=0.12, label="COVID-19 shock window")
    ax.set_title("ACI online target-miscoverage alpha_t (lower alpha => wider intervals)")
    ax.set_ylabel("alpha_t")
    ax.set_xlabel("Date (2020)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aci_alpha.png"), dpi=130)
    plt.close(fig)


def main():
    ts, d = load_arrays()
    plot_intervals(ts, d)
    plot_rolling_coverage(ts, d)
    plot_alpha(ts, d)
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        m = json.load(f)
    print("Figures written to", OUT_DIR)
    print(json.dumps(m["windows"], indent=2))


if __name__ == "__main__":
    main()
