"""Conformal prediction interval methods.

Two methods are provided, both wrapping the SAME frozen base point model
(no retraining at test time):

1. StaticSplitConformal  -- classic split conformal. A single absolute-residual
   quantile is computed once on a calibration set and reused unchanged for every
   test point. This is the static baseline.

2. AdaptiveConformal (ACI) -- Adaptive Conformal Inference (Gibbs & Candes,
   2021). The target miscoverage level alpha_t is updated online from realised
   coverage errors:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

   where err_t = 1 if the true value fell outside the interval at step t.
   The (1 - alpha_t) absolute-residual quantile from the calibration pool is
   used to size each interval. This is a lightweight test-time recalibration:
   the base model is never refit, only the interval width adapts to drift.
"""
from __future__ import annotations

import numpy as np


class StaticSplitConformal:
    """Fixed-width split-conformal intervals from calibration residuals."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_ = None

    def fit(self, calib_residuals: np.ndarray) -> "StaticSplitConformal":
        scores = np.abs(calib_residuals)
        n = len(scores)
        # Finite-sample-adjusted conformal quantile level.
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def intervals(self, y_pred: np.ndarray):
        lo = y_pred - self.q_
        hi = y_pred + self.q_
        return lo, hi


class AdaptiveConformal:
    """Adaptive Conformal Inference (ACI) with online alpha updates."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.02):
        self.alpha = alpha          # target miscoverage (=> 1-alpha coverage)
        self.gamma = gamma          # adaptation step size
        self.scores_ = None

    def fit(self, calib_residuals: np.ndarray) -> "AdaptiveConformal":
        # Calibration pool of nonconformity scores (absolute residuals).
        self.scores_ = np.sort(np.abs(calib_residuals))
        return self

    def _width_for_alpha(self, alpha_t: float) -> float:
        a = float(np.clip(alpha_t, 0.0, 1.0))
        if a <= 0.0:
            # Demand full coverage -> widen beyond observed calibration range.
            return float(self.scores_[-1]) * 1.5
        if a >= 1.0:
            return 0.0
        return float(np.quantile(self.scores_, 1.0 - a, method="higher"))

    def run(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Stream over test points, adapting alpha_t after each observation.

        Returns lo, hi arrays and the trajectory of alpha_t.
        """
        n = len(y_pred)
        lo = np.empty(n)
        hi = np.empty(n)
        alpha_traj = np.empty(n)
        alpha_t = self.alpha
        for t in range(n):
            alpha_traj[t] = alpha_t
            w = self._width_for_alpha(alpha_t)
            lo[t] = y_pred[t] - w
            hi[t] = y_pred[t] + w
            # Realised coverage error at step t.
            err = 0.0 if (lo[t] <= y_true[t] <= hi[t]) else 1.0
            # ACI online update.
            alpha_t = alpha_t + self.gamma * (self.alpha - err)
        return lo, hi, alpha_traj
