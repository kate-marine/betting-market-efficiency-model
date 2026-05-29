"""
Post-hoc calibration of the LightGBM multiclass predictions.

Two methods:
  - Isotonic regression (one-vs-rest): non-parametric, monotone mapping.
    More flexible; may overfit with small calibration sets (<200 samples).
  - Platt scaling (one-vs-rest): logistic regression on logit(raw_p).
    Parametric (2 degrees of freedom per outcome); more stable with small data.

Both use a one-vs-rest approach: calibrate each outcome independently, then
renormalize so the three calibrated probabilities sum to 1. The renormalization
is a mild approximation — it breaks the strict multiclass structure — but works
well in practice and is the standard approach (Zadrozny & Elkan 2002).

Walk-forward integration: for each test season T, the calibrator is fit on
the immediately preceding season T-1 predictions. This is the same calibration
set used by Phase 7's conformal evaluation. We must never use test-season data
to fit the calibrator.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Outcome columns
# ---------------------------------------------------------------------------

OUTCOMES = ("H", "D", "A")
RAW_COLS = {"H": "pred_pH", "D": "pred_pD", "A": "pred_pA"}
CAL_COLS_ISO   = {"H": "iso_pH",   "D": "iso_pD",   "A": "iso_pA"}
CAL_COLS_PLATT = {"H": "platt_pH", "D": "platt_pD", "A": "platt_pA"}


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

def _fit_isotonic(raw_p: np.ndarray, observed: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(raw_p, observed)
    return iso


def _fit_platt(raw_p: np.ndarray, observed: np.ndarray) -> LogisticRegression:
    # Logistic regression on logit(p) — the standard Platt scaling parametrisation.
    # Clip to avoid log(0).
    eps = 1e-6
    logit = np.log(np.clip(raw_p, eps, 1 - eps) / np.clip(1 - raw_p, eps, 1 - eps))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(logit.reshape(-1, 1), observed)
    return lr


def _apply_platt(model: LogisticRegression, raw_p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    logit = np.log(np.clip(raw_p, eps, 1 - eps) / np.clip(1 - raw_p, eps, 1 - eps))
    return model.predict_proba(logit.reshape(-1, 1))[:, 1]


def _renormalize(
    cal_H: np.ndarray, cal_D: np.ndarray, cal_A: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = cal_H + cal_D + cal_A
    # Guard against degenerate rows (all zero — shouldn't happen with real data)
    total = np.where(total < 1e-9, 1.0, total)
    return cal_H / total, cal_D / total, cal_A / total


# ---------------------------------------------------------------------------
# Walk-forward calibration
# ---------------------------------------------------------------------------

def calibrate_walk_forward(
    preds: pd.DataFrame,
    methods: tuple[str, ...] = ("isotonic", "platt"),
    min_cal_samples: int = 100,
) -> pd.DataFrame:
    """
    Add post-hoc calibrated probability columns to the predictions DataFrame.

    For each test season T, fits calibrators on T-1 predictions and applies
    them to T. Returns the same DataFrame with additional columns:
        iso_pH,   iso_pD,   iso_pA   (isotonic)
        platt_pH, platt_pD, platt_pA (Platt scaling)

    Seasons with no prior test season in the data (or too few calibration
    samples) get NaN calibrated probabilities.
    """
    df = preds.copy()
    seasons = sorted(df["test_season"].unique())

    # Initialize calibrated columns as NaN
    for method in methods:
        cal_cols = CAL_COLS_ISO if method == "isotonic" else CAL_COLS_PLATT
        for col in cal_cols.values():
            df[col] = np.nan

    for i, test_season in enumerate(seasons):
        if i == 0:
            continue   # no prior season for calibration

        cal_season = seasons[i - 1]
        cal_mask  = df["test_season"] == cal_season
        test_mask = df["test_season"] == test_season

        if cal_mask.sum() < min_cal_samples:
            warnings.warn(f"Skipping {test_season}: only {cal_mask.sum()} cal samples")
            continue

        cal_df  = df[cal_mask]
        test_df = df[test_mask]

        for method in methods:
            cal_cols = CAL_COLS_ISO if method == "isotonic" else CAL_COLS_PLATT
            raw_cal = {}   # outcome → calibrated probs on test set

            for outcome, raw_col in RAW_COLS.items():
                cal_raw  = cal_df[raw_col].values
                cal_obs  = (cal_df["result"] == outcome).values.astype(float)
                test_raw = test_df[raw_col].values

                if method == "isotonic":
                    fitted = _fit_isotonic(cal_raw, cal_obs)
                    raw_cal[outcome] = fitted.predict(test_raw)
                else:
                    fitted = _fit_platt(cal_raw, cal_obs)
                    raw_cal[outcome] = _apply_platt(fitted, test_raw)

            # Renormalize so the three calibrated probs sum to 1
            norm_H, norm_D, norm_A = _renormalize(
                raw_cal["H"], raw_cal["D"], raw_cal["A"]
            )
            df.loc[test_mask, cal_cols["H"]] = norm_H
            df.loc[test_mask, cal_cols["D"]] = norm_D
            df.loc[test_mask, cal_cols["A"]] = norm_A

    return df
