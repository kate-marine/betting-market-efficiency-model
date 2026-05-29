"""
LightGBM base model with walk-forward (expanding-window) cross-validation.

Walk-forward CV: train on all seasons up to T-1, predict season T.
Never shuffle — time ordering is everything. A random train/test split would
leak future match outcomes into the training set via Elo ratings and form
features, which are computed cumulatively from past matches.

Output: a DataFrame with pred_pH / pred_pD / pred_pA columns appended,
covering only the held-out test seasons. These predictions go into the
calibration evaluation (Phase 8) and the Layer 1 comparison against market
implied probabilities.
"""

from __future__ import annotations

import warnings
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, RESULT_MAP, compute_features


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 0,
    "verbose": -1,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def walk_forward_predict(
    wide: pd.DataFrame,
    test_seasons: Optional[list[str]] = None,
    min_train_seasons: int = 1,
    params: Optional[dict] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Train a LightGBM model with walk-forward expanding-window CV and return
    out-of-sample predictions for each test season.

    Parameters
    ----------
    wide : wide-format DataFrame (from load_soccer), with feature columns
        already added by compute_features(). If features are absent they
        will be computed here.
    test_seasons : seasons to generate predictions for. Defaults to all
        seasons except the earliest (which has nothing to train on).
    min_train_seasons : skip a test season if fewer than this many training
        seasons are available. First seasons have very little training data.
    params : LightGBM parameters (merged into DEFAULT_PARAMS).
    seed : random seed for reproducibility.

    Returns
    -------
    DataFrame with the same rows as the test portion of `wide`, plus columns:
        pred_pH, pred_pD, pred_pA  (predicted probabilities, sum to 1)
        test_season                (which season this row was predicted in)
    """
    lgb_params = {**DEFAULT_PARAMS, "random_state": seed}
    if params:
        lgb_params.update(params)

    # Compute features if not already present
    if "elo_home" not in wide.columns:
        wide = compute_features(wide)

    # Sort by date globally — features were computed in this order
    df = wide.sort_values(["date", "match_id"]).reset_index(drop=True)

    all_seasons = sorted(df["season"].unique())
    if test_seasons is None:
        test_seasons = all_seasons[min_train_seasons:]

    result_parts = []

    for test_season in test_seasons:
        train_df = df[df["season"] < test_season].copy()
        test_df  = df[df["season"] == test_season].copy()

        n_train_seasons = train_df["season"].nunique()
        if n_train_seasons < min_train_seasons or len(train_df) == 0:
            warnings.warn(f"Skipping {test_season}: only {n_train_seasons} training seasons")
            continue

        # Drop rows with missing outcome or features
        train_valid = train_df.dropna(subset=["result"] + FEATURE_COLS)
        if len(train_valid) < 100:
            warnings.warn(f"Skipping {test_season}: too few valid training rows ({len(train_valid)})")
            continue

        X_train = train_valid[FEATURE_COLS]   # keep as DataFrame so LightGBM retains feature names
        y_train = train_valid["result"].map(RESULT_MAP).values

        test_valid = test_df.dropna(subset=FEATURE_COLS)
        X_test = test_valid[FEATURE_COLS]

        model = lgb.LGBMClassifier(**lgb_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)   # shape (n, 3): cols = H, D, A

        test_valid = test_valid.copy()
        test_valid["pred_pH"] = proba[:, 0]
        test_valid["pred_pD"] = proba[:, 1]
        test_valid["pred_pA"] = proba[:, 2]
        test_valid["test_season"] = test_season

        result_parts.append(test_valid)
        print(f"  {test_season}: trained on {n_train_seasons} seasons "
              f"({len(train_valid):,} matches) → predicted {len(test_valid):,} matches",
              flush=True)

    if not result_parts:
        warnings.warn("No predictions generated — all test seasons were skipped")
        return pd.DataFrame(columns=list(wide.columns) + ["pred_pH","pred_pD","pred_pA","test_season"])

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience: add predictions to wide parquet and save
# ---------------------------------------------------------------------------

def run_and_save(
    wide: pd.DataFrame,
    output_path: str,
    test_seasons: Optional[list[str]] = None,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Compute features, run walk-forward CV, save predictions to Parquet."""
    print("Computing features...")
    wide_feat = compute_features(wide)

    print("Running walk-forward CV...")
    preds = walk_forward_predict(wide_feat, test_seasons=test_seasons, seed=seed, **kwargs)

    preds.to_parquet(output_path, index=False)
    print(f"Saved {len(preds):,} predictions to {output_path}")
    return preds
