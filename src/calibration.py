"""
Calibration evaluation: Brier score, log-loss, ECE, and reliability diagrams.

All metrics are computed on the long format — one row per (match, outcome) pair.
This treats each (match × outcome) as a binary prediction task: "did outcome j
occur in match i?" with predicted probability f̂_ij.

Bootstrap CIs use match-level resampling throughout: we resample matches and
pull all 3 outcome rows per match, preserving the within-match constraint that
observed values sum to 1. Row-level resampling would violate this constraint.

Metrics implemented:
  - Brier score: mean squared error of predicted vs. observed (0=perfect, 2=worst)
  - Log-loss (multiclass): -mean log(pred_p_true_class), i.e., only the
    probability assigned to the outcome that actually occurred
  - ECE: expected calibration error — weighted mean |observed_freq − predicted_prob|
    across equal-width probability bins

Why ECE over reliability diagrams alone: ECE is a single number that summarises
calibration gap, useful for tables and bootstrap CIs. Reliability diagrams are
the visual complement.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Long-format conversion
# ---------------------------------------------------------------------------

OUTCOME_COLS_MODEL  = {"H": "pred_pH",  "D": "pred_pD",  "A": "pred_pA"}
OUTCOME_COLS_MARKET = {"H": "norm_pH",  "D": "norm_pD",  "A": "norm_pA"}


def to_long(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot wide-format predictions to long format.
    Returns one row per (match, outcome) with columns:
        match_id, league, test_season, outcome, observed, pred_p, market_p
    """
    parts = []
    for outcome in ("H", "D", "A"):
        chunk = preds[["match_id", "league", "test_season", "result"]].copy()
        chunk["outcome"] = outcome
        chunk["observed"] = (preds["result"] == outcome).astype(int).values
        chunk["pred_p"]   = preds[OUTCOME_COLS_MODEL[outcome]].values
        chunk["market_p"] = preds[OUTCOME_COLS_MARKET[outcome]].values
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Point-estimate metrics
# ---------------------------------------------------------------------------

def brier_score(pred_p: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    return float(np.mean((pred_p - observed) ** 2))


def log_loss_multiclass(
    preds: pd.DataFrame,
    pred_cols: dict[str, str],
    eps: float = 1e-15,
) -> float:
    """
    Multiclass log-loss: -mean log(prob assigned to true class).

    Only the probability of the outcome that occurred contributes — this is
    the natural information-theoretic measure of how surprised the model is.
    """
    df = preds.reset_index(drop=True)
    correct_p = np.zeros(len(df))
    for outcome, col in pred_cols.items():
        mask = df["result"] == outcome
        correct_p[mask.values] = df.loc[mask, col].values
    correct_p = np.clip(correct_p, eps, 1.0 - eps)
    return float(-np.mean(np.log(correct_p)))


def ece(
    pred_p: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error over equal-width bins.

    ECE = Σ_b (n_b / n) |mean(observed_b) - mean(pred_b)|

    Empty bins are skipped. Note: ECE is sensitive to binning choice;
    we use equal-width bins (0, 0.1, ..., 1.0) which is standard.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred_p)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_p >= lo) & (pred_p < hi)
        if mask.sum() == 0:
            continue
        mean_pred = pred_p[mask].mean()
        mean_obs  = observed[mask].mean()
        weight = mask.sum() / total
        ece_val += weight * abs(mean_obs - mean_pred)
    return float(ece_val)


def reliability_diagram_data(
    pred_p: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin data for a reliability diagram.

    Returns DataFrame with columns:
        bin_mid, mean_pred, obs_freq, count, bin_lo, bin_hi
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_p >= lo) & (pred_p < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "bin_mid": (lo + hi) / 2,
            "mean_pred": float(pred_p[mask].mean()),
            "obs_freq": float(observed[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CIs — match-level resampling
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    long_df: pd.DataFrame,
    pred_col: str = "pred_p",
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = 10,
) -> dict:
    """
    Bootstrap confidence intervals on Brier score and ECE.

    Resamples matches (not rows) so within-match structure is preserved.
    Returns dict with point estimates and 95% percentile CIs.
    """
    rng = np.random.default_rng(seed)
    match_ids = long_df["match_id"].unique()

    bs_boots, ece_boots = [], []
    for _ in range(n_boot):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        boot_df = (
            pd.DataFrame({"match_id": sampled})
            .merge(long_df, on="match_id")
        )
        bs_boots.append(brier_score(boot_df[pred_col].values, boot_df["observed"].values))
        ece_boots.append(ece(boot_df[pred_col].values, boot_df["observed"].values, n_bins))

    point_bs  = brier_score(long_df[pred_col].values, long_df["observed"].values)
    point_ece = ece(long_df[pred_col].values, long_df["observed"].values, n_bins)

    if bs_boots:
        bs_boots  = np.array(bs_boots)
        ece_boots = np.array(ece_boots)
        return {
            "brier":      point_bs,
            "brier_lo":   float(np.percentile(bs_boots, 2.5)),
            "brier_hi":   float(np.percentile(bs_boots, 97.5)),
            "ece":        point_ece,
            "ece_lo":     float(np.percentile(ece_boots, 2.5)),
            "ece_hi":     float(np.percentile(ece_boots, 97.5)),
            "n_matches":  int(long_df["match_id"].nunique()),
            "n_obs":      len(long_df),
        }
    return {
        "brier": point_bs, "brier_lo": np.nan, "brier_hi": np.nan,
        "ece":   point_ece, "ece_lo":  np.nan, "ece_hi":  np.nan,
        "n_matches": int(long_df["match_id"].nunique()), "n_obs": len(long_df),
    }


# ---------------------------------------------------------------------------
# Calibration table: pooled and by subgroup
# ---------------------------------------------------------------------------

def calibration_table(
    long_df: pd.DataFrame,
    subgroup_col: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute Brier score and ECE with bootstrap CIs for both estimators.

    If subgroup_col is given (e.g. "league"), computes per-subgroup metrics
    without bootstrapping (too slow per-group); uses the full-sample bootstrap
    only for the pooled row.
    """
    rows = []

    def _add_row(label: str, df: pd.DataFrame, do_boot: bool):
        for pred_col, name in [("pred_p", "model"), ("market_p", "market")]:
            p = df[pred_col].values
            o = df["observed"].values
            bs = brier_score(p, o)
            ec = ece(p, o)
            if do_boot:
                boot = bootstrap_metrics(df, pred_col=pred_col, n_boot=n_boot, seed=seed)
                rows.append({
                    "group": label, "estimator": name,
                    "brier": bs, "brier_lo": boot["brier_lo"], "brier_hi": boot["brier_hi"],
                    "ece":   ec, "ece_lo":   boot["ece_lo"],   "ece_hi":   boot["ece_hi"],
                    "n_matches": boot["n_matches"], "n_obs": boot["n_obs"],
                    "bootstrapped": True,
                })
            else:
                rows.append({
                    "group": label, "estimator": name,
                    "brier": bs, "brier_lo": np.nan, "brier_hi": np.nan,
                    "ece":   ec, "ece_lo":   np.nan, "ece_hi":   np.nan,
                    "n_matches": long_df["match_id"].nunique()
                        if subgroup_col is None else df["match_id"].nunique(),
                    "n_obs": len(df),
                    "bootstrapped": False,
                })

    # Pooled row — with bootstrap
    _add_row("Pooled", long_df, do_boot=True)

    # Subgroup rows — point estimates only (bootstrapping per-league is expensive)
    if subgroup_col is not None:
        for grp, grp_df in long_df.groupby(subgroup_col):
            _add_row(str(grp), grp_df, do_boot=False)

    return pd.DataFrame(rows)
