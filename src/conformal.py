"""
Conformal prediction wrapper for the soccer outcome model.

Two conformal methods are implemented:

  1. Split conformal (marginal guarantee): calibrate on a held-out season,
     apply threshold to test season. Coverage guarantee is marginal —
     P(Y ∈ C(X)) ≥ 1-α holds on average over test matches, not conditionally
     on any subgroup.

  2. Mondrian conformal (group-conditional guarantee): compute separate
     calibration thresholds per league. Coverage guarantee is conditional —
     P(Y ∈ C(X) | league=l) ≥ 1-α for each league l. Stronger claim but
     requires enough calibration data per league.

Nonconformity score: s_i = 1 - f̂(x_i)_{y_i}
  — the probability the model assigned to the true outcome, inverted.
  Low score = high confidence (and correct). High score = surprised by the truth.

The finite-sample correction to the quantile level:
  q̂ = quantile(s_cal, ⌈(n+1)(1-α)⌉/n)
ensures exact marginal coverage ≥ 1-α even in finite samples, under the
assumption that calibration and test scores are exchangeable (i.i.d. draws
from the same distribution). This assumption is *approximately* satisfied in
our walk-forward setup (consecutive seasons) but not guaranteed — temporal
drift violates strict exchangeability. We flag this as a limitation.

Reference: Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification."
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Outcome encoding (matches the model's class order: H=0, D=1, A=2)
# ---------------------------------------------------------------------------

OUTCOME_TO_IDX = {"H": 0, "D": 1, "A": 2}
PROB_COLS_MODEL  = ["pred_pH", "pred_pD", "pred_pA"]
PROB_COLS_MARKET = ["norm_pH", "norm_pD", "norm_pA"]


# ---------------------------------------------------------------------------
# Core conformal functions
# ---------------------------------------------------------------------------

def nonconformity_scores(
    probs: np.ndarray,
    true_idx: np.ndarray,
) -> np.ndarray:
    """
    Compute s_i = 1 − f̂(x_i)_{y_i} for each instance.

    Parameters
    ----------
    probs     : (n, 3) array of predicted probabilities
    true_idx  : (n,) array of true outcome indices (0=H, 1=D, 2=A)
    """
    return 1.0 - probs[np.arange(len(true_idx)), true_idx]


def conformal_threshold(
    cal_scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Finite-sample quantile correction for split conformal.

    Returns the (1-α)-level threshold for prediction sets.
    If the calibration set is empty, returns 1.0 (include everything).
    """
    n = len(cal_scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level))


def prediction_sets(
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Boolean (n, 3) array: True where predicted prob ≥ 1 − threshold.

    Outcome j is included in match i's prediction set when the model is
    confident enough about j relative to the calibration threshold.
    """
    return probs >= (1.0 - threshold)


def coverage_rate(pred_sets: np.ndarray, true_idx: np.ndarray) -> float:
    """Fraction of matches where the true outcome is in the prediction set."""
    return float(pred_sets[np.arange(len(true_idx)), true_idx].mean())


def mean_set_size(pred_sets: np.ndarray) -> float:
    """Average number of outcomes included per match."""
    return float(pred_sets.sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Mondrian (group-conditional) conformal
# ---------------------------------------------------------------------------

def mondrian_thresholds(
    cal_scores: np.ndarray,
    cal_groups: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """
    Compute per-group conformal thresholds.

    Parameters
    ----------
    cal_scores  : (n,) nonconformity scores from calibration set
    cal_groups  : (n,) group labels (e.g., league codes)
    alpha       : miscoverage level

    Returns dict mapping group label → threshold.
    Groups with fewer than 30 calibration examples fall back to the global
    threshold (noted as approximate).
    """
    global_thresh = conformal_threshold(cal_scores, alpha)
    thresholds = {}
    for group in np.unique(cal_groups):
        mask = cal_groups == group
        group_scores = cal_scores[mask]
        if mask.sum() < 30:
            thresholds[group] = global_thresh   # fallback; not a valid guarantee
        else:
            thresholds[group] = conformal_threshold(group_scores, alpha)
    return thresholds


def mondrian_prediction_sets(
    probs: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
    fallback_threshold: float,
) -> np.ndarray:
    """Apply per-group thresholds to produce Mondrian prediction sets."""
    sets = np.zeros(probs.shape, dtype=bool)
    for group in np.unique(groups):
        mask = groups == group
        tau = thresholds.get(group, fallback_threshold)
        sets[mask] = probs[mask] >= (1.0 - tau)
    return sets


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_conformal(
    preds: pd.DataFrame,
    alpha: float = 0.1,
    prob_cols_model: list[str] = PROB_COLS_MODEL,
    prob_cols_market: list[str] = PROB_COLS_MARKET,
) -> pd.DataFrame:
    """
    Walk-forward conformal evaluation: for each test season T, calibrate on T-1.

    Evaluates both the ML model and the market at the same nominal level α,
    under both marginal and Mondrian (by-league) conformal.

    Returns a DataFrame with one row per (season, estimator, method) with
    columns: coverage, mean_set_size, n_cal, n_test, alpha, guarantee.
    """
    seasons = sorted(preds["test_season"].unique())
    rows = []

    for i, test_season in enumerate(seasons):
        if i == 0:
            continue   # no prior season to use as calibration

        cal_season = seasons[i - 1]
        cal = preds[preds["test_season"] == cal_season].copy()
        test = preds[preds["test_season"] == test_season].copy()

        if len(cal) < 10 or len(test) < 10:
            continue

        cal_true = cal["result"].map(OUTCOME_TO_IDX).values
        test_true = test["result"].map(OUTCOME_TO_IDX).values
        cal_leagues = cal["league"].values
        test_leagues = test["league"].values

        for estimator, prob_cols in [("model", prob_cols_model), ("market", prob_cols_market)]:
            cal_probs  = cal[prob_cols].values
            test_probs = test[prob_cols].values

            cal_scores = nonconformity_scores(cal_probs, cal_true)

            # --- Marginal conformal ---
            tau_marginal = conformal_threshold(cal_scores, alpha)
            sets_marginal = prediction_sets(test_probs, tau_marginal)
            rows.append({
                "test_season": test_season,
                "cal_season": cal_season,
                "estimator": estimator,
                "method": "marginal",
                "guarantee": "marginal: P(Y∈C(X)) ≥ 1-α",
                "alpha": alpha,
                "tau": tau_marginal,
                "coverage": coverage_rate(sets_marginal, test_true),
                "mean_set_size": mean_set_size(sets_marginal),
                "n_cal": len(cal),
                "n_test": len(test),
            })

            # --- Mondrian conformal (by league) ---
            global_tau = conformal_threshold(cal_scores, alpha)
            league_thresholds = mondrian_thresholds(cal_scores, cal_leagues, alpha)
            sets_mondrian = mondrian_prediction_sets(
                test_probs, test_leagues, league_thresholds, global_tau
            )
            rows.append({
                "test_season": test_season,
                "cal_season": cal_season,
                "estimator": estimator,
                "method": "mondrian_league",
                "guarantee": "conditional: P(Y∈C(X)|league) ≥ 1-α per league",
                "alpha": alpha,
                "tau": float(np.mean(list(league_thresholds.values()))),
                "coverage": coverage_rate(sets_mondrian, test_true),
                "mean_set_size": mean_set_size(sets_mondrian),
                "n_cal": len(cal),
                "n_test": len(test),
            })

    return pd.DataFrame(rows)


def per_league_coverage(
    preds: pd.DataFrame,
    alpha: float = 0.1,
    prob_cols_model: list[str] = PROB_COLS_MODEL,
    prob_cols_market: list[str] = PROB_COLS_MARKET,
) -> pd.DataFrame:
    """
    Mondrian per-league coverage breakdown pooled across all test seasons.

    Uses all available (cal_season, test_season) pairs, pooling calibration
    and test data across seasons for each league to get stable estimates.

    Note: pooling across seasons means the coverage guarantee is marginal-
    within-league-across-seasons, not strictly conditional on season.
    """
    seasons = sorted(preds["test_season"].unique())
    rows = []

    for estimator, prob_cols in [("model", prob_cols_model), ("market", prob_cols_market)]:
        # Pool all calibration and test data across walk-forward pairs
        cal_parts, test_parts = [], []
        for i, test_season in enumerate(seasons):
            if i == 0:
                continue
            cal_season = seasons[i - 1]
            cal_parts.append(preds[preds["test_season"] == cal_season])
            test_parts.append(preds[preds["test_season"] == test_season])

        if not cal_parts:
            continue

        cal_all = pd.concat(cal_parts)
        test_all = pd.concat(test_parts)

        cal_scores = nonconformity_scores(
            cal_all[prob_cols].values,
            cal_all["result"].map(OUTCOME_TO_IDX).values,
        )
        global_tau = conformal_threshold(cal_scores, alpha)
        league_thresholds = mondrian_thresholds(
            cal_scores, cal_all["league"].values, alpha
        )

        for league in sorted(test_all["league"].unique()):
            test_league = test_all[test_all["league"] == league]
            test_probs = test_league[prob_cols].values
            test_true = test_league["result"].map(OUTCOME_TO_IDX).values
            tau = league_thresholds.get(league, global_tau)

            sets = prediction_sets(test_probs, tau)
            cov = coverage_rate(sets, test_true)
            sz = mean_set_size(sets)
            n_cal_league = (cal_all["league"] == league).sum()

            rows.append({
                "league": league,
                "estimator": estimator,
                "tau": tau,
                "coverage": cov,
                "mean_set_size": sz,
                "n_cal": n_cal_league,
                "n_test": len(test_league),
                "coverage_gap": cov - (1 - alpha),   # positive = over-covers
            })

    return pd.DataFrame(rows)
