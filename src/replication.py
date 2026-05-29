"""
H&W (2024) replication: favorite-longshot bias regression.

Implements equation 13 from Hegarty & Whelan (2024):

    Y_ij - P^N_ij = alpha + gamma * P^N_ij + epsilon_ij

estimated by WLS with weights 1/(P*(1-P)), cluster-robust SEs at the match
level. gamma > 0 → favorite-longshot bias (favorites underpriced).

The module also implements the *inverse-odds* version of the regression —
using raw 1/odds instead of normalized probabilities — to demonstrate the
upward bias that H&W diagnose in the earlier literature (their section 3).

Why cluster at the match level: for a match with K outcomes, the K rows sum
to zero in the dependent variable (outcomes sum to 1, probs sum to 1, so
residuals are mechanically correlated). Ignoring this inflates the effective
sample size and shrinks SEs, making the test anti-conservative.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def _hw_regression(
    df: pd.DataFrame,
    p_col: str,
    y_col: str = "observed",
    match_id_col: str = "match_id",
    weight_floor: float = 1e-4,
) -> dict:
    """
    Run the H&W WLS regression on a single group (league or pooled data).

    Parameters
    ----------
    df : long-format DataFrame with one row per (match, outcome)
    p_col : column name holding the implied probability estimate
    y_col : column name holding the observed outcome (0/1)
    match_id_col : column for clustering SEs
    weight_floor : clip P*(1-P) below this to avoid division by zero

    Returns dict with keys: gamma, alpha, gamma_se, alpha_se, t_stat, p_value,
        n_matches, n_obs, converged
    """
    sub = df[[p_col, y_col, match_id_col]].dropna()
    if len(sub) < 10:
        return _empty_result()

    p = sub[p_col].values
    y = sub[y_col].values
    dep = y - p                               # Y_ij - P^N_ij
    weights = 1.0 / np.clip(p * (1.0 - p), weight_floor, None)

    X = sm.add_constant(p, prepend=True)     # [intercept, P^N_ij]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.WLS(dep, X, weights=weights)
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": sub[match_id_col].values},
            )
    except Exception:
        return _empty_result()

    return {
        "alpha":    float(result.params[0]),
        "gamma":    float(result.params[1]),
        "alpha_se": float(result.bse[0]),
        "gamma_se": float(result.bse[1]),
        "t_stat":   float(result.tvalues[1]),
        "p_value":  float(result.pvalues[1]),
        "n_obs":    int(result.nobs),
        "n_matches": int(sub[match_id_col].nunique()),
        "converged": True,
    }


def _empty_result() -> dict:
    return {k: np.nan for k in
            ("alpha", "gamma", "alpha_se", "gamma_se", "t_stat", "p_value")} | \
           {"n_obs": 0, "n_matches": 0, "converged": False}


# ---------------------------------------------------------------------------
# Pooled and by-league tables
# ---------------------------------------------------------------------------

def run_hw_table(
    long_df: pd.DataFrame,
    p_col: str = "norm_p",
    season_range: Optional[tuple[str, str]] = None,
    league_col: str = "league",
) -> pd.DataFrame:
    """
    Produce an H&W-style results table: pooled + one row per league.

    Parameters
    ----------
    long_df : long-format DataFrame (one row per match×outcome)
    p_col : which probability column to use ('norm_p' for normalized,
            'inv_odds' for raw inverse-odds comparison)
    season_range : optional ("2011-2012", "2021-2022") filter

    Returns DataFrame with columns: league, gamma, gamma_se, t_stat, p_value,
        n_matches, n_obs.
    """
    df = long_df.copy()

    if season_range is not None:
        lo, hi = season_range
        df = df[df["season"].between(lo, hi)]

    rows = []

    # Pooled
    r = _hw_regression(df, p_col)
    rows.append({"league": "Pooled"} | r)

    # By league
    for league, grp in df.groupby(league_col):
        r = _hw_regression(grp, p_col)
        rows.append({"league": league} | r)

    result = pd.DataFrame(rows)
    result = result[[
        "league", "gamma", "gamma_se", "t_stat", "p_value",
        "n_matches", "n_obs", "alpha", "alpha_se", "converged"
    ]]
    return result


# ---------------------------------------------------------------------------
# Normalized vs. inverse-odds comparison (H&W's key methodological point)
# ---------------------------------------------------------------------------

def compare_estimators(long_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Run both the normalized-probability regression (H&W's method) and the
    inverse-odds regression (the biased method used in earlier literature).

    H&W show that regressing on raw 1/odds instead of normalized probs inflates
    gamma mechanically because the overround adds a systematic positive term to
    the regressor — even in a perfectly efficient market, you'd see a positive
    gamma under this specification.

    Returns a merged DataFrame with _norm and _inv suffixes on key columns.
    """
    df = long_df.copy()
    # Construct the raw inverse-odds column
    df["inv_odds"] = 1.0 / df["odds"]

    norm_table = run_hw_table(df, p_col="norm_p", **kwargs)
    inv_table  = run_hw_table(df, p_col="inv_odds", **kwargs)

    merged = norm_table.merge(
        inv_table[["league", "gamma", "gamma_se", "t_stat", "p_value"]],
        on="league",
        suffixes=("_norm", "_inv"),
    )
    return merged


# ---------------------------------------------------------------------------
# Bootstrap CIs on gamma — match-level resampling (H&W cluster point)
# ---------------------------------------------------------------------------

def bootstrap_gamma(
    long_df: pd.DataFrame,
    p_col: str = "norm_p",
    n_boot: int = 1000,
    seed: int = 0,
    league: Optional[str] = None,
    season_range: Optional[tuple[str, str]] = None,
) -> dict:
    """
    Bootstrap confidence intervals on gamma using match-level resampling.

    Row-level resampling would break the mechanical correlation structure within
    a match (the three outcome rows are not independent). We resample *matches*
    instead, then keep all K outcome rows per resampled match.

    Returns dict with keys: gamma_hat, ci_low, ci_high, se_boot.
    """
    df = long_df.copy()
    if season_range is not None:
        df = df[df["season"].between(*season_range)]
    if league is not None:
        df = df[df["league"] == league]

    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()

    gammas = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(match_ids, size=len(match_ids), replace=True)
        # Build a lookup for repeated matches
        id_df = pd.DataFrame({"match_id": sampled_ids}).reset_index(names="boot_id")
        boot = id_df.merge(df, on="match_id")
        # Use boot_id as the cluster key so repeated matches get separate cluster IDs
        r = _hw_regression(boot, p_col, match_id_col="boot_id")
        if r["converged"]:
            gammas.append(r["gamma"])

    gammas = np.array(gammas)
    base = _hw_regression(df, p_col)
    return {
        "gamma_hat": base["gamma"],
        "gamma_se":  base["gamma_se"],
        "ci_low":  float(np.percentile(gammas, 2.5)),
        "ci_high": float(np.percentile(gammas, 97.5)),
        "se_boot": float(gammas.std()),
        "n_boot_success": len(gammas),
    }


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def add_multiple_testing_correction(
    table: pd.DataFrame,
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Add Bonferroni and BH-FDR corrected p-values to a run_hw_table() result.

    Only the per-league rows are corrected (the pooled row is a single test
    and should not be included in the correction). The number of simultaneous
    tests is the number of per-league rows.

    Why both corrections:
      - Bonferroni: conservative, controls FWER (probability of any false positive)
      - BH-FDR: less conservative, controls FDR (expected fraction of false positives)
    Both are reported so the reader can see the range of conclusions.

    Adds columns: p_bonferroni, p_bh, sig_bonferroni, sig_bh.
    """
    from statsmodels.stats.multitest import multipletests

    df = table.copy()
    df["p_bonferroni"] = np.nan
    df["p_bh"]         = np.nan
    df["sig_bonferroni"] = ""
    df["sig_bh"]         = ""

    # Per-league rows only (not Pooled)
    league_mask = df["league"] != "Pooled"
    p_vals = df.loc[league_mask, p_col].values.astype(float)
    n = len(p_vals)

    # Bonferroni
    p_bonf = np.minimum(p_vals * n, 1.0)
    df.loc[league_mask, "p_bonferroni"] = p_bonf

    # BH-FDR
    _, p_bh_vals, _, _ = multipletests(p_vals, method="fdr_bh")
    df.loc[league_mask, "p_bh"] = p_bh_vals

    # Significance stars on corrected values
    def stars(p):
        if pd.isna(p): return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    df.loc[league_mask, "sig_bonferroni"] = [stars(p) for p in p_bonf]
    df.loc[league_mask, "sig_bh"]         = [stars(p) for p in p_bh_vals]

    return df


# ---------------------------------------------------------------------------
# Table formatting utilities
# ---------------------------------------------------------------------------

def format_table(df: pd.DataFrame, sig_stars: bool = True) -> pd.DataFrame:
    """Format a results table for display/export."""
    out = df.copy()
    for col in ("gamma", "gamma_se", "alpha", "alpha_se"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    for col in ("t_stat",):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if "p_value" in out.columns and sig_stars:
        def fmt_p(p):
            if pd.isna(p):
                return "—"
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            return f"{p:.4f}{stars}"
        out["p_value"] = out["p_value"].map(fmt_p)
    for col in ("n_matches", "n_obs"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    return out
