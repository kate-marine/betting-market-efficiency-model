"""
Power analysis for the H&W FLB regression.

Answers the question: given our sample sizes and the observed cluster-robust SEs,
what is the smallest FLB (gamma) we could have detected at 80% power?

The "no FLB detected" result for E0, D1, SP1 is NOT evidence of market efficiency —
it is evidence of insufficient power. This module quantifies how underpowered
we are and puts an honest lower bound on the undetectable effect size.

Two approaches:
  1. Analytical: uses the normal approximation to the t-distribution.
     MDE = (z_{alpha/2} + z_{1-beta}) * SE_cluster
     Valid for large samples (n_matches > 500), which all our leagues satisfy.

  2. Simulation: draws synthetic outcomes with known gamma from the real odds
     distribution, runs the full H&W regression, reports empirical rejection rate.
     Used to validate the analytical formula given the real cluster structure.

The generating process for FLB outcomes:
  pi_ij = -gamma/3 + (1 + gamma) * P^N_ij
These probabilities sum to 1 per match and encode exactly gamma worth of FLB:
E[Y_ij - P^N_ij] = alpha + gamma * P^N_ij  with alpha = -gamma/3.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Analytical power functions
# ---------------------------------------------------------------------------

def analytical_power(
    gamma: float,
    se: float,
    alpha: float = 0.05,
) -> float:
    """
    Analytical power of the two-sided H&W test at a true effect size gamma.

    Uses the normal approximation: under H1: gamma = gamma_1, the t-statistic
    gamma_hat / SE is approximately N(gamma_1/SE, 1). This is accurate for
    n_matches > 500.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp = abs(gamma) / se
    # P(|Z - ncp| > z_alpha) where Z ~ N(0,1) -- equivalent to two-tailed power
    return float(stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha))


def analytical_mde(
    se: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Minimum detectable effect (MDE): the smallest |gamma| detectable at the
    specified power level given SE.

    MDE = (z_{alpha/2} + z_{1-beta}) * SE
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    return (z_alpha + z_beta) * se


def power_table(
    replication_table: pd.DataFrame,
    alpha: float = 0.05,
    power: float = 0.80,
    pooled_gamma: float = 0.0459,
) -> pd.DataFrame:
    """
    Build the full MDE table from run_hw_table() output.

    Adds columns:
      mde          — minimum detectable gamma at (alpha, power)
      power_at_obs — power at the actually observed gamma (how powered were we?)
      power_at_pooled — power at the pooled gamma (reference: could we detect
                         an effect as large as the pooled estimate?)
      adequately_powered — True if power_at_obs >= power threshold
    """
    rows = []
    for _, row in replication_table.iterrows():
        if pd.isna(row.get("gamma_se")) or row.get("n_matches", 0) == 0:
            continue
        se = float(row["gamma_se"])
        gamma_obs = float(row["gamma"])
        n = int(row["n_matches"])
        mde = analytical_mde(se, alpha=alpha, power=power)
        p_obs    = analytical_power(gamma_obs, se, alpha=alpha)
        p_pooled = analytical_power(pooled_gamma, se, alpha=alpha)
        rows.append({
            "league":           row["league"],
            "gamma":            gamma_obs,
            "gamma_se":         se,
            "n_matches":        n,
            "mde":              mde,
            "power_at_obs":     p_obs,
            "power_at_pooled":  p_pooled,
            "adequately_powered": (p_obs >= power),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FLB outcome generation for simulation
# ---------------------------------------------------------------------------

def generate_flb_outcomes(
    wide_df: pd.DataFrame,
    gamma: float,
    rng: np.random.Generator,
    norm_cols: tuple[str, str, str] = ("norm_pH", "norm_pD", "norm_pA"),
) -> pd.DataFrame:
    """
    Generate synthetic match outcomes with known gamma FLB, using real odds.

    The generating probabilities are:
        pi_ij = -gamma/3 + (1 + gamma) * P^N_ij

    which encode exactly E[Y_ij - P^N_ij] = -gamma/3 + gamma * P^N_ij,
    matching the H&W regression specification with alpha = -gamma/3.

    Returns a copy of wide_df with 'result' column replaced by synthetic outcomes.
    """
    df = wide_df.copy()
    pH = df[norm_cols[0]].values
    pD = df[norm_cols[1]].values
    pA = df[norm_cols[2]].values

    pi_H = -gamma / 3 + (1 + gamma) * pH
    pi_D = -gamma / 3 + (1 + gamma) * pD
    pi_A = -gamma / 3 + (1 + gamma) * pA

    # Clip negatives (rare, only for very large gamma + extreme odds)
    pi_H = np.clip(pi_H, 1e-9, None)
    pi_D = np.clip(pi_D, 1e-9, None)
    pi_A = np.clip(pi_A, 1e-9, None)
    total = pi_H + pi_D + pi_A
    pi_H /= total; pi_D /= total; pi_A /= total

    n = len(df)
    u = rng.uniform(size=n)
    results = np.where(u < pi_H, "H", np.where(u < pi_H + pi_D, "D", "A"))
    df["result"] = results
    return df


# ---------------------------------------------------------------------------
# Fast WLS regression (numpy-only, no statsmodels) for simulation
# ---------------------------------------------------------------------------

def _wls_cluster_test(
    p: np.ndarray,
    y_minus_p: np.ndarray,
    cluster_ids: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """
    WLS regression of (y - p) on p with cluster-robust SE. Returns (gamma_hat, t_stat, reject).

    Uses the sandwich estimator: Var(beta) = (X'WX)^{-1} B (X'WX)^{-1}
    where B = sum_c (X_c' W_c e_c) (X_c' W_c e_c)'  (cluster-level scores).
    """
    w = 1.0 / np.clip(p * (1 - p), 1e-6, None)
    X = np.column_stack([np.ones(len(p)), p])  # (n, 2)
    y = y_minus_p

    # WLS estimate: beta = (X'WX)^{-1} X'Wy
    XtW = (X * w[:, None]).T  # (2, n)
    XtWX = XtW @ X            # (2, 2)
    XtWy = XtW @ y            # (2,)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, False

    # Residuals
    e_hat = y - X @ beta  # (n,)

    # Cluster-robust sandwich: sum over clusters
    unique_clusters = np.unique(cluster_ids)
    meat = np.zeros((2, 2))
    for c in unique_clusters:
        mask = cluster_ids == c
        score_c = XtW[:, mask] @ (e_hat[mask] * w[mask] / w[mask])
        # Equivalently: score_c = X_c' W_c e_hat_c
        sc = (X[mask] * w[mask, None]).T @ e_hat[mask]  # (2,)
        meat += np.outer(sc, sc)

    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, False

    vcov = XtWX_inv @ meat @ XtWX_inv
    se_gamma = np.sqrt(vcov[1, 1])

    gamma_hat = float(beta[1])
    t_stat    = gamma_hat / se_gamma if se_gamma > 0 else np.nan
    reject    = abs(t_stat) > stats.norm.ppf(1 - alpha / 2) if not np.isnan(t_stat) else False
    return gamma_hat, float(t_stat), bool(reject)


# ---------------------------------------------------------------------------
# Simulation power curve
# ---------------------------------------------------------------------------

def simulate_power_curve(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    gamma_vals: list[float],
    n_sim: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulation-based power curve. For each gamma in gamma_vals, simulate n_sim
    datasets with that FLB and count empirical rejection rate.

    Uses the fast numpy WLS+cluster sandwich estimator (no statsmodels) for speed.
    Each simulation generates new outcomes from the real odds distribution.

    Parameters
    ----------
    wide_df : wide-format match data with norm_pH/pD/pA and match_id columns
    long_df : long-format data — used to build the regression design (p values, cluster ids)
    gamma_vals : list of true gamma values to evaluate
    n_sim : number of simulations per gamma value

    Returns DataFrame with columns: gamma, power_sim, se_sim, n_sim.
    """
    rng = np.random.default_rng(seed)

    # Pre-build the regression design from the long format (constant across sims)
    # Sort by match_id then outcome so the wide→long mapping is consistent
    long_sorted = long_df.sort_values(["match_id", "outcome"]).reset_index(drop=True)
    p_base = long_sorted["norm_p"].values
    cluster_ids = long_sorted["match_id"].values

    # Build a match_id → (pH, pD, pA) mapping for generating new outcomes
    # We need the wide format's norm_p columns for the generating process
    match_wide = wide_df.set_index("match_id")[["norm_pH", "norm_pD", "norm_pA"]]

    rows = []
    for gamma in gamma_vals:
        rejections = 0
        for sim_idx in range(n_sim):
            # Generate synthetic outcomes with FLB = gamma
            sim_wide = generate_flb_outcomes(wide_df, gamma, rng)

            # Build y - p for the long format (must use simulated outcomes)
            # Map match outcomes to long format
            result_map = sim_wide.set_index("match_id")["result"]
            sim_observed = (long_sorted["outcome"] == long_sorted["match_id"].map(result_map)).astype(float)
            y_minus_p = sim_observed.values - p_base

            # Run WLS + cluster-robust test
            _, _, reject = _wls_cluster_test(p_base, y_minus_p, cluster_ids, alpha=alpha)
            if reject:
                rejections += 1

        power_sim = rejections / n_sim
        # Wilson SE for a proportion
        se_sim = np.sqrt(power_sim * (1 - power_sim) / n_sim)
        rows.append({
            "gamma": gamma,
            "power_sim": power_sim,
            "se_sim": se_sim,
            "n_sim": n_sim,
        })
        print(f"    gamma={gamma:.3f}  empirical power={power_sim:.3f} ± {se_sim:.3f}", flush=True)

    return pd.DataFrame(rows)
