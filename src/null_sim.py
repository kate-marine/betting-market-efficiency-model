"""
Efficient-markets null simulation for the H&W regression.

Under H0 (market efficiency): outcomes for match i are drawn from
Categorical(P^N_iH, P^N_iD, P^N_iA) — i.e., the market-implied probabilities
are the true probabilities. There is no systematic favorite-longshot bias.

For each simulation, we draw new outcomes and run the WLS regression. The
simulation p-value for each league is:
    p_sim = P(|γ̂_null| ≥ |γ̂_obs|)

Comparing p_sim to the parametric p-value checks whether the normal
approximation is well-calibrated for this cluster structure and odds
distribution. Where they diverge, the parametric test is miscalibrated
(Winkelmann et al.'s concern).

The joint test uses the max |t-statistic| across all leagues as the test
statistic. This controls the family-wise error rate without assuming
independence between leagues.

Speed design: (X'WX)^{-1}X'W is precomputed per league once. Each
simulation then requires only:
  1. O(n_matches) to draw new outcomes
  2. O(n_long) matrix-vector multiply to get γ̂_sim
With n_sim=2000 and 9 leagues, this runs in ~30–60 seconds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-league precomputation
# ---------------------------------------------------------------------------

OUTCOME_TO_IDX = {"H": 0, "D": 1, "A": 2}


class LeagueSetup(NamedTuple):
    """Pre-computed regression objects for one league."""
    league: str
    n_matches: int
    n_long: int
    M: np.ndarray          # (2, n_long): (X'WX)^{-1} X'W
    p_long: np.ndarray     # (n_long,): norm_p per long-format row
    outcome_idx: np.ndarray  # (n_long,): 0=H,1=D,2=A per long-format row
    match_idx: np.ndarray  # (n_long,): which match (0..n_matches-1) each row belongs to
    pH: np.ndarray         # (n_matches,): normalized home-win prob
    pD: np.ndarray         # (n_matches,): normalized draw prob
    pA: np.ndarray         # (n_matches,): normalized away-win prob
    gamma_obs: float       # observed gamma from the real regression
    se_obs: float          # cluster-robust SE of observed gamma


def _build_setup(league: str, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> LeagueSetup:
    """
    Pre-compute everything needed for fast null simulation of one league.

    long_df must have columns: match_id, outcome, norm_p.
    wide_df must have columns: match_id, norm_pH, norm_pD, norm_pA.
    """
    # Sort long format by match_id then outcome so match blocks are contiguous
    long = long_df.sort_values(["match_id", "outcome"]).reset_index(drop=True)

    p = long["norm_p"].values.astype(np.float64)
    outcome_idx = long["outcome"].map(OUTCOME_TO_IDX).values.astype(np.int32)

    # Integer match index per long row
    match_ids_sorted = long["match_id"].unique()   # unique match IDs in this subset
    id_to_int = {m: i for i, m in enumerate(match_ids_sorted)}
    match_idx = long["match_id"].map(id_to_int).values.astype(np.int32)
    n_matches = len(match_ids_sorted)
    n_long    = len(long)

    # Wide format: align to same match ordering
    wide = wide_df.set_index("match_id").loc[match_ids_sorted].reset_index()
    pH = wide["norm_pH"].values.astype(np.float64)
    pD = wide["norm_pD"].values.astype(np.float64)
    pA = wide["norm_pA"].values.astype(np.float64)

    # WLS regression: Y = (1, p) * beta, weights = 1/(p*(1-p))
    w = 1.0 / np.clip(p * (1.0 - p), 1e-6, None)
    X = np.column_stack([np.ones(n_long), p])      # (n_long, 2)
    XtW = (X * w[:, None]).T                        # (2, n_long)
    XtWX = XtW @ X                                  # (2, 2)
    M = np.linalg.solve(XtWX, XtW)                 # (2, n_long): the projection matrix

    # Observed gamma and SE (placeholder; caller fills these from the real regression)
    gamma_obs = 0.0
    se_obs    = 0.0

    return LeagueSetup(
        league=league, n_matches=n_matches, n_long=n_long,
        M=M, p_long=p, outcome_idx=outcome_idx, match_idx=match_idx,
        pH=pH, pD=pD, pA=pA,
        gamma_obs=gamma_obs, se_obs=se_obs,
    )


def build_setups(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    reg_table: pd.DataFrame,
    season_range: tuple[str, str] = ("2015-2016", "2021-2022"),
) -> list[LeagueSetup]:
    """
    Build LeagueSetup objects for all leagues, attaching observed gamma and SE
    from the real regression table.
    """
    long_hw = long_df[long_df["season"].between(*season_range)].copy()
    wide_hw = wide_df[wide_df["season"].between(*season_range)].copy()

    reg_idx = reg_table.set_index("league")
    setups = []
    for league in sorted(long_hw["league"].unique()):
        long_l = long_hw[long_hw["league"] == league]
        wide_l = wide_hw[wide_hw["league"] == league]
        setup = _build_setup(league, long_l, wide_l)

        # Attach observed regression stats
        if league in reg_idx.index:
            row = reg_idx.loc[league]
            setup = setup._replace(
                gamma_obs=float(row["gamma"]),
                se_obs=float(row["gamma_se"]),
            )
        setups.append(setup)

    return setups


# ---------------------------------------------------------------------------
# Fast null simulation
# ---------------------------------------------------------------------------

def _simulate_gamma(setup: LeagueSetup, rng: np.random.Generator) -> float:
    """
    Draw one null dataset and return γ̂.

    Outcomes are drawn match-by-match from Categorical(pH, pD, pA).
    The regression uses pre-computed M = (X'WX)^{-1}X'W.
    """
    # Draw outcomes: u < pH → H(0), pH ≤ u < pH+pD → D(1), else A(2)
    u = rng.uniform(size=setup.n_matches)
    new_result = np.where(u < setup.pH, 0,
                 np.where(u < setup.pH + setup.pD, 1, 2))  # (n_matches,): 0=H,1=D,2=A

    # Build observed indicator for each long-format row
    new_obs = (setup.outcome_idx == new_result[setup.match_idx]).astype(np.float64)

    # WLS residual and estimate
    y_sim = new_obs - setup.p_long
    beta_sim = setup.M @ y_sim   # (2,): [alpha_hat, gamma_hat]
    return float(beta_sim[1])


def run_simulation(
    setups: list[LeagueSetup],
    n_sim: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Run the null simulation for all leagues.

    Returns dict mapping league → (n_sim,) array of γ̂ values under the null.
    """
    rng = np.random.default_rng(seed)
    null_gammas: dict[str, list[float]] = {s.league: [] for s in setups}

    for sim_i in range(n_sim):
        for setup in setups:
            null_gammas[setup.league].append(_simulate_gamma(setup, rng))

        if (sim_i + 1) % 500 == 0:
            print(f"  Completed {sim_i+1}/{n_sim} simulations", flush=True)

    return {league: np.array(vals) for league, vals in null_gammas.items()}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def simulation_pvalue(gamma_obs: float, null_gammas: np.ndarray) -> float:
    """Two-sided simulation p-value: P(|γ_null| ≥ |γ_obs|)."""
    return float(np.mean(np.abs(null_gammas) >= abs(gamma_obs)))


def build_results_table(
    setups: list[LeagueSetup],
    null_gammas: dict[str, np.ndarray],
    reg_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare parametric and simulation p-values.

    Columns: league, gamma_obs, se_obs, t_obs, p_parametric, p_simulation,
             ratio (sim/param), null_gamma_mean, null_gamma_sd.
    """
    reg_idx = reg_table.set_index("league")
    rows = []
    for setup in setups:
        league = setup.league
        null = null_gammas[league]
        t_obs = setup.gamma_obs / setup.se_obs if setup.se_obs > 0 else np.nan
        p_param = float(reg_idx.loc[league, "p_value"]) if league in reg_idx.index else np.nan
        p_sim = simulation_pvalue(setup.gamma_obs, null)
        rows.append({
            "league":         league,
            "gamma_obs":      setup.gamma_obs,
            "se_obs":         setup.se_obs,
            "t_obs":          t_obs,
            "p_parametric":   p_param,
            "p_simulation":   p_sim,
            "ratio_sim_param": p_sim / p_param if (p_param > 0 and not np.isnan(p_param)) else np.nan,
            "null_gamma_mean": float(null.mean()),
            "null_gamma_sd":   float(null.std()),
            "n_sim":          len(null),
        })
    return pd.DataFrame(rows)


def joint_test(
    setups: list[LeagueSetup],
    null_gammas: dict[str, np.ndarray],
    method: str = "max_t",
) -> dict:
    """
    Joint test across all leagues.

    method='max_t': test statistic = max(|t_l|) across leagues.
    method='sum_chi2': test statistic = sum(t_l^2) across leagues.

    Returns dict with observed statistic, simulation p-value, and the
    fraction of simulations exceeding the observed value.
    """
    n_sim = len(next(iter(null_gammas.values())))

    # Observed test statistics
    t_obs = np.array([
        abs(s.gamma_obs / s.se_obs) if s.se_obs > 0 else 0.0
        for s in setups
    ])

    # Null test statistics (one value per simulation)
    t_null_matrix = np.column_stack([
        np.abs(null_gammas[s.league]) / s.se_obs for s in setups
    ])  # (n_sim, n_leagues)

    if method == "max_t":
        obs_stat = float(t_obs.max())
        null_stats = t_null_matrix.max(axis=1)
    elif method == "sum_chi2":
        obs_stat = float((t_obs**2).sum())
        null_stats = (t_null_matrix**2).sum(axis=1)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    p_joint = float(np.mean(null_stats >= obs_stat))
    return {
        "method":     method,
        "obs_stat":   obs_stat,
        "p_joint":    p_joint,
        "n_sim":      n_sim,
        "n_leagues":  len(setups),
    }
