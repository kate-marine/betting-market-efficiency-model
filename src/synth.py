"""
Synthetic data generator for testing and validation.

Produces files in the exact on-disk format of football-data.co.uk (CSV) and
tennis-data.co.uk (XLSX) with *known ground truth*. Two market variants:

  - "efficient": outcomes drawn from the true probabilities implied by the odds
    (no favorite-longshot bias). The H&W regression should recover gamma ≈ 0.
  - "flb": systematic FLB — favorites are underpriced (true prob > implied prob
    for favorites). Simulates Winkelmann et al.'s null and H&W's empirical finding.
    gamma should be positive.

Simulation design follows Winkelmann et al. (2024) §3 and H&W (2024) §5.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Odds simulation helpers
# ---------------------------------------------------------------------------

def _draw_raw_probs(rng: np.random.Generator, n: int, k: int) -> NDArray:
    """
    Draw true outcome probabilities for n matches with k outcomes each.
    Uses Dirichlet(alpha=1) so all outcomes are plausible.
    """
    return rng.dirichlet(np.ones(k), size=n)


def _probs_to_odds(probs: NDArray, overround: float = 0.05) -> NDArray:
    """
    Convert true probabilities to bookmaker odds with a given overround.
    Overround is spread proportionally (normalized devigging undoes this exactly).
    """
    raw = probs / (1.0 + overround)      # scale so sum(1/odds) = 1 + overround
    return 1.0 / raw


def _apply_flb(true_probs: NDArray, gamma: float = 0.08) -> NDArray:
    """
    Apply favorite-longshot bias to true probabilities.

    In FLB markets, favorites are *underbet* (their true probability exceeds the
    market-implied probability). We model this by shifting probability mass from
    longshots toward favorites proportional to (p_i - 1/k), where 1/k is the
    uniform baseline.

    gamma > 0 corresponds to H&W's finding of positive gamma (favorites underpriced).
    """
    k = true_probs.shape[1]
    uniform = 1.0 / k
    # Shift: p_i_biased = p_i + gamma * (p_i - uniform)
    biased = true_probs + gamma * (true_probs - uniform)
    # Renormalize (small numeric correction)
    biased = np.clip(biased, 1e-4, None)
    return biased / biased.sum(axis=1, keepdims=True)


def _draw_outcomes(rng: np.random.Generator, probs: NDArray) -> NDArray:
    """
    Draw one outcome per row by sampling from the true probability distribution.
    Returns integer array of shape (n,) with values in {0, 1, ..., k-1}.
    """
    n = probs.shape[0]
    cum = probs.cumsum(axis=1)
    u = rng.uniform(size=(n, 1))
    return (u > cum).sum(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Soccer (football-data.co.uk format)
# ---------------------------------------------------------------------------

SOCCER_LEAGUES = {
    "E0": "England Premier League",
    "D1": "Germany Bundesliga",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
}

_SOCCER_RESULT_MAP = {0: "H", 1: "D", 2: "A"}


def _make_soccer_df(
    rng: np.random.Generator,
    n_matches: int,
    season: str,
    league: str,
    variant: str,
    overround: float,
    flb_gamma: float,
) -> pd.DataFrame:
    """Build one season of soccer data in football-data.co.uk format."""
    true_probs = _draw_raw_probs(rng, n_matches, k=3)

    if variant == "flb":
        market_probs = _apply_flb(true_probs, gamma=flb_gamma)
    else:
        market_probs = true_probs.copy()

    avg_odds = _probs_to_odds(market_probs, overround=overround)
    outcomes = _draw_outcomes(rng, true_probs)

    start_year = int(season[:4])
    dates = pd.date_range(
        start=f"{start_year}-08-01", periods=n_matches, freq="3D"
    )
    teams_home = [f"Home_{i % 20:02d}" for i in range(n_matches)]
    teams_away = [f"Away_{(i + 10) % 20:02d}" for i in range(n_matches)]

    df = pd.DataFrame({
        "Div": league,
        "Date": dates.strftime("%d/%m/%Y"),
        "HomeTeam": teams_home,
        "AwayTeam": teams_away,
        "FTR": [_SOCCER_RESULT_MAP[o] for o in outcomes],
        "AvgH": avg_odds[:, 0].round(2),
        "AvgD": avg_odds[:, 1].round(2),
        "AvgA": avg_odds[:, 2].round(2),
        "B365H": (avg_odds[:, 0] * rng.uniform(0.97, 1.03, n_matches)).round(2),
        "B365D": (avg_odds[:, 1] * rng.uniform(0.97, 1.03, n_matches)).round(2),
        "B365A": (avg_odds[:, 2] * rng.uniform(0.97, 1.03, n_matches)).round(2),
        # Ground-truth columns (not in real data; used only for test validation)
        "_true_pH": true_probs[:, 0].round(6),
        "_true_pD": true_probs[:, 1].round(6),
        "_true_pA": true_probs[:, 2].round(6),
    })
    return df


def generate_soccer(
    output_dir: str | pathlib.Path,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    n_matches: int = 380,
    variant: str = "efficient",
    overround: float = 0.05,
    flb_gamma: float = 0.08,
    seed: int = 0,
) -> list[pathlib.Path]:
    """
    Generate synthetic soccer data in football-data.co.uk format.

    Files are written as <output_dir>/<league>/<season>.csv, mirroring
    one common real-data layout. The loader is built to handle this.

    Returns list of written file paths.
    """
    if variant not in ("efficient", "flb"):
        raise ValueError(f"variant must be 'efficient' or 'flb', got {variant!r}")

    rng = np.random.default_rng(seed)
    output_dir = pathlib.Path(output_dir)

    if leagues is None:
        leagues = list(SOCCER_LEAGUES.keys())
    if seasons is None:
        seasons = ["2018-2019", "2019-2020", "2020-2021"]

    written = []
    for league in leagues:
        league_dir = output_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        for season in seasons:
            df = _make_soccer_df(
                rng, n_matches, season, league, variant, overround, flb_gamma
            )
            path = league_dir / f"{season}.csv"
            df.to_csv(path, index=False)
            written.append(path)

    return written


# ---------------------------------------------------------------------------
# Tennis (tennis-data.co.uk format)
# ---------------------------------------------------------------------------

_SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
_TOURS = ["atp", "wta"]


def _make_tennis_df(
    rng: np.random.Generator,
    n_matches: int,
    year: int,
    tour: str,
    variant: str,
    overround: float,
    flb_gamma: float,
) -> pd.DataFrame:
    """Build one year of tennis data in tennis-data.co.uk format."""
    true_probs_winner = _draw_raw_probs(rng, n_matches, k=2)[:, [0]]
    true_probs = np.concatenate([true_probs_winner, 1.0 - true_probs_winner], axis=1)

    if variant == "flb":
        market_probs = _apply_flb(true_probs, gamma=flb_gamma)
    else:
        market_probs = true_probs.copy()

    avg_odds = _probs_to_odds(market_probs, overround=overround)

    dates = pd.date_range(start=f"{year}-01-01", periods=n_matches, freq="2D")
    surfaces = rng.choice(_SURFACES, size=n_matches)
    players_w = [f"Player_W{i % 100:03d}" for i in range(n_matches)]
    players_l = [f"Player_L{(i + 50) % 100:03d}" for i in range(n_matches)]

    df = pd.DataFrame({
        "Date": dates.strftime("%d/%m/%Y"),
        "Tournament": [f"Tournament_{i % 30}" for i in range(n_matches)],
        "Round": rng.choice(["R128", "R64", "R32", "R16", "QF", "SF", "F"],
                            size=n_matches),
        "Surface": surfaces,
        "Winner": players_w,
        "Loser": players_l,
        "AvgW": avg_odds[:, 0].round(2),
        "AvgL": avg_odds[:, 1].round(2),
        "B365W": (avg_odds[:, 0] * rng.uniform(0.97, 1.03, n_matches)).round(2),
        "B365L": (avg_odds[:, 1] * rng.uniform(0.97, 1.03, n_matches)).round(2),
        "_true_pW": true_probs[:, 0].round(6),
        "_true_pL": true_probs[:, 1].round(6),
    })
    return df


def generate_tennis(
    output_dir: str | pathlib.Path,
    tours: list[str] | None = None,
    years: list[int] | None = None,
    n_matches: int = 500,
    variant: str = "efficient",
    overround: float = 0.05,
    flb_gamma: float = 0.08,
    seed: int = 0,
) -> list[pathlib.Path]:
    """
    Generate synthetic tennis data in tennis-data.co.uk format.

    Files are written as <output_dir>/<tour>/<year>.xlsx. Returns list of paths.
    """
    if variant not in ("efficient", "flb"):
        raise ValueError(f"variant must be 'efficient' or 'flb', got {variant!r}")

    rng = np.random.default_rng(seed)
    output_dir = pathlib.Path(output_dir)

    if tours is None:
        tours = _TOURS
    if years is None:
        years = [2019, 2020, 2021]

    written = []
    for tour in tours:
        tour_dir = output_dir / tour
        tour_dir.mkdir(parents=True, exist_ok=True)
        for year in years:
            df = _make_tennis_df(rng, n_matches, year, tour, variant, overround, flb_gamma)
            path = tour_dir / f"{year}.xlsx"
            df.to_excel(path, index=False)
            written.append(path)

    return written
