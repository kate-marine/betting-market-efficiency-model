"""
Feature engineering for the soccer predictive model.

All features are computed in strict chronological order — each feature value
for match i uses only data from matches played before match i. No lookahead.

Features produced per match (from the home team's perspective):
  - elo_home, elo_away, elo_diff: Elo ratings before the match
  - home_form_{W,D,L}: wins/draws/losses in team's last 5 matches
  - away_form_{W,D,L}: same for away team
  - home_gf5, home_ga5: avg goals scored/conceded in last 5 matches
  - away_gf5, away_ga5: same for away team
  - home_rest_days, away_rest_days: days since last match (NaN if first match)
  - league_id: integer-encoded league (for league fixed effects)

Why Elo per league: teams compete within leagues. A Bundesliga Elo doesn't
mean the same thing as a Premier League Elo — they're separate rating pools.
Cross-league carryover (for promoted/relegated teams) is not modelled here.

Why form over the last 5 matches (not 10 or a full season): 5 matches
balances recency against sample size. Winkelmann et al. use similar short
windows in their simulation; H&W don't use a predictive model so there's no
direct precedent to follow.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELO_START = 1500.0      # initial rating for new teams
ELO_K = 20.0            # update step size
ELO_HOME_ADV = 100.0    # home advantage in Elo points (added to home team's rating)
FORM_WINDOW = 5         # number of recent matches for rolling form


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for team A vs team B (standard logistic formula)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating: float, actual: float, expected: float) -> float:
    return rating + ELO_K * (actual - expected)


def _elo_scores(result: str) -> tuple[float, float]:
    """Map FTR result to (home_score, away_score) for Elo update."""
    if result == "H":
        return 1.0, 0.0
    elif result == "D":
        return 0.5, 0.5
    else:
        return 0.0, 1.0


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Augment wide-format match data with pre-match features.

    Processes matches in chronological order within each league. The output
    has the same row order as the input but with new feature columns appended.
    Features that can't be computed (e.g., form before a team's first match)
    are left as NaN — LightGBM handles missing values natively.

    Parameters
    ----------
    wide : wide-format DataFrame from load_soccer(); must contain at minimum
        league, date, home_team, away_team, result, goals_H, goals_A columns.

    Returns
    -------
    Same DataFrame with feature columns added (does not mutate input).
    """
    df = wide.copy()
    df = df.sort_values(["league", "date", "match_id"]).reset_index(drop=True)

    # Encode leagues as integers for use as a categorical feature
    league_codes = {lg: i for i, lg in enumerate(sorted(df["league"].unique()))}
    df["league_id"] = df["league"].map(league_codes)

    # Pre-allocate feature arrays (NaN = not yet available)
    n = len(df)
    feat_names = [
        "elo_home", "elo_away", "elo_diff",
        "home_form_W", "home_form_D", "home_form_L",
        "away_form_W", "away_form_D", "away_form_L",
        "home_gf5", "home_ga5",
        "away_gf5", "away_ga5",
        "home_rest_days", "away_rest_days",
    ]
    feats = {name: np.full(n, np.nan) for name in feat_names}

    # State tracked per league
    # elo[league][team] = current Elo rating
    elo: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: ELO_START))
    # form[team] = deque of last FORM_WINDOW (goals_for, goals_against, result_code)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    # last_match_date[team] = date of last match played
    last_date: dict[str, pd.Timestamp] = {}

    for idx, row in df.iterrows():
        league = row["league"]
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        result = row["result"]
        gh = row["goals_H"]  # may be NaN for synthetic data without goals
        ga = row["goals_A"]

        # ----------------------------------------------------------------
        # READ features (before this match)
        # ----------------------------------------------------------------
        r_home = elo[league][home]
        r_away = elo[league][away]

        feats["elo_home"][idx] = r_home
        feats["elo_away"][idx] = r_away
        feats["elo_diff"][idx] = r_home - r_away

        # Form for home team
        hf = list(form[home])
        if hf:
            feats["home_form_W"][idx] = sum(1 for g in hf if g[2] == "W")
            feats["home_form_D"][idx] = sum(1 for g in hf if g[2] == "D")
            feats["home_form_L"][idx] = sum(1 for g in hf if g[2] == "L")
            valid_goals = [(g[0], g[1]) for g in hf if not (np.isnan(g[0]) or np.isnan(g[1]))]
            if valid_goals:
                feats["home_gf5"][idx] = np.mean([g[0] for g in valid_goals])
                feats["home_ga5"][idx] = np.mean([g[1] for g in valid_goals])

        # Form for away team
        af = list(form[away])
        if af:
            feats["away_form_W"][idx] = sum(1 for g in af if g[2] == "W")
            feats["away_form_D"][idx] = sum(1 for g in af if g[2] == "D")
            feats["away_form_L"][idx] = sum(1 for g in af if g[2] == "L")
            valid_goals = [(g[0], g[1]) for g in af if not (np.isnan(g[0]) or np.isnan(g[1]))]
            if valid_goals:
                feats["away_gf5"][idx] = np.mean([g[0] for g in valid_goals])
                feats["away_ga5"][idx] = np.mean([g[1] for g in valid_goals])

        # Rest days
        if home in last_date and pd.notna(date):
            feats["home_rest_days"][idx] = (date - last_date[home]).days
        if away in last_date and pd.notna(date):
            feats["away_rest_days"][idx] = (date - last_date[away]).days

        # ----------------------------------------------------------------
        # UPDATE state (after this match)
        # ----------------------------------------------------------------
        if pd.notna(result) and result in ("H", "D", "A"):
            s_home, s_away = _elo_scores(result)
            e_home = _elo_expected(r_home + ELO_HOME_ADV, r_away)
            e_away = 1.0 - e_home
            elo[league][home] = _elo_update(r_home, s_home, e_home)
            elo[league][away] = _elo_update(r_away, s_away, e_away)

            # Home team's perspective in the form deque
            home_result = "W" if result == "H" else ("D" if result == "D" else "L")
            away_result = "W" if result == "A" else ("D" if result == "D" else "L")
            form[home].append((gh if pd.notna(gh) else np.nan,
                               ga if pd.notna(ga) else np.nan,
                               home_result))
            form[away].append((ga if pd.notna(ga) else np.nan,
                               gh if pd.notna(gh) else np.nan,
                               away_result))

        if pd.notna(date):
            last_date[home] = date
            last_date[away] = date

    for name, arr in feats.items():
        df[name] = arr

    return df


# ---------------------------------------------------------------------------
# Feature column list (for model training)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_W", "home_form_D", "home_form_L",
    "away_form_W", "away_form_D", "away_form_L",
    "home_gf5", "home_ga5",
    "away_gf5", "away_ga5",
    "home_rest_days", "away_rest_days",
    "league_id",
]

RESULT_MAP = {"H": 0, "D": 1, "A": 2}
RESULT_MAP_INV = {0: "H", 1: "D", 2: "A"}
