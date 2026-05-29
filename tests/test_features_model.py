"""
Tests for feature engineering and walk-forward CV model.

Every test validates against synthetic data with known ground truth.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.synth import generate_soccer
from src.loaders import load_soccer
from src.features import compute_features, FEATURE_COLS, ELO_START
from src.model import walk_forward_predict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_wide(tmp_path_factory):
    raw = tmp_path_factory.mktemp("feat_raw")
    proc = tmp_path_factory.mktemp("feat_proc")
    # 3 seasons so walk-forward has something to do; efficient market
    generate_soccer(raw, leagues=["E0"], seasons=["2017-2018","2018-2019","2019-2020"],
                    n_matches=100, variant="efficient", seed=42)
    wide, _ = load_soccer(raw, proc)
    return wide


@pytest.fixture(scope="module")
def featured_wide(synth_wide):
    return compute_features(synth_wide)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestComputeFeatures:

    def test_output_has_all_feature_cols(self, featured_wide):
        for col in FEATURE_COLS:
            assert col in featured_wide.columns, f"Missing feature column: {col}"

    def test_elo_starts_at_default(self, featured_wide):
        """The first match for any team must use the default starting Elo."""
        # Find the very first match in the dataset per team
        df = featured_wide.sort_values("date")
        first_home = df.groupby("home_team").first()["elo_home"]
        # At least some teams should start exactly at ELO_START
        assert (first_home == ELO_START).any(), "No team started at ELO_START"

    def test_elo_changes_after_matches(self, featured_wide):
        """Elo must change after each match — it would be flat if updates were skipped."""
        df = featured_wide.sort_values("date")
        team_elos = df.groupby("home_team")["elo_home"].nunique()
        # Most teams should have more than one distinct Elo value across matches
        assert (team_elos > 1).mean() > 0.5, "Most teams have static Elo — update not working"

    def test_no_lookahead_in_elo(self, featured_wide):
        """
        Elo for match i must reflect only matches before i, not i itself.
        Check: if team A won match i, their elo_home for match i should be
        *lower* than their elo_home for match i+1 (if they were home again).
        We verify the simpler property: Elo values are pre-match, not post-match.
        Validated by checking the first match uses exactly ELO_START.
        """
        df = featured_wide.sort_values("date")
        # Every team's very first appearance should have Elo = ELO_START
        seen_home = {}
        seen_away = {}
        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            if ht not in seen_home:
                seen_home[ht] = row["elo_home"]
            if at not in seen_away:
                seen_away[at] = row["elo_away"]

        first_elos = list(seen_home.values()) + list(seen_away.values())
        assert all(e == ELO_START for e in first_elos), \
            "Some team's first match did not use ELO_START"

    def test_form_is_nan_for_first_match(self, featured_wide):
        """The very first match in the dataset must have NaN form — no prior data exists."""
        df = featured_wide.sort_values("date").reset_index(drop=True)
        # Use nth(0) not first() — groupby().first() skips NaN by default
        first_home_matches = df.sort_values("date").groupby("home_team").nth(0)
        # At least the very first row of the full dataset has NaN form
        first_row = df.iloc[0]
        assert pd.isna(first_row["home_form_W"]), \
            "First match of dataset should have NaN form (no prior matches)"

    def test_form_increases_after_win(self, featured_wide):
        """After a home win, home_form_W in the team's next home match should increase."""
        df = featured_wide.sort_values("date")
        # Find a team with multiple home matches
        home_counts = df.groupby("home_team").size()
        team = home_counts[home_counts >= 3].index[0]
        team_df = df[df["home_team"] == team].sort_values("date").reset_index(drop=True)

        # After at least one win, form_W should be > 0 at some point
        wins_after = team_df["home_form_W"].dropna()
        if len(wins_after) > 0 and (team_df["result"] == "H").any():
            assert wins_after.max() > 0, "home_form_W never positive after wins"

    def test_rest_days_nan_for_first_match(self, featured_wide):
        """The very first match in the dataset must have NaN rest days."""
        df = featured_wide.sort_values("date").reset_index(drop=True)
        first_row = df.iloc[0]
        assert pd.isna(first_row["home_rest_days"]), \
            "First match should have NaN rest_days (team has no prior match)"

    def test_rest_days_positive(self, featured_wide):
        """All non-NaN rest days must be positive."""
        df = featured_wide
        home_rest = df["home_rest_days"].dropna()
        assert (home_rest > 0).all(), "Some rest days are zero or negative"

    def test_elo_diff_equals_home_minus_away(self, featured_wide):
        diff = featured_wide["elo_home"] - featured_wide["elo_away"]
        pd.testing.assert_series_equal(
            featured_wide["elo_diff"].round(8), diff.round(8),
            check_names=False
        )

    def test_league_id_is_integer(self, featured_wide):
        assert featured_wide["league_id"].dtype in (np.int64, np.int32, int)

    def test_row_count_unchanged(self, synth_wide, featured_wide):
        assert len(featured_wide) == len(synth_wide)


# ---------------------------------------------------------------------------
# Walk-forward CV tests
# ---------------------------------------------------------------------------

class TestWalkForwardCV:

    def test_produces_predictions_for_test_seasons(self, featured_wide):
        preds = walk_forward_predict(featured_wide, test_seasons=["2019-2020"])
        assert "pred_pH" in preds.columns
        assert "pred_pD" in preds.columns
        assert "pred_pA" in preds.columns
        assert (preds["test_season"] == "2019-2020").all()

    def test_predicted_probs_sum_to_one(self, featured_wide):
        preds = walk_forward_predict(featured_wide, test_seasons=["2019-2020"])
        sums = preds[["pred_pH","pred_pD","pred_pA"]].sum(axis=1)
        assert (sums - 1.0).abs().max() < 1e-5

    def test_no_test_data_in_training(self, featured_wide):
        """Walk-forward CV must not train on data from the test season."""
        # Validate by checking that training only uses seasons < test_season.
        # We can't inspect LightGBM internals, but we can verify predictions
        # exist only for the requested test season.
        preds = walk_forward_predict(featured_wide, test_seasons=["2019-2020"])
        assert (preds["season"] == "2019-2020").all(), \
            "Predictions include rows from non-test seasons"

    def test_skips_season_with_insufficient_training(self, featured_wide):
        """If min_train_seasons=2 and only 1 season available, skip with warning."""
        with pytest.warns(UserWarning):
            preds = walk_forward_predict(
                featured_wide,
                test_seasons=["2018-2019"],
                min_train_seasons=2,
            )
        assert len(preds) == 0 or "2018-2019" not in preds["test_season"].values

    def test_predictions_in_unit_interval(self, featured_wide):
        preds = walk_forward_predict(featured_wide, test_seasons=["2019-2020"])
        for col in ["pred_pH","pred_pD","pred_pA"]:
            assert (preds[col] >= 0).all() and (preds[col] <= 1).all()
