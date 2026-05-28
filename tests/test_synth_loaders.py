"""
Tests for synthetic data generator and loaders.

Each test validates against synthetic data with known ground truth —
following the principle that "if it doesn't pass on data where I know the
true answer, the result isn't trustworthy yet."
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.synth import generate_soccer, generate_tennis
from src.loaders import load_soccer, load_tennis


# ---------------------------------------------------------------------------
# Synthetic data generator tests
# ---------------------------------------------------------------------------

class TestGenerateSoccer:
    def test_creates_expected_files(self, tmp_path):
        paths = generate_soccer(
            tmp_path / "raw", leagues=["E0"], seasons=["2019-2020"], seed=0
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_csv_has_required_columns(self, tmp_path):
        paths = generate_soccer(
            tmp_path / "raw", leagues=["E0"], seasons=["2019-2020"], seed=0
        )
        df = pd.read_csv(paths[0])
        for col in ("Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"):
            assert col in df.columns, f"Missing column: {col}"

    def test_result_values_are_valid(self, tmp_path):
        paths = generate_soccer(
            tmp_path / "raw", leagues=["E0"], seasons=["2019-2020"], seed=0
        )
        df = pd.read_csv(paths[0])
        assert set(df["FTR"].unique()).issubset({"H", "D", "A"})

    def test_efficient_variant_outcome_frequency(self, tmp_path):
        """
        In an efficient market, favorites (lowest odds) should win more often.
        With 380 matches, we expect the lowest-odds team to win ~40–60% —
        not a strict bound, but a sanity check on the simulation direction.
        """
        paths = generate_soccer(
            tmp_path / "raw", leagues=["E0"], seasons=["2019-2020"],
            n_matches=2000, variant="efficient", seed=42
        )
        df = pd.read_csv(paths[0])
        fav_is_home = df["AvgH"] < df[["AvgA", "AvgD"]].min(axis=1)
        home_wins = df["FTR"] == "H"
        fav_win_rate = (fav_is_home & home_wins).sum() / fav_is_home.sum()
        assert 0.30 < fav_win_rate < 0.75, f"Unexpected favorite win rate: {fav_win_rate:.3f}"

    def test_flb_variant_increases_favorite_probability(self, tmp_path):
        """
        FLB variant: the market-implied probability of the favorite should be
        *lower* than the true probability — that's the bias. Since we have access
        to _true_* columns in synthetic data, we can verify this directly.
        """
        paths = generate_soccer(
            tmp_path / "raw", leagues=["E0"], seasons=["2019-2020"],
            n_matches=2000, variant="flb", flb_gamma=0.10, seed=7
        )
        df = pd.read_csv(paths[0])
        from src.devig import normalized
        odds = df[["AvgH", "AvgD", "AvgA"]].values
        market_p = normalized(odds)

        # Identify per-row favorite (col with highest true prob)
        true_p = df[["_true_pH", "_true_pD", "_true_pA"]].values
        fav_idx = true_p.argmax(axis=1)
        row_idx = np.arange(len(df))
        true_fav = true_p[row_idx, fav_idx]
        market_fav = market_p[row_idx, fav_idx]

        # In FLB markets, favorites are underpriced: true > market on average
        mean_gap = (true_fav - market_fav).mean()
        assert mean_gap > 0, f"Expected true_fav > market_fav on avg, got gap={mean_gap:.4f}"

    def test_multiple_leagues_and_seasons(self, tmp_path):
        paths = generate_soccer(
            tmp_path / "raw",
            leagues=["E0", "D1"],
            seasons=["2019-2020", "2020-2021"],
            seed=0,
        )
        assert len(paths) == 4
        assert all(p.exists() for p in paths)


class TestGenerateTennis:
    def test_creates_expected_files(self, tmp_path):
        paths = generate_tennis(
            tmp_path / "raw", tours=["atp"], years=[2020], seed=0
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_xlsx_has_required_columns(self, tmp_path):
        paths = generate_tennis(
            tmp_path / "raw", tours=["atp"], years=[2020], seed=0
        )
        df = pd.read_excel(paths[0])
        for col in ("Date", "Winner", "Loser", "AvgW", "AvgL"):
            assert col in df.columns, f"Missing column: {col}"

    def test_reproducibility(self, tmp_path):
        raw1 = tmp_path / "raw1"
        raw2 = tmp_path / "raw2"
        [p1] = generate_soccer(raw1, leagues=["E0"], seasons=["2019-2020"], seed=0)
        [p2] = generate_soccer(raw2, leagues=["E0"], seasons=["2019-2020"], seed=0)
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def soccer_data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("soccer_raw")
    proc = tmp_path_factory.mktemp("soccer_proc")
    generate_soccer(raw, leagues=["E0", "D1"], seasons=["2019-2020", "2020-2021"], seed=0)
    wide, long = load_soccer(raw, proc)
    return wide, long, proc


@pytest.fixture(scope="module")
def tennis_data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("tennis_raw")
    proc = tmp_path_factory.mktemp("tennis_proc")
    generate_tennis(raw, tours=["atp"], years=[2020, 2021], seed=0)
    wide, long = load_tennis(raw, proc)
    return wide, long, proc


class TestSoccerLoader:
    def test_wide_shape(self, soccer_data):
        wide, long, _ = soccer_data
        # 2 leagues × 2 seasons × 380 matches
        assert len(wide) == 2 * 2 * 380

    def test_long_shape(self, soccer_data):
        wide, long, _ = soccer_data
        # 3 outcomes per match
        assert len(long) == len(wide) * 3

    def test_wide_required_columns(self, soccer_data):
        wide, _, _ = soccer_data
        for col in ("match_id", "league", "season", "date", "home_team", "away_team",
                    "result", "norm_pH", "norm_pD", "norm_pA"):
            assert col in wide.columns, f"Missing: {col}"

    def test_long_required_columns(self, soccer_data):
        _, long, _ = soccer_data
        for col in ("match_id", "league", "season", "outcome", "observed", "norm_p", "odds"):
            assert col in long.columns, f"Missing: {col}"

    def test_long_observed_sums_to_one_per_match(self, soccer_data):
        _, long, _ = soccer_data
        sums = long.groupby("match_id")["observed"].sum()
        assert (sums == 1).all(), "Each match must have exactly one observed outcome"

    def test_norm_probs_sum_to_one_wide(self, soccer_data):
        wide, _, _ = soccer_data
        row_sums = wide[["norm_pH", "norm_pD", "norm_pA"]].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-5

    def test_parquet_files_written(self, soccer_data):
        _, _, proc = soccer_data
        assert (proc / "soccer_wide.parquet").exists()
        assert (proc / "soccer_long.parquet").exists()

    def test_parquet_roundtrip(self, soccer_data):
        wide, long, proc = soccer_data
        wide2 = pd.read_parquet(proc / "soccer_wide.parquet")
        assert len(wide2) == len(wide)

    def test_season_recovery_from_date(self, tmp_path):
        """Loader must recover season from Date when it's not in the filename."""
        raw = tmp_path / "raw"
        raw.mkdir()
        # Generate data, then rename file to remove season info
        generate_soccer(raw, leagues=["E0"], seasons=["2019-2020"], seed=0)
        src = raw / "E0" / "2019-2020.csv"
        dst = raw / "E0" / "matches.csv"  # no season in name
        src.rename(dst)
        wide, _ = load_soccer(raw, tmp_path / "proc")
        assert (wide["season"] == "2019-2020").all()


class TestTennisLoader:
    def test_wide_shape(self, tennis_data):
        wide, long, _ = tennis_data
        # 1 tour × 2 years × 500 matches
        assert len(wide) == 1 * 2 * 500

    def test_long_shape(self, tennis_data):
        wide, long, _ = tennis_data
        # 2 sides per match
        assert len(long) == len(wide) * 2

    def test_long_observed_sum_per_match(self, tennis_data):
        _, long, _ = tennis_data
        sums = long.groupby("match_id")["observed"].sum()
        assert (sums == 1).all(), "Each tennis match must have exactly one winner"

    def test_norm_probs_sum_to_one_wide(self, tennis_data):
        wide, _, _ = tennis_data
        row_sums = wide[["norm_pW", "norm_pL"]].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-5

    def test_parquet_files_written(self, tennis_data):
        _, _, proc = tennis_data
        assert (proc / "tennis_wide.parquet").exists()
        assert (proc / "tennis_long.parquet").exists()

    def test_winner_has_lower_odds(self, tennis_data):
        """In the loader's long format, winner side (observed=1) should on average
        have lower odds than loser side — consistent with favorites winning more."""
        _, long, _ = tennis_data
        mean_odds_winner = long[long["observed"] == 1]["odds"].mean()
        mean_odds_loser  = long[long["observed"] == 0]["odds"].mean()
        assert mean_odds_winner < mean_odds_loser, (
            f"Expected winners to have lower avg odds: {mean_odds_winner:.2f} vs {mean_odds_loser:.2f}"
        )
