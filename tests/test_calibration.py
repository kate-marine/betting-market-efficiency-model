"""
Tests for calibration metrics.

Key principle: every test validates against data with known ground truth
so we can verify the metric does the right thing, not just that it runs.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.calibration import (
    brier_score, ece, reliability_diagram_data,
    log_loss_multiclass, bootstrap_metrics, to_long,
)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def test_brier_score_perfect():
    """Perfect predictions → Brier score = 0."""
    pred = np.array([1.0, 0.0, 1.0, 0.0])
    obs  = np.array([1,   0,   1,   0])
    assert brier_score(pred, obs) == pytest.approx(0.0)


def test_brier_score_worst():
    """Assigning probability 1 to the wrong outcome → score = 1 per prediction."""
    pred = np.array([1.0, 1.0])
    obs  = np.array([0,   0])
    assert brier_score(pred, obs) == pytest.approx(1.0)


def test_brier_score_uniform():
    """Uniform 1/3 predictor on 3-class → Brier per cell = (1/3)² + (1/3)² ≈ 0.222."""
    # Two matches: H, A. Predict 1/3 for each.
    # Match 1 (H=1, D=0, A=0): scores = (1-1/3)², (0-1/3)², (0-1/3)² = 4/9, 1/9, 1/9
    # Mean across all (match, outcome) = (4/9 + 1/9 + 1/9) / 3 = 6/27 = 2/9 ≈ 0.222
    pred = np.full(6, 1.0 / 3.0)
    obs  = np.array([1, 0, 0, 0, 0, 1])   # H won match 1, A won match 2
    expected = ((1 - 1/3)**2 + (0 - 1/3)**2 * 2) / 3
    assert brier_score(pred, obs) == pytest.approx(expected, abs=1e-6)


def test_brier_score_ordering():
    """Better calibrated model should have lower Brier score."""
    obs = np.array([1, 0, 1, 0, 1])
    good_pred = np.array([0.8, 0.2, 0.9, 0.1, 0.7])
    bad_pred  = np.array([0.2, 0.8, 0.1, 0.9, 0.3])
    assert brier_score(good_pred, obs) < brier_score(bad_pred, obs)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    """
    If predicted probs match observed frequencies exactly within each bin,
    ECE should be 0.
    """
    # 100 observations: predicted probs all 0.5, 50 observed as 1 → ECE = 0
    pred = np.full(100, 0.5)
    obs  = np.array([1]*50 + [0]*50)
    assert ece(pred, obs) == pytest.approx(0.0, abs=1e-10)


def test_ece_overconfident():
    """
    Always predicting 0.9 but only 50% are correct → large ECE.
    ECE ≈ |0.5 - 0.9| = 0.4 (all in one bin).
    """
    pred = np.full(200, 0.9)
    obs  = np.array([1]*100 + [0]*100)
    e = ece(pred, obs)
    assert e == pytest.approx(0.4, abs=0.01)


def test_ece_nonnegative():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 500)
    obs  = rng.integers(0, 2, 500)
    assert ece(pred, obs) >= 0


def test_ece_decreases_with_better_calibration():
    """
    A clearly miscalibrated predictor should have higher ECE than a calibrated one.
    Use a scenario where the miscalibration is unambiguous:
      - good: predict 0.7, true frequency is 0.7 → ECE ≈ 0
      - bad:  predict 0.3 (inverted), true frequency is 0.7 → ECE ≈ 0.4
    """
    n = 500
    obs = np.array([1]*350 + [0]*150)   # 70% observed = 1
    good_pred = np.full(n, 0.7)         # matches frequency exactly
    bad_pred  = np.full(n, 0.3)         # inverted — clearly wrong
    assert ece(good_pred, obs) < ece(bad_pred, obs)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def test_reliability_diagram_data_shape():
    pred = np.linspace(0.05, 0.95, 100)
    obs  = (pred > 0.5).astype(int)
    df = reliability_diagram_data(pred, obs, n_bins=10)
    assert "bin_mid" in df.columns
    assert "mean_pred" in df.columns
    assert "obs_freq" in df.columns
    assert len(df) <= 10   # at most one row per bin


def test_reliability_diagram_counts_sum_to_total():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 300)
    obs  = rng.integers(0, 2, 300)
    df = reliability_diagram_data(pred, obs, n_bins=10)
    assert df["count"].sum() == 300


# ---------------------------------------------------------------------------
# Log-loss
# ---------------------------------------------------------------------------

def test_log_loss_perfect():
    """Assigning probability 1 to the true class → log-loss = 0."""
    preds_df = pd.DataFrame({
        "result":   ["H", "D", "A"],
        "pred_pH":  [1.0, 0.0, 0.0],
        "pred_pD":  [0.0, 1.0, 0.0],
        "pred_pA":  [0.0, 0.0, 1.0],
    })
    from src.calibration import OUTCOME_COLS_MODEL
    ll = log_loss_multiclass(preds_df.reset_index(), OUTCOME_COLS_MODEL, eps=1e-15)
    # log(1) = 0, so log-loss = 0
    assert ll == pytest.approx(0.0, abs=0.01)


def test_log_loss_uniform():
    """Uniform predictor (1/3 each) → log-loss = log(3) ≈ 1.099."""
    preds_df = pd.DataFrame({
        "result":   ["H", "D", "A"] * 100,
        "pred_pH":  [1/3] * 300,
        "pred_pD":  [1/3] * 300,
        "pred_pA":  [1/3] * 300,
    })
    from src.calibration import OUTCOME_COLS_MODEL
    ll = log_loss_multiclass(preds_df.reset_index(), OUTCOME_COLS_MODEL)
    assert ll == pytest.approx(np.log(3), abs=0.01)


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_point_estimate(tmp_path):
    """Bootstrap 95% CI should contain the point estimate."""
    rng = np.random.default_rng(5)
    n = 200
    match_ids = np.arange(n)
    pred_p = rng.uniform(0.2, 0.8, n * 3)
    observed = rng.integers(0, 2, n * 3)
    long_df = pd.DataFrame({
        "match_id": np.repeat(match_ids, 3),
        "pred_p": pred_p,
        "observed": observed,
    })
    result = bootstrap_metrics(long_df, pred_col="pred_p", n_boot=200, seed=0)
    assert result["brier_lo"] <= result["brier"] <= result["brier_hi"]
    assert result["ece_lo"]   <= result["ece"]   <= result["ece_hi"]


def test_bootstrap_ci_width_positive():
    """CIs should have positive width — point estimate is not at the boundary."""
    rng = np.random.default_rng(6)
    n = 150
    long_df = pd.DataFrame({
        "match_id": np.repeat(np.arange(n), 3),
        "pred_p": rng.uniform(0.2, 0.8, n * 3),
        "observed": rng.integers(0, 2, n * 3),
    })
    result = bootstrap_metrics(long_df, n_boot=100, seed=0)
    assert result["brier_hi"] > result["brier_lo"]
    assert result["ece_hi"]   > result["ece_lo"]


# ---------------------------------------------------------------------------
# to_long integration
# ---------------------------------------------------------------------------

def test_to_long_row_count(tmp_path_factory):
    """to_long should produce exactly 3 rows per match."""
    import pandas as pd
    preds = pd.DataFrame({
        "match_id": [1, 2, 3],
        "league": ["E0"] * 3,
        "test_season": ["2021-2022"] * 3,
        "result": ["H", "D", "A"],
        "pred_pH": [0.6, 0.3, 0.2],
        "pred_pD": [0.2, 0.5, 0.3],
        "pred_pA": [0.2, 0.2, 0.5],
        "norm_pH": [0.5, 0.4, 0.3],
        "norm_pD": [0.3, 0.3, 0.3],
        "norm_pA": [0.2, 0.3, 0.4],
    })
    long = to_long(preds)
    assert len(long) == 9   # 3 matches × 3 outcomes
    assert set(long["outcome"].unique()) == {"H", "D", "A"}


def test_to_long_observed_sums_to_one():
    """Each match must have exactly one observed outcome."""
    import pandas as pd
    preds = pd.DataFrame({
        "match_id": [10, 20],
        "league": ["D1", "D1"],
        "test_season": ["2020-2021", "2020-2021"],
        "result": ["H", "A"],
        "pred_pH": [0.6, 0.2], "pred_pD": [0.2, 0.3], "pred_pA": [0.2, 0.5],
        "norm_pH": [0.5, 0.3], "norm_pD": [0.3, 0.2], "norm_pA": [0.2, 0.5],
    })
    long = to_long(preds)
    obs_per_match = long.groupby("match_id")["observed"].sum()
    assert (obs_per_match == 1).all()
