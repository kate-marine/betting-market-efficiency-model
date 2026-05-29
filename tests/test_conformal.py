"""
Tests for conformal prediction wrapper.

Key property to validate: at the nominal level α, empirical coverage ≥ 1-α
holds on data where we can compute the true answer. We test this on synthetic
data with known distributions, not just structural checks.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.conformal import (
    nonconformity_scores,
    conformal_threshold,
    prediction_sets,
    coverage_rate,
    mean_set_size,
    mondrian_thresholds,
    mondrian_prediction_sets,
    evaluate_conformal,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic calibration and test data
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_probs():
    """Model that assigns probability 1 to the true class (oracle)."""
    rng = np.random.default_rng(0)
    n = 500
    true_idx = rng.integers(0, 3, size=n)
    probs = np.zeros((n, 3))
    probs[np.arange(n), true_idx] = 1.0
    return probs, true_idx


@pytest.fixture
def uniform_probs():
    """Model that assigns 1/3 to each outcome (random baseline)."""
    n = 500
    true_idx = np.random.default_rng(1).integers(0, 3, size=n)
    probs = np.full((n, 3), 1.0 / 3.0)
    return probs, true_idx


@pytest.fixture
def calibrated_probs():
    """
    Reasonably calibrated model: probabilities are noisy versions of true probs.
    True probs drawn from Dirichlet(2,2,2), model probs are true + noise.
    Ground truth: coverage at any α should ≈ 1-α with enough data.
    """
    rng = np.random.default_rng(42)
    n = 2000
    true_probs = rng.dirichlet([2, 2, 2], size=n)
    # Draw true outcome from the true distribution
    true_idx = np.array([rng.choice(3, p=p) for p in true_probs])
    # Model probs: add Dirichlet noise (miscalibration)
    noise = rng.dirichlet([5, 5, 5], size=n)
    model_probs = 0.8 * true_probs + 0.2 * noise
    model_probs /= model_probs.sum(axis=1, keepdims=True)
    return model_probs, true_idx


# ---------------------------------------------------------------------------
# nonconformity_scores tests
# ---------------------------------------------------------------------------

def test_nonconformity_scores_perfect_model(perfect_probs):
    probs, true_idx = perfect_probs
    scores = nonconformity_scores(probs, true_idx)
    assert_allclose(scores, 0.0, atol=1e-10), "Oracle model should have score = 0"


def test_nonconformity_scores_uniform_model(uniform_probs):
    probs, true_idx = uniform_probs
    scores = nonconformity_scores(probs, true_idx)
    assert_allclose(scores, 1.0 - 1.0/3.0, atol=1e-10), "Uniform model score = 2/3"


def test_nonconformity_scores_in_unit_interval(calibrated_probs):
    probs, true_idx = calibrated_probs
    scores = nonconformity_scores(probs, true_idx)
    assert np.all(scores >= 0) and np.all(scores <= 1)


# ---------------------------------------------------------------------------
# conformal_threshold tests
# ---------------------------------------------------------------------------

def test_threshold_empty_set():
    """Empty calibration set should return threshold = 1 (include everything)."""
    assert conformal_threshold(np.array([]), alpha=0.1) == 1.0


def test_threshold_all_zero_scores():
    """If all scores are 0 (oracle), threshold should be at or near 0."""
    scores = np.zeros(100)
    tau = conformal_threshold(scores, alpha=0.1)
    assert tau <= 0.01, f"Expected near-zero threshold, got {tau}"


def test_threshold_increases_with_alpha():
    """Higher alpha (more permissive) → lower threshold → bigger prediction sets."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200)
    tau_low  = conformal_threshold(scores, alpha=0.05)
    tau_high = conformal_threshold(scores, alpha=0.20)
    assert tau_low >= tau_high, "Lower alpha should give higher (stricter) threshold"


# ---------------------------------------------------------------------------
# Finite-sample coverage guarantee
# ---------------------------------------------------------------------------

def test_marginal_coverage_at_least_nominal(calibrated_probs):
    """
    Core guarantee: empirical coverage ≥ 1-α with high probability.
    We split calibrated_probs into cal/test and verify.
    This test can fail with low probability (~α) even if the code is correct,
    but with n=1000 cal, n=1000 test, and α=0.1, failures are very rare.
    """
    probs, true_idx = calibrated_probs
    n = len(probs)
    n_cal = n // 2
    cal_p, test_p = probs[:n_cal], probs[n_cal:]
    cal_y, test_y = true_idx[:n_cal], true_idx[n_cal:]

    alpha = 0.10
    cal_scores = nonconformity_scores(cal_p, cal_y)
    tau = conformal_threshold(cal_scores, alpha)
    sets = prediction_sets(test_p, tau)
    cov = coverage_rate(sets, test_y)

    # Coverage should be ≥ 1-α; allow tiny slack for finite-sample noise
    assert cov >= (1 - alpha) - 0.02, \
        f"Coverage {cov:.3f} is below nominal {1-alpha} by more than 2pp"


def test_perfect_model_sets_size_one(perfect_probs):
    """Oracle model: each prediction set should contain exactly 1 outcome."""
    probs, true_idx = perfect_probs
    n = len(probs)
    cal_p, test_p = probs[:n//2], probs[n//2:]
    cal_y, test_y = true_idx[:n//2], true_idx[n//2:]

    cal_scores = nonconformity_scores(cal_p, cal_y)
    tau = conformal_threshold(cal_scores, alpha=0.1)
    sets = prediction_sets(test_p, tau)

    # With perfect scores = 0, threshold ≈ 0, so only outcomes with prob ≥ 1 are included
    assert mean_set_size(sets) <= 1.05, "Oracle model should give ~1-outcome prediction sets"


def test_uniform_model_set_size_three():
    """
    Uniform model (1/3, 1/3, 1/3): all nonconformity scores = 2/3.
    Threshold = 2/3, so 1 - threshold = 1/3, meaning all outcomes with prob ≥ 1/3
    are included — all 3 outcomes. Set size should be 3.
    """
    n = 300
    rng = np.random.default_rng(5)
    probs = np.full((n, 3), 1.0/3.0)
    true_idx = rng.integers(0, 3, size=n)

    cal_scores = nonconformity_scores(probs[:150], true_idx[:150])
    tau = conformal_threshold(cal_scores, alpha=0.1)
    sets = prediction_sets(probs[150:], tau)
    sz = mean_set_size(sets)

    assert sz == pytest.approx(3.0, abs=0.01), \
        f"Uniform model should give 3-outcome sets, got {sz:.2f}"


# ---------------------------------------------------------------------------
# Mondrian conformal tests
# ---------------------------------------------------------------------------

def test_mondrian_thresholds_per_group(calibrated_probs):
    """Mondrian should produce distinct thresholds for distinct groups."""
    probs, true_idx = calibrated_probs
    groups = np.array(["A"] * 1000 + ["B"] * 1000)
    scores = nonconformity_scores(probs, true_idx)
    thresholds = mondrian_thresholds(scores, groups, alpha=0.1)

    assert "A" in thresholds and "B" in thresholds
    # Groups A and B may have different thresholds (they're from same dist here,
    # but both should be non-trivial)
    assert 0 < thresholds["A"] < 1
    assert 0 < thresholds["B"] < 1


def test_mondrian_coverage_per_group(calibrated_probs):
    """Mondrian should achieve ≥ 1-α coverage in each group separately."""
    probs, true_idx = calibrated_probs
    n = len(probs)
    # Interleave groups so both A and B appear in cal and test halves
    groups = np.array(["A" if i % 2 == 0 else "B" for i in range(n)])

    # Split into cal/test
    n_cal = n // 2
    cal_p, test_p = probs[:n_cal], probs[n_cal:]
    cal_y, test_y = true_idx[:n_cal], true_idx[n_cal:]
    cal_g = groups[:n_cal]
    test_g = groups[n_cal:]

    cal_scores = nonconformity_scores(cal_p, cal_y)
    global_tau = conformal_threshold(cal_scores, 0.1)
    league_thresholds = mondrian_thresholds(cal_scores, cal_g, 0.1)
    sets = mondrian_prediction_sets(test_p, test_g, league_thresholds, global_tau)

    for group in ["A", "B"]:
        mask = test_g == group
        cov = coverage_rate(sets[mask], test_y[mask])
        assert cov >= 0.88, f"Group {group} coverage {cov:.3f} below 0.88"


# ---------------------------------------------------------------------------
# evaluate_conformal integration test
# ---------------------------------------------------------------------------

def test_evaluate_conformal_returns_expected_columns(tmp_path_factory):
    """End-to-end: evaluate_conformal on a small synthetic preds DataFrame."""
    import pandas as pd
    rng = np.random.default_rng(7)
    n_per_season = 200
    seasons = ["2018-2019", "2019-2020", "2020-2021"]
    rows = []
    for s in seasons:
        probs_m = rng.dirichlet([2, 2, 2], size=n_per_season)
        probs_k = rng.dirichlet([2, 2, 2], size=n_per_season)
        true_idx = np.array([rng.choice(3, p=p) for p in probs_m])
        results = ["H", "D", "A"]
        for i in range(n_per_season):
            rows.append({
                "test_season": s,
                "league": rng.choice(["E0", "D1"]),
                "result": results[true_idx[i]],
                "pred_pH": probs_m[i, 0], "pred_pD": probs_m[i, 1], "pred_pA": probs_m[i, 2],
                "norm_pH": probs_k[i, 0], "norm_pD": probs_k[i, 1], "norm_pA": probs_k[i, 2],
            })
    preds = pd.DataFrame(rows)

    result = evaluate_conformal(preds, alpha=0.1)

    assert len(result) > 0
    for col in ["test_season", "estimator", "method", "coverage", "mean_set_size", "alpha"]:
        assert col in result.columns, f"Missing column: {col}"
    assert set(result["estimator"].unique()) == {"model", "market"}
    assert set(result["method"].unique()) == {"marginal", "mondrian_league"}

    # Coverage should be ≥ 0.85 for both estimators under marginal conformal
    marginal = result[result["method"] == "marginal"]
    assert (marginal["coverage"] >= 0.85).all(), "Some seasons below 0.85 coverage"
