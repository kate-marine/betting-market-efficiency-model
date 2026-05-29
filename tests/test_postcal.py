"""
Tests for post-hoc calibration.

Core property: fitting a calibrator on in-distribution data and applying it
to held-out data from the same distribution should reduce ECE. Validated on
synthetic data with controlled miscalibration.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.postcal import calibrate_walk_forward, _renormalize
from src.calibration import ece


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preds(n_per_season: int, n_seasons: int, overconfidence: float, seed: int) -> pd.DataFrame:
    """
    Synthetic predictions DataFrame with controlled overconfidence.

    True probs are drawn from Dirichlet(2,2,2). Model probs are the true probs
    pushed toward the argmax by `overconfidence` — mimicking a model that is
    right about which outcome is most likely but too certain about it.
    """
    rng = np.random.default_rng(seed)
    season_labels = [f"{2017+i}-{2018+i}" for i in range(n_seasons)]
    rows = []
    for i, season in enumerate(season_labels):
        true_p = rng.dirichlet([2, 2, 2], size=n_per_season)
        argmax = true_p.argmax(axis=1)

        # Model is overconfident: push mass toward argmax
        model_p = true_p.copy()
        for j in range(n_per_season):
            model_p[j, argmax[j]] += overconfidence * (1 - true_p[j, argmax[j]])
            other = [k for k in range(3) if k != argmax[j]]
            scale = (1 - model_p[j, argmax[j]]) / (true_p[j, other].sum() + 1e-9)
            model_p[j, other] *= scale
        model_p = np.clip(model_p, 1e-4, None)
        model_p /= model_p.sum(axis=1, keepdims=True)

        outcomes_idx = np.array([rng.choice(3, p=p) for p in true_p])
        outcome_map  = {0: "H", 1: "D", 2: "A"}
        results = [outcome_map[o] for o in outcomes_idx]

        for j in range(n_per_season):
            rows.append({
                "match_id":   f"{season}_{j}",
                "test_season": season,
                "league":      "E0",
                "result":      results[j],
                "pred_pH":     model_p[j, 0],
                "pred_pD":     model_p[j, 1],
                "pred_pA":     model_p[j, 2],
                "norm_pH":     true_p[j, 0],   # pretend market = true probs
                "norm_pD":     true_p[j, 1],
                "norm_pA":     true_p[j, 2],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenormalize:
    def test_sums_to_one(self):
        h = np.array([0.5, 0.3, 0.2])
        d = np.array([0.3, 0.4, 0.3])
        a = np.array([0.2, 0.3, 0.5])
        rh, rd, ra = _renormalize(h, d, a)
        np.testing.assert_allclose(rh + rd + ra, np.ones(3), atol=1e-10)

    def test_preserves_order(self):
        """Outcome with highest raw cal score should remain highest after renorm."""
        h = np.array([0.7, 0.1])
        d = np.array([0.2, 0.6])
        a = np.array([0.1, 0.3])
        rh, rd, ra = _renormalize(h, d, a)
        assert rh[0] > rd[0] and rh[0] > ra[0]   # H is largest for match 0
        assert rd[1] > rh[1] and rd[1] > ra[1]   # D is largest for match 1


class TestCalibrateWalkForward:
    def test_output_has_calibrated_columns(self):
        preds = _make_preds(200, 3, overconfidence=0.3, seed=0)
        result = calibrate_walk_forward(preds)
        for col in ["iso_pH","iso_pD","iso_pA","platt_pH","platt_pD","platt_pA"]:
            assert col in result.columns, f"Missing {col}"

    def test_first_season_is_nan(self):
        """First season has no calibration data → calibrated probs must be NaN."""
        preds = _make_preds(200, 3, overconfidence=0.3, seed=0)
        result = calibrate_walk_forward(preds)
        first = sorted(result["test_season"].unique())[0]
        assert result[result["test_season"] == first]["iso_pH"].isna().all()

    def test_calibrated_probs_sum_to_one(self):
        preds = _make_preds(300, 3, overconfidence=0.3, seed=1)
        result = calibrate_walk_forward(preds)
        # Drop first season (NaN)
        valid = result[result["iso_pH"].notna()]
        iso_sum = valid[["iso_pH","iso_pD","iso_pA"]].sum(axis=1)
        np.testing.assert_allclose(iso_sum.values, np.ones(len(valid)), atol=1e-6)

    def test_calibrated_probs_in_unit_interval(self):
        preds = _make_preds(300, 3, overconfidence=0.3, seed=2)
        result = calibrate_walk_forward(preds)
        valid = result[result["iso_pH"].notna()]
        for col in ["iso_pH","iso_pD","iso_pA","platt_pH","platt_pD","platt_pA"]:
            assert (valid[col] >= 0).all() and (valid[col] <= 1).all(), f"{col} out of [0,1]"

    def test_isotonic_reduces_ece_on_overconfident_model(self):
        """
        Core property: isotonic calibration applied to an overconfident model
        should reduce ECE on held-out data drawn from the same distribution.
        With large enough sample sizes, this is reliable (not random).
        """
        preds = _make_preds(n_per_season=800, n_seasons=4, overconfidence=0.4, seed=42)
        result = calibrate_walk_forward(preds)
        valid = result[result["iso_pH"].notna()]

        # Compute ECE pooled across all calibrated test seasons (one-vs-rest, all outcomes)
        def pooled_ece(df, pred_col, obs_col):
            return ece(df[pred_col].values, df[obs_col].values)

        # Convert to long for ECE comparison
        raw_ece_vals, iso_ece_vals = [], []
        for outcome, raw_col, iso_col in [
            ("H", "pred_pH", "iso_pH"),
            ("D", "pred_pD", "iso_pD"),
            ("A", "pred_pA", "iso_pA"),
        ]:
            obs = (valid["result"] == outcome).astype(int)
            raw_ece_vals.append(ece(valid[raw_col].values, obs.values))
            iso_ece_vals.append(ece(valid[iso_col].values, obs.values))

        mean_raw_ece = np.mean(raw_ece_vals)
        mean_iso_ece = np.mean(iso_ece_vals)
        assert mean_iso_ece < mean_raw_ece, (
            f"Isotonic calibration did not reduce ECE: "
            f"raw={mean_raw_ece:.4f}, iso={mean_iso_ece:.4f}"
        )

    def test_row_count_unchanged(self):
        preds = _make_preds(200, 3, overconfidence=0.3, seed=3)
        result = calibrate_walk_forward(preds)
        assert len(result) == len(preds)
