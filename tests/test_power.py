"""
Tests for power analysis module.

Core property: the analytical formula and simulation must agree with each other
and with known results from statistics textbooks.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.power import (
    analytical_power, analytical_mde,
    generate_flb_outcomes, _wls_cluster_test, power_table,
)


# ---------------------------------------------------------------------------
# Analytical power / MDE
# ---------------------------------------------------------------------------

class TestAnalyticalPower:

    def test_power_at_zero_is_alpha(self):
        """At true gamma=0, power equals alpha (type I error rate)."""
        for alpha in [0.05, 0.10]:
            p = analytical_power(0.0, se=0.05, alpha=alpha)
            assert abs(p - alpha) < 0.005, f"power at gamma=0 should be alpha={alpha}, got {p:.4f}"

    def test_power_increases_with_gamma(self):
        """Larger true effect → higher power."""
        se = 0.05
        powers = [analytical_power(g, se) for g in [0.05, 0.10, 0.20, 0.40]]
        assert all(powers[i] < powers[i+1] for i in range(len(powers)-1))

    def test_power_increases_with_smaller_se(self):
        """Smaller SE (larger sample) → higher power for same gamma."""
        gamma = 0.10
        p_large_se = analytical_power(gamma, se=0.08)
        p_small_se = analytical_power(gamma, se=0.03)
        assert p_small_se > p_large_se

    def test_power_at_mde_equals_target(self):
        """analytical_power(mde, se) should equal the target power."""
        se = 0.04
        for target_power in [0.70, 0.80, 0.90]:
            mde = analytical_mde(se, alpha=0.05, power=target_power)
            computed_power = analytical_power(mde, se, alpha=0.05)
            assert abs(computed_power - target_power) < 0.001, (
                f"power at MDE should be {target_power:.2f}, got {computed_power:.4f}"
            )

    def test_mde_formula_value(self):
        """MDE = (1.96 + 0.842) * SE for alpha=0.05, power=0.80."""
        se = 0.03
        expected = 2.8024 * se   # z_{0.025} + z_{0.20} = 1.96 + 0.842
        assert abs(analytical_mde(se) - expected) < 1e-4

    def test_mde_increases_with_se(self):
        """Larger SE (smaller sample) → harder to detect → larger MDE."""
        assert analytical_mde(0.03) < analytical_mde(0.05) < analytical_mde(0.10)

    def test_power_at_1_se_is_reasonable(self):
        """At gamma = 1*SE, power should be around 17% (well below 80%)."""
        se = 0.05
        p = analytical_power(se, se)
        assert 0.12 < p < 0.25, f"power at 1*SE should be ~17%, got {p:.4f}"

    def test_power_at_3_se_is_above_80pct(self):
        """At gamma = 3*SE, power should be above 80% (it's ~85%)."""
        se = 0.05
        p = analytical_power(3 * se, se)
        assert p > 0.80, f"power at 3*SE should be >80%, got {p:.4f}"

    def test_power_at_4_se_is_above_95pct(self):
        """At gamma = 4*SE, power should be above 95% (ncp=4, well above threshold)."""
        se = 0.05
        p = analytical_power(4 * se, se)
        assert p > 0.95, f"power at 4*SE should be >95%, got {p:.4f}"


# ---------------------------------------------------------------------------
# FLB outcome generation
# ---------------------------------------------------------------------------

class TestGenerateFlbOutcomes:

    def _make_wide(self, n=500, seed=0):
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet([2, 2, 2], size=n)
        return pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "league": "E0",
            "season": "2019-2020",
            "norm_pH": probs[:, 0],
            "norm_pD": probs[:, 1],
            "norm_pA": probs[:, 2],
            "result": ["H"] * n,  # placeholder
        })

    def test_results_are_valid_codes(self):
        wide = self._make_wide()
        rng = np.random.default_rng(0)
        out = generate_flb_outcomes(wide, gamma=0.10, rng=rng)
        assert set(out["result"].unique()).issubset({"H", "D", "A"})

    def test_row_count_unchanged(self):
        wide = self._make_wide(300)
        rng = np.random.default_rng(1)
        out = generate_flb_outcomes(wide, gamma=0.05, rng=rng)
        assert len(out) == 300

    def test_efficient_market_gives_near_zero_gamma(self):
        """
        Under gamma=0 (efficient market), running the H&W regression on
        generated outcomes should recover gamma ≈ 0 on average.
        Validated with large n and many simulations to reduce noise.
        """
        n = 1000
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([2, 2, 2], size=n)
        wide = pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "norm_pH": probs[:, 0],
            "norm_pD": probs[:, 1],
            "norm_pA": probs[:, 2],
            "result": ["H"] * n,
        })

        gamma_estimates = []
        for i in range(50):
            out = generate_flb_outcomes(wide, gamma=0.0, rng=rng)
            # Build long-format arrays — keep types separate to avoid numpy string coercion
            p_list, ymp_list, id_list = [], [], []
            for outcome, p_col in [("H","norm_pH"),("D","norm_pD"),("A","norm_pA")]:
                obs = (out["result"] == outcome).astype(float).values
                p   = out[p_col].values.astype(float)
                ids = out["match_id"].values
                p_list.append(p); ymp_list.append(obs - p); id_list.append(ids)
            p_arr   = np.concatenate(p_list)
            ymp_arr = np.concatenate(ymp_list)
            cids    = np.concatenate(id_list)
            gamma_hat, _, _ = _wls_cluster_test(p_arr, ymp_arr, cids)
            if not np.isnan(gamma_hat):
                gamma_estimates.append(gamma_hat)

        mean_gamma = np.mean(gamma_estimates)
        assert abs(mean_gamma) < 0.05, f"gamma should be ~0 under efficiency, got {mean_gamma:.4f}"

    def test_flb_market_gives_positive_gamma(self):
        """Under gamma=0.15, the regression should recover a positive estimate on average."""
        n = 800
        rng = np.random.default_rng(7)
        probs = rng.dirichlet([2, 2, 2], size=n)
        wide = pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "norm_pH": probs[:, 0],
            "norm_pD": probs[:, 1],
            "norm_pA": probs[:, 2],
            "result": ["H"] * n,
        })

        gamma_estimates = []
        for _ in range(30):
            out = generate_flb_outcomes(wide, gamma=0.15, rng=rng)
            p_list, ymp_list, id_list = [], [], []
            for outcome, p_col in [("H","norm_pH"),("D","norm_pD"),("A","norm_pA")]:
                obs = (out["result"] == outcome).astype(float).values
                p   = out[p_col].values.astype(float)
                p_list.append(p); ymp_list.append(obs - p); id_list.append(out["match_id"].values)
            p_arr   = np.concatenate(p_list)
            ymp_arr = np.concatenate(ymp_list)
            cids    = np.concatenate(id_list)
            g, _, _ = _wls_cluster_test(p_arr, ymp_arr, cids)
            if not np.isnan(g):
                gamma_estimates.append(g)

        assert np.mean(gamma_estimates) > 0.05, "Expected positive gamma under FLB"


# ---------------------------------------------------------------------------
# WLS cluster test
# ---------------------------------------------------------------------------

class TestWlsClusterTest:

    def test_rejects_under_strong_signal(self):
        """Very strong FLB should always be rejected."""
        rng = np.random.default_rng(0)
        n = 500
        p = rng.uniform(0.1, 0.9, n * 3)
        # Force strong FLB: y - p = 0.3 * p (gamma = 0.3, very strong)
        y_minus_p = 0.3 * p + rng.normal(0, 0.01, n * 3)
        cluster_ids = np.repeat(np.arange(n), 3)
        _, t, reject = _wls_cluster_test(p, y_minus_p, cluster_ids)
        assert reject, f"Strong FLB should be rejected, t={t:.2f}"

    def test_does_not_reject_under_null(self):
        """Under the null (gamma=0), rejection rate should be near alpha=0.05."""
        rng = np.random.default_rng(10)
        rejections = 0
        n_trials = 100
        for _ in range(n_trials):
            n = 300
            p = rng.uniform(0.1, 0.9, n * 3)
            # Null: y - p = noise only
            y_minus_p = rng.normal(0, np.sqrt(p * (1 - p)), n * 3)
            cluster_ids = np.repeat(np.arange(n), 3)
            _, _, reject = _wls_cluster_test(p, y_minus_p, cluster_ids)
            if reject:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should be near 5%; allow generous range due to small n_trials
        assert 0.00 <= rejection_rate <= 0.20, (
            f"Null rejection rate {rejection_rate:.2f} outside [0, 0.20]"
        )
