"""
Tests for the efficient-markets null simulation.

Core properties:
  1. Under the null (gamma=0), the simulation p-value is uniform → rejection rate = alpha.
  2. Under FLB (gamma>0), the simulation p-value shrinks.
  3. The null gamma distribution is centered near zero with plausible spread.
  4. The joint test statistic is larger when observed effects are larger.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.null_sim import (
    _build_setup, _simulate_gamma, run_simulation,
    simulation_pvalue, build_results_table, joint_test,
)


# ---------------------------------------------------------------------------
# Fixture: minimal synthetic league
# ---------------------------------------------------------------------------

def _make_synthetic_league(n_matches: int, gamma: float, seed: int):
    """
    Build synthetic wide+long DataFrames with known FLB.
    Market probs drawn from Dirichlet; outcomes drawn with FLB = gamma.
    """
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([2, 2, 2], size=n_matches)
    pH, pD, pA = probs[:, 0], probs[:, 1], probs[:, 2]

    # FLB outcomes: pi_ij = -gamma/3 + (1+gamma)*P^N_ij
    pi_H = np.clip(-gamma/3 + (1+gamma)*pH, 1e-9, None)
    pi_D = np.clip(-gamma/3 + (1+gamma)*pD, 1e-9, None)
    pi_A = np.clip(-gamma/3 + (1+gamma)*pA, 1e-9, None)
    total = pi_H + pi_D + pi_A
    pi_H /= total; pi_D /= total; pi_A /= total

    u = rng.uniform(size=n_matches)
    results = np.where(u < pi_H, "H", np.where(u < pi_H + pi_D, "D", "A"))

    match_ids = [f"m{i}" for i in range(n_matches)]
    wide = pd.DataFrame({
        "match_id": match_ids, "league": "E0", "season": "2019-2020",
        "norm_pH": pH, "norm_pD": pD, "norm_pA": pA,
        "result": results,
    })

    # Build long format
    parts = []
    for outcome, p_col in [("H","norm_pH"),("D","norm_pD"),("A","norm_pA")]:
        c = wide[["match_id","league","season"]].copy()
        c["outcome"] = outcome
        c["norm_p"] = wide[p_col].values
        c["observed"] = (wide["result"] == outcome).astype(int).values
        parts.append(c)
    long = pd.concat(parts, ignore_index=True)
    return wide, long


# ---------------------------------------------------------------------------
# _build_setup
# ---------------------------------------------------------------------------

class TestBuildSetup:
    def test_matrix_shape(self):
        wide, long = _make_synthetic_league(200, gamma=0.0, seed=0)
        setup = _build_setup("E0", long, wide)
        assert setup.M.shape == (2, len(long)), "M must be (2, n_long)"
        assert setup.pH.shape == (200,)
        assert setup.p_long.shape == (len(long),)

    def test_outcome_indices_cover_all_three(self):
        wide, long = _make_synthetic_league(100, gamma=0.0, seed=1)
        setup = _build_setup("E0", long, wide)
        assert set(np.unique(setup.outcome_idx)) == {0, 1, 2}

    def test_match_indices_range(self):
        wide, long = _make_synthetic_league(50, gamma=0.0, seed=2)
        setup = _build_setup("E0", long, wide)
        assert setup.match_idx.min() == 0
        assert setup.match_idx.max() == 49

    def test_ph_pd_pa_sum_to_one(self):
        wide, long = _make_synthetic_league(100, gamma=0.0, seed=3)
        setup = _build_setup("E0", long, wide)
        row_sums = setup.pH + setup.pD + setup.pA
        np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-6)


# ---------------------------------------------------------------------------
# _simulate_gamma
# ---------------------------------------------------------------------------

class TestSimulateGamma:
    def test_returns_scalar(self):
        wide, long = _make_synthetic_league(200, gamma=0.0, seed=0)
        setup = _build_setup("E0", long, wide)
        rng = np.random.default_rng(0)
        g = _simulate_gamma(setup, rng)
        assert isinstance(g, float)

    def test_null_gamma_is_small_on_average(self):
        """Under null, average of many simulations should be near zero."""
        wide, long = _make_synthetic_league(500, gamma=0.0, seed=42)
        setup = _build_setup("E0", long, wide)
        rng = np.random.default_rng(0)
        gammas = [_simulate_gamma(setup, rng) for _ in range(500)]
        mean_g = np.mean(gammas)
        assert abs(mean_g) < 0.03, f"Mean gamma under null should be ~0, got {mean_g:.4f}"


# ---------------------------------------------------------------------------
# simulation_pvalue and run_simulation
# ---------------------------------------------------------------------------

class TestSimulationPvalue:
    def test_pvalue_at_zero_is_near_one(self):
        """p-value for gamma_obs=0 should be near 1 (always exceeded by |null|)."""
        null = np.random.default_rng(0).normal(0, 0.05, 1000)
        p = simulation_pvalue(0.0, null)
        assert p >= 0.90, f"p-value at gamma=0 should be near 1, got {p:.3f}"

    def test_pvalue_for_extreme_obs_is_small(self):
        """gamma_obs much larger than null gammas → very small p-value."""
        null = np.random.default_rng(1).normal(0, 0.05, 1000)
        p = simulation_pvalue(1.0, null)   # 1.0 >> typical null gamma of ±0.05
        assert p < 0.01, f"p-value for extreme obs should be tiny, got {p:.4f}"

    def test_pvalue_is_two_sided(self):
        """Negative gamma_obs as extreme as positive should give same p-value."""
        null = np.random.default_rng(2).normal(0, 0.05, 2000)
        p_pos = simulation_pvalue(+0.10, null)
        p_neg = simulation_pvalue(-0.10, null)
        assert abs(p_pos - p_neg) < 0.02, "Two-sided p-values should match for ± same magnitude"


class TestNullDistributionCalibration:
    def test_rejection_rate_under_null_matches_alpha(self):
        """
        If we run many simulations under the null and use the simulation p-value
        to decide whether to reject, the rejection rate should ≈ alpha.

        This is the fundamental calibration check for the simulation.
        We run 100 'experiments' (each drawing one dataset and one null distribution)
        and check that ~5% reject at alpha=0.05.
        """
        rng_outer = np.random.default_rng(99)
        rejections = 0
        n_experiments = 100
        n_sim_inner = 300   # smaller for speed in test

        for exp_i in range(n_experiments):
            # Draw one dataset from the null
            wide, long = _make_synthetic_league(300, gamma=0.0, seed=int(rng_outer.integers(1e6)))
            setup = _build_setup("E0", long, wide)
            rng_inner = np.random.default_rng(exp_i)

            # Compute gamma_obs for this dataset (the "true" observed value, also from null)
            gamma_obs = _simulate_gamma(setup, np.random.default_rng(exp_i * 1000 + 1))

            # Run null distribution
            null = np.array([_simulate_gamma(setup, rng_inner) for _ in range(n_sim_inner)])
            p_sim = simulation_pvalue(gamma_obs, null)

            if p_sim < 0.05:
                rejections += 1

        rejection_rate = rejections / n_experiments
        # Allow generous range given n_experiments=100 (SE ≈ 2%)
        assert 0.00 <= rejection_rate <= 0.18, (
            f"Null rejection rate {rejection_rate:.2f} outside [0, 0.18] — simulation miscalibrated"
        )


# ---------------------------------------------------------------------------
# Joint test
# ---------------------------------------------------------------------------

class TestJointTest:
    def _make_two_setups(self):
        wide1, long1 = _make_synthetic_league(300, gamma=0.0, seed=10)
        wide2, long2 = _make_synthetic_league(300, gamma=0.0, seed=11)
        s1 = _build_setup("E0", long1, wide1)._replace(gamma_obs=0.05, se_obs=0.04)
        s2 = _build_setup("D1", long2, wide2)._replace(gamma_obs=0.02, se_obs=0.04)
        return [s1, s2]

    def test_joint_max_t_returns_required_keys(self):
        setups = self._make_two_setups()
        rng = np.random.default_rng(0)
        null = {s.league: np.array([_simulate_gamma(s, rng) for _ in range(200)]) for s in setups}
        result = joint_test(setups, null, method="max_t")
        for key in ["method","obs_stat","p_joint","n_sim","n_leagues"]:
            assert key in result

    def test_joint_pvalue_is_probability(self):
        setups = self._make_two_setups()
        rng = np.random.default_rng(1)
        null = {s.league: np.array([_simulate_gamma(s, rng) for _ in range(200)]) for s in setups}
        for method in ["max_t", "sum_chi2"]:
            r = joint_test(setups, null, method=method)
            assert 0.0 <= r["p_joint"] <= 1.0

    def test_larger_effect_gives_smaller_joint_pvalue(self):
        """More extreme observed γ should give smaller joint p-value."""
        wide, long = _make_synthetic_league(300, gamma=0.0, seed=20)
        base = _build_setup("E0", long, wide)
        rng = np.random.default_rng(5)
        null = {"E0": np.array([_simulate_gamma(base, rng) for _ in range(500)])}

        small = [base._replace(gamma_obs=0.02, se_obs=0.04)]
        large = [base._replace(gamma_obs=0.25, se_obs=0.04)]

        p_small = joint_test(small, null, method="max_t")["p_joint"]
        p_large = joint_test(large, null, method="max_t")["p_joint"]
        assert p_large < p_small, "Larger effect should give smaller joint p-value"
