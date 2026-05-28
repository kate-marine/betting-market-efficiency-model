"""
Tests for src/devig.py — each test validates a property against synthetic data
with known ground truth, not just structural checks.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.devig import normalized, additive, power, shin, all_methods, METHODS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def soccer_odds():
    """Typical 3-way soccer odds with ~5% overround."""
    return np.array([
        [2.10, 3.40, 3.60],   # balanced-ish match
        [1.30, 5.50, 9.50],   # heavy favorite
        [3.80, 3.60, 1.95],   # away favorite
    ])


@pytest.fixture
def tennis_odds():
    """Typical 2-way tennis odds."""
    return np.array([
        [1.50, 2.60],
        [1.05, 12.00],
        [2.10, 1.80],
    ])


@pytest.fixture
def fair_tennis_odds():
    """Perfectly fair 2-way odds (no overround) — sum of inv-odds = 1."""
    # True probs [0.6, 0.4] → odds [1/0.6, 1/0.4]
    return np.array([[1 / 0.6, 1 / 0.4]])


# ---------------------------------------------------------------------------
# Shared property tests — run for every method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name", list(METHODS.keys()))
def test_probs_sum_to_one_soccer(method_name, soccer_odds):
    probs = METHODS[method_name](soccer_odds)
    assert_allclose(probs.sum(axis=1), np.ones(len(soccer_odds)), atol=1e-6,
                    err_msg=f"{method_name}: row sums not 1")


@pytest.mark.parametrize("method_name", list(METHODS.keys()))
def test_probs_sum_to_one_tennis(method_name, tennis_odds):
    probs = METHODS[method_name](tennis_odds)
    assert_allclose(probs.sum(axis=1), np.ones(len(tennis_odds)), atol=1e-6,
                    err_msg=f"{method_name}: row sums not 1")


@pytest.mark.parametrize("method_name", ["normalized", "power", "shin"])
def test_probs_in_open_unit_interval(method_name, soccer_odds):
    """Additive excluded — it can push extreme longshots slightly below 0."""
    probs = METHODS[method_name](soccer_odds)
    assert np.all(probs > 0), f"{method_name}: some prob <= 0"
    assert np.all(probs < 1), f"{method_name}: some prob >= 1"


@pytest.mark.parametrize("method_name", list(METHODS.keys()))
def test_favorite_has_highest_prob(method_name, soccer_odds):
    """The team with the lowest odds (highest raw inv-odds) must have highest prob."""
    probs = METHODS[method_name](soccer_odds)
    raw = 1.0 / soccer_odds
    fav_by_odds = raw.argmax(axis=1)
    fav_by_prob = probs.argmax(axis=1)
    np.testing.assert_array_equal(
        fav_by_prob, fav_by_odds,
        err_msg=f"{method_name}: favorite ordering not preserved"
    )


# ---------------------------------------------------------------------------
# Method-specific tests
# ---------------------------------------------------------------------------

def test_normalized_matches_definition(soccer_odds):
    """Normalized: p_i == (1/o_i) / sum(1/o_j) exactly."""
    raw = 1.0 / soccer_odds
    expected = raw / raw.sum(axis=1, keepdims=True)
    assert_allclose(normalized(soccer_odds), expected, atol=1e-12)


def test_normalized_fair_odds_unchanged(fair_tennis_odds):
    """With no overround, normalized should recover the exact true probs."""
    probs = normalized(fair_tennis_odds)
    assert_allclose(probs[0], [0.6, 0.4], atol=1e-10)


def test_additive_fair_odds_unchanged(fair_tennis_odds):
    """With no overround, additive devig should leave probs unchanged."""
    probs = additive(fair_tennis_odds)
    assert_allclose(probs[0], [0.6, 0.4], atol=1e-10)


def test_power_fair_odds(fair_tennis_odds):
    """Power devig with no overround: exponent k~1, probs recover true probs."""
    probs = power(fair_tennis_odds)
    assert_allclose(probs[0], [0.6, 0.4], atol=1e-6)


def test_shin_fair_odds(fair_tennis_odds):
    """Shin with no overround: z~0, probs recover true probs."""
    probs = shin(fair_tennis_odds)
    assert_allclose(probs[0], [0.6, 0.4], atol=1e-6)


def test_all_methods_returns_all_four(soccer_odds):
    result = all_methods(soccer_odds)
    assert set(result.keys()) == {"normalized", "additive", "power", "shin"}
    for name, arr in result.items():
        assert arr.shape == soccer_odds.shape, f"{name}: wrong shape"


def test_methods_agree_on_rank_order(soccer_odds):
    """All methods should agree on which outcome is most probable."""
    results = all_methods(soccer_odds)
    rankings = {name: arr.argmax(axis=1) for name, arr in results.items()}
    ref = rankings["normalized"]
    for name, ranks in rankings.items():
        np.testing.assert_array_equal(ranks, ref,
            err_msg=f"{name} disagrees with normalized on argmax")


def test_single_row_input(soccer_odds):
    """Single-match input (shape (1,3)) should work for all methods."""
    single = soccer_odds[[0]]
    for name, fn in METHODS.items():
        probs = fn(single)
        assert probs.shape == (1, 3), f"{name}: wrong shape for single row"
        assert_allclose(probs.sum(axis=1), [1.0], atol=1e-6)
