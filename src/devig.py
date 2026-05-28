"""
Devigging (margin removal) methods for converting bookmaker odds to true probabilities.

H&W (2024) show normalized probabilities are the theoretically correct estimator
under strong-form efficiency. All other methods are robustness checks.

All functions accept odds as a 2-D array of shape (n_matches, k_outcomes)
and return probabilities of the same shape. Odds are decimal (European) format.
"""

import numpy as np
from numpy.typing import NDArray


def _inv_odds(odds: NDArray) -> NDArray:
    """Raw inverse-odds (overround retained)."""
    return 1.0 / odds


def normalized(odds: NDArray) -> NDArray:
    """
    Normalized (Shin-style rescaling): divide each 1/o_i by the row sum.
    This is H&W's primary devig method and the theoretically preferred one.
    """
    raw = _inv_odds(odds)
    return raw / raw.sum(axis=1, keepdims=True)


def additive(odds: NDArray) -> NDArray:
    """
    Additive devig: subtract a uniform share of the overround from each 1/o_i.
    Simplest margin-removal; can produce negatives for extreme longshots.
    """
    raw = _inv_odds(odds)
    k = odds.shape[1]
    overround = raw.sum(axis=1, keepdims=True) - 1.0
    return raw - overround / k


def power(odds: NDArray, tol: float = 1e-8, maxiter: int = 100) -> NDArray:
    """
    Power devig: find exponent k per row such that sum((1/o_i)^k) = 1.
    For decimal odds > 1, all raw inv-odds are < 1, so k > 1 is needed to
    push the overround sum back down to 1. Bisects in [1, k_max].
    """
    raw = _inv_odds(odds)
    n = raw.shape[0]
    probs = np.empty_like(raw)

    for i in range(n):
        row = raw[i]

        def f(k: float) -> float:
            return float(np.sum(row**k)) - 1.0

        # k=1 → f > 0 (overround); k→∞ → f → -1. Find upper bound where f < 0.
        hi = 2.0
        while f(hi) > 0:
            hi *= 2.0
        lo = 1.0
        for _ in range(maxiter):
            mid = (lo + hi) / 2.0
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        probs[i] = row ** ((lo + hi) / 2.0)

    return probs


def shin(odds: NDArray, tol: float = 1e-8, maxiter: int = 100) -> NDArray:
    """
    Shin (1992) model: accounts for insider trading via parameter z in [0,1).
    Solves for z per row such that the Shin implied probabilities sum to 1.

    Shin formula: p_i = (sqrt(z^2 + 4(1-z) * q_i^2 / Q) - z) / (2(1-z))
    where q_i = 1/o_i and Q = sum(q_i).
    """
    raw = _inv_odds(odds)
    n = raw.shape[0]
    probs = np.empty_like(raw)

    for i in range(n):
        q = raw[i]
        Q = q.sum()

        def shin_probs(z: float) -> NDArray:
            disc = z**2 + 4.0 * (1.0 - z) * q**2 / Q
            return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

        def residual(z: float) -> float:
            return float(shin_probs(z).sum()) - 1.0

        # z=0 → normalized; z→1 → degenerate. Bisect in [0, 0.99].
        lo, hi = 0.0, 0.99
        for _ in range(maxiter):
            mid = (lo + hi) / 2.0
            if residual(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break

        probs[i] = shin_probs((lo + hi) / 2.0)

    return probs


# Convenience registry so callers can iterate over all methods
METHODS: dict[str, callable] = {
    "normalized": normalized,
    "additive": additive,
    "power": power,
    "shin": shin,
}


def all_methods(odds: NDArray) -> dict[str, NDArray]:
    """Return a dict of {method_name: prob_array} for all four devig methods."""
    return {name: fn(odds) for name, fn in METHODS.items()}
