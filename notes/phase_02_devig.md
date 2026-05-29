# Phase 2: Devigging Module

**Status:** Complete  
**Files created:** `src/devig.py`, `tests/test_devig.py`  
**Tests:** 23 passing

---

## What we built

Four devigging (margin-removal) methods as pure numpy functions, vectorized over an `(n_matches, k_outcomes)` odds array. All methods take decimal (European) odds and return true-probability estimates.

| Method | Formula | Notes |
|--------|---------|-------|
| `normalized` | `p_i = (1/o_i) / Σ(1/o_j)` | H&W's theoretically correct method |
| `additive` | `p_i = 1/o_i − overround/k` | Simple; can go slightly negative on long shots |
| `power` | Find `k > 1` s.t. `Σ(1/o_i)^k = 1` | Log-probability compression |
| `shin` | Shin (1992) insider-trading model | Solves for `z` via bisection |

All four methods are exposed via `METHODS` dict and `all_methods()` for easy iteration across robustness checks.

**Why four methods?** Winkelmann et al. (2024) flag devig choice as an analyst-discretion lever that can shift results. Having all four lets us show the main results are robust to the choice.

---

## Decisions made

**Pure functions, no classes.** State isn't needed here — each function is a pure numerical transform. Follows the CLAUDE.md convention.

**Bisection for power and Shin.** Both require solving a 1-D equation for a parameter. Scipy's `brentq` would work too, but bisection is transparent, dependency-free, and fast enough (< 50ms on 500 rows).

**`all_methods()` returns a dict.** Callers (loaders, replication) need to attach each method's probabilities as named columns. A dict with consistent string keys makes that mapping straightforward.

---

## What didn't work

**Power devig bisection searched the wrong interval.** The first implementation searched for `k` in `(0, 1]`. This was wrong. For decimal odds > 1, all raw inverse-odds values are < 1. Raising a number < 1 to a power > 1 shrinks it, which is what we need to bring `Σ(1/o_i)^k` from (overround > 1) down to 1. So we need `k > 1`, not `k < 1`. The bisection always had both endpoints with `f > 0`, so it never converged.

Fix: search in `[1, hi]` where `hi` is doubled until `f(hi) < 0`. This guarantees the root is bracketed:
```python
hi = 2.0
while f(hi) > 0:
    hi *= 2.0
lo = 1.0
# ... bisect in [lo, hi]
```

Four tests caught this immediately (`test_probs_sum_to_one_soccer[power]` and three others), which is exactly why we test against synthetic data with known properties rather than just running the code.

---

## Test philosophy

Tests validate structural properties against hand-crafted odds fixtures:
- Row sums = 1 (for all methods)
- All values in (0, 1) (for methods that guarantee this)
- Favorite by odds = favorite by probability (rank ordering preserved)
- Normalized matches definition exactly
- Fair odds (no overround) → each method recovers true probabilities
