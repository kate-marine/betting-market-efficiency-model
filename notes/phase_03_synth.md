# Phase 3: Synthetic Data Generator

**Status:** Complete  
**Files created:** `src/synth.py`, (tests in `tests/test_synth_loaders.py`)  
**Tests:** See Phase 4 (synth and loaders tested together)

---

## What we built

A generator that produces files in the *exact on-disk formats* of the two real data sources:

- **Soccer:** `football-data.co.uk` CSV format with `Div`, `Date`, `HomeTeam`, `AwayTeam`, `FTR`, `AvgH/D/A`, `B365H/D/A` columns
- **Tennis:** `tennis-data.co.uk` XLSX format with `Date`, `Tournament`, `Surface`, `Winner`, `Loser`, `AvgW/L`, `B365W/L`

Two market variants:
- `"efficient"`: outcomes drawn from the true probabilities implied by the odds — no bias. The H&W regression should recover `gamma ≈ 0`.
- `"flb"`: the market *compresses* the probability distribution toward uniform, making favorites underpriced in the odds. The H&W regression should recover `gamma > 0`.

Synthetic files also include `_true_pH/_pD/_pA` columns with ground-truth probabilities so tests can verify statistical properties without relying on asymptotic arguments.

**Why build this before loading real data?** Winkelmann et al.'s entire methodological contribution is simulation-based validation — showing that seemingly significant season-level results arise purely by chance under efficiency. We follow the same principle: every downstream method should pass on data with known ground truth before we trust it on real data.

---

## Decisions made

**Files match real format exactly.** This forces the loader to be tested against realistic file structures (column names, date formats, etc.), not a simplified mock. If the loader works on synthetic files that look exactly like the real ones, it'll almost certainly work on the real ones.

**FLB implemented as market flattening, not outcome enrichment.** The bias lives in the odds, not in which outcomes we sample. `_apply_flb` compresses the market-implied distribution toward uniform before computing odds, then outcomes are still drawn from the *true* (unbiased) distribution. This correctly models the FLB scenario: the bookmaker offers worse-than-fair odds on favorites.

**Dirichlet(α=1) for true probabilities.** This gives a uniform distribution over the probability simplex — all probability vectors equally likely. That produces realistic-looking match odds without imposing structure.

---

## What didn't work

**FLB bias direction was backwards initially.** The first implementation applied `biased = true_probs + gamma * (true_probs - uniform)` as the market probabilities. This *increases* the favorite's implied probability, making favorites *overpriced* in the market — which is the opposite of FLB. The test `test_flb_variant_increases_favorite_probability` caught this: it expected `true_fav > market_fav` but got `mean_gap = -0.025`.

Fix: the market should *underestimate* favorites, so compress toward uniform:
```python
biased = true_probs - gamma * (true_probs - uniform)
```
This reduces `market_p` for the favorite below `true_p`, which is the correct FLB direction (and matches H&W's `gamma > 0` interpretation).

**Date range spanned multiple seasons.** The original code used `freq="3D"` starting from August. With 380 matches at 3-day intervals, the dates ran from August 2019 through September 2022 — spanning three seasons. The `_season_from_dates` function would then assign the wrong season.

Fix: use `pd.date_range(start=..., end=..., periods=n)` to spread matches naturally within one August–May window:
```python
dates = pd.date_range(
    start=f"{start_year}-08-01", end=f"{start_year + 1}-05-31",
    periods=n_matches,
)
```
