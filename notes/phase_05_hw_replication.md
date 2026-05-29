# Phase 5: H&W Replication

**Status:** Complete  
**Files created:** `src/replication.py`, `scripts/run_replication.py`  
**Outputs:** `results/tables/hw_replication_normalized.{csv,md}`, `results/tables/hw_comparison_norm_vs_inv.{csv,md}`, `results/tables/hw_bootstrap_cis.csv`

---

## What we built

A replication of Hegarty & Whelan (2024)'s core empirical finding: the favorite-longshot bias regression.

### The regression (H&W equation 13)

```
Y_ij − P^N_ij = α + γ * P^N_ij + ε_ij
```

- `Y_ij` = 1 if outcome j occurred in match i, else 0
- `P^N_ij` = normalized (devigged) implied probability for outcome j
- Estimated by WLS with weights `1/(P*(1-P))`
- SEs clustered at the match level
- `γ > 0` → favorite-longshot bias (favorites are underpriced relative to their true win probability)

**Why WLS?** The variance of `Y_ij − P^N_ij` is not constant — it depends on `P`. A Bernoulli outcome with mean `P` has variance `P(1-P)`. The WLS weight `1/(P*(1-P))` corrects for heteroskedasticity.

**Why cluster at the match level?** For a match with K outcomes, the K rows in the long format are mechanically correlated: `Σ Y_ij = 1` and `Σ P^N_ij = 1`, so `Σ(Y_ij − P^N_ij) = 0` by construction. The residuals within a match sum to zero — they are not independent. Ignoring this inflates the effective sample size, which shrinks SEs and makes the test anti-conservative (too many "significant" results).

### Normalized vs. inverse-odds comparison

The module also runs the *inverse-odds* specification: regress `Y_ij − (1/o_ij)` on `(1/o_ij)` directly, without normalizing. This is H&W's diagnosis of what the earlier literature did wrong.

**Why does inverse-odds bias the test?** The raw inverse-odds `q_ij = 1/o_ij` satisfy `Σ_j q_ij = 1 + overround > 1`. The overround is a systematic positive component of `q_ij` that is unrelated to market efficiency. Because the overround is larger for favorites (bookmakers take proportionally more margin on favorites), it creates a mechanical negative relationship between `Y_ij − q_ij` and `q_ij` — even under full efficiency. This pushes `γ` toward zero (or negative), making it appear that no FLB exists when it does.

### Bootstrap CIs

Bootstrap CIs use match-level resampling. We resample *matches* (keeping all K outcome rows per match), not individual rows. This preserves the within-match correlation structure. With row-level resampling, some bootstrap samples would contain outcome rows from a match without the matching rows, which violates the constraint `Σ Y_ij = 1`.

---

## Results (2015-16 to 2021-22)

### Table 1 equivalent: normalized probabilities

| League | γ | SE | t | p | Matches |
|--------|---|----|---|---|---------|
| **Pooled** | **+0.0459** | 0.0131 | 3.50 | 0.0005*** | 18,538 |
| D1 (Bundesliga) | −0.0228 | 0.0370 | −0.61 | 0.539 | 2,142 |
| E0 (Premier League) | +0.0154 | 0.0307 | 0.50 | 0.617 | 2,660 |
| E1 (Championship) | +0.0877 | 0.0449 | 1.95 | 0.051* | 3,863 |
| E2 (League 1) | +0.2205 | 0.0891 | 2.47 | 0.013** | 1,104 |
| F1 (Ligue 1) | +0.0571 | 0.0371 | 1.54 | 0.123 | 2,279 |
| **I1 (Serie A)** | **+0.1141** | 0.0322 | 3.55 | 0.0004*** | 2,279 |
| N1 (Eredivisie) | +0.0502 | 0.0541 | 0.93 | 0.353 | 612 |
| SC0 (Scottish) | +0.0498 | 0.0411 | 1.21 | 0.226 | 1,319 |
| SP1 (La Liga) | +0.0217 | 0.0348 | 0.62 | 0.532 | 2,280 |

### Table 2 equivalent: normalized vs. inverse-odds

| League | γ (normalized) | p | γ (inv-odds) | p |
|--------|---------------|---|-------------|---|
| Pooled | +0.0459*** | 0.0005 | −0.0030 | 0.814 |
| E0 | +0.0154 | 0.617 | −0.0181 | 0.542 |
| I1 | +0.1141*** | 0.0004 | +0.0618* | 0.043 |

**Key finding:** Pooled inverse-odds γ = −0.003 (p=0.81) vs. normalized γ = +0.046 (p<0.001). The overround suppresses the bias signal to statistical insignificance — exactly H&W's methodological point.

### Bootstrap CIs (95%, match-level resampling, n=1000)

| League | γ | 95% CI |
|--------|---|--------|
| Pooled | +0.046 | [+0.021, +0.070] *** |
| I1 | +0.114 | [+0.051, +0.180] *** |
| E2 | +0.221 | [+0.045, +0.389] *** |
| E1 | +0.088 | [−0.001, +0.179] |
| F1 | +0.057 | [−0.017, +0.129] |
| SC0 | +0.050 | [−0.026, +0.125] |
| N1 | +0.050 | [−0.051, +0.148] |
| SP1 | +0.022 | [−0.044, +0.089] |
| E0 | +0.015 | [−0.047, +0.074] |
| D1 | −0.023 | [−0.097, +0.047] |

### Interpretation

The results are consistent with H&W:
- **Top-flight, liquid markets (E0, D1, SP1):** CIs include zero — no evidence of FLB. These are the most efficient markets.
- **Lower tiers (E2) and certain top-flight leagues (I1):** Clear positive FLB. Favorites are systematically underpriced.
- **Middle tier (E1):** Borderline — CI just barely includes zero.

This matches H&W's finding that bias is heterogeneous across leagues and is not simply a function of league prestige.

---

## Limitations vs. H&W

H&W use 22 European leagues from 2011/12–2021/22 (11 seasons, ~84,000 matches). We have:
- 9 leagues
- 2015/16–2021/22 (7 seasons for most leagues, fewer for E2/N1)
- ~18,500 matches in the H&W window

This means:
1. We cannot directly compare per-league γ estimates to H&W's Table 1 values (different league sets, fewer seasons)
2. We cannot replicate their pooled γ magnitude exactly
3. The direction and significance pattern should match, and it does

A full replication would require downloading additional leagues and the 2011-12 through 2014-15 seasons.

---

## What didn't work

**`tabulate` missing.** The script used `DataFrame.to_markdown()` which requires the `tabulate` package. It wasn't in the base install. Added to `pyproject.toml` and installed. The error was clear (`ImportError: Import tabulate failed`) so this was a quick fix.

No other issues — the regression logic itself worked correctly on the first attempt.
