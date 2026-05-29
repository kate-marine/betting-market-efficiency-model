# Phase 8: Calibration Evaluation

**Status:** Complete  
**Files created:** `src/calibration.py`, `tests/test_calibration.py`, `scripts/run_calibration.py`  
**Outputs:** `results/tables/calibration_pooled.{csv,md}`, `results/tables/calibration_by_league.csv`, `results/tables/reliability_diagram_data.csv`, `results/figures/reliability_diagrams_soccer_*.png`  
**Tests:** 16 passing (added to 91 total)

---

## What we built

Calibration evaluation comparing the LightGBM model against the betting market across three metrics:

| Metric | What it measures |
|--------|-----------------|
| Brier score | Mean squared error between predicted probs and outcomes (lower = better; 0 = perfect, 2 = worst) |
| Log-loss | Negative mean log-probability of the true outcome (lower = better; information-theoretic measure) |
| ECE | Expected calibration error — weighted mean gap between predicted and observed frequency within bins |

Plus **reliability diagrams** — visual calibration plots showing mean predicted probability vs. observed frequency per bin. Perfect calibration = points on the diagonal.

All pooled metrics have **bootstrap CIs** (match-level resampling, n=1000). Per-league metrics are point estimates only (bootstrapping per-league would be ~10× slower with marginal gain given sample sizes).

---

## Results

### Pooled metrics (2016-17 to 2021-22, 15,856 matches)

| Estimator | Brier | 95% CI | ECE | 95% CI | Log-loss |
|-----------|-------|--------|-----|--------|---------|
| **Market** | **0.1939** | [0.1923, 0.1956] | **0.0075** | [0.0050, 0.0123] | **0.9776** |
| Model | 0.2134 | [0.2113, 0.2157] | 0.0738 | [0.0689, 0.0787] | 1.0846 |

**The CIs for Brier score do not overlap** — the market's advantage is statistically unambiguous. The ECE gap is even more striking: market ECE of 0.0075 vs. model ECE of 0.0738 (roughly 10× better). The market's probabilities closely track actual outcome frequencies; the model is meaningfully miscalibrated.

### Per-league Brier score

| League | Market | Model | Gap |
|--------|--------|-------|-----|
| I1 (Serie A) | **0.183** | 0.206 | −0.024 |
| N1 (Eredivisie) | **0.184** | 0.198 | −0.014 |
| E0 (Premier League) | **0.186** | 0.206 | −0.021 |
| SC0 (Scottish) | **0.187** | 0.210 | −0.023 |
| SP1 (La Liga) | **0.191** | 0.214 | −0.022 |
| D1 (Bundesliga) | **0.196** | 0.211 | −0.015 |
| E2 (League 1) | **0.204** | 0.219 | −0.015 |
| F1 (Ligue 1) | **0.196** | 0.214 | −0.019 |
| E1 (Championship) | **0.207** | 0.225 | −0.018 |

Market dominates in every single league. The gap is slightly smaller in lower tiers (E2, N1), consistent with Phase 5's FLB finding and Phase 7's conformal set-size result.

### Per-league ECE

| League | Market | Model |
|--------|--------|-------|
| E1 | **0.009** | 0.075 |
| E0 | **0.014** | 0.072 |
| D1 | **0.015** | 0.074 |
| E2 | **0.019** | 0.040 |
| N1 | **0.023** | 0.039 |

The model's ECE of ~0.07–0.09 across top-flight leagues means its predictions are systematically off by about 7–9 percentage points per bin. The market's ECE of ~0.01 is remarkably good — it prices outcomes with close to actuarial accuracy.

---

## Decisions made

**Three separate metrics rather than one.** Brier score, log-loss, and ECE capture different failure modes. A predictor can have low Brier score but be poorly calibrated (if it correctly identifies who wins but is overconfident). ECE specifically measures calibration — whether the predicted probabilities match empirical frequencies. Log-loss is the strictest: it penalizes overconfident wrong predictions most harshly.

**Equal-width bins for ECE.** The standard choice (10 bins of width 0.1). Known limitation: bins with few points give unreliable estimates, and the choice of bin count affects the absolute ECE value. We use 10 bins consistently for all comparisons.

**Bootstrap at match level, not row level.** The long format has 3 rows per match. Row-level resampling would produce bootstrap samples with broken within-match structure (some matches missing one or two outcomes). Match-level resampling keeps all 3 rows per sampled match.

**Per-league metrics without bootstrap.** Bootstrapping all 9 leagues × 2 estimators would require 18,000 bootstrap iterations and take ~15–20 minutes. The point estimates are informative enough for the per-league comparison, and the pooled CI establishes that the overall market advantage is not sampling noise.

---

## What didn't work

**`log_loss_multiclass` shape error.** The original implementation tried to build a 2D numpy array by stacking the per-outcome series from the DataFrame — but the three outcome groups (H, D, A) have different sizes (437 home wins ≠ 312 away wins ≠ 251 draws per 1000 matches), so numpy couldn't form the array. Fixed by iterating over outcomes and assigning correct-class probabilities by index mask:

```python
correct_p = np.zeros(len(df))
for outcome, col in pred_cols.items():
    mask = df["result"] == outcome
    correct_p[mask.values] = df.loc[mask, col].values
```

**`n_boot=0` → empty array in `np.percentile`.** Passing `n_boot=0` to `calibration_table` (to skip bootstrapping per-league) caused `bootstrap_metrics` to try `np.percentile([], 2.5)` which raises `IndexError` in numpy 2.x. Fixed by guarding: if `bs_boots` is empty, return NaN for CI bounds without calling `np.percentile`.

**ECE test with "better calibration" failed.** The test assumed `ece(good_pred, obs) < ece(bad_pred, obs)` where `good_pred ≈ true_p` and `bad_pred = 0.5`. But ECE is sensitive to the marginal distribution: if the true frequency is ~50% overall, a constant 0.5 predictor can have *lower* ECE than a well-calibrated variable predictor (because ECE is computed within bins, and within the 0.5 bin the constant predictor is perfectly calibrated). Fixed by using a scenario with unambiguous miscalibration: `good_pred = 0.7` vs `bad_pred = 0.3` when true frequency is 70%.

---

## Interpretation for the writeup

**The market is remarkably well-calibrated** (ECE ≈ 0.008 pooled, ≈ 0.01 per league). Its predicted probabilities track true outcome frequencies with high accuracy. This is the strongest evidence that betting markets price outcomes accurately and that the FLB detected in Phase 5 is a subtle, league-level effect rather than systematic gross mispricing.

**The model is moderately miscalibrated** (ECE ≈ 0.07). Its raw LightGBM probabilities are off by about 7 percentage points per bin on average. This is expected — LightGBM's outputs are not naturally calibrated probabilities, and our feature set (Elo + 5-match form + rest days) captures only a fraction of the information priced into the market.

**The three-phase evidence is consistent:**
- Phase 5: the market has a positive FLB (γ = 0.046), but the bias is subtle
- Phase 7: the market achieves the same 90% coverage with ~0.2 fewer outcomes per prediction set
- Phase 8: the market has ~10× lower ECE

All three point to the same conclusion: betting markets are highly informationally efficient, with room for improvement primarily in lower-tier leagues.
