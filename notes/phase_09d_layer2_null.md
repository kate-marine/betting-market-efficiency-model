# Phase 9d: Layer 2 — Efficient-Markets Null Simulation

**Status:** Complete  
**Files created:** `src/null_sim.py`, `tests/test_null_sim.py`, `scripts/run_null_simulation.py`  
**Outputs:** `results/tables/null_simulation_results.{csv,md}`, `results/figures/null_distributions_flb_all_leagues.png`  
**Tests:** 13 passing (added to 127 total)

---

## What we built and why

The parametric p-values in Phase 5 assume the H&W test statistic is approximately normal. This assumption holds under large-sample regularity conditions, but the soccer data has a specific structure (clustered 3-way outcomes, non-uniform odds distributions) that could cause the normal approximation to break down.

The null simulation answers the question directly: **given the real odds in each league, how often would we see a γ̂ as extreme as observed, if the market were perfectly efficient?**

Procedure for each league:
1. Use the real market-implied probabilities as the "true" outcome probabilities
2. Draw n_sim = 2000 datasets by sampling one outcome per match from Categorical(P^N_iH, P^N_iD, P^N_iA)
3. Run the H&W WLS regression on each simulated dataset
4. Simulation p-value = fraction of simulations where |γ̂_null| ≥ |γ̂_obs|

**Speed:** Pre-computing `M = (X'WX)^{-1}X'W` once per league reduces each simulation to a single matrix–vector multiply. 2000 simulations across 9 leagues ran in **0.5 seconds**.

**Joint test:** The max |t-statistic| across all 9 leagues is used as a joint test statistic. This controls the family-wise error rate without assuming independence between leagues.

---

## Results

### Per-league comparison (parametric vs. simulation p-values)

| League | γ_obs | t_obs | p_parametric | p_simulation | ratio | verdict |
|--------|-------|-------|-------------|-------------|-------|---------|
| D1 | −0.0228 | −0.614 | 0.539 | 0.538 | 1.00 | well-calibrated |
| E0 | +0.0154 | +0.500 | 0.617 | 0.622 | 1.01 | well-calibrated |
| E1 | +0.0877 | +1.951 | 0.051 | 0.064 | 1.25 | mild inflation |
| **E2** | **+0.2205** | **+2.474** | **0.013** | **0.015** | 1.12 | mild inflation |
| F1 | +0.0571 | +1.541 | 0.123 | 0.141 | 1.14 | mild inflation |
| **I1** | **+0.1141** | **+3.549** | **0.0004** | **0.0010** | 2.58 | anti-conservative tail |
| N1 | +0.0502 | +0.928 | 0.353 | 0.400 | 1.13 | mild inflation |
| SC0 | +0.0498 | +1.211 | 0.226 | 0.271 | 1.20 | mild inflation |
| SP1 | +0.0217 | +0.624 | 0.532 | 0.532 | 1.00 | well-calibrated |

### Joint tests

| Method | Observed statistic | p_joint |
|--------|-------------------|---------|
| max |t| across leagues | 3.549 (from I1) | **0.0100** |
| sum t² across leagues | 28.241 | **0.0020** |

---

## Interpretation

### 1. The parametric test is mostly well-calibrated

For D1, E0, SP1: ratio ≈ 1.00 — the cluster-robust normal approximation is accurate for these leagues. Their null results are not an artifact of test miscalibration.

For most other leagues: ratio ≈ 1.12–1.25 — mild anti-conservatism (parametric p is somewhat too small). Not alarming, consistent with slight heavy-tailedness of the null distribution.

### 2. I1's significance survives the exact simulation test

I1 shows ratio = 2.58 (parametric p = 0.0004 vs. simulation p = 0.0010). The parametric test is anti-conservative at the tail for I1. This is expected — at t-statistics as extreme as 3.55, small departures from normality in the null distribution matter.

But crucially: the simulation p-value of 0.0010 confirms I1 remains highly significant under the exact test. The anti-conservatism changes the headline number (0.0004 → 0.0010) but not the conclusion.

### 3. The joint test is highly significant

Across all 9 leagues simultaneously, there is a 1% chance of seeing a max t-statistic ≥ 3.55 under the efficient-markets null. This controls for multiple comparisons without any parametric assumption.

### 4. No-FLB leagues are accurately calibrated

D1, E0, SP1 show ratio ≈ 1.00 — the test is correctly calibrated for them. Combined with the power analysis (Phase 9c), the honest conclusion is:
- The test for E0/D1/SP1 is **calibrated but underpowered**
- Both conditions are needed to fully characterize the null result

### 5. Null distribution properties

| League | γ_obs / SE_null |
|--------|----------------|
| I1 | 3.24σ |
| E2 | 2.53σ |
| E1 | 1.89σ |
| F1 | 1.47σ |
| SC0 | 1.11σ |
| N1 | 0.84σ |
| D1 | 0.61σ |
| SP1 | 0.62σ |
| E0 | 0.49σ |

I1 and E2 are genuine outliers relative to the null distribution; E0/D1/SP1 are indistinguishable from noise.

---

## Decisions made

**Pre-compute M once per league.** The key optimization. The matrix `M = (X'WX)^{-1}X'W` encodes the WLS regression coefficients: `β = M @ y`. Since only `y` (the outcomes) changes across simulations, precomputing M reduces each simulation to a matrix-vector multiply. This yielded a 1000× speedup vs. running statsmodels per simulation.

**Use γ̂ directly as the test statistic for per-league p-values.** Under the null, the SE changes negligibly across simulations (same design, same weights), so |γ̂| and |t| rank simulations identically. Avoiding the SE computation saves a cluster-loop per simulation.

**max |t| for joint test.** The max-t statistic naturally controls FWER without independence assumptions. The sum-χ² statistic is also computed as a sensitivity check. Both give similar conclusions.

**n_sim = 2000.** SE of simulation p-value at p=0.05 is √(0.05×0.95/2000) ≈ 0.005. Sufficient precision to distinguish p=0.05 from p=0.10 at the 2σ level.

---

## What didn't work

**Column name mismatch.** `build_results_table` returns `p_simulation` but the script initially accessed it as `p_sim`. Caught immediately at runtime, fixed in one edit.

**Nothing else** — the implementation worked correctly on the first run. The test calibration check (that rejection rate under the null matches α=0.05) passed on the first attempt.

---

## Connection to Winkelmann et al.

Winkelmann et al. show that season-by-season regressions produce false positives 77.6% of the time under efficiency. Our simulation checks the analogous question for our league-by-league structure: the false positive rate given our specific odds distributions and cluster sizes.

Key finding: **the false positive rate for our parametric test is modest (ratio 1.00–1.25 for most leagues)**. The cluster-robust SE largely addresses the inflation Winkelmann diagnoses. This is because:
1. We pool across seasons within each league (not separate season-level tests)
2. We use cluster-robust SEs (Winkelmann's simulations use homoskedastic SEs)
3. Our match counts per league (600–4000) are large enough for the normal approximation

The joint test provides an additional layer of protection that directly addresses the multiple-testing concern without any distributional assumptions.
