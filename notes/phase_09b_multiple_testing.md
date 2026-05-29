# Phase 9b: Multiple-Testing Correction for Per-League H&W Table

**Status:** Complete  
**Files modified:** `src/replication.py` (added `add_multiple_testing_correction()`), `scripts/run_replication.py`  
**Outputs:** Updated `results/tables/hw_replication_normalized.{csv,md}` with corrected p-values

---

## What we built and why

The Phase 5 replication ran 9 simultaneous league-level tests and reported per-test p-values without any correction. The entire framing of this project — directly responding to Winkelmann et al. (2024) — is about being more honest with uncertainty. Running uncorrected multiple tests while citing Winkelmann's multiple-testing critique would be a glaring inconsistency.

Two corrections added:
- **Bonferroni**: divide the significance threshold by 9, equivalently multiply each p-value by 9. Controls FWER (family-wise error rate) — probability of any false positive ≤ α. Most conservative.
- **Benjamini-Hochberg (BH-FDR)**: rank-based correction. Controls FDR (expected fraction of discoveries that are false positives). Less conservative than Bonferroni, appropriate when we expect some true effects.

Both applied only to the 9 per-league rows; the pooled row is a single test and is not corrected.

---

## Results

| League | γ | p (uncorrected) | p (Bonferroni) | sig | p (BH) | sig |
|--------|---|-----------------|---------------|-----|---------|-----|
| D1 | −0.0228 | 0.539 | 1.000 | | 0.606 | |
| E0 | +0.0154 | 0.617 | 1.000 | | 0.617 | |
| E1 | +0.0877 | 0.051 | 0.459 | | 0.153 | |
| **E2** | **+0.2205** | **0.013** | 0.120 | | **0.060** | * |
| F1 | +0.0571 | 0.123 | 1.000 | | 0.277 | |
| **I1** | **+0.1141** | **0.0004** | **0.0035** | *** | **0.0035** | *** |
| N1 | +0.0502 | 0.353 | 1.000 | | 0.530 | |
| SC0 | +0.0498 | 0.226 | 1.000 | | 0.407 | |
| SP1 | +0.0217 | 0.532 | 1.000 | | 0.606 | |

**After Bonferroni correction:** Only I1 (Serie A) remains significant at any conventional level.

**After BH-FDR:** I1 survives clearly (q=0.0035***). E2 is borderline at α=0.10 (q=0.060). All others lose significance.

**E1's apparent significance (p=0.051) vanishes under both corrections** — exactly illustrating Winkelmann's point that borderline per-test findings in multi-league analyses are likely false positives.

---

## Implications for the writeup

The corrected table strengthens the paper's intellectual coherence. The honest conclusions:

1. **I1 (Serie A)** shows robust FLB: significant after Bonferroni, replicated under BH. γ = +0.114, bootstrap CI [+0.051, +0.180].
2. **E2 (League 1)** shows suggestive FLB: survives BH-FDR at q=0.06, fails Bonferroni. γ = +0.221, but with a wide CI ([+0.045, +0.389]) reflecting only 1,104 matches across 2–3 seasons.
3. **All other leagues:** no statistically reliable FLB after correction. E1's borderline uncorrected p=0.051 is exactly the kind of result Winkelmann warns about.
4. **Pooled:** γ = +0.046, p=0.0005 (this is a single test, no correction needed). The pooled finding is robust.

This is a more cautious conclusion than the uncorrected table implied, but it's more honest and more defensible. The specific pattern — I1 and E2 showing bias, top-flight English and German leagues not — is consistent with the calibration and conformal results and tells a coherent story.

---

## What didn't work

No bugs. `statsmodels.stats.multitest.multipletests` with `method='fdr_bh'` returned correct BH q-values. The only design decision was whether to apply correction to the pooled row — the answer is no (it's a single test), and the function correctly excludes it.

---

## Connection to Winkelmann et al.

Winkelmann et al. (2024) show via Monte Carlo that season-by-season analysis of 14 seasons produces at least one false-positive season 77.6% of the time under full efficiency. Our league-by-league analysis is the cross-sectional analogue. The correction demonstrates their point concretely: E1, which looked borderline significant, is plausibly a false positive. I1, which survives Bonferroni, is not.
