# Phase 9c: Power Analysis for Per-League H&W Regressions

**Status:** Complete  
**Files created:** `src/power.py`, `tests/test_power.py`, `scripts/run_power_analysis.py`  
**Outputs:** `results/tables/power_analysis_mde_table.{csv,md}`, `results/tables/power_simulation_e0.csv`, `results/figures/power_curves_flb_by_league.png`, `results/figures/power_validation_e0_simulation.png`  
**Tests:** 15 passing (added to 114 total)

---

## What we built and why

Phase 5 reported "no detectable FLB" for E0, D1, SP1. This is an under-claim: it confounds two distinct situations that are currently indistinguishable in the data:
1. These leagues genuinely have no FLB (market efficiency)
2. These leagues have FLB, but we don't have enough data to detect it

The power analysis quantifies the boundary between these: the minimum detectable effect (MDE), i.e., the smallest γ we can detect at 80% power given each league's sample size.

**Two approaches:**
- **Analytical MDE** (primary result): `MDE = (z_{α/2} + z_{1-β}) × SE_cluster` — exact formula for two-sided normal test, valid for large samples
- **Simulation validation** (sanity check): generate synthetic outcomes with known γ from real E0 odds, run the full regression, check empirical power vs. analytical

---

## Results

### MDE table (α=0.05, 80% power)

| League | γ_obs | SE | n_matches | MDE | power@obs | power@pooled | Powered? |
|--------|-------|----|-----------|-----|-----------|-------------|----------|
| **Pooled** | +0.0459 | 0.0131 | 18,538 | **0.0368** | **0.938** | **0.938** | **YES** |
| D1 | −0.0228 | 0.0370 | 2,142 | 0.1038 | 0.094 | 0.236 | no |
| E0 | +0.0154 | 0.0307 | 2,660 | 0.0861 | 0.079 | 0.321 | no |
| E1 | +0.0877 | 0.0449 | 3,863 | 0.1259 | 0.497 | 0.175 | no |
| E2 | +0.2205 | 0.0891 | 1,104 | 0.2497 | 0.696 | 0.081 | no |
| F1 | +0.0571 | 0.0371 | 2,279 | 0.1038 | 0.338 | 0.236 | no |
| **I1** | **+0.1141** | 0.0322 | 2,279 | 0.0901 | **0.944** | 0.297 | **YES** |
| N1 | +0.0502 | 0.0541 | 612 | 0.1516 | 0.153 | 0.136 | no |
| SC0 | +0.0498 | 0.0411 | 1,319 | 0.1153 | 0.228 | 0.200 | no |
| SP1 | +0.0217 | 0.0348 | 2,280 | 0.0974 | 0.096 | 0.262 | no |

**power@pooled** = probability of detecting FLB if the true effect were equal to the pooled estimate (γ=0.046). All individual leagues except Pooled and I1 have power below 40%.

### Simulation validation (E0, n_sim=200)

| γ | Analytical power | Simulation power | Difference |
|---|-----------------|-----------------|-----------|
| 0.000 | 0.050 | 0.055 | +0.005 |
| 0.043 | 0.288 | 0.295 | +0.007 |
| **0.086 (MDE)** | **0.800** | **0.825** | **+0.025** |
| 0.129 | 0.988 | 0.990 | +0.002 |
| 0.172 | 1.000 | 1.000 | 0.000 |

Analytical formula is validated: at the MDE, empirical power is 82.5% vs. analytical 80.0%. The +2.5pp difference is within simulation noise (SE ≈ 2.7pp). The formula is sufficiently accurate for the primary result.

---

## The honest claims this enables

**Before power analysis (Phase 5):** "E0, D1, SP1 show no detectable FLB."

**After power analysis (Phase 9c):** 
- "For E0 (2,660 matches), we can rule out FLB ≥ 0.086 at 80% power. We cannot distinguish between no FLB and FLB in the range 0–0.086, which includes the pooled estimate (0.046)."
- "For D1 and SP1, the detectable threshold (MDE ≈ 0.10) is similarly above the pooled estimate."
- "The null result for these leagues is entirely consistent with the league having FLB at the pooled level — we simply don't have enough data to see it."

This is Winkelmann's underpowering concern applied directly to our own analysis. We are not immune to the problem we're critiquing.

---

## Decisions made

**Analytical formula as the primary result.** The simulation validates it and is more transparent about the derivation than a pure black-box simulation.

**SE from the actual regression (cluster-robust).** The MDE uses the same SE that comes out of the fitted regression — it automatically accounts for the cluster structure and the actual odds distribution in each league.

**E0 as the validation target.** Most policy-relevant: the Premier League is the most-studied market and the "no FLB" conclusion there matters most for the efficiency literature. Validating the formula for E0 covers the primary use case.

**200 simulations per gamma value.** SE of a proportion ≈ 2.7pp with n=200, giving about ±5pp 95% CI. Sufficient to confirm the analytical formula isn't badly wrong.

---

## What didn't work

**`test_power_at_3_se_is_high` was wrong.** I wrote the test expecting power > 95% at 3×SE, but the analytical formula gives 85% at 3×SE. Power reaches 95% at ≈3.6×SE. Fixed by splitting into two separate tests: ">80% at 3×SE" and ">95% at 4×SE."

**Numpy dtype coercion in simulation tests.** The tests built a `np.array` from a list of `(float, float, string)` tuples — numpy coerced everything to string, causing the WLS function to fail on string arithmetic. Fixed by keeping numeric arrays and cluster ID arrays separate (using `np.concatenate` on each type separately).

---

## Connection to Winkelmann et al.

Winkelmann et al. (2024) show that 14-season analyses of single leagues produce false positives 77.6% of the time. Our per-league analysis compounds this: not only can we produce false positives (Phase 9b), we can also produce false negatives for leagues that genuinely have FLB but too small a sample to detect it. The power analysis quantifies the false-negative risk exactly.

The three-way result is now internally coherent:
- **False positives controlled:** multiple-testing correction (Phase 9b) removes E1's borderline significance
- **True positives identified:** I1 and pooled survive correction with adequate power
- **False negatives quantified:** E0/D1/SP1 are underpowered; their null results are not evidence of efficiency

This is the kind of honest uncertainty quantification the project set out to produce.
