# Phase 9a: Model Calibration Ablation

**Status:** Complete  
**Files created:** `src/postcal.py`, `tests/test_postcal.py`, `scripts/run_calibration_ablation.py`  
**Outputs:** `results/tables/calibration_ablation.{csv,md}`, `results/figures/reliability_diagrams_calibration_ablation_soccer.png`  
**Tests:** 8 passing (added to 99 total)

---

## What we built and why

Phase 8 compared raw LightGBM probabilities (ECE 0.074) against the betting market (ECE 0.008) and concluded the market was ~10× better calibrated. This comparison was unfair: the market has been price-discovered by millions of bets over time; the model outputs are raw LightGBM probabilities, which are known to be poorly calibrated out of the box.

Any reviewer will immediately ask: *did you try post-hoc calibration?* This phase answers that.

Two calibrators, both one-vs-rest with renormalization:

- **Isotonic regression** (`IsotonicRegression(out_of_bounds="clip")`): non-parametric, monotone mapping from raw probability to calibrated probability. More flexible; fits the data exactly over binned regions.
- **Platt scaling** (logistic regression on `logit(raw_p)`): parametric (2 degrees of freedom per outcome). More stable with small calibration sets.

Walk-forward: for test season T, fit calibrators on season T-1 predictions. Same holdout used for conformal thresholds in Phase 7. No data from test season T ever touches the calibrator.

---

## Results

### Pooled metrics (5 evaluable seasons: 2017-18 to 2021-22, n=13,265 matches)

| Estimator | Brier | 95% CI | ECE | 95% CI | Log-loss |
|-----------|-------|--------|-----|--------|---------|
| Raw model | 0.2122 | [0.2098, 0.2146] | 0.0685 | [0.0632, 0.0740] | 1.0744 |
| Isotonic | 0.2052 | [0.2037, 0.2067] | 0.0124 | [0.0090, 0.0168] | 1.0508 |
| Platt | 0.2049 | [0.2033, 0.2064] | 0.0110 | [0.0075, 0.0158] | 1.0254 |
| **Market** | **0.1953** | [0.1935, 0.1972] | **0.0058** | [0.0041, 0.0105] | **0.9836** |

### ECE gap decomposition

| | ECE | Gap to raw | % of total gap |
|-|-----|-----------|---------------|
| Raw model | 0.0685 | — | — |
| Isotonic | 0.0124 | −0.0561 | **89.5% closed** |
| Market | 0.0058 | −0.0627 | 100% (baseline) |

Isotonic calibration closes 89% of the ECE gap. The residual 10.5% (ECE 0.0124 vs. 0.0058) is the market's genuine information advantage even against a calibrated model.

### Conformal set sizes after calibration (90% nominal)

| Estimator | Mean set size |
|-----------|-------------|
| Raw model | 2.560 |
| Isotonic | 2.490 |
| Platt | 2.538 |
| Market | 2.350 |

Calibration shrinks the model's set size by 0.07 (isotonic). The market–model gap narrows from 0.21 to 0.14, but persists.

---

## The revised story for the writeup

The comparison changes from "market is 10× better calibrated" to something more accurate and more honest:

> Raw LightGBM outputs are, as expected, poorly calibrated (ECE 0.07). After standard isotonic post-hoc calibration — a two-line fix that any practitioner would apply — the ECE gap closes to 2× (0.012 vs. 0.006). The residual calibration gap and the Brier score gap (0.205 vs. 0.195, non-overlapping CIs) represent the market's genuine information advantage: prices aggregated from thousands of informed bettors capture injury news, lineup information, and betting-flow signals that our team-level features do not.

This is both more honest and more interesting than the uncalibrated comparison. It separates two distinct claims:
1. Raw ML probabilities are miscalibrated — obvious and fixable.
2. Even properly calibrated ML beats a well-informed market — hard, and our model doesn't do it.

---

## Decisions made

**Isotonic over Dirichlet calibration.** Dirichlet calibration (Kull et al. 2019) is the theoretically ideal multiclass calibrator — it handles the joint probability structure. But it requires more code and is harder to validate. One-vs-rest isotonic with renormalization is the standard practice and good enough to answer the question.

**Walk-forward calibration holdout = T-1.** Using any in-sample data would overfit the calibrator and give optimistically low ECE estimates. Using T-1 means our calibration eval and conformal eval use the same holdout — internally consistent.

**Renormalization after independent calibration.** Each outcome's calibrator is fit independently; the three resulting probabilities don't necessarily sum to 1. Dividing by their sum (renormalization) is the standard fix. This breaks the strict multiclass structure but works well in practice.

---

## What didn't work

No bugs — the implementation worked on the first attempt. The test `test_isotonic_reduces_ece_on_overconfident_model` required `n_per_season=800` to be reliable; with smaller samples (200) the improvement was noisy and the test was flaky.

---

## ECE trend by season (raw model, before calibration)

Note from the data: raw model ECE improved season-by-season as training data grew:
- 2016-17: ECE 0.101 (3 training seasons)
- 2021-22: ECE 0.045 (8 training seasons)

This means some of the raw ECE gap was training-data-size, not inherent LightGBM miscalibration. Even so, isotonic calibration brings all seasons to ~0.01 ECE — the calibration benefit is robust.
