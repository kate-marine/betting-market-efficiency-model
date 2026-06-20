# Conformal Prediction Meets Sports Betting Market Efficiency

Betting markets are often claimed to be efficient — meaning prices accurately reflect true outcome probabilities. One known violation is **favorite-longshot bias (FLB)**: favorites are underpriced relative to their true win probability, while longshots are overpriced. The existing literature documents this bias, but Hegarty & Whelan (2024) show the standard test is methodologically flawed, and Winkelmann et al. (2024) show the per-league, per-season analyses that populate this literature have severe multiple-testing and power problems.

This project replicates H&W's FLB regression across 9 European soccer leagues, applies Winkelmann's critique to its own results (Bonferroni correction, power analysis, null simulation), and adds a Layer 1 comparison of market probabilities against a conformal-wrapped LightGBM model. Every claim has a bootstrap CI; every null result has a power analysis; the test calibration is verified by direct simulation.

---

## Key findings

### 1. FLB exists in the pooled data, but most league-level findings are fragile

Pooled across **18,538 matches**: γ = +0.046 (95% CI [+0.021, +0.070], p < 0.001). Favorites are systematically underpriced; returns on favorites exceed returns on longshots.

After Bonferroni correction for 9 simultaneous league-level tests, **only Serie A (I1, γ = +0.114)** remains clearly significant. League 1 England (E2, γ = +0.221) is borderline under BH-FDR (q = 0.06). The English Championship's apparent p = 0.051 vanishes under any reasonable correction — exactly the false-positive pattern Winkelmann warns about.

The inverse-odds estimator used in much of the earlier literature gives a pooled γ of −0.003 (p = 0.81), absorbing the FLB signal into the overround. Normalized probabilities recover the correct sign and magnitude.

*Robustness:* A direct null simulation (2,000 draws per league from market-implied probabilities) confirms the cluster-robust normal approximation is accurate for 8 of 9 leagues (simulation-to-parametric p-value ratio 1.0–1.25). The joint test across all 9 leagues yields p = 0.010, confirming the full dataset is collectively inconsistent with market efficiency without any distributional assumption.

### 2. "No FLB in the Premier League" is a power statement, not an efficiency result

Premier League (E0): MDE = **0.086** at 80% power. The pooled effect (0.046) is smaller than this threshold — even if the Premier League has FLB at exactly the pooled level, we'd detect it only **32% of the time** with this sample. Bundesliga (D1) and La Liga (SP1) face similar constraints (MDE ≈ 0.10).

The honest claim is: "we can rule out FLB ≥ 0.086 in the Premier League; we cannot distinguish between no FLB and FLB at or below the pooled level." This is Winkelmann's underpowering concern applied to our own analysis.

### 3. The betting market outperforms a calibrated ML model — and the gap is real, not an artifact

A LightGBM model on Elo ratings, 5-match rolling form, and rest days achieves Brier = **0.213 raw**, **0.205 after isotonic calibration** (walk-forward CV, 2016–2022). The market achieves **0.195**, with non-overlapping bootstrap CIs. ECE: market 0.006 vs. calibrated model 0.012.

Raw LightGBM outputs are predictably miscalibrated — isotonic post-hoc calibration closes 89% of the ECE gap. The remaining gap and the full Brier difference represent genuine information the market has that team-level features don't: current injuries, lineup decisions, and betting flow. At 90% conformal coverage, market prediction sets are 0.14–0.21 outcomes smaller than the calibrated model's, with the smallest gap in League 1 (E2) — consistent with E2 showing the strongest FLB.

---

## Data

**Soccer:** [football-data.co.uk](https://www.football-data.co.uk/data.php) — one CSV per league per season. 9 European leagues (D1, E0, E1, E2, F1, I1, N1, SC0, SP1), seasons 2013/14–2025/26. H&W replication window: 2015/16–2021/22.

**Tennis:** [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php) — loaders and synthetic validation complete; real data analysis planned.

Raw data is gitignored. See [Adding real data](#adding-real-data) below.

---

## Methods

| Component | Approach |
|-----------|---------|
| **Devigging** | Four methods: normalized (H&W's recommended), additive, power, Shin (1992). All four computed per match; normalized used as the primary estimator. |
| **H&W replication** | WLS regression of `Y_ij − P^N_ij` on `P^N_ij`, weights `1/(P(1−P))`. Match-level cluster-robust SEs (H/D/A outcomes within a match sum to zero mechanically). Bonferroni and BH-FDR correction for 9 simultaneous league tests. |
| **ML model** | LightGBM multiclass (H/D/A). Walk-forward expanding-window CV — never shuffled. Post-hoc isotonic calibration fitted on season T−1, applied to season T. |
| **Conformal wrapper** | Split conformal (marginal guarantee) and Mondrian by league (group-conditional guarantee). Same walk-forward holdout as ML calibration. |
| **Calibration** | Brier score, log-loss, ECE. 1,000-bootstrap CIs via match-level resampling. Reliability diagrams saved as PNG + CSV. |
| **Power analysis** | MDE = (z_{α/2} + z_{1−β}) × SE_cluster. Simulation validation for E0 (n=200, empirical power matches analytical to within 2.5pp at the MDE). |
| **Null simulation** | 2,000 draws per league from Categorical(P^N). Pre-computed projection matrix makes each draw a single matrix–vector multiply; 2,000 simulations × 9 leagues runs in ~0.5s. |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                          # 127 tests, ~10 seconds
```

## Smoke test (no real data required)

Generates synthetic data in the real file formats, runs the full pipeline, verifies all invariants:

```bash
python scripts/smoke_test.py
```

## Adding real data

Drop soccer CSVs and tennis XLSXs anywhere under `data/raw/`. The loaders recover the league from the `Div` column (or folder name) and the season from the filename or `Date` column:

```bash
python -c "
from src.loaders import load_soccer, load_tennis
load_soccer('data/raw', 'data/processed', recursive=False)
load_tennis('data/raw', 'data/processed')
"
```

## Running the full analysis

Each script reads Parquet written by earlier steps. Approximate runtimes on a 2020 MacBook Pro:

```bash
python scripts/run_replication.py          # ~1 min  — H&W regression + multiple-testing correction
python scripts/run_model.py                # ~1 min  — LightGBM walk-forward CV
python scripts/run_conformal.py            # ~5 sec  — conformal prediction set sizes
python scripts/run_calibration.py          # ~2 min  — Brier, ECE, reliability diagrams (n_boot=1000)
python scripts/run_calibration_ablation.py # ~3 min  — post-hoc calibration comparison
python scripts/run_power_analysis.py       # ~5 min  — MDE table + simulation validation
python scripts/run_null_simulation.py      # ~5 sec  — efficient-markets null simulation
```

---

## Project layout

```
src/
  devig.py        — four devigging methods
  synth.py        — synthetic data generator (efficient + FLB variants)
  loaders.py      — soccer and tennis loaders → Parquet
  replication.py  — H&W WLS regression, multiple-testing correction
  features.py     — Elo, rolling form, rest days (strict chronological order)
  model.py        — LightGBM walk-forward CV
  conformal.py    — split conformal and Mondrian conformal
  calibration.py  — Brier, ECE, reliability diagram data
  postcal.py      — isotonic and Platt post-hoc calibration
  power.py        — analytical MDE, simulation-based power curves
  null_sim.py     — efficient-markets null simulation

tests/            — 127 tests; every statistical function validated against
                    synthetic data with known ground truth

scripts/          — one script per analysis step
notes/            — per-phase log: decisions made, bugs found, what failed

results/
  tables/         — CSV + Markdown for every reported number
  figures/        — reliability diagrams, power curves, null distributions
```

---

## References

- **Hegarty & Whelan (2024)**, *Sports Economics Review* 8.100042. Shows the standard inverse-odds test for market efficiency is biased toward accepting the null; recommends normalized probabilities; finds FLB in soccer and tennis.
- **Winkelmann et al. (2024)**, *Journal of Sports Economics* 25(1). Shows via Monte Carlo that season-by-season betting analyses routinely produce false positives (77.6% chance of at least one spurious significant season over 14 seasons). This project applies their critique to its own league-level tests.
