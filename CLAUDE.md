# Calibrated Claims: Conformal Prediction Meets Sports Betting Market Efficiency

## What this project is

A two-layer empirical study of sports betting market efficiency using modern
uncertainty quantification. Portfolio piece for analytics internship
applications (target: DraftKings and similar), with a stretch goal of being
workshop-paper publishable.

**Layer 1 (prediction-level):** Train an ML model per sport, wrap in conformal
prediction, benchmark against sportsbook market-implied probabilities.

**Layer 2 (claim-level):** Replicate Hegarty & Whelan (2024)'s favorite-longshot
bias regressions, then use conformal-style and simulation-based methods to put
honest uncertainty around "this league shows a significant bias" claims —
directly responding to Winkelmann et al. (2024)'s diagnosis that the
sports-efficiency literature has a multiple-testing + publication-bias problem.

## Anchor papers (assume the reader knows them)

- **Hegarty & Whelan (2024)**, *Sports Economics Review* 8.100042. Recommends
  normalized probabilities over inverse odds; finds favorite-longshot bias in
  soccer (84k matches, 22 leagues, 2011/12–2021/22) and tennis (56k matches,
  ATP/WTA, 2011–2022).
- **Winkelmann et al. (2024)**, *Journal of Sports Economics* 25(1). Monte
  Carlo simulation: 77.6% chance of at least one false-positive season in a
  14-season span under full efficiency. Flags publication bias.

If you find yourself doing something either paper has already done, you're not
contributing — go up a level.

## Where the project is right now (Phase 8 complete)

**91 passing tests. 8 phases done. Real data flowing end-to-end.**

What's built:
- **Devigging** (`src/devig.py`): four methods — normalized, additive, power,
  Shin — with full unit tests.
- **Synthetic data** (`src/synth.py`): generates files in the real
  on-disk formats (football-data.co.uk CSV, tennis-data.co.uk XLSX) with known
  ground truth (efficient or FLB markets, with goals).
- **Loaders** (`src/loaders.py`): tolerant of layout variation, produce wide
  and long format Parquet. Dedup on match identity. Defense-in-depth against
  zero/inf odds.
- **H&W replication** (`src/replication.py`): WLS with `1/(P(1-P))` weights,
  cluster-robust SE at match level, match-level bootstrap. Pooled γ = +0.046
  (p < 0.001), per-league results consistent with H&W's heterogeneity finding.
- **Features + model** (`src/features.py`, `src/model.py`): Elo (per-league,
  +100 home advantage, K=20), 5-match rolling form, goals, rest days, league
  fixed effect. LightGBM multiclass with walk-forward CV. 15,856 OOS
  predictions across 2016–2022.
- **Conformal** (`src/conformal.py`): split conformal + Mondrian by league.
  Calibration = prior season. Both market and model achieve ≥ 90% marginal
  coverage; market sets are 0.14–0.29 outcomes smaller at the same nominal
  level.
- **Calibration evaluation** (`src/calibration.py`): Brier, log-loss, ECE,
  reliability diagrams. Match-level bootstrap CIs on pooled metrics. Pooled
  Brier CIs do not overlap (market 0.194 vs. model 0.213).

Sample is 9 leagues × 2015/16–2021/22, ~18.5k matches in the H&W regression
window — smaller than H&W's 22 leagues × 11 seasons. We do not directly compare
per-league γ magnitudes to H&W's Table 1.

Detailed per-phase notes live in `notes/phase_NN_*.md` files. Read those if you
need decisions, failure modes, or specific numbers. Don't repeat them here.

## The empirical story so far

Three independent lines of evidence converge on the same finding:

1. **H&W replication:** pooled FLB γ = +0.046 (95% CI [+0.021, +0.070]).
   Significant but small. Heterogeneous across leagues — strongest in I1 and
   E2, absent in E0/D1/SP1.
2. **Conformal set sizes:** at 90% coverage, market sets 0.14–0.29 smaller than
   model sets across all leagues. Gap smallest in E2 (the league with the
   strongest FLB).
3. **Calibration metrics:** market Brier 0.194 vs. model 0.213 (CIs disjoint);
   market ECE 0.008 vs. model 0.074 (~10× better). Market wins in every league.

**The honest summary:** sportsbook markets are remarkably well-calibrated. The
FLB documented in the literature is real but subtle, mostly concentrated in
specific leagues, and gets smaller (or vanishes) as market liquidity rises.
The model — Elo + form + rest + league FE on LightGBM — does not beat the
market on any pooled metric we've measured.

This is a substantive finding even though it's "negative" relative to the
"beat the market with ML" framing. Treat it that way. Don't soft-pedal.

## North Star

**Honesty about uncertainty is the whole point.** The reason this project
exists is that the existing literature is too quick to declare significance and
too slow to acknowledge multiple testing, publication bias, and underpowered
designs. Every claim in the final writeup should pass three filters:

1. Is it backed by a passing test, a robustness check, or a CI?
2. Have I asked "what would Winkelmann say?" Would a multiple-testing or
   power analysis change the conclusion?
3. Would I be comfortable defending this claim in front of someone who knows
   both anchor papers cold?

If the answer to any of those is no, the claim isn't ready.

Corollary: a smaller scope with airtight execution always beats a larger scope
with hand-wavy claims. We have already produced enough genuine signal to fill
a strong writeup. Resist scope creep.

## Open methodological questions (the things that actually need work)

These are the items the project would benefit from before we declare Layer 1
done. They're listed in priority order, but you (Claude Code) should look at
the code and the data and propose how to tackle them — including whether the
order is right and whether anything else should be added or dropped.

### 1. Model calibration ablation

The model's raw LightGBM probabilities are uncalibrated. Standard practice is
to fit a post-hoc calibrator (isotonic regression or Platt scaling) on the
calibration set. We haven't done this. Without it, the "market beats model on
calibration" claim has an obvious gap that any reviewer will point out:
*you compared a calibrated estimator to an uncalibrated one*.

Run isotonic and/or Platt on the model outputs, recompute all Phase 7 + Phase 8
metrics, and report both raw and calibrated numbers. The question to answer:
how much of the model's ECE gap to the market closes with simple post-hoc
calibration? If it closes most of the gap, the story changes from "market is
better" to "raw LightGBM is uncalibrated but easily fixable" — which is more
honest and more interesting.

### 2. Multiple-testing correction for the H&W per-league table

The per-league regression in Phase 5 runs 9 simultaneous tests. We report
per-test p-values without correction. Apply Bonferroni (or BH-FDR) and show
the table both ways. Some currently-significant findings will likely lose
significance. That's the point.

### 3. Empirical exchangeability check for conformal coverage

The split conformal guarantee requires calibration and test data to be
exchangeable. Consecutive soccer seasons aren't strictly exchangeable. We note
this in Phase 7 but don't check it empirically.

The check: does coverage degrade as a function of distance between calibration
season and test season? If yes, the marginal guarantee is overstated and we
should narrow our claims. If no, we have empirical comfort that
near-exchangeability holds in practice.

### 4. Power analysis for the "no FLB detected" leagues

E0, D1, SP1 have wide per-league CIs that include zero. We currently report
this as "no detectable FLB." That's an under-claim that hides the real
question: what's the smallest FLB we could have detected at 80% power given
the sample size? If that floor is γ = 0.05, the honest claim is "we can rule
out FLB ≥ 0.05," not "no FLB."

### 5. Layer 2 lite

Two analyses, both leveraging existing infrastructure:

**5a. Out-of-sample γ prediction with conformal intervals.** Use seasons 1..t
to predict γ in season t+1 with a conformal interval. Baseline: γ_{t+1} = γ_t.
The interesting finding will probably be that the conformal intervals are wide
enough to overlap with γ = 0 most of the time — directly supporting
Winkelmann's claim that single-season FLB estimates are mostly noise.

**5b. Efficient-markets null simulation, per league.** Calibrate a Dirichlet to
each league's odds distribution, simulate efficient-market outcomes, run the
H&W regression, ask: what fraction of simulations produce |γ| ≥ observed |γ|?
Compare to the reported p-values. Where they diverge, that's the inflation
Winkelmann warns about. Extend to a joint test across all 9 leagues.

## What to do next session

Read the phase docs in `notes/` to understand specific decisions and failure
modes. Look at the code. Then propose a plan for the next session — which of
the open questions above to tackle first, whether any of them should be
combined or split, and whether anything else should be on the list that isn't.

Don't just execute. Push back if you think the priority is wrong or if there's
something more important you've spotted in the code.

## Conventions (unchanged from earlier)

- Reproducibility: every script accepts `--seed`, default 0.
- Output naming: descriptive filenames, not `fig1.png`.
- Bootstrap CIs by default for any reported metric. Match-level resampling.
- Pure functions over classes unless state is genuinely needed.
- Type hints throughout, but don't go overboard.
- Docstrings explain *why*, not *what*.
- Validate statistical methods against synthetic data with known ground truth
  before trusting on real data.

## Style for the eventual writeup

Two versions will likely make sense:

1. **Long form (~10 pages):** methods-paper style. Could go to a workshop or
   arXiv if the work warrants it. Full uncertainty quantification, robustness
   checks, honest discussion of limits.

2. **Portfolio form (~2 pages):** for recruiters. Headline finding, one or two
   charts, clear statement of method, link to repo. The version that gets the
   interview.

Don't write either yet — but keep in mind that we're building toward both.

## Summary docs

Write summary docs of each phase of work that you do. Describe in detail what you did and and why. Include failures if you ran into any and changes made as well as results.

Update this document CLAUDE.md if key things in the project change and context needs updating.