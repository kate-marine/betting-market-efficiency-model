# Calibrated Claims: Conformal Prediction Meets Sports Betting Market Efficiency

## What we're building

A two-layer empirical study of sports betting market efficiency using modern
uncertainty quantification. This is a portfolio piece for analytics internship
applications (target: DraftKings and similar), with a stretch goal of being
workshop-paper publishable.

**Layer 1 (prediction-level):** Train a base ML model per sport, wrap it in
conformal prediction, and benchmark its calibrated probabilities against
sportsbook market-implied probabilities. Where does the market dominate? Where
does conformal-wrapped ML hold its own or beat it? Slice by market thickness
(top-flight vs. lower-tier leagues, Grand Slam vs. regular tour, etc.).

**Layer 2 (claim-level):** Replicate Hegarty & Whelan (2024)'s favorite-longshot
bias regressions on the same data they used, then use conformal-style and
simulation-based methods to put honest uncertainty around "this league shows a
significant bias" claims. This directly responds to Winkelmann et al. (2024)'s
diagnosis that the sports-efficiency literature has a multiple-testing +
publication-bias problem.

## Anchor papers (assume the reader knows them)

- **Hegarty & Whelan (2024)**, *Sports Economics Review* 8.100042. Shows that
  the standard test of strong-form market efficiency (regress outcomes on
  inverse odds) is biased toward accepting the null. Recommends normalized
  probabilities instead. Finds substantial favorite-longshot bias in soccer
  (84,230 matches across 22 European leagues) and tennis (55,988 matches on
  ATP/WTA tours). Their datasets are public and we use them directly.
- **Winkelmann et al. (2024)**, *Journal of Sports Economics* 25(1). Uses Monte
  Carlo simulation to show that season-by-season analyses of betting markets
  routinely produce "significant" bias findings under full market efficiency —
  77.6% chance of at least one false-positive season in a 14-season span. Flags
  publication bias as a contributor.

If you find yourself doing something either paper has already done, you're not
contributing — go up a level.

## Data sources

- **Soccer:** https://www.football-data.co.uk/data.php — one CSV per league per
  season. We follow H&W and use 22 European leagues, 2011/12–2021/22. Key
  columns: `Date`, `HomeTeam`, `AwayTeam`, `FTR` (full-time result H/D/A),
  `AvgH`/`AvgD`/`AvgA` (average closing odds across bookmakers — H&W's primary
  source), with `B365H`/`B365D`/`B365A` as fallback if averages are absent.
- **Tennis:** http://www.tennis-data.co.uk/alldata.php — one XLSX per tour per
  year. ATP and WTA, 2011–2022. Key columns: `Date`, `Tournament`, `Round`,
  `Surface`, `Winner`, `Loser`, `AvgW`/`AvgL`, `B365W`/`B365L` fallback.

The user will download these and drop them somewhere under `data/raw/`. Be
flexible about the exact layout — they may organize by season-folder, by flat
filename, or just dump everything. Use the `Date` column to recover the season
if it's not in the path.

Note: tennis-data.co.uk pre-labels `Winner` as the actual winner of the match
(not a side label). For symmetry with the soccer setup, generate one row per
(match, side) where side 1 = labelled "Winner" with their odds (always wins,
outcome=1) and side 2 = labelled "Loser" with their odds (outcome=0).

## Key methodological points to get right

These are the easy things to get subtly wrong, and the project lives or dies on
them. Do not improvise on any of these without thinking carefully:

1. **Devigging.** The default everywhere is *normalized probabilities*
   (inverse-odds rescaled to sum to 1). H&W show this is the theoretically
   correct estimator under strong-form efficiency. Implement at least three
   alternatives — additive, power, and Shin (1992) — for robustness checks,
   since Winkelmann flags devig choice as an analyst-discretion lever.

2. **Cluster-robust standard errors.** Each match contributes K rows to the
   regression (K=3 for soccer, K=2 for tennis), and the outcomes within a match
   are mechanically correlated (they sum to 1). Cluster SEs at the match level.
   H&W do this; we follow.

3. **The H&W regression** (their equation 13):
   `Y_ij - P^N_ij = alpha + gamma * P^N_ij + epsilon_ij`,
   estimated by WLS with weights `1 / (P(1-P))`. gamma > 0 means
   favorite-longshot bias. Test gamma = 0. Replicating their Table 1 and Table 2
   on real data is the first sanity check — if our gammas don't match within
   rounding, the loader has a bug.

4. **The favorite-longshot bias direction.** Returns on *favorites* exceed
   returns on *longshots* (favorites are "underbet"). This is the opposite of
   what naive intuition suggests. Get this right in writeups and labels.

5. **Conformal prediction caveats.** Standard split conformal gives *marginal*
   coverage — coverage holds on average over the test set, not conditionally on
   subgroups. For per-league or per-segment claims, use Mondrian / group-
   conditional conformal. Be precise about which guarantee applies to which
   claim.

## What to build, roughly in order

1. **Project skeleton** — `src/`, `tests/`, `data/raw/` (gitignored),
   `data/processed/`, `results/figures/`, `results/tables/`, `notebooks/`.
   `requirements.txt` or `pyproject.toml` (your call).

2. **Devigging module** — four methods (normalized, additive, power, Shin) as
   pure-numpy functions vectorized over matches. Unit tests verifying: probs
   sum to 1, are in (0,1), favorite-by-odds has highest prob, normalized
   matches definition.

3. **Synthetic data generator** — produce files in the *real* football-data and
   tennis-data on-disk formats with known ground truth (efficient market and
   FLB market variants). This lets every downstream method be validated against
   ground truth before we trust it on real data. Follows the simulation design
   in Winkelmann et al. and H&W section 5.

4. **Loaders** — soccer and tennis, tolerant of file-layout variation. Produce
   harmonized parquet files with a wide format (one row per match, all odds and
   probability columns alongside) and a long format (one row per (match,
   outcome), which is what the regressions consume).

5. **H&W replication** — pooled and by-league. Cluster-robust SE. Compare
   normalized vs. inverse-odds estimators side-by-side to reproduce the
   methodological point of H&W's paper. Sanity check: gammas should be close
   to H&W's reported values on the same data.

6. **Base predictive model** — start with gradient boosting (LightGBM) on team-
   strength features. Soccer first: rolling form (last N matches), Elo-style
   rating, rest days, home indicator, league fixed effects. Tennis: player Elo
   on surface, recent form, head-to-head. Walk-forward / expanding-window split
   (train on past seasons, predict future seasons — never random shuffle).

7. **Conformal wrapper** — split conformal first, then Mondrian (group-
   conditional) for subgroup coverage. MAPIE is fine; or write it directly,
   it's not that much code for the binary case.

8. **Calibration evaluation** — reliability diagrams (binned by predicted prob),
   Brier score, log-loss, expected calibration error, and coverage rate at the
   nominal level. Compute pooled and sliced by subgroup. Bootstrap CIs on all
   metrics — point estimates without intervals will not be taken seriously.

9. **Layer 2 lite: claim-level uncertainty.** Two analyses:
   - For each league, use seasons 1..t to predict the gamma you'd see in season
     t+1, with a conformal prediction interval. Check empirical coverage. If
     intervals fail to cover, that itself is a meaningful finding.
   - For each league, simulate the efficient-markets null calibrated to that
     league's odds distribution, and ask: how often would we see a gamma as
     extreme as H&W's, after multiple-testing correction across all 22 leagues?

10. **Stretch (only if 1-9 are solid):** apply the same framework to NBA player
    props as a sport-agnostic generality check. Player props are newer, less
    studied, and DraftKings-relevant. The Odds API has a free tier.

## Conventions

- **Reproducibility:** every script accepts `--seed`, defaults to 0. Random
  state propagates through.
- **Output naming:** filenames should be descriptive
  (`reliability_diagram_soccer_E0_2019-2020.png`, not `fig1.png`).
- **Tables:** save as both CSV (machine-readable) and a rendered markdown/LaTeX
  version. Final writeup eats the rendered versions.
- **Bootstrap CIs by default** for any reported metric. Default 1000 resamples,
  match-level resampling (not row-level — see the cluster point above).
- **Pure functions over classes** unless state is genuinely needed.
- **Type hints throughout**, but don't go overboard with generics.
- **Docstrings explain *why* the function exists**, not what each line does.
  This is research code, not a tutorial.

## Testing philosophy

Every nontrivial statistical function should have at least one test that
validates it against synthetic data with known ground truth, not just unit-level
property checks. The synthetic data generator is part of the test infrastructure
for the same reason.

When a regression or simulation produces a result, the first question is "would
this pass on data where I know the true answer?" If you haven't run that check,
the result isn't trustworthy yet.


## The North Star

Would this hold up to a careful reader who knows the two anchor papers? If a
claim isn't backed by a passing test, a robustness check, or an explicit
acknowledgment of its limits, it's not ready to be in the writeup.

Bias toward saying less but more confidently. A medium-scope project with
beautiful execution and honest uncertainty quantification beats an ambitious
project with hand-wavy claims, every time — especially given that the entire
*point* of the project is being more honest about uncertainty than the existing
literature is.

## First session

When the user starts the first session, they'll likely say something like
"let's get started." Your job in the first session is:

1. Set up the project skeleton (directories, requirements, gitignore, README).
2. Build the devigging module with tests.
3. Build the synthetic data generator.
4. Build the loaders against the synthetic data.
5. Show the user end-to-end: synthetic data → loaders → harmonized parquet.

Real data won't be there yet. Don't wait for it — everything in steps 1–5 can be
built and validated on synthetic data, and that's the right way to do it
regardless. The user downloads real data in parallel.

Do NOT skip the synthetic data step. It is the test infrastructure for
everything downstream, and Winkelmann et al.'s entire contribution rests on
exactly this kind of simulation-based validation. Build it properly the first
time.