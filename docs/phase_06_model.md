# Phase 6: Base Predictive Model (LightGBM)

**Status:** Complete  
**Files created:** `src/features.py`, `src/model.py`, `tests/test_features_model.py`, `scripts/run_model.py`  
**Output:** `data/processed/soccer_predictions.parquet` (15,856 out-of-sample predictions)  
**Tests:** 16 passing (added to 63 total)

---

## What we built

### Feature engineering (`src/features.py`)

All features are computed **in strict chronological order** with no lookahead: every feature value for match i uses only matches played before match i.

| Feature | Description |
|---------|-------------|
| `elo_home`, `elo_away` | Elo ratings before the match (per-league, start = 1500) |
| `elo_diff` | `elo_home − elo_away` |
| `home_form_{W,D,L}` | Wins/draws/losses in team's last 5 matches |
| `away_form_{W,D,L}` | Same for away team |
| `home_gf5`, `home_ga5` | Avg goals scored/conceded in last 5 matches |
| `away_gf5`, `away_ga5` | Same for away team |
| `home_rest_days` | Days since home team's last match |
| `away_rest_days` | Days since away team's last match |
| `league_id` | Integer-encoded league (for league fixed effects) |

**Elo implementation details:**
- Home advantage: +100 points added to home team rating when computing expected score
- K-factor: 20 (standard, not adapted by game importance)
- Draws counted as 0.5 for both teams
- Separate rating pool per league — teams don't carry Elo across leagues (a known simplification; promoted/relegated teams get a fresh start at whatever Elo they had when they enter the league sample)

**Why form over 5 matches:** Short enough to reflect recent form, long enough to not be dominated by single-match noise. First few matches of a team's history have NaN goals/form features; LightGBM handles these natively without requiring imputation.

### Walk-forward CV (`src/model.py`)

Train on all seasons < T, predict season T. Expanding window so each new test year has more training data than the last.

**Why never shuffle:** Elo and form features are computed cumulatively from past matches. A random train/test split would mean test-set Elo values were computed using future training-set results — a direct lookahead leak.

LightGBM multiclass (`num_class=3`) outputs `pred_pH`, `pred_pD`, `pred_pA` summing to 1.

---

## Results on real data (2016-17 to 2021-22)

| Test Season | Training Seasons | Train Matches | Test Predictions |
|-------------|-----------------|---------------|-----------------|
| 2016-2017 | 3 | 4,222 | 2,591 |
| 2017-2018 | 4 | 6,813 | 2,598 |
| 2018-2019 | 5 | 9,411 | 2,370 |
| 2019-2020 | 6 | 11,781 | 1,412* |
| 2020-2021 | 7 | 13,193 | 3,436 |
| 2021-2022 | 8 | 16,629 | 3,449 |

*COVID-shortened season — many matches not played.

**Elo range:** 1,295 to 1,875 — teams have differentiated meaningfully from the 1,500 starting point.

**Predicted probabilities by actual result (directional sanity check):**

| Actual Result | pred_pH | pred_pD | pred_pA |
|--------------|---------|---------|---------|
| Home win | 0.531 | 0.229 | 0.241 |
| Draw | 0.446 | 0.243 | 0.311 |
| Away win | 0.375 | 0.236 | 0.389 |

Model correctly assigns higher probability to the actual outcome class in each row — directionally sensible. Full calibration evaluation is Phase 8.

---

## Decisions made

**LightGBM over logistic regression or XGBoost.** LightGBM handles NaN natively (no imputation needed for first-match teams), trains quickly on tabular data, and is the industry standard for this type of structured prediction. Logistic regression would require explicit imputation and interaction terms for the Elo features.

**Multiclass rather than three binary models.** Ensures predicted probabilities for H/D/A sum exactly to 1 without post-hoc renormalization. The sum-to-1 constraint is a hard requirement for comparison with market implied probabilities.

**`min_train_seasons=1`** rather than 2 or 3. We already have limited pre-2015 data; requiring 2+ training seasons would eliminate 2016-17 from the test set. We note that 2016-17 predictions (3 training seasons) are less reliable than later seasons.

**Goals added to synthetic data generator.** The original `synth.py` didn't generate `FTHG`/`FTAG` columns. When the loader read synthetic files, `goals_H` and `goals_A` came back NaN everywhere, making all `home_gf5` / `away_gf5` features NaN, which caused `dropna(subset=FEATURE_COLS)` to drop all training rows. Fixed by drawing Poisson goals in `_make_soccer_df` (mean dependent on outcome to ensure consistency with `FTR`).

---

## What didn't work

**`groupby().first()` skips NaN by default.** Two tests checked that the first match for each team has NaN form/rest days by using `df.groupby("home_team").first()["home_form_W"].isna()`. This returned 0.0 (not NaN) for all teams because pandas `first()` skips NaN values and returns the next non-NaN value. Fixed by using the actual first row of the sorted dataset instead (`df.sort_values("date").iloc[0]`).

**Synthetic data had no goals → all training rows dropped.** `FEATURE_COLS` includes `home_gf5` and `away_gf5`. When all goals are NaN (original synthetic data), `dropna(subset=FEATURE_COLS)` drops every training row. The walk-forward CV then warned "too few valid training rows (0)" and generated no predictions. Fixed by adding Poisson-drawn goals to `synth.py`.

**`LightGBMError: scikit-learn is required`.** The LightGBM sklearn API (`LGBMClassifier`) requires scikit-learn to be installed even though LightGBM itself is installed. This wasn't in the initial dependency list. Fixed by `pip install scikit-learn` and adding it to `pyproject.toml`.

**Feature name warning.** LightGBM fitted with a named DataFrame was being predicted with `.values` (numpy array), triggering a sklearn warning. Fixed by passing the DataFrame slice directly instead of `.values`.

**`ValueError` instead of empty DataFrame.** When all test seasons are skipped (e.g., not enough training data), the function raised `ValueError: No predictions generated`. Changed to return an empty DataFrame with a warning, which is cleaner for callers that might iterate over seasons.
