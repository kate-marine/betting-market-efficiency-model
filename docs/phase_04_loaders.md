# Phase 4: Loaders

**Status:** Complete  
**Files created:** `src/loaders.py`, `tests/test_synth_loaders.py`, `scripts/smoke_test.py`  
**Tests:** 24 passing (combined synth + loader tests)

---

## What we built

Two loaders — soccer and tennis — that scan a directory for data files and produce harmonized Parquet output in two formats:

- **Wide format** (`soccer_wide.parquet`, `tennis_wide.parquet`): one row per match, with all four devigged probability columns alongside the raw odds. Used for feature engineering and exploratory analysis.
- **Long format** (`soccer_long.parquet`, `tennis_long.parquet`): one row per (match, outcome). This is what the regressions consume — the H&W regression operates on one row per (match, side) pair.

### Layout tolerance

The loaders handle several layout variations without requiring a specific folder structure:

| Source | League | Season |
|--------|--------|--------|
| `Div` column in CSV | Filename prefix (e.g. "E0") | Path pattern `2019-2020` |
| Parent folder name | — | `Date` column (July-June split) |

**Why the fallback chain?** Real users organize downloads differently. The football-data.co.uk site offers files by league and season, but people may flatten them, rename them, or use different folder structures. The loader tries each source in order and falls back gracefully.

### Soccer: 3-way outcome

One row per match in wide format. Long format adds three rows (H, D, A) per match.

### Tennis: winner/loser → symmetric

tennis-data.co.uk pre-labels the `Winner` column as the actual winner. For symmetry with the soccer setup (and because the H&W regression treats both sides equivalently), the long format creates two rows per match: Winner side (`observed=1`) and Loser side (`observed=0`).

---

## Decisions made

**Parquet over CSV for processed output.** Parquet is column-oriented, typed, and typically 5-10× smaller than CSV for numerical data. Reading a 35k-row Parquet with pandas is nearly instant; a CSV of the same data would need type inference on every load.

**`recursive=True` default, `recursive=False` for real data.** The synthetic data lives under `data/raw/synthetic/`. Using `rglob("*.csv")` by default finds everything, which is fine for tests. But when loading real data alongside synthetic data, we need `recursive=False` to avoid contaminating the real dataset with synthetic rows.

**Dedup on (league, season, date, home_team, away_team).** The user downloaded some seasons twice (e.g. `D1 (6).csv` and `D1 copy.csv` both covered Bundesliga 2019-20). Rather than pre-processing the files, we deduplicate after concatenation. Match identity is uniquely determined by the four key columns; keeping the first occurrence is safe because the duplicates were byte-for-byte identical on overlap.

---

## What didn't work

**`iterrows()` was O(n) Python loops — catastrophically slow.** The first long-format builder looped over every row:
```python
for _, row in wide.iterrows():
    for outcome, ...:
        long_rows.append(...)
```
On 35,000 matches × 3 outcomes, that's 105,000 Python loop iterations plus dict construction. The command timed out at 2 minutes before producing any output. Replaced with vectorized concat:
```python
for outcome, prob_col, odds_col in [...]:
    chunk = wide[base_cols].copy()
    chunk["observed"] = (wide["result"] == outcome).astype(int)
    chunk["norm_p"] = wide[prob_col].values
    long_parts.append(chunk)
long = pd.concat(long_parts, ignore_index=True)
```
This runs in milliseconds.

**Zero-valued odds caused an infinite loop.** One row in `E1 (1) copy.csv` had `B365H = 0.0`. The validity filter `np.isfinite(odds_arr)` passed it through because `0.0` is finite. Then `1/0.0 = inf` in the devig, and the power devig's bisection loop `while Σ(row^k) > 1: hi *= 2` ran forever because `inf^k = inf` for any positive `k`.

The loader hung silently with no error message — particularly hard to debug because Python subprocess timeouts went to background mode before printing anything.

Fix in the loader:
```python
valid = np.isfinite(odds_arr).all(axis=1) & (odds_arr > 0).all(axis=1)
```
Fix in `devig.power()` as a defense-in-depth guard:
```python
if not np.all(np.isfinite(row)) or np.any(row <= 0):
    probs[i] = np.full(len(row), 1.0 / len(row))
    continue
```

**Season strings came out as floats.** `_season_from_dates` returned `"2014.0-2015.0"` instead of `"2014-2015"`. Root cause: pandas `.mode()` returns a float when the year series has any NaT values (which produce NaN years). Fixed with `int()` cast:
```python
dominant_year = int(year.where(month >= 7, year - 1).mode()[0])
```

**Duplicate season files doubled row counts.** D1 2019-20 showed 612 rows (2×306) and E1 2019-20 showed 1,104 rows (2×552). The user downloaded the same season under two filenames. Detected by comparing date ranges across all files and confirmed with `diff`. Fixed by post-load deduplication on match identity columns.

---

## Smoke test

`scripts/smoke_test.py` runs the full pipeline on synthetic data:
1. Generate synthetic soccer (3 leagues × 3 seasons) and tennis (2 tours × 3 years)
2. Run loaders
3. Assert prob sums, observed-per-match counts, and Parquet existence

This is the canonical "does the pipeline work end-to-end" check before touching real data.
