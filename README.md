# Calibrated Claims: Conformal Prediction Meets Sports Betting Market Efficiency

Empirical study of sports betting market efficiency using conformal prediction and
simulation-based uncertainty quantification. See `CLAUDE.md` for the full research design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## End-to-end smoke test (no real data needed)

```bash
python scripts/smoke_test.py
```

Generates synthetic data in both football-data.co.uk and tennis-data.co.uk formats,
runs them through the loaders, and writes harmonized Parquet to `data/processed/`.

## Adding real data

Drop real data anywhere under `data/raw/`:

- **Soccer** (football-data.co.uk CSVs): the loader will find them recursively.
  League is taken from the `Div` column or parent folder name; season from the
  filename or `Date` column.
- **Tennis** (tennis-data.co.uk XLSXs): place under `data/raw/atp/` and
  `data/raw/wta/`. Year is taken from the filename or `Date` column.

Then run:
```bash
python -c "
from src.loaders import load_soccer, load_tennis
load_soccer('data/raw', 'data/processed')
load_tennis('data/raw', 'data/processed')
"
```

## Project layout

```
src/
  devig.py      — four devigging methods (normalized, additive, power, Shin)
  synth.py      — synthetic data generator (efficient + FLB variants)
  loaders.py    — soccer and tennis loaders → harmonized Parquet
tests/
  test_devig.py
  test_synth_loaders.py
scripts/
  smoke_test.py — end-to-end pipeline validation on synthetic data
data/
  raw/          — gitignored; drop real data here
  processed/    — gitignored; Parquet output
```
