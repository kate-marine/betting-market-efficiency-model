"""Quick diagnostic: load each CSV individually, report timing and errors."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import glob, time, warnings
import pandas as pd, pathlib as pl
from src.loaders import _load_soccer_file, _season_from_path, _season_from_dates

warnings.filterwarnings('ignore')

files = sorted(glob.glob('data/raw/*.csv'))
print(f'Total CSV files: {len(files)}', flush=True)

for f in files:
    path = pl.Path(f)
    t = time.time()
    try:
        peek = pd.read_csv(f, nrows=2, low_memory=False)
        league = peek['Div'].iloc[0] if 'Div' in peek.columns and pd.notna(peek['Div'].iloc[0]) else path.parent.name
        season = _season_from_path(path)
        if season is None:
            peek2 = pd.read_csv(f, usecols=['Date'], low_memory=False)
            season = _season_from_dates(peek2['Date'])
        df = _load_soccer_file(f, league, season)
        elapsed = time.time() - t
        print(f'OK  {path.name:35s} {league:5s} {season}  {len(df):4d} rows  {elapsed:.2f}s', flush=True)
    except Exception as e:
        print(f'ERR {path.name:35s} {str(e)[:80]}', flush=True)

print('Done.', flush=True)
