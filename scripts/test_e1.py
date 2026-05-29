import warnings; warnings.filterwarnings('ignore')
import glob, time, sys, pathlib as pl, pandas as pd
sys.path.insert(0, '.')
from src.loaders import _load_soccer_file, _season_from_path, _season_from_dates

for f in sorted(glob.glob('data/raw/E1*.csv')):
    fname = pl.Path(f).name
    try:
        peek = pd.read_csv(f, nrows=2, low_memory=False)
        league = peek['Div'].iloc[0] if 'Div' in peek.columns else 'E1'
        season = _season_from_path(pl.Path(f))
        if season is None:
            season = _season_from_dates(pd.read_csv(f, usecols=['Date'], low_memory=False)['Date'])
        t = time.time()
        df = _load_soccer_file(f, league, str(season))
        print(f'OK {fname:30s} {season}  {len(df):4d} rows  {time.time()-t:.2f}s', flush=True)
    except Exception as e:
        print(f'ERR {fname:30s} {e}', flush=True)
print('Done', flush=True)
