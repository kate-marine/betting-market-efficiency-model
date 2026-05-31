"""
Phase 6: Train LightGBM with walk-forward CV on real soccer data.

Saves predictions to data/processed/soccer_predictions.parquet.
These predictions feed into Phase 8 (calibration evaluation).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import time
import pandas as pd
from src.features import compute_features
from src.model import walk_forward_predict

WIDE  = pathlib.Path("data/processed/soccer_wide.parquet")
OUT   = pathlib.Path("data/processed/soccer_predictions.parquet")

# test on seasons with enough training data
HW_TEST_SEASONS = [
    "2016-2017", "2017-2018", "2018-2019",
    "2019-2020", "2020-2021", "2021-2022",
]

print("Loading data...")
wide = pd.read_parquet(WIDE)
print(f"  {len(wide):,} matches across {wide['league'].nunique()} leagues")

print("\nComputing features (Elo, form, rest days)...")
t = time.time()
wide_feat = compute_features(wide)
print(f"  Done in {time.time()-t:.1f}s")

# Elo should vary
print(f"  Elo range: [{wide_feat['elo_home'].min():.0f}, {wide_feat['elo_home'].max():.0f}]")
print(f"  NaN rate in features:")
for col in ["elo_home", "home_form_W", "home_gf5", "home_rest_days"]:
    print(f"    {col}: {wide_feat[col].isna().mean():.1%}")

print("\nRunning walk-forward CV...")
t = time.time()
preds = walk_forward_predict(
    wide_feat,
    test_seasons=HW_TEST_SEASONS,
    min_train_seasons=1,
    seed=0,
)
print(f"\nDone in {time.time()-t:.1f}s")
print(f"Total predictions: {len(preds):,}")

# Quick calibration check: are predicted probs in the right ballpark?
print("\nPredicted probability averages by declared result:")
for result, label in [("H","Home win"), ("D","Draw"), ("A","Away win")]:
    mask = preds["result"] == result
    print(f"  {label}: pred_pH={preds.loc[mask,'pred_pH'].mean():.3f}  "
          f"pred_pD={preds.loc[mask,'pred_pD'].mean():.3f}  "
          f"pred_pA={preds.loc[mask,'pred_pA'].mean():.3f}")

preds.to_parquet(OUT, index=False)
print(f"\nSaved to {OUT}")
