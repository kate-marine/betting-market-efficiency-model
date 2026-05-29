"""
Calibration ablation: compare raw model vs. post-hoc calibrated model vs. market.

Answers the key question: how much of the ECE gap between model and market
is due to uncalibrated raw LightGBM outputs, vs. genuine information advantage?
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.postcal import calibrate_walk_forward
from src.calibration import (
    to_long, brier_score, log_loss_multiclass, ece,
    reliability_diagram_data, bootstrap_metrics,
    OUTCOME_COLS_MODEL, OUTCOME_COLS_MARKET,
)
from src.conformal import evaluate_conformal

PREDS  = pathlib.Path("data/processed/soccer_predictions.parquet")
TABLES = pathlib.Path("results/tables")
FIGS   = pathlib.Path("results/figures")
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and calibrate
# ---------------------------------------------------------------------------
print("Loading predictions and applying post-hoc calibration...")
preds = pd.read_parquet(PREDS)
preds = calibrate_walk_forward(preds, methods=("isotonic", "platt"))

# Evaluable seasons: those with calibrated probs (all except first)
evaluable = preds[preds["iso_pH"].notna()].copy()
print(f"  Evaluable matches (calibrated): {len(evaluable):,}")
print(f"  Seasons: {sorted(evaluable['test_season'].unique())}")

# ---------------------------------------------------------------------------
# 2. Calibration metrics for all three estimators
# ---------------------------------------------------------------------------
OUTCOME_COLS_ISO   = {"H": "iso_pH",   "D": "iso_pD",   "A": "iso_pA"}
OUTCOME_COLS_PLATT = {"H": "platt_pH", "D": "platt_pD", "A": "platt_pA"}

# Build long-format for each estimator
def long_for(df, pred_cols):
    parts = []
    for outcome in ("H", "D", "A"):
        c = df[["match_id", "league", "test_season", "result"]].copy()
        c["outcome"]  = outcome
        c["observed"] = (df["result"] == outcome).astype(int).values
        c["pred_p"]   = df[pred_cols[outcome]].values
        parts.append(c)
    return pd.concat(parts, ignore_index=True)

long_raw    = long_for(evaluable, {"H":"pred_pH","D":"pred_pD","A":"pred_pA"})
long_iso    = long_for(evaluable, OUTCOME_COLS_ISO)
long_platt  = long_for(evaluable, OUTCOME_COLS_PLATT)
long_market = long_for(evaluable, {"H":"norm_pH","D":"norm_pD","A":"norm_pA"})

print("\nComputing calibration metrics with bootstrap CIs (n=1000)...")
rows = []
for label, long_df, pred_col_map in [
    ("Raw model",    long_raw,    {"H":"pred_pH","D":"pred_pD","A":"pred_pA"}),
    ("Isotonic",     long_iso,    OUTCOME_COLS_ISO),
    ("Platt",        long_platt,  OUTCOME_COLS_PLATT),
    ("Market",       long_market, {"H":"norm_pH","D":"norm_pD","A":"norm_pA"}),
]:
    boot = bootstrap_metrics(long_df, pred_col="pred_p", n_boot=1000, seed=0)
    ll   = log_loss_multiclass(
        evaluable.reset_index(drop=True),
        {k: v for k, v in pred_col_map.items()},
    )
    rows.append({
        "estimator":  label,
        "brier":      boot["brier"],
        "brier_lo":   boot["brier_lo"],
        "brier_hi":   boot["brier_hi"],
        "ece":        boot["ece"],
        "ece_lo":     boot["ece_lo"],
        "ece_hi":     boot["ece_hi"],
        "log_loss":   ll,
        "n_matches":  boot["n_matches"],
    })
    print(f"  {label:15s}  Brier={boot['brier']:.4f} [{boot['brier_lo']:.4f},{boot['brier_hi']:.4f}]"
          f"  ECE={boot['ece']:.4f} [{boot['ece_lo']:.4f},{boot['ece_hi']:.4f}]"
          f"  LL={ll:.4f}")

results = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. ECE gap analysis
# ---------------------------------------------------------------------------
raw_ece    = results.loc[results["estimator"]=="Raw model", "ece"].iloc[0]
iso_ece    = results.loc[results["estimator"]=="Isotonic",  "ece"].iloc[0]
platt_ece  = results.loc[results["estimator"]=="Platt",     "ece"].iloc[0]
market_ece = results.loc[results["estimator"]=="Market",    "ece"].iloc[0]

gap_total    = raw_ece - market_ece
gap_iso      = raw_ece - iso_ece
gap_residual = iso_ece - market_ece
pct_closed   = gap_iso / gap_total * 100 if gap_total > 0 else 0

print(f"\n=== ECE gap analysis ===")
print(f"  Raw model ECE:            {raw_ece:.4f}")
print(f"  After isotonic cal ECE:   {iso_ece:.4f}")
print(f"  After Platt scaling ECE:  {platt_ece:.4f}")
print(f"  Market ECE:               {market_ece:.4f}")
print(f"  Total ECE gap (raw→mkt):  {gap_total:.4f}")
print(f"  Isotonic closed:          {gap_iso:.4f}  ({pct_closed:.1f}%)")
print(f"  Residual gap (iso→mkt):   {gap_residual:.4f}  ({100-pct_closed:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Conformal set sizes with calibrated probabilities
# ---------------------------------------------------------------------------
print("\nRunning conformal evaluation with calibrated probabilities...")

# Add calibrated columns to preds under market-compatible column names for reuse
ALPHA = 0.10

def conformal_set_sizes(df, pred_cols, alpha=0.10):
    """Compute mean set sizes via split conformal, walk-forward."""
    from src.conformal import nonconformity_scores, conformal_threshold, prediction_sets, mean_set_size
    seasons = sorted(df["test_season"].unique())
    sizes = []
    for i, test_s in enumerate(seasons):
        if i == 0: continue
        cal_s = seasons[i-1]
        cal = df[df["test_season"]==cal_s]
        test = df[df["test_season"]==test_s]
        if len(cal)<10 or len(test)<10: continue
        cal_true  = cal["result"].map({"H":0,"D":1,"A":2}).values
        test_true = test["result"].map({"H":0,"D":1,"A":2}).values
        cal_probs  = cal[list(pred_cols.values())].values
        test_probs = test[list(pred_cols.values())].values
        scores = nonconformity_scores(cal_probs, cal_true)
        tau = conformal_threshold(scores, alpha)
        sets = prediction_sets(test_probs, tau)
        sizes.append(mean_set_size(sets))
    return float(np.mean(sizes))

print(f"  Nominal coverage = {1-ALPHA:.0%}, α = {ALPHA}")
for label, pred_cols in [
    ("Raw model", {"H":"pred_pH","D":"pred_pD","A":"pred_pA"}),
    ("Isotonic",  OUTCOME_COLS_ISO),
    ("Platt",     OUTCOME_COLS_PLATT),
    ("Market",    {"H":"norm_pH","D":"norm_pD","A":"norm_pA"}),
]:
    sz = conformal_set_sizes(evaluable, pred_cols, alpha=ALPHA)
    print(f"  {label:15s}  mean set size = {sz:.3f}")

# ---------------------------------------------------------------------------
# 5. Reliability diagram: raw vs. calibrated vs. market
# ---------------------------------------------------------------------------
print("\nGenerating reliability diagram comparing all three estimators...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Calibration Ablation: Reliability Diagrams (pooled, all leagues)", fontsize=11)
OUTCOMES = [("H","Home win"), ("D","Draw"), ("A","Away win")]

for ax, (outcome, label) in zip(axes, OUTCOMES):
    obs = (evaluable["result"] == outcome).astype(int).values
    for pred_col, color, name, ls in [
        (f"pred_p{outcome}",  "#AAAAAA", "Raw model",  "--"),
        (f"iso_p{outcome}",   "#2166AC", "Isotonic",   "-"),
        (f"norm_p{outcome}",  "#D6604D", "Market",     "-"),
    ]:
        rd = reliability_diagram_data(evaluable[pred_col].values, obs, n_bins=10)
        ax.plot(rd["mean_pred"], rd["obs_freq"], f"o{ls}", color=color, label=name,
                ms=5, lw=1.5)
    ax.plot([0,1],[0,1],"--",color="gray",lw=0.8,alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8)

plt.tight_layout()
fp = FIGS / "reliability_diagrams_calibration_ablation_soccer.png"
plt.savefig(fp, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {fp}")

# ---------------------------------------------------------------------------
# 6. Save tables
# ---------------------------------------------------------------------------
results.to_csv(TABLES / "calibration_ablation.csv", index=False)

with open(TABLES / "calibration_ablation.md", "w") as f:
    f.write("## Calibration Ablation: Raw Model vs. Post-Hoc Calibrated vs. Market\n\n")
    f.write(f"Evaluable seasons: {sorted(evaluable['test_season'].unique())}\n\n")
    f.write(f"Bootstrap CIs: 95% percentile, n=1000, match-level resampling.\n\n")
    disp = results.copy()
    for col in ["brier","brier_lo","brier_hi","ece","ece_lo","ece_hi","log_loss"]:
        disp[col] = disp[col].map("{:.4f}".format)
    f.write(disp.to_markdown(index=False))
    f.write(f"\n\n**ECE gap closed by isotonic calibration:** {pct_closed:.1f}% of raw→market gap\n")

print(f"\nAll outputs saved to {TABLES}/ and {FIGS}/")
print(f"\n=== Summary ===")
print(f"  Raw model ECE:   {raw_ece:.4f}")
print(f"  Isotonic ECE:    {iso_ece:.4f}  ({pct_closed:.0f}% of gap closed)")
print(f"  Market ECE:      {market_ece:.4f}")
print(f"\n  Brier improvement (raw→iso): {results.loc[0,'brier'] - results.loc[1,'brier']:.4f}")
print(f"  Brier gap (iso→market):      {results.loc[1,'brier'] - results.loc[3,'brier']:.4f}")
