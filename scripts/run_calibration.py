"""
Phase 8: Calibration evaluation — Brier score, log-loss, ECE, reliability diagrams.
Compares LightGBM model vs. betting market on out-of-sample predictions.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration import (
    to_long, brier_score, log_loss_multiclass, ece,
    reliability_diagram_data, calibration_table,
    OUTCOME_COLS_MODEL, OUTCOME_COLS_MARKET,
)

PREDS   = pathlib.Path("data/processed/soccer_predictions.parquet")
TABLES  = pathlib.Path("results/tables")
FIGURES = pathlib.Path("results/figures")
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("Loading predictions...")
preds = pd.read_parquet(PREDS)
long  = to_long(preds)
print(f"  {len(preds):,} matches  →  {len(long):,} (match × outcome) rows")

# -----------------------------------------------------------------------
# Table 1: Pooled metrics with bootstrap CIs
# -----------------------------------------------------------------------
print("\nComputing pooled metrics (n_boot=1000)...")
table_pool = calibration_table(long, n_boot=1000, seed=0)

print("\n=== Pooled calibration metrics (model vs. market) ===")
for _, row in table_pool[table_pool["group"] == "Pooled"].iterrows():
    print(f"  {row['estimator']:8s}  "
          f"Brier={row['brier']:.4f} [{row['brier_lo']:.4f}, {row['brier_hi']:.4f}]  "
          f"ECE={row['ece']:.4f} [{row['ece_lo']:.4f}, {row['ece_hi']:.4f}]")

# Also compute log-loss
for estimator, pred_cols in [("model", OUTCOME_COLS_MODEL), ("market", OUTCOME_COLS_MARKET)]:
    ll = log_loss_multiclass(preds.reset_index(), pred_cols)
    print(f"  {estimator:8s}  Log-loss={ll:.4f}")

# -----------------------------------------------------------------------
# Table 2: Per-league metrics
# -----------------------------------------------------------------------
print("\nComputing per-league metrics...")
table_league = calibration_table(long, subgroup_col="league", n_boot=0, seed=0)
league_rows = table_league[table_league["group"] != "Pooled"].copy()

print("\n=== Per-league calibration (Brier score) ===")
pivot_brier = league_rows.pivot(index="group", columns="estimator", values="brier").round(4)
print(pivot_brier.to_string())

print("\n=== Per-league calibration (ECE) ===")
pivot_ece = league_rows.pivot(index="group", columns="estimator", values="ece").round(4)
print(pivot_ece.to_string())

# Save tables
table_pool.to_csv(TABLES / "calibration_pooled.csv", index=False)
table_league.to_csv(TABLES / "calibration_by_league.csv", index=False)

with open(TABLES / "calibration_pooled.md", "w") as f:
    f.write("## Calibration Metrics: Model vs. Market\n\n")
    f.write("Bootstrap CIs (95%, n=1000, match-level resampling).\n\n")
    disp = table_pool[table_pool["group"]=="Pooled"][
        ["estimator","brier","brier_lo","brier_hi","ece","ece_lo","ece_hi","n_matches"]
    ].copy()
    for col in ["brier","brier_lo","brier_hi","ece","ece_lo","ece_hi"]:
        disp[col] = disp[col].map("{:.4f}".format)
    f.write(disp.to_markdown(index=False))
    f.write("\n")

# -----------------------------------------------------------------------
# Reliability diagrams
# -----------------------------------------------------------------------
print("\nGenerating reliability diagrams...")

OUTCOMES = [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]
N_BINS = 10

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Reliability Diagrams: LightGBM Model vs. Betting Market", fontsize=12)

reldiag_data = []

for ax, (outcome, label) in zip(axes, OUTCOMES):
    sub = long[long["outcome"] == outcome]

    for pred_col, color, name in [
        ("pred_p",   "#2166AC", "Model"),
        ("market_p", "#D6604D", "Market"),
    ]:
        rd = reliability_diagram_data(sub[pred_col].values, sub["observed"].values, n_bins=N_BINS)
        ax.plot(rd["mean_pred"], rd["obs_freq"], "o-", color=color, label=name,
                ms=5, lw=1.5)
        for _, r in rd.iterrows():
            reldiag_data.append({
                "outcome": outcome, "estimator": name.lower(),
                **r.to_dict()
            })

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Perfect")
    ax.set_title(label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

plt.tight_layout()
fig_path = FIGURES / "reliability_diagrams_soccer_pooled_all_leagues.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {fig_path}")

# Save reliability diagram data as CSV
pd.DataFrame(reldiag_data).to_csv(TABLES / "reliability_diagram_data.csv", index=False)

# Per-league reliability diagrams for E0 and I1 (most interesting contrast)
for league in ["E0", "I1", "E2"]:
    sub_l = long[long["league"] == league]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Reliability Diagrams: {league}", fontsize=12)

    for ax, (outcome, label) in zip(axes, OUTCOMES):
        sub = sub_l[sub_l["outcome"] == outcome]
        for pred_col, color, name in [("pred_p","#2166AC","Model"), ("market_p","#D6604D","Market")]:
            rd = reliability_diagram_data(sub[pred_col].values, sub["observed"].values, n_bins=8)
            if len(rd):
                ax.plot(rd["mean_pred"], rd["obs_freq"], "o-", color=color, label=name, ms=5)
        ax.plot([0,1],[0,1],"--",color="gray",lw=0.8)
        ax.set_title(label); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8)

    plt.tight_layout()
    fp = FIGURES / f"reliability_diagrams_soccer_{league}_all_seasons.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fp}")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
model_row  = table_pool[(table_pool["group"]=="Pooled") & (table_pool["estimator"]=="model")].iloc[0]
market_row = table_pool[(table_pool["group"]=="Pooled") & (table_pool["estimator"]=="market")].iloc[0]
model_ll   = log_loss_multiclass(preds.reset_index(), OUTCOME_COLS_MODEL)
market_ll  = log_loss_multiclass(preds.reset_index(), OUTCOME_COLS_MARKET)

print("\n=== Summary ===")
print(f"{'Metric':12s}  {'Model':>20s}  {'Market':>20s}  {'Winner':>8s}")
print(f"{'Brier':12s}  {model_row['brier']:.4f} [{model_row['brier_lo']:.4f},{model_row['brier_hi']:.4f}]"
      f"  {market_row['brier']:.4f} [{market_row['brier_lo']:.4f},{market_row['brier_hi']:.4f}]"
      f"  {'market' if market_row['brier'] < model_row['brier'] else 'model':>8s}")
print(f"{'ECE':12s}  {model_row['ece']:.4f} [{model_row['ece_lo']:.4f},{model_row['ece_hi']:.4f}]"
      f"  {market_row['ece']:.4f} [{market_row['ece_lo']:.4f},{market_row['ece_hi']:.4f}]"
      f"  {'market' if market_row['ece'] < model_row['ece'] else 'model':>8s}")
print(f"{'Log-loss':12s}  {model_ll:.4f}                    {market_ll:.4f}                    "
      f"{'market' if market_ll < model_ll else 'model':>8s}")

print(f"\nAll outputs saved to {TABLES}/ and {FIGURES}/")
