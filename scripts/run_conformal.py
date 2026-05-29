"""
Phase 7: Conformal prediction evaluation on real soccer predictions.

Compares ML model vs. market under marginal and Mondrian conformal.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from src.conformal import evaluate_conformal, per_league_coverage

PREDS  = pathlib.Path("data/processed/soccer_predictions.parquet")
TABLES = pathlib.Path("results/tables")
TABLES.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10   # 90% nominal coverage

print(f"Loading predictions... (nominal coverage = {1-ALPHA:.0%})")
preds = pd.read_parquet(PREDS)
print(f"  {len(preds):,} matches, seasons {sorted(preds['test_season'].unique())}")

# -----------------------------------------------------------------------
# Table A: Season-by-season marginal and Mondrian coverage
# -----------------------------------------------------------------------
print("\nRunning conformal evaluation...")
results = evaluate_conformal(preds, alpha=ALPHA)

print("\n=== Marginal conformal (P(Y∈C(X)) ≥ 1-α guarantee) ===")
marginal = results[results["method"] == "marginal"].copy()
marginal_fmt = marginal[["test_season","estimator","tau","coverage","mean_set_size","n_cal","n_test"]].copy()
marginal_fmt["coverage"] = marginal_fmt["coverage"].map("{:.3f}".format)
marginal_fmt["tau"] = marginal_fmt["tau"].map("{:.3f}".format)
marginal_fmt["mean_set_size"] = marginal_fmt["mean_set_size"].map("{:.3f}".format)
print(marginal_fmt.to_string(index=False))

print("\n=== Mondrian conformal by league (conditional guarantee per league) ===")
mondrian = results[results["method"] == "mondrian_league"].copy()
mondrian_fmt = mondrian[["test_season","estimator","coverage","mean_set_size"]].copy()
mondrian_fmt["coverage"] = mondrian_fmt["coverage"].map("{:.3f}".format)
mondrian_fmt["mean_set_size"] = mondrian_fmt["mean_set_size"].map("{:.3f}".format)
print(mondrian_fmt.to_string(index=False))

# -----------------------------------------------------------------------
# Table B: Per-league coverage breakdown (Mondrian, pooled across seasons)
# -----------------------------------------------------------------------
print("\nComputing per-league coverage (Mondrian, pooled across seasons)...")
league_cov = per_league_coverage(preds, alpha=ALPHA)

print("\n=== Per-league coverage: model vs. market ===")
pivot = league_cov.pivot_table(
    index="league",
    columns="estimator",
    values=["tau","coverage","mean_set_size","n_test"]
).round(3)
print(pivot.to_string())

# Highlight leagues where model and market substantially differ
model_cov = league_cov[league_cov["estimator"] == "model"].set_index("league")["coverage"]
market_cov = league_cov[league_cov["estimator"] == "market"].set_index("league")["coverage"]
model_sz = league_cov[league_cov["estimator"] == "model"].set_index("league")["mean_set_size"]
market_sz = league_cov[league_cov["estimator"] == "market"].set_index("league")["mean_set_size"]

print(f"\n=== Summary comparison (nominal = {1-ALPHA:.0%}) ===")
print(f"{'League':6s}  {'Model cov':>10s}  {'Market cov':>11s}  "
      f"{'Model size':>11s}  {'Market size':>12s}  {'Size diff':>10s}")
for lg in sorted(model_cov.index):
    mc = model_cov.get(lg, float('nan'))
    kc = market_cov.get(lg, float('nan'))
    ms = model_sz.get(lg, float('nan'))
    ks = market_sz.get(lg, float('nan'))
    flag = " ← model smaller" if ms < ks - 0.05 else (" ← market smaller" if ks < ms - 0.05 else "")
    print(f"{lg:6s}  {mc:>10.3f}  {kc:>11.3f}  {ms:>11.3f}  {ks:>12.3f}  {ms-ks:>+10.3f}{flag}")

# Save
results.to_csv(TABLES / "conformal_season_results.csv", index=False)
league_cov.to_csv(TABLES / "conformal_league_coverage.csv", index=False)

with open(TABLES / "conformal_league_coverage.md", "w") as f:
    f.write(f"## Mondrian Per-League Coverage (nominal = {1-ALPHA:.0%})\n\n")
    f.write("Calibration: pooled across all walk-forward cal seasons.\n\n")
    disp = league_cov[["league","estimator","tau","coverage","mean_set_size","coverage_gap","n_test"]].copy()
    for col in ["tau","coverage","mean_set_size","coverage_gap"]:
        disp[col] = disp[col].map("{:.3f}".format)
    f.write(disp.to_markdown(index=False))
    f.write("\n\n*coverage_gap = coverage − (1−α); positive = over-covers*\n")

print(f"\nTables saved to {TABLES}/")
