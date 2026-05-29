"""
H&W replication script. Run after load_real_data.py (or the loader) to
produce Table 1-equivalent results.

Saves CSV + Markdown tables to results/tables/.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from src.replication import run_hw_table, compare_estimators, bootstrap_gamma, format_table, add_multiple_testing_correction

LONG  = pathlib.Path("data/processed/soccer_long.parquet")
TABLES = pathlib.Path("results/tables")
TABLES.mkdir(parents=True, exist_ok=True)

# H&W period: filter to what we have (2015/16–2021/22)
HW_WINDOW = ("2015-2016", "2021-2022")

print("Loading long-format data...")
long = pd.read_parquet(LONG)
hw = long[long["season"].between(*HW_WINDOW)].copy()

n_matches = hw["match_id"].nunique()
n_obs     = len(hw)
print(f"  {n_matches:,} matches, {n_obs:,} match-outcome rows")
print(f"  Leagues: {sorted(hw['league'].unique())}")
print(f"  Seasons: {sorted(hw['season'].unique())}")
print()

# -----------------------------------------------------------------------
# Table 1 equivalent: H&W regression, normalized probabilities
# -----------------------------------------------------------------------
print("Running H&W regression (normalized)...")
table_norm = run_hw_table(hw, p_col="norm_p")
table_norm = add_multiple_testing_correction(table_norm)

table_norm_fmt = format_table(
    table_norm[["league","gamma","gamma_se","t_stat","p_value","n_matches","n_obs"]]
)
print(table_norm_fmt.to_string(index=False))
print()

# Show which leagues survive multiple-testing correction
print("Multiple-testing correction (per-league rows only, n=9 tests):")
league_rows = table_norm[table_norm["league"] != "Pooled"][
    ["league","gamma","p_value","p_bonferroni","sig_bonferroni","p_bh","sig_bh"]
].copy()
for col in ["gamma","p_value","p_bonferroni","p_bh"]:
    league_rows[col] = league_rows[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
print(league_rows.to_string(index=False))
print()

table_norm.to_csv(TABLES / "hw_replication_normalized.csv", index=False)
with open(TABLES / "hw_replication_normalized.md", "w") as f:
    f.write("## H&W Replication: Normalized Probabilities\n\n")
    f.write(f"Seasons: {HW_WINDOW[0]}–{HW_WINDOW[1]}\n\n")
    f.write(table_norm_fmt.to_markdown(index=False))
    f.write("\n\n*: p<0.10  **: p<0.05  ***: p<0.01 (uncorrected)\n\n")
    f.write("### Multiple-testing correction (n=9 league-level tests)\n\n")
    f.write(league_rows.to_markdown(index=False))
    f.write("\n\nBonferroni controls FWER; BH controls FDR (less conservative).\n")

# -----------------------------------------------------------------------
# Table 2 equivalent: normalized vs. inverse-odds comparison
# -----------------------------------------------------------------------
print("Running normalized vs. inverse-odds comparison...")
comp = compare_estimators(hw)
comp_fmt = format_table(comp[[
    "league","gamma_norm","gamma_se_norm","t_stat_norm","p_value_norm",
    "gamma_inv","gamma_se_inv","t_stat_inv","p_value_inv",
]])
print(comp_fmt.to_string(index=False))
print()

comp.to_csv(TABLES / "hw_comparison_norm_vs_inv.csv", index=False)
with open(TABLES / "hw_comparison_norm_vs_inv.md", "w") as f:
    f.write("## Normalized vs. Inverse-Odds Estimator Comparison\n\n")
    f.write(f"Seasons: {HW_WINDOW[0]}–{HW_WINDOW[1]}\n\n")
    f.write(comp_fmt.to_markdown(index=False))
    f.write("\n\n*H&W's key finding: inverse-odds estimator suppresses gamma toward 0 due to overround.*\n")

# -----------------------------------------------------------------------
# Bootstrap CIs on gamma (match-level resampling)
# -----------------------------------------------------------------------
print("Computing bootstrap CIs (n=1000, match-level resampling)...")
boot_rows = []

# Pooled
r = bootstrap_gamma(hw, n_boot=1000, seed=0)
boot_rows.append({"league": "Pooled"} | r)

for league in sorted(hw["league"].unique()):
    r = bootstrap_gamma(hw, n_boot=1000, seed=0, league=league)
    boot_rows.append({"league": league} | r)

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(TABLES / "hw_bootstrap_cis.csv", index=False)

# Print summary
print()
print("Bootstrap CIs (95%, percentile method):")
for _, row in boot_df.iterrows():
    sig = " ***" if row["ci_low"] > 0 else (" *" if row["ci_high"] < 0 else "")
    print(f"  {row['league']:8s}  gamma={row['gamma_hat']:+.4f}  "
          f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]{sig}")

print()
print(f"Tables saved to {TABLES}/")
