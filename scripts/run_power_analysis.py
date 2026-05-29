"""
Phase 9c: Power analysis for the H&W per-league FLB regressions.

Shows that "no detectable FLB" for E0, D1, SP1 is NOT evidence of market
efficiency — it is evidence of insufficient power.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.replication import run_hw_table, add_multiple_testing_correction
from src.power import analytical_power, analytical_mde, power_table, simulate_power_curve

LONG   = pathlib.Path("data/processed/soccer_long.parquet")
WIDE   = pathlib.Path("data/processed/soccer_wide.parquet")
TABLES = pathlib.Path("results/tables")
FIGS   = pathlib.Path("results/figures")
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

HW_WINDOW    = ("2015-2016", "2021-2022")
ALPHA        = 0.05
POWER_TARGET = 0.80
POOLED_GAMMA = 0.0459   # from Phase 5

# ---------------------------------------------------------------------------
# 1. MDE table
# ---------------------------------------------------------------------------
print("Loading data and running H&W regressions...")
long = pd.read_parquet(LONG)
hw = long[long["season"].between(*HW_WINDOW)].copy()

reg_table = run_hw_table(hw, p_col="norm_p")
reg_table = add_multiple_testing_correction(reg_table)

mde_tbl = power_table(reg_table, alpha=ALPHA, power=POWER_TARGET, pooled_gamma=POOLED_GAMMA)

print(f"\n=== Power table (α={ALPHA}, target power={POWER_TARGET:.0%}) ===")
print(f"Reference: pooled γ = {POOLED_GAMMA:.4f}\n")
header = f"{'League':8s}  {'γ_obs':>8s}  {'SE':>8s}  {'n_match':>8s}  "
header += f"{'MDE':>8s}  {'power@obs':>10s}  {'power@pool':>11s}  {'powered?':>9s}"
print(header)
for _, row in mde_tbl.iterrows():
    powered = "YES" if row["adequately_powered"] else "no"
    print(f"{row['league']:8s}  {row['gamma']:+8.4f}  {row['gamma_se']:8.4f}  "
          f"{row['n_matches']:8,}  {row['mde']:8.4f}  "
          f"{row['power_at_obs']:10.3f}  {row['power_at_pooled']:11.3f}  {powered:>9s}")

# Save
mde_tbl.to_csv(TABLES / "power_analysis_mde_table.csv", index=False)

with open(TABLES / "power_analysis_mde_table.md", "w") as f:
    f.write(f"## Power Analysis: Minimum Detectable Effect per League\n\n")
    f.write(f"α = {ALPHA}, target power = {POWER_TARGET:.0%}. "
            f"MDE = (z_{{α/2}} + z_{{1-β}}) × SE_cluster.\n\n")
    f.write(f"Reference: pooled γ = {POOLED_GAMMA:.4f} (robust across bootstrap CIs).\n\n")
    disp = mde_tbl.copy()
    for col in ["gamma","gamma_se","mde","power_at_obs","power_at_pooled"]:
        disp[col] = disp[col].map("{:.4f}".format)
    disp["n_matches"] = disp["n_matches"].map("{:,}".format)
    f.write(disp.to_markdown(index=False))
    f.write("\n\n*power@pool*: power at the true effect if γ = pooled estimate.\n"
            "Leagues with power@pool < 0.80 would miss a pooled-sized effect most of the time.\n")

# ---------------------------------------------------------------------------
# 2. Power curves
# ---------------------------------------------------------------------------
print("\nGenerating power curves...")
gamma_grid = np.linspace(0, 0.30, 200)

fig, ax = plt.subplots(figsize=(9, 5.5))

cmap = plt.cm.tab10
leagues_of_interest = ["E0", "D1", "SP1", "I1", "E2", "Pooled"]

for k, row in enumerate(mde_tbl[mde_tbl["league"].isin(leagues_of_interest)].itertuples()):
    se = row.gamma_se
    powers = [analytical_power(g, se, alpha=ALPHA) for g in gamma_grid]
    color  = cmap(k)
    ax.plot(gamma_grid, powers, color=color, lw=1.8, label=row.league)
    # Mark observed gamma
    p_obs = analytical_power(abs(row.gamma), se, alpha=ALPHA)
    ax.scatter([abs(row.gamma)], [p_obs], color=color, s=50, zorder=5)

ax.axhline(POWER_TARGET, color="black", lw=0.8, ls="--", label=f"{POWER_TARGET:.0%} power threshold")
ax.axvline(POOLED_GAMMA, color="gray",  lw=0.8, ls=":",  label=f"Pooled γ = {POOLED_GAMMA:.3f}")

ax.set_xlabel("True γ (favorite-longshot bias)")
ax.set_ylabel(f"Power (α = {ALPHA})")
ax.set_title("Power curves: what FLB could we detect in each league?")
ax.set_xlim(0, 0.30)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9, loc="upper left")
ax.annotate("Dots = observed |γ|", xy=(0.68, 0.05), xycoords="axes fraction", fontsize=8,
            color="gray")

plt.tight_layout()
fp = FIGS / "power_curves_flb_by_league.png"
plt.savefig(fp, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {fp}")

# ---------------------------------------------------------------------------
# 3. Simulation validation (E0 only — most policy-relevant "no FLB" league)
# ---------------------------------------------------------------------------
print(f"\nRunning simulation validation for E0 (n_sim=200)...")
wide = pd.read_parquet(WIDE)
wide_e0 = wide[wide["league"] == "E0"].copy()
long_e0 = hw[(hw["league"] == "E0") & (hw["outcome"].isin(["H","D","A"]) if "outcome" in hw.columns else True)].copy()

# Use the soccer_long format
long_e0 = long[(long["league"] == "E0") & long["season"].between(*HW_WINDOW)].copy()

# Gamma values to test: span the E0 MDE region
e0_se  = float(mde_tbl.loc[mde_tbl["league"]=="E0", "gamma_se"].iloc[0])
e0_mde = analytical_mde(e0_se)
gamma_sim_vals = [0.0, e0_mde * 0.5, e0_mde, e0_mde * 1.5, e0_mde * 2.0]
gamma_sim_vals = [round(g, 4) for g in gamma_sim_vals]

print(f"  E0 SE={e0_se:.4f}, analytical MDE={e0_mde:.4f}")
print(f"  Gamma values: {gamma_sim_vals}")
print(f"  Running simulation...")

sim_results = simulate_power_curve(
    wide_df=wide_e0.reset_index(drop=True),
    long_df=long_e0,
    gamma_vals=gamma_sim_vals,
    n_sim=200,
    alpha=ALPHA,
    seed=0,
)

# Compare analytical vs simulation
print(f"\n  Analytical vs. simulation validation for E0:")
print(f"  {'gamma':>8s}  {'analytical':>12s}  {'simulation':>12s}  {'diff':>8s}")
for _, row in sim_results.iterrows():
    analytical = analytical_power(row["gamma"], e0_se, alpha=ALPHA)
    diff = row["power_sim"] - analytical
    print(f"  {row['gamma']:8.4f}  {analytical:12.3f}  {row['power_sim']:12.3f}  {diff:+8.3f}")

sim_results.to_csv(TABLES / "power_simulation_e0.csv", index=False)

# Add simulation to power curve plot
fig, ax = plt.subplots(figsize=(8, 5))
gamma_fine = np.linspace(0, 0.30, 200)
powers_analytical = [analytical_power(g, e0_se, alpha=ALPHA) for g in gamma_fine]
ax.plot(gamma_fine, powers_analytical, "b-", lw=2, label="Analytical (E0)")
ax.errorbar(sim_results["gamma"], sim_results["power_sim"],
            yerr=1.96 * sim_results["se_sim"], fmt="ro", ms=6, capsize=4,
            label="Simulation ± 95% CI")
ax.axhline(POWER_TARGET, color="black", lw=0.8, ls="--", label=f"{POWER_TARGET:.0%} power")
ax.axvline(e0_mde, color="blue", lw=0.8, ls=":", label=f"Analytical MDE = {e0_mde:.3f}")
ax.axvline(POOLED_GAMMA, color="gray", lw=0.8, ls=":", label=f"Pooled γ = {POOLED_GAMMA:.3f}")
ax.set_xlabel("True γ"); ax.set_ylabel("Power"); ax.set_xlim(0, 0.30); ax.set_ylim(0, 1.02)
ax.set_title("E0 (Premier League): Analytical vs. Simulation Power")
ax.legend(fontsize=9)
plt.tight_layout()
fp = FIGS / "power_validation_e0_simulation.png"
plt.savefig(fp, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {fp}")

print(f"\n=== Key takeaways ===")
for _, row in mde_tbl[mde_tbl["league"] != "Pooled"].sort_values("mde").iterrows():
    pap = row["power_at_pooled"]
    verdict = "✗ underpowered" if pap < POWER_TARGET else "✓ adequately powered"
    print(f"  {row['league']:4s}  MDE={row['mde']:.4f}  power@pooled={pap:.3f}  {verdict}")

print(f"\nAll outputs saved to {TABLES}/ and {FIGS}/")
