"""
Phase 9d: Efficient-markets null simulation.

For each league: simulate outcomes under H0 (market is efficient), run H&W
regression, compare distribution of null gammas to the observed value.
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
from src.null_sim import build_setups, run_simulation, simulation_pvalue, build_results_table, joint_test

LONG   = pathlib.Path("data/processed/soccer_long.parquet")
WIDE   = pathlib.Path("data/processed/soccer_wide.parquet")
TABLES = pathlib.Path("results/tables")
FIGS   = pathlib.Path("results/figures")
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

HW_WINDOW = ("2015-2016", "2021-2022")
N_SIM     = 2000
SEED      = 0

# ---------------------------------------------------------------------------
# 1. Load data and run real regression
# ---------------------------------------------------------------------------
print("Loading data...")
long = pd.read_parquet(LONG)
wide = pd.read_parquet(WIDE)
hw_long = long[long["season"].between(*HW_WINDOW)].copy()

print("Running H&W regression (real data)...")
reg_table = run_hw_table(hw_long, p_col="norm_p")
reg_table = add_multiple_testing_correction(reg_table)

league_rows = reg_table[reg_table["league"] != "Pooled"].copy()
print(f"  {len(league_rows)} leagues, pooled gamma = {reg_table.loc[reg_table['league']=='Pooled','gamma'].iloc[0]:.4f}")

# ---------------------------------------------------------------------------
# 2. Precompute and run simulation
# ---------------------------------------------------------------------------
print(f"\nPrecomputing regression matrices...")
setups = build_setups(long, wide, reg_table, season_range=HW_WINDOW)
print(f"  {len(setups)} leagues set up")
for s in setups:
    print(f"  {s.league}: {s.n_matches} matches, gamma_obs={s.gamma_obs:+.4f}, SE={s.se_obs:.4f}")

print(f"\nRunning {N_SIM} null simulations...")
import time
t0 = time.time()
null_gammas = run_simulation(setups, n_sim=N_SIM, seed=SEED)
print(f"Done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. Per-league comparison table
# ---------------------------------------------------------------------------
results = build_results_table(setups, null_gammas, reg_table)

print("\n=== Per-league: parametric vs. simulation p-values ===")
print(f"{'League':6s}  {'gamma':>8s}  {'t_obs':>7s}  {'p_param':>10s}  {'p_sim':>8s}  {'ratio':>7s}  note")
for _, row in results.iterrows():
    ratio = row["ratio_sim_param"]
    note = ""
    if not pd.isna(ratio):
        if ratio > 2.0:
            note = "← parametric anti-conservative"
        elif ratio < 0.5:
            note = "← parametric too conservative"
    print(f"{row['league']:6s}  {row['gamma_obs']:+8.4f}  {row['t_obs']:7.3f}  "
          f"{row['p_parametric']:10.4f}  {row['p_simulation']:8.4f}  {ratio:7.2f}  {note}")

# ---------------------------------------------------------------------------
# 4. Joint tests
# ---------------------------------------------------------------------------
print("\n=== Joint tests ===")
for method in ["max_t", "sum_chi2"]:
    jt = joint_test(setups, null_gammas, method=method)
    print(f"  {method}: obs_stat={jt['obs_stat']:.3f}  p_joint={jt['p_joint']:.4f}  "
          f"(n_sim={jt['n_sim']}, n_leagues={jt['n_leagues']})")

# Null distribution properties
print("\n=== Null distribution properties ===")
print(f"{'League':6s}  {'null_mean':>10s}  {'null_SD':>8s}  {'|γ_obs|/SD':>10s}")
for _, row in results.iterrows():
    ratio_to_sd = abs(row["gamma_obs"]) / row["null_gamma_sd"] if row["null_gamma_sd"] > 0 else np.nan
    print(f"{row['league']:6s}  {row['null_gamma_mean']:+10.4f}  {row['null_gamma_sd']:8.4f}  {ratio_to_sd:10.2f}σ")

# ---------------------------------------------------------------------------
# 5. Figures: null distributions with observed gamma marked
# ---------------------------------------------------------------------------
print("\nGenerating figures...")
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
fig.suptitle(f"Null distributions of γ̂ under market efficiency\n"
             f"(n_sim={N_SIM}, H&W window {HW_WINDOW[0]}–{HW_WINDOW[1]})", fontsize=11)

for ax, setup in zip(axes.flat, setups):
    null = null_gammas[setup.league]
    ax.hist(null, bins=50, color="#4292C6", alpha=0.7, density=True, edgecolor="none")
    ax.axvline(setup.gamma_obs, color="#D6604D", lw=2, label=f"γ_obs={setup.gamma_obs:+.3f}")
    ax.axvline(-abs(setup.gamma_obs), color="#D6604D", lw=1, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls=":")

    p_sim = simulation_pvalue(setup.gamma_obs, null)
    p_param = float(results.loc[results["league"]==setup.league, "p_parametric"].iloc[0])
    ax.set_title(f"{setup.league}  p_sim={p_sim:.3f}  p_param={p_param:.3f}", fontsize=9)
    ax.set_xlabel("γ̂ (null)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

plt.tight_layout()
fp = FIGS / "null_distributions_flb_all_leagues.png"
plt.savefig(fp, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {fp}")

# ---------------------------------------------------------------------------
# 6. Save tables
# ---------------------------------------------------------------------------
results.to_csv(TABLES / "null_simulation_results.csv", index=False)

jt_max = joint_test(setups, null_gammas, method="max_t")
jt_chi = joint_test(setups, null_gammas, method="sum_chi2")

with open(TABLES / "null_simulation_results.md", "w") as f:
    f.write(f"## Efficient-Markets Null Simulation: Per-League Results\n\n")
    f.write(f"n_sim = {N_SIM}. SE of simulation p-value at p=0.05 ≈ {np.sqrt(0.05*0.95/N_SIM):.3f}.\n\n")
    disp = results[["league","gamma_obs","t_obs","p_parametric","p_simulation",
                    "ratio_sim_param","null_gamma_mean","null_gamma_sd"]].copy()
    for col in ["gamma_obs","t_obs","p_parametric","p_simulation",
                "ratio_sim_param","null_gamma_mean","null_gamma_sd"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    f.write(disp.to_markdown(index=False))
    f.write(f"\n\n### Joint tests\n\n")
    f.write(f"- max |t| across leagues: obs={jt_max['obs_stat']:.3f}, p={jt_max['p_joint']:.4f}\n")
    f.write(f"- sum t² across leagues:  obs={jt_chi['obs_stat']:.3f}, p={jt_chi['p_joint']:.4f}\n")
    f.write(f"\n*ratio_sim_param > 2 → parametric test anti-conservative; < 0.5 → too conservative*\n")

print(f"\nAll outputs saved to {TABLES}/ and {FIGS}/")
