"""
End-to-end smoke test: synthetic data → loaders → harmonized parquet.
Run this to confirm the pipeline works before real data arrives.

Usage:
    python scripts/smoke_test.py
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
from src.synth import generate_soccer, generate_tennis
from src.loaders import load_soccer, load_tennis

SYNTH_DIR = pathlib.Path("data/raw/synthetic")
PROC_DIR  = pathlib.Path("data/processed")

def main():
    print("=== Generating synthetic data ===")
    soccer_paths = generate_soccer(
        SYNTH_DIR / "soccer",
        leagues=["E0", "D1", "SP1"],
        seasons=["2019-2020", "2020-2021", "2021-2022"],
        variant="flb",
        seed=0,
    )
    print(f"  Soccer: {len(soccer_paths)} files")

    tennis_paths = generate_tennis(
        SYNTH_DIR / "tennis",
        tours=["atp", "wta"],
        years=[2019, 2020, 2021],
        variant="efficient",
        seed=0,
    )
    print(f"  Tennis: {len(tennis_paths)} files")

    print("\n=== Loading soccer ===")
    soccer_wide, soccer_long = load_soccer(SYNTH_DIR / "soccer", PROC_DIR)
    print(f"  Wide:  {soccer_wide.shape[0]:,} rows × {soccer_wide.shape[1]} cols")
    print(f"  Long:  {soccer_long.shape[0]:,} rows × {soccer_long.shape[1]} cols")
    print(f"  Leagues:  {sorted(soccer_wide['league'].unique())}")
    print(f"  Seasons:  {sorted(soccer_wide['season'].unique())}")

    print("\n=== Loading tennis ===")
    tennis_wide, tennis_long = load_tennis(SYNTH_DIR / "tennis", PROC_DIR)
    print(f"  Wide:  {tennis_wide.shape[0]:,} rows × {tennis_wide.shape[1]} cols")
    print(f"  Long:  {tennis_long.shape[0]:,} rows × {tennis_long.shape[1]} cols")
    print(f"  Tours: {sorted(tennis_wide['tour'].unique())}")
    print(f"  Years: {sorted(tennis_wide['year'].unique())}")

    print("\n=== Sanity checks ===")
    # Normalized probs sum to 1 per match
    soccer_prob_sums = soccer_wide[["norm_pH", "norm_pD", "norm_pA"]].sum(axis=1)
    assert (soccer_prob_sums - 1.0).abs().max() < 1e-5, "Soccer prob sums broken"
    print("  Soccer prob sums: OK")

    tennis_prob_sums = tennis_wide[["norm_pW", "norm_pL"]].sum(axis=1)
    assert (tennis_prob_sums - 1.0).abs().max() < 1e-5, "Tennis prob sums broken"
    print("  Tennis prob sums: OK")

    # Each match has exactly one observed outcome
    soccer_obs = soccer_long.groupby("match_id")["observed"].sum()
    assert (soccer_obs == 1).all(), "Soccer observed counts broken"
    print("  Soccer observed=1 per match: OK")

    tennis_obs = tennis_long.groupby("match_id")["observed"].sum()
    assert (tennis_obs == 1).all(), "Tennis observed counts broken"
    print("  Tennis observed=1 per match: OK")

    print("\n=== Parquet output ===")
    for name in ("soccer_wide", "soccer_long", "tennis_wide", "tennis_long"):
        path = PROC_DIR / f"{name}.parquet"
        df = pd.read_parquet(path)
        print(f"  {name}.parquet: {len(df):,} rows")

    print("\nAll checks passed. Pipeline is ready for real data.")


if __name__ == "__main__":
    main()
