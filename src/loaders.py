"""
Loaders for football-data.co.uk (soccer) and tennis-data.co.uk (tennis) data.

Tolerant of file-layout variation: accepts league/season subfolders, flat
directories, or a mix. Season is recovered from the Date column when not in
the path.

Output: harmonized Parquet in two formats per sport:
  - wide: one row per match, all odds and computed probability columns
  - long: one row per (match, outcome) — what regressions consume

Both formats include normalized devigged probabilities (H&W's preferred method).
All four devig methods are in wide format; long format carries normalized only
to keep it manageable for the regression step.
"""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.devig import all_methods

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_SEASON_RE = re.compile(r"(\d{4})[_\-](\d{2,4})")


def _season_from_path(path: pathlib.Path) -> Optional[str]:
    """Try to extract a season string like '2019-2020' from a file path."""
    for part in reversed(path.parts):
        m = _SEASON_RE.search(part)
        if m:
            y1 = int(m.group(1))
            y2_raw = m.group(2)
            y2 = y1 + 1 if len(y2_raw) == 2 else int(y2_raw)
            return f"{y1}-{y2}"
    return None


def _season_from_dates(dates: pd.Series) -> str:
    """Derive season string from the Date column (July–June split)."""
    dates_parsed = pd.to_datetime(dates, dayfirst=True, errors="coerce")
    year = dates_parsed.dt.year
    month = dates_parsed.dt.month
    # Matches in Aug–Dec belong to season starting that year
    dominant_year = int(year.where(month >= 7, year - 1).mode()[0])
    return f"{dominant_year}-{dominant_year + 1}"


def _write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Soccer loader
# ---------------------------------------------------------------------------

_SOCCER_ODDS_COLS = {
    "home": ("AvgH", "B365H"),
    "draw": ("AvgD", "B365D"),
    "away": ("AvgA", "B365A"),
}


def _pick_odds_col(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    """Use primary column; fall back to secondary if primary is all-NaN."""
    if primary in df.columns and df[primary].notna().any():
        return pd.to_numeric(df[primary], errors="coerce")
    if fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    raise KeyError(f"Neither {primary!r} nor {fallback!r} found or non-null")


def _load_soccer_file(path: pathlib.Path, league: str, season: str) -> pd.DataFrame:
    """Load one soccer CSV and return a cleaned, minimally harmonized DataFrame."""
    df = pd.read_csv(path, low_memory=False)

    # Required columns
    for col in ("Date", "HomeTeam", "AwayTeam", "FTR"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col!r}")

    # Odds columns — primary Avg*, fallback B365*
    odds_h = _pick_odds_col(df, "AvgH", "B365H")
    odds_d = _pick_odds_col(df, "AvgD", "B365D")
    odds_a = _pick_odds_col(df, "AvgA", "B365A")

    odds_arr = np.column_stack([odds_h, odds_d, odds_a]).astype(float)

    # Drop rows with any missing, zero, or negative odds.
    # Zero odds are rare data errors; they produce inf in inv-odds and infinite
    # loops in the power/shin bisections.
    valid = np.isfinite(odds_arr).all(axis=1) & (odds_arr > 0).all(axis=1)
    df = df[valid].copy()
    odds_arr = odds_arr[valid]

    # Devig all four methods
    devigs = all_methods(odds_arr)

    result = pd.DataFrame({
        "match_id": range(len(df)),
        "league": league,
        "season": season,
        "date": pd.to_datetime(df["Date"].values, dayfirst=True, errors="coerce"),
        "home_team": df["HomeTeam"].values,
        "away_team": df["AwayTeam"].values,
        "result": df["FTR"].values,
        "odds_H": odds_arr[:, 0],
        "odds_D": odds_arr[:, 1],
        "odds_A": odds_arr[:, 2],
        "norm_pH": devigs["normalized"][:, 0],
        "norm_pD": devigs["normalized"][:, 1],
        "norm_pA": devigs["normalized"][:, 2],
        "add_pH": devigs["additive"][:, 0],
        "add_pD": devigs["additive"][:, 1],
        "add_pA": devigs["additive"][:, 2],
        "pow_pH": devigs["power"][:, 0],
        "pow_pD": devigs["power"][:, 1],
        "pow_pA": devigs["power"][:, 2],
        "shin_pH": devigs["shin"][:, 0],
        "shin_pD": devigs["shin"][:, 1],
        "shin_pA": devigs["shin"][:, 2],
    })

    # Carry through any _true_* columns from synthetic data (for validation)
    for col in df.columns:
        if col.startswith("_true_"):
            result[col] = df[col].values

    # Assign a global unique match_id after knowing league/season
    result["match_id"] = (
        league + "_" + season + "_" + result["match_id"].astype(str)
    )
    return result


def load_soccer(
    raw_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    recursive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan raw_dir for soccer CSVs and produce harmonized parquet.

    Layout-agnostic: discovers league from the Div column (if present) or the
    parent folder name; season from path or Date column.

    recursive=False scans only the top-level directory, which is useful when
    synthetic data lives in a subdirectory alongside real data.

    Returns (wide_df, long_df). Also writes Parquet to output_dir.
    """
    raw_dir = pathlib.Path(raw_dir)
    output_dir = pathlib.Path(output_dir)

    csv_files = sorted(raw_dir.rglob("*.csv") if recursive else raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    wide_parts = []
    for path in csv_files:
        try:
            # Peek at Div column for league; fall back to parent dir name
            peek = pd.read_csv(path, nrows=2, low_memory=False)
            league = (
                peek["Div"].iloc[0]
                if "Div" in peek.columns and pd.notna(peek["Div"].iloc[0])
                else path.parent.name
            )
            season = _season_from_path(path)
            if season is None:
                peek2 = pd.read_csv(path, usecols=["Date"], low_memory=False)
                season = _season_from_dates(peek2["Date"])

            df = _load_soccer_file(path, league, season)
            wide_parts.append(df)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")

    if not wide_parts:
        raise ValueError("No soccer files could be loaded")

    wide = pd.concat(wide_parts, ignore_index=True)

    # Deduplicate: if the user downloaded the same season twice under different
    # filenames (e.g. "D1 (6).csv" and "D1 copy.csv"), the loader will have
    # loaded both. Drop exact match duplicates keyed on league+season+teams+date.
    n_before = len(wide)
    wide = wide.drop_duplicates(
        subset=["league", "season", "date", "home_team", "away_team"],
        keep="first",
    ).reset_index(drop=True)
    n_dropped = n_before - len(wide)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} duplicate match rows "
            "(same league/season/teams/date seen in multiple files)"
        )

    # Build long format: one row per (match, outcome) with outcome ∈ {H, D, A}
    # Vectorized: stack the three outcomes, avoid iterrows which is O(n) Python loops.
    base_cols = ["match_id", "league", "season", "date"]
    long_parts = []
    for outcome, prob_col, odds_col in [
        ("H", "norm_pH", "odds_H"),
        ("D", "norm_pD", "odds_D"),
        ("A", "norm_pA", "odds_A"),
    ]:
        chunk = wide[base_cols].copy()
        chunk["outcome"] = outcome
        chunk["observed"] = (wide["result"] == outcome).astype(int)
        chunk["norm_p"] = wide[prob_col].values
        chunk["odds"] = wide[odds_col].values
        long_parts.append(chunk)
    long = pd.concat(long_parts, ignore_index=True)

    _write_parquet(wide, output_dir / "soccer_wide.parquet")
    _write_parquet(long, output_dir / "soccer_long.parquet")

    return wide, long


# ---------------------------------------------------------------------------
# Tennis loader
# ---------------------------------------------------------------------------

def _load_tennis_file(path: pathlib.Path, tour: str, year: int) -> pd.DataFrame:
    """
    Load one tennis XLSX and return per-(match, side) rows.

    tennis-data.co.uk labels Winner/Loser explicitly, so winner always has
    outcome=1. We expand to two rows per match: Winner side (outcome=1) and
    Loser side (outcome=0), symmetric with soccer's long format.
    """
    df = pd.read_excel(path, engine="openpyxl")

    for col in ("Date", "Winner", "Loser"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col!r}")

    odds_w = _pick_odds_col(df, "AvgW", "B365W")
    odds_l = _pick_odds_col(df, "AvgL", "B365L")

    odds_arr = np.column_stack([odds_w, odds_l]).astype(float)
    valid = np.isfinite(odds_arr).all(axis=1)
    df = df[valid].copy()
    odds_arr = odds_arr[valid]

    devigs = all_methods(odds_arr)

    # Wide row per match
    n = len(df)
    wide = pd.DataFrame({
        "match_id": (
            tour + "_" + str(year) + "_" + np.arange(n).astype(str)
        ),
        "tour": tour,
        "year": year,
        "date": pd.to_datetime(df["Date"].values, dayfirst=True, errors="coerce"),
        "tournament": df.get("Tournament", pd.Series([""] * n)).values,
        "surface": df.get("Surface", pd.Series([np.nan] * n)).values,
        "winner": df["Winner"].values,
        "loser": df["Loser"].values,
        "odds_W": odds_arr[:, 0],
        "odds_L": odds_arr[:, 1],
        "norm_pW": devigs["normalized"][:, 0],
        "norm_pL": devigs["normalized"][:, 1],
        "add_pW": devigs["additive"][:, 0],
        "add_pL": devigs["additive"][:, 1],
        "pow_pW": devigs["power"][:, 0],
        "pow_pL": devigs["power"][:, 1],
        "shin_pW": devigs["shin"][:, 0],
        "shin_pL": devigs["shin"][:, 1],
    })

    # Carry through ground-truth columns from synthetic data
    for col in df.columns:
        if col.startswith("_true_"):
            wide[col] = df[col].values

    # Long format: two rows per match — vectorized, no iterrows
    base_cols = ["match_id", "date"]
    w_chunk = wide[base_cols].copy()
    w_chunk["tour"] = tour
    w_chunk["year"] = year
    w_chunk["side"] = "W"
    w_chunk["player"] = wide["winner"].values
    w_chunk["observed"] = 1
    w_chunk["norm_p"] = wide["norm_pW"].values
    w_chunk["odds"] = wide["odds_W"].values

    l_chunk = wide[base_cols].copy()
    l_chunk["tour"] = tour
    l_chunk["year"] = year
    l_chunk["side"] = "L"
    l_chunk["player"] = wide["loser"].values
    l_chunk["observed"] = 0
    l_chunk["norm_p"] = wide["norm_pL"].values
    l_chunk["odds"] = wide["odds_L"].values

    return wide, pd.concat([w_chunk, l_chunk], ignore_index=True)


def load_tennis(
    raw_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan raw_dir recursively for tennis XLSX files and produce harmonized parquet.

    Tour is taken from the parent folder name ("atp"/"wta"); year from the
    filename stem or Date column.

    Returns (wide_df, long_df). Also writes Parquet to output_dir.
    """
    raw_dir = pathlib.Path(raw_dir)
    output_dir = pathlib.Path(output_dir)

    xlsx_files = sorted(raw_dir.rglob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found under {raw_dir}")

    wide_parts, long_parts = [], []
    for path in xlsx_files:
        try:
            tour = path.parent.name.lower()
            year_match = re.search(r"(\d{4})", path.stem)
            if year_match:
                year = int(year_match.group(1))
            else:
                peek = pd.read_excel(path, usecols=["Date"], engine="openpyxl")
                year = pd.to_datetime(
                    peek["Date"], dayfirst=True, errors="coerce"
                ).dt.year.mode()[0]

            wide, long = _load_tennis_file(path, tour, year)
            wide_parts.append(wide)
            long_parts.append(long)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")

    if not wide_parts:
        raise ValueError("No tennis files could be loaded")

    wide = pd.concat(wide_parts, ignore_index=True)
    long = pd.concat(long_parts, ignore_index=True)

    _write_parquet(wide, output_dir / "tennis_wide.parquet")
    _write_parquet(long, output_dir / "tennis_long.parquet")

    return wide, long
