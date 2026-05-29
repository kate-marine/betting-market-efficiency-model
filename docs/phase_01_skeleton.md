# Phase 1: Project Skeleton

**Status:** Complete  
**Files created:** `pyproject.toml`, `.gitignore`, `README.md`, `src/__init__.py`, `tests/__init__.py`  
**Directories:** `src/`, `tests/`, `data/raw/`, `data/processed/`, `results/figures/`, `results/tables/`, `notebooks/`, `scripts/`

---

## What we built

Standard Python project layout with `pyproject.toml` (modern build system, no `setup.py`) and a venv at `.venv/`. All dependencies declared up front so the project is reproducible.

`.gitignore` excludes `data/raw/` and `data/processed/` entirely — the raw data files can be large and are re-downloadable; the processed Parquet files are derivable from raw data + code.

---

## Decisions made

**pyproject.toml over requirements.txt.** The user chose this. It's the current Python standard, works cleanly with `pip install -e ".[dev]"`, and separates dev dependencies (pytest, ruff) from runtime ones.

**Virtual environment at `.venv/`.** Python 3.14 was the system version. All subsequent commands use `.venv/bin/python` explicitly to avoid any ambiguity about which interpreter is active.

**No `setup.py`.** Deprecated; `pyproject.toml` with setuptools backend handles everything.

---

## What didn't work

**VS Code interpreter mismatch.** VS Code's Python extension was pointing at the system Python (`/opt/homebrew/bin/python3`) rather than `.venv`. This caused spurious "package not installed" diagnostics in the editor even though everything ran correctly from the terminal. Fixed later (Phase 5) by adding `.vscode/settings.json` with `"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"`.

---

## Setup instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
