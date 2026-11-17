# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Installable Python package `libs_viewer` for loading, processing and comparing LIBS spectra from
**Applied Spectra `.tdms` files only**, with a local NIST Handbook line database. [README.md](README.md)
holds the CONOPS (actors, operating principles, loading contract, tool access tiers) and the
phased roadmap; read it before adding features. Later: a simple Streamlit UI, an MCP server over
the same library, ICP-MS (Agilent 7000).

## Environment and commands

- Conda env `libs_viewer` (Python 3.12): `C:\Users\Jack\anaconda3\envs\libs_viewer\python.exe`.
  The package is installed editable there (`pip install -e ".[dev]"`). The base conda Python is
  older (numpy 1.x, no pybaselines) and will not work.
- Tests: `python -m pytest` from the repo root (no network; NIST tests build from `tests/fixtures/nist`).
- CLI: `libs-viewer analyze | paths | nist status | nist build` (`Scripts\libs-viewer.exe` in the env).
- `data/` (real `.tdms` files) is gitignored and may be empty; tests use synthetic files from
  [tests/conftest.py](tests/conftest.py).

## Layout

- [src/libs_viewer/libs/](src/libs_viewer/libs/): `loading` (LIBSFile, LoadLIBSData, study_plan, temperature_from_name),
  `processing` (per-shot scaling, baseline, saturation), `metrics` (total / line areas),
  `lines` (NIST line selection), `analysis` (analyze_sample, ratio tables), `study` (run_study: the
  whole pipeline in one call, mirrored step by step in examples/end_to_end_temp_study.ipynb),
  `plotting` (returns Figure).
  `libs/__init__` re-exports everything except plotting.
- [src/libs_viewer/nist/](src/libs_viewer/nist/): `build` (Handbook scraper/parsers, user-only),
  `query` (read-only lookups; wavelengths stored in Å as published, returned in nm).
- [src/libs_viewer/core/paths.py](src/libs_viewer/core/paths.py): per-user data folder
  (`%LOCALAPPDATA%\libs_viewer`, override `LIBS_VIEWER_DATA`) holding the NIST DB and page cache.
- A legacy `nist_handbook.sqlite` / `nist_cache/` may still sit in the repo root (gitignored).

## Rules

- Raw `.tdms` files are read-only. Never write, move or delete them.
- Analysis code returns data (arrays/DataFrames); plotting returns a `matplotlib.figure.Figure`
  built without pyplot. No `plt.show()`, no print-only functions.
- Nothing in `core`, `nist` or the analysis modules imports matplotlib, Streamlit or MCP.
- Tier 3 (user-only) actions, such as building the NIST DB, must never be exposed as LLM/MCP tools.
- Study-level loading must be explicit (manifest CSV or file-name pattern), raising if neither is
  given. Don't add filename guessing.
- Keep the NIST parsers' behaviour pinned by `tests/test_nist.py`; validate parser changes across
  all 99 elements (Ek − Ei = 1/λ_vac for persistent lines), not just Ce/Gd.
