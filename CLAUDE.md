# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Early-stage Python project for loading and plotting LIBS (laser-induced breakdown spectroscopy) data from Applied Spectra `.tdms` files, with ICP-MS (Agilent 7000) support planned. There is no build system, `requirements.txt`/`pyproject.toml`, test suite, or linter yet. Dependencies in use: `nptdms`, `pandas`, `numpy`, `scipy`, `matplotlib`. Work is exercised interactively through [example.ipynb](example.ipynb) in VS Code (`.vscode/` is gitignored).

The roadmap in [README.md](README.md) is the intended direction: a directory-pointing loader that maps filenames to data (dict of filename → data), shot normalization/accumulation, fast full-spectrum or windowed plotting with a LIBS line database plus custom wavelengths, ICP-MS integration, a UI, and an MCP/tool layer for LLM use. Detailed docstrings are an explicit goal for future sensor/technique integrations.

## Direction and priorities

This is meant to become a reusable lab tool for LIBS (and later ICP-MS) data, possibly exposed as an MCP server for lab work. Current priority is the **loading layer** (point at a file or directory, get data keyed by filename). Plotting, a database of atomic line data (to be scraped from an atomic handbook), and the UI/MCP come after. Design loaders and plotters as importable, UI-independent functions/classes that return data (DataFrames) rather than only printing or calling `plt.show()`, so they can later be wrapped as MCP tools.

## Layout

- `src/LIBS/` — [tdms_loader.py](src/LIBS/tdms_loader.py) (`LoadLIBSData`: reads one TDMS file, `to_df()` builds a DataFrame with one column per TDMS channel, `param_table()` prints the instrument parameter block) and [spectra_viewer.py](src/LIBS/spectra_viewer.py) (`PlotLIBSData`: matplotlib spectrum plot with peak labelling, log-y, wavelength zoom, and dashed reference lines).
- `src/ICPMS/` — empty placeholder.
- `data/libs_data/*.tdms`, `data/icpms_data/*.xlsx` — sample data.

Note: git status shows the modules as moved from `src/` into `src/LIBS/` (old paths deleted, new directory untracked). Use the `src/LIBS/` paths.

## Things to know

- `param_table()` assumes the Applied Spectra format: the parameter table occupies rows 0–31 of the DataFrame (first column = names with a leading `'`, second = values).
- `PlotLIBSData.plot_tdms` is work in progress and currently broken: it references `wavlengths` (typo for `wavelengths`) and calls a bare `to_nm(...)`, while the helper is the method `self._to_nm`. It also hard-codes "Cerium Reference lines" and a "Ce Sample" default title. `self.df` is stored but unused. Fix these when touching it.
- Peak detection runs on the full spectrum (`prominence=median(counts)`) even when zoomed, so zooming doesn't change which peaks exist; label spacing scales with the zoom window.
