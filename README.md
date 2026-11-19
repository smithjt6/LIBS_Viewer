# LIBS Viewer

Python tools for loading, processing and comparing LIBS (laser-induced breakdown spectroscopy)
spectra from **Applied Spectra `.tdms` files**, with a local NIST atomic-line database for line
identification. The core library will be used by people through a simple UI and by LLM clients
through an MCP server, and the same pattern extended to ICP-MS (Agilent 7000).

> Status: early, working library (installable package `libs_viewer`, pytest suite) used from
> the notebooks in [examples/](examples/) (start with
> [quick_study.ipynb](examples/quick_study.ipynb)) and `libs-viewer analyze`. No UI or MCP server yet. See [Roadmap](#roadmap).

---

## Contents

1. [Concept of operations (CONOPS)](#concept-of-operations-conops)
2. [What works today](#what-works-today)
3. [Reference workflow: temperature-series study](#reference-workflow-temperature-series-study)
4. [Installation](#installation)
5. [Package layout](#package-layout)
6. [Roadmap](#roadmap)
7. [NIST line database](#nist-line-database)

---

## Concept of operations (CONOPS)

### Purpose

Turn a folder of raw Applied Spectra `.tdms` files into defensible, reproducible numbers and
figures (continuum-corrected spectra, line areas, area ratios across conditions) with as little
hand work as possible, and let an LLM assistant run the same analysis through the same code,
so results don't depend on who (or what) pushed the buttons.

### Scope

- **In scope:** Applied Spectra LIBS `.tdms` files (one file = one acquisition, 31-row
  parameter block, one stored spectrum per channel `Grid Location # L-S`: either the sum of
  N shots at location L with Accumulate = 1, or single shot S at location L with Accumulate = 0;
  files mixing both are handled). Spectra comparison across conditions (temperature,
  delay, energy, sample).
- **Not in scope:** other LIBS vendors' formats. The loader is deliberately format-specific
  and validates the layout instead of guessing. It is possible to adjust the data loading
  logic in the future for other formats, as long as the input matches the post-loaded tdms inputs.
- **Later:** ICP-MS (Agilent 7000), then electrochemistry / corrosion tools, built on the same
  shared core (results + provenance, plotting style, tool registry).

### Who uses it

| Actor | Interface | Can do |
|---|---|---|
| **Researcher** (lab user) | Simple UI (Streamlit), Python / notebook | Everything, including the user-only tools below |
| **LLM assistant** (Claude, Gemini, ChatGPT, or any MCP client) | MCP server | Read data, look up lines, run analyses, make figures and tables, inside the data folder it is given |
| **Reader** (colleague, advisor, reviewer) | Exported tables / figures / slides | Read results and their recorded settings |

### Operating principles

1. **Raw data is read-only.** Nothing ever writes to, moves or deletes a `.tdms` file.
2. **One code path.** The notebook, the UI and the MCP server all call the same library
   functions. MCP tools and UI pages are thin wrappers with no analysis logic of their own.
3. **Functions return data, not side effects.** Analysis returns arrays / DataFrames; plotting
   returns a figure object. Nothing calls `plt.show()` or prints as its only output.
4. **Every result carries its provenance:** source files and columns, shot scaling, baseline
   method and parameters, line windows (pixels and nm), saturation mask, code version. A number
   produced by an LLM must be reproducible by a person from that record.
5. **Defaults are physical, not magic.** Every default (SNIP window, saturation level, peak
   search tolerance) is documented with the reasoning and units, and can be overridden.
6. **Inputs are explicit.** A study never guesses which file is which sample or condition (see
   [Loading contract](#loading-contract)).

### Loading contract

A study needs to know, for every file, its **element**, **sample** and **temperature** (the
condition being compared). `study_plan` / `run_study` take that mapping one of two ways; if
neither is given (or both are), they raise an error instead of falling back to guesses.

| Mode | Input | Use when |
|---|---|---|
| **Manifest** (`manifest=`) | CSV or DataFrame, one row per file: `file` (relative to the folder), `element`, `sample`, `temperature`, optional `use_columns` (stored-spectrum names such as `Grid Location # 1-1`, separated by `;`, empty = choose automatically) | Names are inconsistent, or you want the grouping written down next to the data |
| **Pattern** (`pattern=`) | Template matched against every `.tdms` path under the folder: `{element}` and `{temperature}` required, `{sample}` optional, `*` = any text within one folder or file name, e.g. `{element}/{sample}/*{temperature}*`. Pass `temperatures=[300, 350, 400]` when a temperature is glued to other digits (`Ce1400.tdms`) | Names follow a fixed convention |

Rules for both modes:
- A file listed in the manifest that doesn't exist, or a `.tdms` file in the folder that
  doesn't match the pattern, is an **error** listing the offending files. Nothing is skipped
  silently. *(Planned: an explicit `ignore=` list for files you mean to leave out.)*
- Two files for the same sample and temperature, a file name containing two candidate
  temperatures, or a sample with more than one element is an error.
- Every sample needs a file at the reference temperature (default 300 °C) or the study fails.
- The resolved plan (file → element, sample, temperature, stored spectra used) is saved with
  the results (`plan.csv`, `settings.json`).
- Stored spectra acquired with a different gate delay or laser energy than the rest of their
  sample are dropped automatically (unless `use_columns` is given) and listed in `notes`.
  Samples whose temperatures were acquired differently (accumulated vs single shots, number of
  locations) are also listed: their per-shot averages compare, their scatter values don't.

Low-level loading of single files (`LIBSFile.from_tdms`) stays available for inspection and
doesn't need a manifest.

### Tool access tiers

Tools are grouped by what they are allowed to change. The MCP server only registers tiers 0-2;
tier 3 tools don't exist from the LLM's point of view (they are not hidden behind a flag, they
are simply not served).

| Tier | Who | What | Examples |
|---|---|---|---|
| 0: Look | User, LLM | Read-only queries, no computation of note | list files, file parameters, NIST line lookup, element lines in a window |
| 1: Analyse | User, LLM | Compute and return results; no files written | per-shot spectrum, baseline correction, line areas, ratio tables, line selection |
| 2: Report | User, LLM | Write new files to the output folder only | export tables (CSV/XLSX), save figures (PNG/SVG), session report |
| 3: User only | User (UI / CLI) | Change shared state, use the network, or run long / costly jobs | **build / rebuild the NIST database and manage its page cache**, change project defaults, baseline benchmark sweeps, anything outside the data/output folders |

The NIST database and its raw-page cache live in a per-user data folder (see
[NIST line database](#nist-line-database)) and are managed from the UI. The LLM only ever
opens the database read-only.

### How a session runs (target state)

```
 researcher (UI) / LLM (MCP)
       │  "compare Ce lines across temperatures for samples 1-4"
       ▼
 ┌──────────────┐ manifest or ┌──────────────┐  per-shot,   ┌──────────────┐
 │ data folder  │ ──pattern──▶│  LIBSFile(s) │ ──baseline──▶│   metrics    │
 │ (.tdms, RO)  │  (else err) │ + sample map │  saturation  │ areas, ratios│
 └──────────────┘             └──────────────┘              └──────┬───────┘
                                     ▲  NIST lines (RO)            │
                              ┌──────┴───────┐               ┌──────▼───────┐
                              │ nist_handbook│               │ tables, figs │──▶ output/
                              │   .sqlite    │               │ + provenance │
                              └──────────────┘               └──────────────┘
                               built/managed by user only
```

---

## What works today

Everything below except plotting is importable from `libs_viewer.libs`
(e.g. `from libs_viewer.libs import LoadLIBSData, analyze_sample`).

| Area | Function / class | Module | Notes |
|---|---|---|---|
| Load a file or folder | `LoadLIBSData(path, recursive=False)`, `.overview()` | [loading.py](src/libs_viewer/libs/loading.py) | Dict-like `{relative path: LIBSFile}`; `overview()` = one row of key settings per file; unreadable files are skipped with a warning (will become strict, see [Loading contract](#loading-contract)) |
| One file | `LIBSFile` (`spectra`, `acquisitions`, `params`, `positions`, `wavelengths`, `n_locations`, `storage`, `shots_per_location()`, `describe()`, `subset(columns=, locations=)`, `accumulate()`, `param_table()`, `varying_params()`) | [loading.py](src/libs_viewer/libs/loading.py) | `acquisitions` = one row per stored spectrum: `location`, `shot`, `shots_summed`, X/Y/Z and every instrument parameter. Locations come from the channel names (checked against the stage position, else grouped by position). `accumulate()` sums each location's spectra into one (as if Accumulate = 1). `SettingsWarning` when settings differ between stored spectra; `LayoutWarning` when one location's spectra sit at different positions, two locations share a position, a location holds repeated accumulated spectra, or `accumulate()` has nothing to do (`run_study` puts layout warnings in `notes`) |
| File names | `temperature_from_name` | [plan.py](src/libs_viewer/libs/plan.py) | `..._300C.tdms`, `..._300_3.5delay.tdms` |
| Which file is which | `study_plan(folder, pattern=... \| manifest=..., temperatures=None)` | [plan.py](src/libs_viewer/libs/plan.py) | Pattern placeholders `{element}` `{sample}` `{temperature}` `*`; manifest CSV `file, element, sample, temperature[, use_columns]`; raises if neither, on unmatched files, duplicates or ambiguous temperatures |
| Whole study in one call | `run_study(folder, elements, baseline, pattern \| manifest, ...)` → `StudyResult`; CLI `libs-viewer analyze` | [study.py](src/libs_viewer/libs/study.py) | Plan → drop stored spectra with other gate/energy → per-shot spectra + saturation → NIST lines → areas, R_direct, R_rel, location and shot scatter. `notes` lists every automatic decision; `save()` writes CSVs + `settings.json` (provenance) + `summary.txt` |
| Baseline check / study figure | `plot_baseline`, `plot_study` | [plotting.py](src/libs_viewer/libs/plotting.py) | Spectrum with fitted continuum + corrected spectrum; R_rel / R_total vs temperature per sample |
| Shot scaling | `per_shot_spectrum`, `per_shot_columns`, `location_spectra` | [processing.py](src/libs_viewer/libs/processing.py) | All counts ÷ all shots (every shot weighs the same, accumulated or not); per stored spectrum; per location |
| Continuum removal | `global_baseline` (`'snip'`, `'airpls'`) | [processing.py](src/libs_viewer/libs/processing.py) | via `pybaselines` |
| Saturation | `saturated_mask`, `exclusion_mask` | [processing.py](src/libs_viewer/libs/processing.py) | Flags clipped pixels across all files of a sample (single shots at full scale; accumulated spectra from ~80% of it, to catch shots that clipped only some of the time) |
| Total area | `total_area` | [metrics.py](src/libs_viewer/libs/metrics.py) | Optional exclusion of saturated pixels / nm ranges |
| Line area | `peak_bounds`, `line_window`, `integrate_local_peak` | [metrics.py](src/libs_viewer/libs/metrics.py) | Valley-to-valley (or fixed) window, straight local baseline through the mean of 2 edge pixels each side |
| Line choice | `select_lines` | [lines.py](src/libs_viewer/libs/lines.py) | NIST candidates filtered by detection, blending, saturation, interferents |
| Sample tables | `analyze_sample`, `areas_table`, `ratio_tables` | [analysis.py](src/libs_viewer/libs/analysis.py) | Same pixel windows at every temperature; `areas_table` measures any set of spectra (e.g. per location); `R_direct` and `R_rel` tables. `RatioWarning` when the Total is largely saturated (`saturated_share`) or a reference area is ≤ 0 (ratios NaN) |
| Spectrum plot | `plot_spectrum` | [plotting.py](src/libs_viewer/libs/plotting.py) | Log-y, zoom, peak labels, reference lines; returns a matplotlib `Figure` |
| Line lookups | `strong_lines`, `database_info` | [query.py](src/libs_viewer/nist/query.py) | Read-only; wavelengths returned in nm |
| Line database | `libs-viewer nist build` | [build.py](src/libs_viewer/nist/build.py) | User only. See [NIST line database](#nist-line-database) |

---

## Reference workflow: temperature-series study

The study this library is built around first, and how each step maps to code.

**Question:** how do LIBS line intensities of Ce and Gd samples change with sample
temperature (300 → 350 → 400 °C), both overall and line by line?

| # | Step | Code | Status |
|---|---|---|---|
| 1 | Put every spectrum on a per-shot scale (accumulated or single shots) | `per_shot_spectrum` | ✅ |
| 2 | Estimate and subtract the plasma continuum (no background spectra exist per temperature) | `global_baseline` (SNIP / airPLS) | ✅ |
| 3 | Total integrated area per temperature, ratio to 300 °C: overall intensity change | `total_area` → `ratio_tables(...)['R_direct'].loc['Total']` | ✅ |
| 4 | Pick 3-4 strong, clean lines per element (e.g. Ce II 394.20 nm) | `select_lines` | ✅ |
| 5 | Area of each line above a local straight baseline | `integrate_local_peak` | ✅ |
| 6 | Line area at T ÷ same line at 300 °C: absolute change | `R_direct` | ✅ |
| 7 | (Line ÷ total at T) ÷ (line ÷ total at 300 °C): which lines change *relative to the whole spectrum* | `R_rel` | ✅ |
| 8 | One table per sample: rows = lines + total, columns = temperatures | `analyze_sample` + `ratio_tables` | ✅ |
| 9 | Map files to sample / element / temperature from a manifest or file-name pattern | `study_plan` ([loading contract](#loading-contract)) | ✅ |
| 10 | Grouped bar chart of key ratios across samples (one for Ce, one for Gd) | `plot_study` (line plots per sample) | 🟡 bar chart planned |
| 11 | Export tables and figures sized for slides | `StudyResult.save` (CSV + `settings.json` + `summary.txt`) | 🟡 XLSX and slide-size figures planned |
| 12 | One call that runs the whole study for a folder and records provenance | `run_study`, `libs-viewer analyze` | ✅ (provenance partial, see Phase 1) |

---

## Installation

Python 3.11+. From the repository root, in the environment you use (e.g. the `libs_viewer`
conda env):

```powershell
pip install -e .              # library + `libs-viewer` command
pip install -e ".[dev]"       # + pytest
pip install -e ".[ui]"        # + Streamlit UI (phase 3)
pip install -e ".[mcp]"       # + MCP server (phase 5)
pip install -e ".[all]"
```

Then build the NIST database once:

```powershell
libs-viewer paths             # shows where the database and page cache live
libs-viewer nist build        # tier 3: downloads Tables 1-7 politely, keeps a page cache
libs-viewer nist status       # row counts
python -m pytest              # test suite (synthetic .tdms files + real NIST pages, no network)
```

Once the UI exists, its first launch will also offer to build the database if it is missing.

*Why not build it during `pip install`?* pip has no post-install step: wheels can't run code
when installed, and pip is non-interactive, so there is no way to ask "build the NIST database?"
at install time. The equivalent is a one-time setup command, plus the UI checking on first launch.
A prebuilt database download is another option for later.

---

## Package layout

One installable package. Nothing in `core`, `nist` or the analysis modules of `libs` imports
matplotlib, Streamlit or MCP. Entries marked *(planned)* don't exist yet.

```
pyproject.toml          package metadata, extras [dev] [ui] [mcp] [all], console script `libs-viewer`
src/libs_viewer/
  core/                 shared by LIBS, ICP-MS, ...
    paths.py            per-user data folder (NIST database + page cache)
    results.py          (planned) Result objects: data + provenance record
    tools.py            (planned) tool registry (name, tier, schema, function)
    style.py            (planned) figure style
  nist/
    build.py            database builder (formerly scraper.py), tier 3
    query.py            read-only line lookups, database status
  libs/
    loading.py          LIBSFile (spectra + one row per stored spectrum), LoadLIBSData, overview
    plan.py             study_plan (manifest / pattern), file-name parsing
    processing.py       per-shot scaling, location spectra, saturation mask, baseline correction
    metrics.py          total area, peak bounds, local-baseline line area
    lines.py            line selection and interferent checks (uses nist.query)
    analysis.py         analyze_sample, ratio tables
    study.py            run_study / StudyResult: the whole temperature study in one call
    plotting.py         plot_spectrum, plot_baseline, plot_study; (planned) overlays
    benchmarks.py       (planned) baseline-method comparison, tier 3
  icpms/                (planned) same layering
  ui/                   (planned) Streamlit app (all tiers, incl. NIST management)
  mcp/                  (planned) MCP server over core/tools.py (tiers 0-2 only)
  cli.py                `libs-viewer paths | nist build | nist status`
tests/                  pytest; synthetic .tdms files, real NIST pages in tests/fixtures/nist
examples/               notebooks: quick_study, loading_data, end_to_end_temp_study (data/ is not in git)
```

---

## Roadmap

### Phase 1: Package and refactor
- [x] `pyproject.toml` (`src/` layout, extras, `libs-viewer` console script), minimum dependency
  versions (`nptdms`, `numpy`, `pandas`, `scipy`, `pybaselines`, `matplotlib`, `requests`, `platformdirs`)
- [x] Move into `libs_viewer` per the layout above; notebook imports updated
- [x] NIST builder as `libs_viewer.nist.build`; database and page cache in the per-user data folder
- [x] pytest suite with synthetic spectra and `.tdms` files, plus an offline NIST build from real pages
- [x] Plotting returns figures; remove hard-coded titles / element names
- [ ] Result objects with provenance, used by `analyze_sample` and the ratio tables. *Partly done:*
  `run_study` saves the plan, inputs, version, picked lines and notes (`settings.json`). Still
  missing: line windows (pixels and nm), saturation mask, shot scaling per file, baseline
  parameters, and provenance on `analyze_sample` / `ratio_tables` when called directly

### Phase 2: Complete the reference workflow
- [x] Loading contract: manifest loader, pattern loader, strict errors when neither is given (`study_plan`)
- [x] One-call study workflow: `run_study` / `libs-viewer analyze` (the obvious single MCP tool later)
- [ ] Grouped bar charts of line and total ratios across samples
- [ ] Overlay plot of one line window across temperatures, with its local baseline
- [ ] Baseline diagnostic plot (raw, continuum, corrected, masked pixels). *Partly done:*
  `plot_baseline` shows raw, continuum and corrected; masked pixels still to add
- [ ] Export: XLSX tables, PNG / SVG figures at slide size. *Done:* CSV tables and the JSON
  provenance sidecar (`StudyResult.save`)
- [ ] `ignore=` list for files to leave out of a pattern scan
- [ ] Document every default with its reasoning and units (`min_prominence`, `peak_tol`,
  saturation level, SNIP window), per operating principle 5

### Phase 3: Simple UI (Streamlit)
Kept deliberately simple; it only calls the library.
- [ ] Load: pick folder + manifest or pattern, show the resolved plan (`study_plan`)
- [ ] Inspect: file parameters, spectrum plot with NIST line overlay
- [ ] Study: choose elements / lines / baseline, show ratio tables and bar charts, export
- [ ] Settings (user only): NIST database status, build / rebuild, page cache location and size

### Phase 4: Baseline benchmark (user-only)
Choose a continuum method on evidence rather than habit. There's no ground-truth continuum in
real data, so score methods two ways:
- [ ] **Synthetic:** known lines (Voigt profiles) on a known continuum with realistic noise,
  built from line-free regions of real spectra; score continuum error and line-area recovery
- [ ] **Real data:** residual in line-free regions (should sit at zero, not go negative),
  fraction of negative corrected area, stability of the ratio tables across parameter values,
  shot-to-shot scatter of ratios
- [ ] Methods from `pybaselines` (SNIP, airPLS, arPLS, asLS, iarPLS, morphological, rolling ball,
  modified polynomial, ...), each with a parameter sweep; output a ranked table and plots
- [ ] Run from the UI or CLI

### Phase 5: MCP server
- [ ] Official MCP Python SDK; tools generated from the tier 0-2 registry
- [ ] Session handles for loaded data (`dataset_id`) so tools pass references, not arrays
- [ ] Small JSON results, tables as records, figures as images
- [ ] Path sandbox: one configured data folder (read-only) and one output folder; NIST database read-only
- [ ] Tested with Claude Desktop / Claude Code first, then other MCP-capable clients

### Phase 6: ICP-MS and beyond
- [ ] Agilent 7000 export loader (`.xlsx`) on the same core
- [ ] Electrochemistry / corrosion tools

---

## NIST line database

[nist/build.py](src/libs_viewer/nist/build.py) builds `nist_handbook.sqlite` from the
[NIST Handbook of Basic Atomic Spectroscopic Data](https://physics.nist.gov/PhysRefData/Handbook/),
Tables 1-7 for Z = 1-99. This will be released in a future update or potentially a download link.
We want to be friendly to NIST servers so the DB is built once and stored on disk.

Building or rebuilding it is a **user-only** (tier 3) action. The database and the raw-page
cache live in the per-user data folder (`%LOCALAPPDATA%\libs_viewer\` on Windows; set
`LIBS_VIEWER_DATA` to use another folder; `libs-viewer paths` shows it). The UI's settings page
will manage them too. Analysis code opens the database read-only.

```powershell
libs-viewer nist build                         # full build, keeps raw pages for fast rebuilds
libs-viewer nist build --elements Ce Gd --force  # re-parse some elements (from cache if present)
libs-viewer nist build --no-cache              # don't read or keep raw pages
libs-viewer nist status
```

With every page cached, a rebuild needs no network at all (robots.txt is only requested when
a page has to be downloaded).

| Table | Contents |
|---|---|
| `atomic_data` | Atomic number and weight, ground configuration / term / J / g, ionization energy (I, II) |
| `strong_lines` | Strong lines with intensity and flags |
| `persistent_lines` | Persistent lines with Aki, lower / upper level energies, configurations, terms, J, g |
| `energy_levels` | Energy levels (I, II), including ionization limits |
| `bibliographic_refs` | Citation for each reference code |

**Wavelengths** (`wavelength_A`) are in **Å, exactly as published** (divide by 10 for nm).
The Handbook uses vacuum wavelengths below 2000 Å and air above; each line's `medium` column
says which, and `wavelength_air_A` gives the air value where defined.

If the schema changes, delete `nist_handbook.sqlite` before rebuilding (tables are only created
when missing).

---

## Contact Information

If you encounter a bug or have a feature you would like to add, please leave an issue or make a pull request

Author: John Smith
Affiliation: University of Utah
