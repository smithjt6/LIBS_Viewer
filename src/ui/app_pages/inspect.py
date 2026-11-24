# ui/app_pages/inspect.py

'''
The Inspect page: pick one loaded file and look at its spectra, settings and saturation.
Reads st.session_state["source"] (set on the Load page).
'''

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs_viewer.core.manifest import (
    clean_manifest,
    draft_manifest,
    labelled_rows,
    manifest_csv,
    missing_labels,
)
from libs_viewer.libs import (
    BASELINES,
    baseline_settings,
    guess_labels,
    location_spectra,
    per_shot_columns,
    per_shot_spectrum,
    saturated_pixels,
    mask_ranges,
)
from libs_viewer.libs.processing import global_baseline
from libs_viewer.nist.query import elements as nist_elements, strong_lines


# Fixed colours for the single-spectrum (Mean) view: each role reads at a glance.
SPECTRUM_COLOR = "#4DA3FF"    # blue: the measured spectrum
BASELINE_COLOR = "#FF3B6B"    # red/pink: the fitted baseline, drawn thick and dashed
CORRECTED_COLOR = "#3DDC84"   # green: spectrum minus baseline
LINE_COLORS = ["#FFB000", "#C77DFF", "#00D1D1", "#FF7F50", "#A0E426", "#F5F5F5"]
SERIES_COLORS = ["#4DA3FF", "#3DDC84", "#B388FF", "#FFD23F", "#2EC4B6", "#F78FB3"]


def _y_range(arrays: list, log: bool) -> list[float] | None:
    '''Axis range (in plotly's units: log10 for a log axis) that fits `arrays` with a margin.'''
    if not arrays:
        return None
    y = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    y = y[np.isfinite(y)]
    if log:
        y = y[y > 0]
    if y.size == 0:
        return None
    if log:
        lo, hi = np.log10(y.min()), np.log10(y.max())
        pad = 0.05 * max(hi - lo, 0.1)
    else:
        lo, hi = float(y.min()), float(y.max())
        pad = 0.05 * (hi - lo or 1.0)
    return [lo - pad, hi + pad]


def spectrum_figure(f, view: str, method: str, params: dict,
                    show_baseline: bool, log_scale: bool = True,
                    wl_bounds: tuple[float, float] | None = None,
                    marked: dict[str, pd.DataFrame] | None = None) -> go.Figure:
    wl = f.wavelengths
    # Baselines are always fitted on the full spectrum; the bounds only crop what is drawn.
    keep = np.ones(len(wl), dtype=bool) if wl_bounds is None else (
        (wl >= wl_bounds[0]) & (wl <= wl_bounds[1]))

    if view == "Mean":
        series = {"Mean": per_shot_spectrum(f)}
    elif view == "Locations":
        table = location_spectra(f)
        series = {f"Location {column}": table[column].to_numpy(dtype=float)
                  for column in table.columns}
    else:
        table = per_shot_columns(f)
        series = {column: table[column].to_numpy(dtype=float)
                  for column in table.columns}

    saturated = saturated_pixels(f).to_numpy().any(axis=1)
    shaded_ranges = mask_ranges(wl, saturated)

    if show_baseline:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[2, 1],
            vertical_spacing=0.1,
            subplot_titles=("Spectrum and baseline", "Baseline-corrected"),
        )
    else:
        fig = go.Figure()

    single = len(series) == 1
    top_vals, bottom_vals = [], []     # visible data, used to fit the y axes to the crop
    for i, (name, intensities) in enumerate(series.items()):
        color = SPECTRUM_COLOR if single else SERIES_COLORS[i % len(SERIES_COLORS)]
        if show_baseline:
            corrected, baseline = global_baseline(
                wl, intensities, method=method, **params
            )
            corrected, baseline = np.asarray(corrected)[keep], np.asarray(baseline)[keep]
            intensities = np.asarray(intensities)[keep]
            top_vals += [intensities, baseline]
            bottom_vals.append(corrected)
            fig.add_trace(go.Scatter(
                x=wl[keep],
                y=intensities,
                mode="lines",
                name=f"{name} spectrum",
                line={"color": color, "width": 1},
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=wl[keep],
                y=baseline,
                mode="lines",
                name=f"{name} baseline",
                line={"color": BASELINE_COLOR if single else "#FFFFFF",
                      "width": 2.5, "dash": "dash"},
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=wl[keep],
                y=corrected,
                mode="lines",
                name=f"{name} corrected",
                line={"color": CORRECTED_COLOR if single else color, "width": 1},
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
                showlegend=False,
            ), row=2, col=1)
        else:
            intensities = np.asarray(intensities)[keep]
            top_vals.append(intensities)
            fig.add_trace(go.Scatter(
                x=wl[keep],
                y=intensities,
                mode="lines",
                name=name,
                line={"color": color, "width": 1},
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
            ))

    x_refs = ["x", "x2"] if show_baseline else ["x"]
    for lo, hi in shaded_ranges:
        for x_ref in x_refs:
            fig.add_shape(
                type="rect",
                x0=lo,
                x1=hi,
                fillcolor="rgba(255, 165, 0, 0.18)",
                line_width=0,
                xref=x_ref,
                yref="paper",
                y0=0,
                y1=1,
                layer="below",
            )

    # NIST line markers: one colour per element, drawn on the spectrum panel.
    for k, (element, table) in enumerate((marked or {}).items()):
        color = LINE_COLORS[k % len(LINE_COLORS)]
        for row in table.itertuples():
            fig.add_shape(type="line", x0=row.wavelength_nm, x1=row.wavelength_nm,
                          xref="x", yref="y domain", y0=0, y1=1,
                          line={"color": color, "width": 1, "dash": "dot"}, opacity=0.7,
                          layer="below")
            fig.add_annotation(x=row.wavelength_nm, xref="x", y=1, yref="y domain",
                               text=f"{element} {row.species.split()[-1]} {row.wavelength_nm:.2f}",
                               textangle=-90, showarrow=False, xanchor="center",
                               yanchor="top", font={"size": 10, "color": color})

    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin={"l": 55, "r": 20, "t": 50, "b": 50},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    # Shared x-axis: label only the bottom axis so the top one does not collide
    # with the lower subplot's title.
    fig.update_xaxes(title_text="Wavelength (nm)", row=2 if show_baseline else None,
                     col=1 if show_baseline else None)
    axis_type = "log" if log_scale else "linear"
    if wl_bounds is not None:
        fig.update_xaxes(range=list(wl_bounds))
    top_range = _y_range(top_vals, log_scale)
    bottom_range = _y_range(bottom_vals, False)
    if show_baseline:
        fig.update_yaxes(title_text="Intensity (counts)", type=axis_type,
                         range=top_range, row=1, col=1)
        fig.update_yaxes(title_text="Corrected", range=bottom_range, row=2, col=1)
    else:
        fig.update_yaxes(title_text="Intensity (counts)", type=axis_type, range=top_range)

    fig.update_layout(
        dragmode="zoom",
        modebar_remove=["lasso2d", "select2d"],
    )
    return fig

st.title("Inspect")

source = st.session_state.get("source")

if source is None:
    st.info("No data loaded yet.", icon=":material/info:")
    st.page_link("app_pages/load.py", label="Go to the Load page", icon=":material/upload_file:")
    st.stop()

# -----------------------------------------------------------------
# File picking logic

files = list(source)                                   # keys like "Ce/1_wt%/Ce1_300C.tdms"
if st.session_state.get("inspect_file") not in files:  # nothing picked yet, or new data loaded
    st.session_state.pop("inspect_file", None)
choice = st.selectbox(f"File ({len(files)} loaded)", files, key="inspect_file",
                      persist_state="session")

f = source[choice]                                     # the LIBSFile
st.caption(f"{f.describe()} · {f.path}")

# ------------------------------------------------------------------
# Draft manifest: label the loaded files here so the Study page needs no manifest file.
# Paths come from the loaded files, so nobody types them.

def _fill_from_names() -> None:
    '''Button callback: guess element / sample / temperature from the pattern.'''
    try:
        guess = guess_labels(files, st.session_state["draft_pattern"])
    except ValueError as e:
        st.session_state["draft_error"] = str(e)
        return
    st.session_state.pop("draft_error", None)
    st.session_state["draft"] = clean_manifest(draft_manifest(files, prefill=guess))
    st.session_state.pop("draft_editor", None)         # drop edits made to the old table


if source.manifest is None:
    if st.session_state.get("draft_for") != files:     # new data loaded: start a fresh draft
        st.session_state["draft"] = clean_manifest(draft_manifest(files))
        st.session_state["draft_for"] = files
        st.session_state.pop("draft_editor", None)
    with st.expander("Label the files for a Study", icon=":material/label:"):
        c_text, c_info = st.columns([5, 1], vertical_alignment="center")
        c_text.caption("Tell the Study page which file is which. Files left blank are skipped.")
        with c_info.popover("What goes where?", icon=":material/info:", type="tertiary"):
            st.markdown(
                "**One row per file. Fill in three things:**\n"
                "- **element**: the element being measured, as a chemical symbol "
                "(`Ce`, `Gd`).\n"
                "- **sample**: any name that tells your samples apart, e.g. `1 wt%`. "
                "Files of the same sample at different temperatures share the same name.\n"
                "- **temperature**: a number, e.g. `300` (no `C` or units).\n\n"
                "Each sample needs exactly one file per temperature, and one at the "
                "*reference temperature* chosen on the Study page.\n\n"
                "**Shortcut:** if your folders or file names already hold this, type a pattern "
                "in *Fill from file names* and press **Fill**, then fix any gaps by hand. "
                "`{element}`, `{sample}` and `{temperature}` mark where each lives and `*` "
                "matches any other text. The `file` column is read-only; it is filled in for you.")
            st.dataframe(
                {"file": ["Ce/1_wt%/Ce1_300C.tdms", "Ce/1_wt%/Ce1_350C.tdms",
                          "Gd/1_wt%/Gd1_300C.tdms"],
                 "element": ["Ce", "Ce", "Gd"], "sample": ["Ce 1 wt%", "Ce 1 wt%", "Gd 1 wt%"],
                 "temperature": [300, 350, 300]}, hide_index=True)
            st.caption("The pattern `{element}/{sample}/*{temperature}*` would fill this table "
                       "from the file paths (it labels samples '<element> <sample>').")
        p1, p2 = st.columns([5, 1], vertical_alignment="bottom")
        p1.text_input("Fill from file names (optional)", key="draft_pattern",
                      persist_state="session", placeholder="{element}/{sample}/*{temperature}*",
                      help="Describe how your folder and file names are built; {element}, "
                           "{sample} and {temperature} mark where each one sits, and * matches "
                           "any other text. Example: Ce/1_wt%/Ce1_300C.tdms matches "
                           "{element}/{sample}/*{temperature}*")
        p2.button("Fill", on_click=_fill_from_names, width="stretch",
                  disabled=not st.session_state.get("draft_pattern"))
        if st.session_state.get("draft_error"):
            st.error(st.session_state["draft_error"])
        edited = st.data_editor(
            st.session_state["draft"], key="draft_editor", hide_index=True, width="stretch",
            disabled=["file"],
            column_config={
                "file": st.column_config.TextColumn(
                    "file", help="Path of the file, filled in for you."),
                "element": st.column_config.TextColumn(
                    "element", help="Chemical symbol of the element measured, e.g. Ce."),
                "sample": st.column_config.TextColumn(
                    "sample", help="Name of the sample, e.g. '1 wt%'. Use the same name for "
                                   "every temperature of one sample."),
                "temperature": st.column_config.NumberColumn(
                    "temperature", format="%g",
                    help="Number only, e.g. 300. Each sample needs a file at the reference "
                         "temperature.")})
        st.session_state["draft_manifest"] = edited
        gaps = missing_labels(edited)
        if gaps:
            st.caption("Not ready yet: " + "; ".join(
                f"{col} missing for {len(fs)} file(s)" if fs else f"{col} is empty"
                for col, fs in gaps.items()))
        else:
            n = len(labelled_rows(edited))
            st.caption(f":material/check_circle: {n} file(s) labelled and ready for the Study page.")
        st.download_button("Download as manifest.csv", manifest_csv(edited), "manifest.csv",
                           "text/csv", icon=":material/download:",
                           help="Keep it next to your data to load the same labels next time.")

# ------------------------------------------------------------------ 

st.dataframe(f.spectra, key="spectra_table",
             alt="The spectra data of the selected file")

method = st.segmented_control("Baseline", list(BASELINES), default="snip",
                              required=True)

baseline = BASELINES[method]
if baseline.description:
    st.caption(baseline.description)

params: dict[str, int | float] = {}
for name, default in baseline.defaults.items():
    param_key = f"inspect_baseline_{method}_{name}"
    if name == "lam":
        options = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
        value = st.select_slider(
            "lam",
            options=options,
            value=default,
            format_func=lambda v: f"{v:.0e}",
            key=param_key,
            persist_state="session",
        )
        params[name] = value
    elif isinstance(default, int):
        if name == "max_half_window":
            minimum, maximum, step = 1, 100, 1
        else:
            minimum, maximum, step = 0, 1_000_000, 1
        value = st.slider(name, min_value=minimum, max_value=maximum, value=default,
                          step=step, key=param_key, persist_state="session")
        params[name] = value
    elif isinstance(default, float):
        if name == "p":
            minimum, maximum, step = 0.0, 1.0, 0.01
        else:
            minimum, maximum, step = 0.0, 1_000_000.0, default / 100 or 0.01
        value = st.slider(name, min_value=minimum, max_value=maximum, value=default,
                          step=step, key=param_key, persist_state="session")
        params[name] = value
    else:
        st.number_input(name, value=float(default), key=param_key,
                        persist_state="session")
        continue

params = baseline_settings(method, params)

view = st.segmented_control(
    "View",
    ["Mean", "Locations", "Single shot"],
    default="Mean",
    key="inspect_view",
    required=True,
    persist_state="session",
)
show_baseline = st.checkbox("Show baseline", key="show_baseline",
                            persist_state="session")

wl_all = np.asarray(f.wavelengths, dtype=float)
wl_min, wl_max = float(wl_all.min()), float(wl_all.max())
for _key in ("inspect_wl_min", "inspect_wl_max"):   # stale value from another file
    _v = st.session_state.get(_key)
    if _v is not None and not (wl_min <= _v <= wl_max):
        st.session_state.pop(_key)
lo_col, hi_col = st.columns(2)
lo = lo_col.number_input("Wavelength min (nm)", min_value=wl_min, max_value=wl_max,
                         value=wl_min, step=1.0, format="%.2f",
                         key="inspect_wl_min", persist_state="session")
hi = hi_col.number_input("Wavelength max (nm)", min_value=wl_min, max_value=wl_max,
                         value=wl_max, step=1.0, format="%.2f",
                         key="inspect_wl_max", persist_state="session")
if lo >= hi:
    st.warning("Min must be below max; showing the full range.", icon=":material/warning:")
    wl_bounds = (wl_min, wl_max)
else:
    wl_bounds = (float(lo), float(hi))

# NIST line markers (persistent = strong lines, Handbook Table 2)
marked: dict[str, pd.DataFrame] = {}
try:
    available = nist_elements()
except Exception:                  # database not built yet
    available = []
    st.caption("NIST line markers unavailable: build the database with `libs-viewer nist build`.")
if available:
    picked = st.multiselect("Mark NIST lines for", available, key="inspect_line_elements",
                            persist_state="session")
    if picked:
        c1, c2 = st.columns(2)
        species = c1.multiselect("Ion stages", ["I", "II"], default=["I", "II"],
                                 key="inspect_line_species", persist_state="session")
        n_lines = c2.slider("Strongest lines per element (in view)", 1, 40, 10,
                            key="inspect_line_count", persist_state="session")
        for element in picked:
            table = strong_lines(element, species=tuple(species), wl_range=wl_bounds)
            marked[element] = table.head(n_lines)

# The recipe the Study page applies: baseline settings plus the lines marked here.
st.session_state["recipe"] = {
    "baseline": method,
    "baseline_params": params,
    "lines": {el: [round(float(w), 4) for w in t.wavelength_nm] for el, t in marked.items()},
}

log_scale = st.toggle("Log intensity scale", value=True, key="inspect_log_scale",
                      persist_state="session")

fig = spectrum_figure(f, view, method, params, show_baseline, log_scale, wl_bounds, marked)
st.plotly_chart(
    fig,
    key="inspect_spectrum_plot",
    width="stretch",
    height=650,
    alt=f"{view} spectrum for {choice}, with saturated regions shaded.",
    config={"scrollZoom": True, "responsive": True},
)

