# ui/app_pages/inspect.py

'''
The Inspect page: pick one loaded file and look at its spectra, settings and saturation.
Reads st.session_state["source"] (set on the Load page).
'''

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs_viewer.libs import (
    BASELINES,
    baseline_settings,
    location_spectra,
    per_shot_columns,
    per_shot_spectrum,
    saturated_pixels,
    mask_ranges,
)
from libs_viewer.libs.processing import global_baseline


def spectrum_figure(f, view: str, method: str, params: dict,
                    show_baseline: bool) -> go.Figure:
    wl = f.wavelengths

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
            vertical_spacing=0.06,
            subplot_titles=("Spectrum and baseline", "Baseline-corrected"),
        )
    else:
        fig = go.Figure()

    for name, intensities in series.items():
        if show_baseline:
            corrected, baseline = global_baseline(
                wl, intensities, method=method, **params
            )
            fig.add_trace(go.Scatter(
                x=wl,
                y=intensities,
                mode="lines",
                name=f"{name} spectrum",
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=wl,
                y=baseline,
                mode="lines",
                name=f"{name} baseline",
                line={"dash": "dash"},
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=wl,
                y=corrected,
                mode="lines",
                name=f"{name} corrected",
                hovertemplate="%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>",
                showlegend=False,
            ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=wl,
                y=intensities,
                mode="lines",
                name=name,
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

    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin={"l": 55, "r": 20, "t": 50, "b": 50},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    fig.update_xaxes(title_text="Wavelength (nm)")
    if show_baseline:
        fig.update_yaxes(title_text="Intensity (counts)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Corrected", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Intensity (counts)", type="log")

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
st.session_state["recipe"] = {"baseline": method, "baseline_params": params}

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

fig = spectrum_figure(f, view, method, params, show_baseline)
st.plotly_chart(
    fig,
    key="inspect_spectrum_plot",
    width="stretch",
    height=650,
    alt=f"{view} spectrum for {choice}, with saturated regions shaded.",
    config={"scrollZoom": True, "responsive": True},
)

