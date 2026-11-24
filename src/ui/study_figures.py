# ui/study_figures.py

'''
Plotly figures for the Study page, mirroring libs_viewer.libs.plotting (matplotlib, for notebooks):

    ratio_figure      R_rel of every line and R_total against temperature, one panel per sample
    baseline_figure   one spectrum with its fitted baseline, and the baseline-corrected spectrum
    lines_figure      one spectrum with the measured lines marked

The library's plotting stays matplotlib-only; plotly lives here, in the UI.
'''

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs_viewer.libs import TOTAL, TOTAL_EXCL
from libs_viewer.libs.processing import global_baseline

SPECTRUM_COLOR = "#4DA3FF"
BASELINE_COLOR = "#FF3B6B"
CORRECTED_COLOR = "#3DDC84"
LINE_MARK_COLOR = "#FFB000"
TOTAL_COLOR = "#8C8C8C"           # grey reads on both light and dark themes
HOVER = "%{x:.3f} nm<br>%{y:.3g} counts<extra></extra>"


def _fit(y: np.ndarray, log: bool) -> list[float] | None:
    '''Axis range (log10 units for a log axis) fitting `y` with a margin.'''
    y = np.asarray(y, dtype=float)
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


def ratio_figure(result, log_y: bool = True) -> go.Figure:
    '''One panel per sample: every line's R_rel and the total's R_direct against temperature.'''
    samples = result.samples
    total_row = result.settings.get("total_row", TOTAL)
    fig = make_subplots(rows=1, cols=len(samples), shared_yaxes=True, subplot_titles=samples)
    for col, sample in enumerate(samples, 1):
        rel = result.r_rel.loc[sample]
        temps = [t for t in rel.columns if rel[t].notna().any()]
        for label in rel.index.drop([TOTAL, TOTAL_EXCL], errors="ignore"):
            fig.add_trace(go.Scatter(
                x=temps, y=rel.loc[label, temps], mode="lines+markers", name=label,
                legendgroup=f"{sample}/{label}", showlegend=True,
                hovertemplate=f"{label}<br>%{{x:g}} °C: %{{y:.3f}}<extra></extra>"),
                row=1, col=col)
        fig.add_trace(go.Scatter(
            x=temps, y=result.r_direct.loc[(sample, total_row), temps], mode="lines+markers",
            name="R_total" + (" (excl. sat.)" if total_row == TOTAL_EXCL else ""),
            line={"color": TOTAL_COLOR, "dash": "dash"}, marker={"symbol": "square"},
            hovertemplate="R_total<br>%{x:g} °C: %{y:.3f}<extra></extra>"),
            row=1, col=col)
        fig.add_hline(y=1, line={"color": TOTAL_COLOR, "width": 0.5}, row=1, col=col)
        fig.update_xaxes(title_text="Temperature (°C)", tickvals=temps, row=1, col=col)
    fig.update_yaxes(title_text=f"ratio to {result.settings['ref']:g} °C", col=1)
    fig.update_yaxes(type="log" if log_y else "linear")
    fig.update_layout(height=420, margin={"l": 55, "r": 20, "t": 50, "b": 50},
                      legend={"orientation": "h", "y": -0.25})
    return fig


def baseline_figure(wl, intensities, method: str, params: dict, log_y: bool = True,
                    title: str | None = None) -> go.Figure:
    '''Spectrum with its fitted baseline (top) and the baseline-corrected spectrum (bottom).'''
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(intensities, dtype=float)
    corrected, baseline = global_baseline(wl, y, method=method, **params)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[2, 1],
                        vertical_spacing=0.1,
                        subplot_titles=(title or "Spectrum and baseline", "Baseline-corrected"))
    detail = ", ".join(f"{k}={v:g}" for k, v in params.items())
    fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name="spectrum",
                             line={"color": SPECTRUM_COLOR, "width": 1}, hovertemplate=HOVER),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=wl, y=baseline, mode="lines",
                             name=f"{method} baseline ({detail})",
                             line={"color": BASELINE_COLOR, "width": 2.5, "dash": "dash"},
                             hovertemplate=HOVER), row=1, col=1)
    fig.add_trace(go.Scatter(x=wl, y=np.asarray(corrected), mode="lines", name="corrected",
                             line={"color": CORRECTED_COLOR, "width": 1}, showlegend=False,
                             hovertemplate=HOVER), row=2, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (counts)", type="log" if log_y else "linear",
                     range=_fit(np.concatenate([y, baseline]), log_y), row=1, col=1)
    fig.update_yaxes(title_text="Corrected", row=2, col=1)
    fig.update_layout(height=600, hovermode="x unified", dragmode="zoom",
                      margin={"l": 55, "r": 20, "t": 50, "b": 50},
                      legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
                      modebar_remove=["lasso2d", "select2d"])
    return fig


def lines_figure(wl, intensities, lines: dict[str, float], wl_bounds: tuple[float, float],
                 log_y: bool = True, title: str | None = None) -> go.Figure:
    '''One spectrum cropped to `wl_bounds`, with every measured line {label: nm} inside it marked.'''
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(intensities, dtype=float)
    keep = (wl >= wl_bounds[0]) & (wl <= wl_bounds[1])
    fig = go.Figure(go.Scatter(x=wl[keep], y=y[keep], mode="lines", name="spectrum",
                               line={"color": SPECTRUM_COLOR, "width": 1}, hovertemplate=HOVER))
    for label, nm in lines.items():
        if not wl_bounds[0] <= nm <= wl_bounds[1]:
            continue
        fig.add_shape(type="line", x0=nm, x1=nm, xref="x", yref="y domain", y0=0, y1=1,
                      line={"color": LINE_MARK_COLOR, "width": 1, "dash": "dot"}, layer="below")
        fig.add_annotation(x=nm, xref="x", y=1, yref="y domain", text=f"{label} {nm:.2f}",
                           textangle=-90, showarrow=False, xanchor="center", yanchor="top",
                           font={"size": 10, "color": LINE_MARK_COLOR})
    fig.update_xaxes(title_text="Wavelength (nm)", range=list(wl_bounds))
    fig.update_yaxes(title_text="Intensity (counts)", type="log" if log_y else "linear",
                     range=_fit(y[keep], log_y))
    fig.update_layout(height=450, hovermode="x unified", dragmode="zoom",
                      margin={"l": 55, "r": 20, "t": 50 if title else 20, "b": 50},
                      title={"text": title} if title else None,
                      modebar_remove=["lasso2d", "select2d"])
    return fig
