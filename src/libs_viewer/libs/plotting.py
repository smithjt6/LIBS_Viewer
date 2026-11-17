'''
Plots of LIBS spectra. Every function returns a matplotlib Figure and never calls
`plt.show()`, so the same figure works in a notebook (displayed when returned), the UI
(`st.pyplot(fig)`) and the MCP server (`fig.savefig(...)`).

Figures are built with `matplotlib.figure.Figure` rather than pyplot, so nothing is left
open in pyplot's global state and notebooks don't show a figure twice. Such a figure only
renders in Jupyter once pyplot's inline backend is active, so they are created as
`NotebookFigure`, which also renders itself as a PNG.
'''

import warnings
from io import BytesIO

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.signal import find_peaks

from .analysis import TOTAL, TOTAL_EXCL
from .processing import global_baseline

LINE_COLOR = '#1f5fa8'
LABEL_COLOR = '#444444'
REF_COLOR = '#d62728'
BASELINE_COLOR = '#e08a00'


class NotebookFigure(Figure):
    """
    A plain matplotlib Figure that Jupyter / VS Code show as an image when it is the last
    expression of a cell, without `import matplotlib.pyplot` or `%matplotlib inline`.
    """

    def _repr_png_(self) -> bytes:
        buf = BytesIO()
        self.savefig(buf, format='png', bbox_inches='tight')
        return buf.getvalue()


def to_nm(x: float | str) -> float:
    '''Converts a value to nm. Accepts numbers or strings like '394.22nm'.'''
    return float(str(x).lower().replace('nm', ''))


def plot_spectrum(wavelengths, intensities, log_y: bool = True, n_labels: int = 15,
                  min_label_gap: float = 12, title: str | None = None, ref_lines=None,
                  peak_tol: float = 0.2, wl_range=None, ax: Axes | None = None) -> Figure:
    '''
    One spectrum with its strongest peaks labelled and optional reference lines.

    Args:
        wavelengths, intensities: 1-D arrays of equal length (nm, counts).
        log_y: log intensity axis.
        n_labels: maximum number of peak labels.
        min_label_gap: minimum spacing (nm, for the full range) between labels; scales with the zoom.
        title: axes title (none by default).
        ref_lines: reference wavelengths to mark, either a sequence (labelled by wavelength)
            or a mapping {label: wavelength}. Values may be numbers or strings like '394.22nm'.
        peak_tol: a reference line with no detected peak within this many nm triggers a warning.
        wl_range: (lo, hi) nm zoom window; each end snaps to the nearest measured wavelength.
        ax: draw into an existing Axes instead of a new figure.

    Peaks are found on the full spectrum (prominence = median intensity), so zooming does
    not change what counts as a peak.

    Returns:
        The Figure.
    '''
    wl = np.asarray(wavelengths, dtype=float)
    counts = np.asarray(intensities, dtype=float)
    if wl.ndim != 1 or counts.ndim != 1 or len(wl) != len(counts):
        raise ValueError('wavelengths and intensities must be one-dimensional and have equal length')
    if len(wl) == 0:
        raise ValueError('wavelengths and intensities cannot be empty')

    # A mapping gives reference labels; a sequence uses the wavelength as its label.
    if hasattr(ref_lines, 'items'):
        references = [(str(label), to_nm(value)) for label, value in ref_lines.items()]
    else:
        references = [(f'{to_nm(value):.2f} nm', to_nm(value)) for value in (ref_lines or [])]

    # zoom window: whole spectrum, or each end snapped to the nearest measured wavelength
    if wl_range is None:
        lo, hi = wl.min(), wl.max()
    else:
        lo, hi = sorted(wl[np.argmin(np.abs(wl - to_nm(x)))] for x in wl_range)
    in_view = (wl >= lo) & (wl <= hi)
    span = hi - lo

    # peaks are found on the full spectrum so zooming doesn't change what counts as a peak
    peaks, props = find_peaks(counts, prominence=np.median(counts))
    peaks = peaks[np.argsort(props['prominences'])[::-1]]
    # label spacing shrinks with the window so a zoom still gets labels
    label_gap = min_label_gap * span / (wl.max() - wl.min())
    labelled = []
    for p in peaks[in_view[peaks]]:
        if all(abs(wl[p] - wl[q]) >= label_gap for q in labelled):
            labelled.append(p)
        if len(labelled) == n_labels:
            break

    if ax is None:
        fig = NotebookFigure(figsize=(14, 5), layout='constrained')
        ax = fig.subplots()
    else:
        fig = ax.figure
    ax.plot(wl[in_view], counts[in_view], color=LINE_COLOR, linewidth=0.6 if span > 100 else 1)

    decimals = 1 if span > 50 else 2
    for p in labelled:
        ax.annotate(f'{wl[p]:.{decimals}f}', (wl[p], counts[p]), xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', rotation=90, fontsize=8, color=LABEL_COLOR,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

    view_counts = counts[in_view]
    visible_refs = [(label, x) for label, x in references if lo <= x <= hi]
    for i, (label, x) in enumerate(visible_refs):
        nearest_index = np.argmin(np.abs(wl - x))
        actual_x = wl[nearest_index]
        ax.axvline(actual_x, color=REF_COLOR, linewidth=0.8, linestyle='--',
                   label='Reference lines' if i == 0 else None)
        ax.annotate(label, xy=(actual_x, counts[nearest_index]),
                    xytext=(actual_x, view_counts.max() * 0.88),
                    ha='center', va='top', rotation=90, fontsize=8, color=REF_COLOR,
                    arrowprops=dict(arrowstyle='->', color=REF_COLOR, lw=0.8))
    if visible_refs:
        ax.legend(loc='upper right', frameon=False)

    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(view_counts.min() * 0.8, view_counts.max() * 3)
    else:
        ax.set_ylim(0, view_counts.max() * 1.15)

    ax.set_xlim(lo, hi)
    if span > 200:
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', color='#e6e6e6', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (counts)')
    if title:
        ax.set_title(title)

    # reference lines with no measured peak nearby
    peak_wl = wl[peaks]
    for _, x in references:
        if not len(peak_wl):
            warnings.warn(f'no peaks detected; cannot check reference line {x} nm', stacklevel=2)
            break
        nearest = peak_wl[np.argmin(np.abs(peak_wl - x))]
        if abs(nearest - x) > peak_tol:
            warnings.warn(f'no peak within {peak_tol} nm of {x} nm (nearest peak: {nearest:.2f} nm)', stacklevel=2)
    return fig


def plot_baseline(wavelengths, intensities, method: str = 'snip', max_half_window: int = 40,
                  lam: float = 1e6, wl_range=None, log_y: bool = True, title: str | None = None) -> Figure:
    """
    How well a baseline method follows the continuum: the spectrum with the fitted baseline
    on top, and the baseline-corrected spectrum (what the total area integrates) below.

    Args:
        method, max_half_window, lam: as in `processing.global_baseline` ('snip' or 'airpls').
        wl_range: (lo, hi) nm zoom window (the baseline is always fitted on the full spectrum).

    Returns:
        The Figure (two axes sharing the wavelength axis).
    """
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(intensities, dtype=float)
    corrected, base = global_baseline(wl, y, method, max_half_window, lam)
    lo, hi = (wl.min(), wl.max()) if wl_range is None else sorted(to_nm(x) for x in wl_range)
    view = (wl >= lo) & (wl <= hi)

    fig = NotebookFigure(figsize=(14, 6), layout='constrained')
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    top.plot(wl[view], y[view], color=LINE_COLOR, lw=0.6, label='spectrum')
    top.plot(wl[view], base[view], color=BASELINE_COLOR, lw=1.4, label=f'{method} baseline')
    if log_y:
        top.set_yscale('log')
        top.set_ylim(max(y[view].min(), 1e-3) * 0.8, y[view].max() * 1.5)
    top.set_ylabel('Intensity (counts)')
    top.legend(loc='upper right', frameon=False)
    bottom.plot(wl[view], corrected[view], color=LINE_COLOR, lw=0.6)
    bottom.axhline(0, color='grey', lw=0.5)
    bottom.set_ylabel('Corrected')
    bottom.set_xlabel('Wavelength (nm)')
    bottom.set_xlim(lo, hi)
    for ax in (top, bottom):
        ax.grid(color='#e6e6e6', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    if title:
        top.set_title(title)
    return fig


def plot_study(result, log_y: bool = True) -> Figure:
    """
    One panel per sample of a `study.StudyResult`: R_rel of every line (markers) and
    R_total (black dashed) against temperature, relative to the reference temperature.
    """
    samples = result.samples
    fig = NotebookFigure(figsize=(4 * len(samples), 4), layout='constrained')
    axes = np.atleast_1d(fig.subplots(1, len(samples), sharey=True))
    for ax, sample in zip(axes, samples):
        rel = result.r_rel.loc[sample]
        temps = [t for t in rel.columns if rel[t].notna().any()]
        for label in rel.index.drop([TOTAL, TOTAL_EXCL]):
            ax.plot(temps, rel.loc[label, temps], marker='o', label=label)
        ax.plot(temps, result.r_direct.loc[(sample, TOTAL), temps], color='k', ls='--', marker='s',
                label='R_total')
        ax.axhline(1, color='grey', lw=0.5)
        ax.set_xticks(temps)
        ax.set_title(sample)
        ax.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
        ax.legend(fontsize=8, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    if log_y:
        axes[0].set_yscale('log')
    axes[0].set_ylabel(f"ratio to {result.settings['ref']} \N{DEGREE SIGN}C")
    return fig
