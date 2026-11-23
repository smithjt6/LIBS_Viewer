'''
Compare one sample across temperatures: how big is each emission line, how much light is
there in total, and how do both change relative to a reference temperature?

The pipeline, start to finish:
    1. processing.per_shot_spectrum   put every spectrum on a "counts per single shot" scale
    2. processing.global_baseline     estimate the broad background glow (continuum), subtract it
    3. metrics.total_area             area under the whole background-free spectrum
    4. metrics.integrate_local_peak   area of one line above a straight line across its base
    5. analyze_sample / ratio_tables  (this module) put the areas in a table, then divide
                                      by the reference temperature

The key idea: each line's pixel window is found ONCE, on the reference spectrum, and the
same pixels are reused for every other spectrum. A ratio then compares like with like.
If the window moved with each spectrum, part of the change would come from the window.

Everything here returns numbers or DataFrames. Nothing plots.
'''

import warnings

import numpy as np
import pandas as pd

from .metrics import EDGE_PTS, integrate_local_peak, line_window, total_area
from .processing import baseline_settings, exclusion_mask, global_baseline

# Row names for the two total-area rows in every areas table
TOTAL = 'Total'                    # area of the whole background-free spectrum
TOTAL_EXCL = 'Total (excl. sat.)'  # the same, with the excluded (e.g. saturated) pixels left out


class RatioWarning(UserWarning):
    '''An area or ratio may mislead: saturated pixels in a line window or in the Total used
    for R_rel, or a zero / negative reference area.'''


def line_windows(wl: np.ndarray, y_ref: np.ndarray, lines: dict[str, float],
                 half_width: float | None = None,
                 edge_pts: int = EDGE_PTS) -> dict[str, tuple[int, int]]:
    '''
    Find where each line sits in the reference spectrum, as a (left, right) pixel range.

    Pass the result to `areas_table` so every spectrum is integrated over exactly these pixels.

    Args:
        wl: wavelength axis (nm).
        y_ref: the reference spectrum (per shot) that the windows are found on.
        lines: {label: wavelength in nm}, e.g. {'Ce II 418.66': 418.66}.
        half_width: None (default) means the window runs from the line's peak down to the
            valley on each side. A number in nm means a fixed window of
            center ± half_width instead.
        edge_pts: pixels at each edge used for the local baseline (see
            `metrics.integrate_local_peak`); a window must be wide enough for both edges.

    Returns:
        {label: (left, right)}. Both ends are included, so the line is y[left:right + 1].

    Raises:
        ValueError: a line is outside the wavelength range, or its window is too narrow.
    '''
    min_px = max(3, 2 * edge_pts + 1)
    return {label: line_window(wl, y_ref, center, half_width, min_px=min_px)
            for label, center in lines.items()}


def areas_table(wl: np.ndarray, spectra, lines: dict[str, float], windows: dict[str, tuple[int, int]],
                method: str = 'snip', baseline_params: dict | None = None,
                edge_pts: int = EDGE_PTS, exclude=None) -> pd.DataFrame:
    '''
    Measure every line and the total area in several spectra. One column per spectrum.

    For each spectrum:
        - each line's area is measured over its fixed pixel window (from `line_windows`)
          above a straight line across the window's edges. This uses the per-shot
          spectrum as given, because the local baseline already removes the background.
        - the background (continuum) is removed with `processing.global_baseline`, and the
          area of what is left gives the 'Total' row and the 'Total (excl. sat.)' row.

    Args:
        wl: wavelength axis (nm), shared by every spectrum.
        spectra: the per-shot spectra, as any of
            - a DataFrame with one spectrum per column (e.g. `processing.location_spectra(f)`
              or `processing.per_shot_columns(f)`)
            - a dict {name: spectrum}
            - a list of (name, spectrum) pairs
        lines: {label: wavelength in nm}.
        windows: {label: (left, right)} pixel windows from `line_windows`.
        method, baseline_params: the background method (a name in `processing.BASELINES`)
            and its parameters, e.g. {'max_half_window': 60}; left-out ones take their
            defaults. Passed to `processing.global_baseline`.
        edge_pts: pixels at each window edge used for the local line baseline.
        exclude: pixels to leave out of 'Total (excl. sat.)'. Either a boolean array the
            length of `wl` (e.g. `processing.saturated_mask`) or a list of (lo, hi) nm
            ranges. None leaves nothing out.

    Returns:
        DataFrame. Rows: one per line, then 'Total', then 'Total (excl. sat.)'.
        Columns: one per spectrum, named as in `spectra`. Units: counts·nm per shot.
        'Total' still contains the saturated pixels; it is kept for reference only.

    Warns (RatioWarning):
        a line window contains excluded pixels (that line's area is unreliable).
    '''

    pairs = spectra.items() if isinstance(spectra, (dict, pd.DataFrame)) else spectra
    params = baseline_settings(method, baseline_params)  # fail before measuring anything
    excluded = exclusion_mask(wl, exclude)  # build the mask once, not once per spectrum

    # a line window over excluded (saturated) pixels measures the detector's ceiling, not the line
    clipped = [label for label, (left, right) in windows.items()
               if label in lines and excluded[left:right + 1].any()]
    if clipped:
        warnings.warn(f'line window(s) {clipped} contain excluded (saturated) pixels; '
                      'their areas are unreliable', RatioWarning, stacklevel=2)

    columns = {}

    for name, y in pairs:
        y = np.asarray(y, dtype=float)
        column = {label: integrate_local_peak(wl, y, center, edge_pts=edge_pts, bounds=windows[label])['area']
                  for label, center in lines.items()}
        corrected, _ = global_baseline(wl, y, method, **params)
        column[TOTAL] = total_area(wl, corrected)
        column[TOTAL_EXCL] = total_area(wl, corrected, excluded)
        columns[name] = column

    return pd.DataFrame(columns, index=[*lines, TOTAL, TOTAL_EXCL])

def analyze_sample(wl: np.ndarray, spectra_by_temp: dict[float, np.ndarray],
                   lines: dict[str, float], method: str = 'snip',
                   baseline_params: dict | None = None, half_width: float | None = None,
                   edge_pts: int = EDGE_PTS, exclude=None, ref: float = 300) -> pd.DataFrame:
    '''
    Line areas and total areas of one sample at every temperature.

    This is `line_windows` on the reference temperature followed by `areas_table` on all
    temperatures, so every temperature is measured over the same pixels.

    Args:
        wl: wavelength axis (nm), shared by every spectrum.
        spectra_by_temp: {temperature in °C: per-shot spectrum}, e.g. {300: y300, 400: y400}.
        lines: {label: wavelength in nm} of the lines to measure.
        half_width: None finds each window from the data (peak to valley). A number in nm
            uses a fixed center ± half_width window instead.
        exclude: pixels to leave out of 'Total (excl. sat.)' (boolean array or list of
            (lo, hi) nm ranges).
        ref: the temperature whose spectrum sets the line windows. Must be a key of
            `spectra_by_temp`.
        method, baseline_params, edge_pts: passed on to `areas_table`.

    Returns:
        DataFrame. Rows: one per line, then 'Total' and 'Total (excl. sat.)'.
        Columns: temperatures, sorted from low to high (column axis named 'Temperature (C)').
        Feed it to `ratio_tables` to get ratios.

    Raises:
        ValueError: `ref` is not one of the temperatures.
    '''
    if ref not in spectra_by_temp:
        raise ValueError(f'reference temperature {ref} not in the data {sorted(spectra_by_temp)}')
    
    windows = line_windows(wl, spectra_by_temp[ref], lines, half_width, edge_pts)
    by_temp = {t: spectra_by_temp[t] for t in sorted(spectra_by_temp)}

    df = areas_table(wl, by_temp, lines, windows, method, baseline_params, edge_pts, exclude)

    df.columns.name = 'Temperature (C)'

    return df


def saturated_share(areas: pd.DataFrame) -> pd.Series:
    '''
    Share (0 to 1) of the Total light that sits in the excluded (saturated) pixels, per column.

    Clipped pixels cannot get brighter, so when this share is large the 'Total' row
    understates how the light changed, and R_rel divided by it is pulled toward 1.
    '''
    excluded = areas.loc[TOTAL_EXCL]
    total = areas.loc[TOTAL]

    if not isinstance(excluded, pd.Series) or not isinstance(total, pd.Series):
        raise ValueError("areas must have exactly one row for each total")

    return 1 - excluded / total


def ratio_tables(areas: pd.DataFrame, ref: float, total_row: str = TOTAL_EXCL,
                 max_saturated_share: float = 0.05) -> dict[str, pd.DataFrame]:
    '''
    How much each area changed compared with the reference temperature (1.0 = no change).

    Args:
        areas: the table from `analyze_sample` (rows = lines + the two totals,
            columns = temperatures).
        ref: the reference temperature column. Every ratio is 1.0 there.
        total_row: which total to divide the lines by for 'R_rel'. TOTAL_EXCL (default)
            leaves the saturated pixels out: clipped pixels report the detector's ceiling,
            not the light. TOTAL keeps them, for comparison only.
        max_saturated_share: warn if total_row is TOTAL and more than this share of the
            Total light is in saturated pixels at any temperature (see `saturated_share`).

    Returns:
        Two tables shaped like `areas`:
            'R_direct'  area(T) / area(ref), for every row.
                        "The line got 1.5x bigger."
            'R_rel'     for line rows, the line's share of the total light, compared with
                        that share at ref:
                            [line(T) / total(T)] / [line(ref) / total(ref)]
                        "The line grew 1.2x MORE than the light as a whole."
                        The two Total rows are copied from R_direct, so both tables end
                        with the same total ratios.

    Raises:
        ValueError: `ref` is not a column of `areas`, or `total_row` is not a total row.

    Warns (RatioWarning):
        - the Total used for R_rel has more than `max_saturated_share` saturated light.
        - a row's area at `ref` is zero or negative: its ratios are NaN instead of inf.
    '''
    if ref not in areas.columns:
        raise ValueError(f'reference temperature {ref} not in the columns {list(areas.columns)}')
    if total_row not in (TOTAL, TOTAL_EXCL):
        raise ValueError(f'total_row must be {TOTAL!r} or {TOTAL_EXCL!r}, got {total_row!r}')

    share = saturated_share(areas)
    if total_row == TOTAL and share.max() > max_saturated_share:
        warnings.warn(f'{share.max():.0%} of the Total light is in saturated pixels (at {share.idxmax()} C), '
                      f'so R_rel divided by {TOTAL!r} is pulled toward 1; '
                      f'use total_row={TOTAL_EXCL!r} to leave those pixels out', RatioWarning, stacklevel=2)

    ref_area = areas[ref].where(areas[ref] > 0)          # zero / negative -> NaN, so no inf ratios
    if ref_area.isna().any():
        warnings.warn(f'area at {ref} C is zero, negative or missing for {list(ref_area.index[ref_area.isna()])}; '
                      'their ratios are NaN', RatioWarning, stacklevel=2)
    r_direct = areas.div(ref_area, axis=0)

    line_share = areas.drop(index=[TOTAL, TOTAL_EXCL]) / areas.loc[total_row]
    ref_share = line_share[ref].where(line_share[ref] > 0)
    r_rel = pd.concat([line_share.div(ref_share, axis=0),
                       r_direct.loc[[TOTAL, TOTAL_EXCL]]])
    return {'R_direct': r_direct, 'R_rel': r_rel}
