'''
Sample-level analysis: line areas and total areas across conditions, and ratio tables.

Pipeline (the temperature-series comparison):
    1. processing.per_shot_spectrum   -> every spectrum on a "counts per single shot" scale
    2. processing.global_baseline     -> SNIP or airPLS continuum estimate, subtracted
    3. metrics.total_area             -> integrated area of the continuum-corrected spectrum
    4. metrics.integrate_local_peak   -> line area above a straight local baseline
    5. analyze_sample / ratio_tables  -> raw areas and ratios vs a reference temperature

All functions are UI-independent and return arrays / DataFrames (no plotting).
'''

import numpy as np
import pandas as pd

from .loading import LIBSFile
from .metrics import integrate_local_peak, total_area
from .processing import global_baseline, shots_per_column

TOTAL = 'Total'
TOTAL_EXCL = 'Total (excl. sat.)'


def line_windows(wl: np.ndarray, y_ref: np.ndarray, lines: dict[str, float],
                 half_width: float | None = None, edge_pts: int = 1) -> dict[str, tuple[int, int]]:
    '''(left, right) pixel window of each line, found on one reference spectrum.'''
    windows = {}
    for label, c in lines.items():
        r = integrate_local_peak(wl, y_ref, c, half_width, edge_pts)
        windows[label] = (int(np.searchsorted(wl, r['wl'][0])), int(np.searchsorted(wl, r['wl'][-1])))
    return windows


def _areas(wl, y, lines, windows, method, max_half_window, lam, edge_pts, exclude) -> dict:
    '''Line areas (fixed windows) and total areas of one per-shot spectrum.'''
    corrected, _ = global_baseline(wl, y, method, max_half_window, lam)
    col = {label: integrate_local_peak(wl, y, c, edge_pts=edge_pts, bounds=windows[label])['area']
           for label, c in lines.items()}
    col[TOTAL] = total_area(wl, corrected)
    col[TOTAL_EXCL] = total_area(wl, corrected, exclude)
    return col


def analyze_sample(wl: np.ndarray, spectra_by_temp: dict[int, np.ndarray],
                   lines: dict[str, float], method: str = 'snip', max_half_window: int = 40,
                   lam: float = 1e6, half_width: float | None = None, edge_pts: int = 1,
                   exclude=None, ref: int = 300) -> pd.DataFrame:
    '''
    Raw areas for one sample.

    Each line's integration window is found once, on the `ref` temperature spectrum
    (valley to valley, or fixed if `half_width` is given), and the same pixels are then
    used at every temperature so the ratios compare like with like.

    Args:
        wl: wavelength axis shared by all spectra.
        spectra_by_temp: {temperature: per-shot spectrum}.
        lines: {label: wavelength_nm} of lines to integrate.
        exclude: pixels masked out for the 'Total (excl. sat.)' row (boolean array or
            list of (lo, hi) nm ranges).
        ref: temperature whose spectrum defines the line windows.

    Returns:
        DataFrame, rows = line labels + 'Total' + 'Total (excl. sat.)', columns = temperatures.
    '''
    windows = line_windows(wl, spectra_by_temp[ref], lines, half_width, edge_pts)
    df = pd.DataFrame({temp: _areas(wl, spectra_by_temp[temp], lines, windows, method,
                                    max_half_window, lam, edge_pts, exclude)
                       for temp in sorted(spectra_by_temp)})
    df.columns.name = 'Temperature (C)'
    return df


def per_column_areas(f: LIBSFile, lines: dict[str, float], windows: dict[str, tuple[int, int]],
                     columns: list[str] | None = None, method: str = 'snip',
                     max_half_window: int = 40, lam: float = 1e6, edge_pts: int = 1,
                     exclude=None) -> pd.DataFrame:
    '''
    The same areas as `analyze_sample`, but for every column of one file separately
    (each on the per-shot scale), to show the shot-to-shot / location-to-location scatter.

    Returns:
        DataFrame, rows = line labels + totals, one column per TDMS column.
    '''
    spectra = f.spectra if columns is None else f.spectra[columns]
    n_shots = shots_per_column(f)
    wl = f.wavelengths
    return pd.DataFrame({name: _areas(wl, (spectra[name] / n_shots).to_numpy(dtype=float), lines, windows,
                                      method, max_half_window, lam, edge_pts, exclude)
                         for name in spectra.columns})


def ratio_tables(areas: pd.DataFrame, ref: int = 300, total_row: str = TOTAL) -> dict[str, pd.DataFrame]:
    '''
    Ratios of each area to the same area at the reference temperature.

    Returns:
        'R_direct': A(T) / A(ref) for every row (the Total rows give R_total).
        'R_rel':    [A_peak(T)/A_total(T)] / [A_peak(ref)/A_total(ref)] for line rows,
                    with the Total rows carried over as R_total so both tables share them.
    '''
    r_direct = areas.div(areas[ref], axis=0)
    line_rows = [r for r in areas.index if r not in (TOTAL, TOTAL_EXCL)]
    frac = areas.loc[line_rows] / areas.loc[total_row]
    r_rel = pd.concat([frac.div(frac[ref], axis=0), r_direct.loc[[TOTAL, TOTAL_EXCL]]])
    return {'R_direct': r_direct, 'R_rel': r_rel}
