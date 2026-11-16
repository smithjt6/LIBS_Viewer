'''
Shot normalization, continuum removal, and peak / total-area ratios for LIBS spectra.

Pipeline (used for the temperature-series comparison):
    1. per_shot_spectrum   -> every spectrum on a "counts per single shot" scale
    2. global_baseline     -> SNIP or airPLS continuum estimate, subtracted
    3. total_area          -> integrated area of the continuum-corrected spectrum
    4. integrate_local_peak-> line area above a straight local baseline
    5. analyze_sample / ratio_tables -> raw areas and ratios vs a reference temperature

All functions are UI-independent and return arrays / DataFrames (no plotting).
'''

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import find_peaks

from .tdms_loader import LIBSFile

TOTAL = 'Total'
TOTAL_EXCL = 'Total (excl. sat.)'


# ---------------------------------------------------------------- 1. shot scale
def per_shot_spectrum(f: LIBSFile, columns: list[str] | None = None) -> np.ndarray:
    '''
    Mean single-shot spectrum of one file.

    Accumulated files (Accumulate = 1) store the sum of `Number Of Shots` shots in each
    column, so each column is divided by that number. Non-accumulated files already store
    one shot per column. The selected columns are then averaged.

    Args:
        f: a loaded LIBSFile.
        columns: column names to use (default: all). Use this to drop columns that were
            acquired with different settings than the first column, since `f.params`
            only describes the first column.

    Returns:
        1-D array of intensities (counts per shot), aligned with `f.wavelengths`.
    '''
    spectra = f.spectra if columns is None else f.spectra[columns]
    n_shots = float(f.params['Number Of Shots']) if f.accumulated else 1.0
    return (spectra / n_shots).mean(axis=1).to_numpy(dtype=float)


# ---------------------------------------------------------- 2. global baseline
def global_baseline(wl: np.ndarray, y: np.ndarray, method: str = 'snip',
                    max_half_window: int = 40, lam: float = 1e6) -> tuple[np.ndarray, np.ndarray]:
    '''
    Estimates the broad plasma continuum and subtracts it.

    Args:
        wl, y: wavelength axis (nm) and per-shot intensity.
        method: 'snip' (default; good for sharp lines on a broad continuum) or 'airpls'.
        max_half_window: SNIP clipping half-window in pixels (~0.07 nm/pixel here, so 40 px
            ~ 2.8 nm). Too small and SNIP eats into the bases of broad lines.
        lam: airPLS smoothness penalty (1e5-1e7 typical).

    Returns:
        (corrected, baseline) arrays.
    '''
    fitter = Baseline(x_data=wl)
    m = method.lower()
    if m == 'snip':
        baseline, _ = fitter.snip(y, max_half_window=max_half_window, decreasing=True)
    elif m == 'airpls':
        baseline, _ = fitter.airpls(y, lam=lam)
    else:
        raise ValueError("method must be 'snip' or 'airpls'")
    return y - baseline, baseline


# --------------------------------------------------------------- 3. total area
def _exclusion_mask(wl: np.ndarray, exclude) -> np.ndarray:
    '''Boolean pixel mask from either a boolean array or a list of (lo, hi) nm ranges.'''
    if exclude is None:
        return np.zeros(len(wl), dtype=bool)
    if isinstance(exclude, np.ndarray) and exclude.dtype == bool:
        return exclude
    mask = np.zeros(len(wl), dtype=bool)
    for lo, hi in exclude:
        mask |= (wl >= lo) & (wl <= hi)
    return mask


def saturated_mask(files: list[LIBSFile], columns: list[list[str] | None] | None = None,
                   level: float = 64900.0, pad_px: int = 2) -> np.ndarray:
    '''
    Pixels where the detector clipped in any of the given files.

    A pixel is flagged if its per-shot intensity reaches `level` counts in any used column
    of any file (the 16-bit detector tops out at ~65000-65500 counts per shot), and the flag
    is widened by `pad_px` pixels each side. Pass all temperatures of one sample together so
    the same pixels are excluded at every temperature.

    Limitation: an accumulated column is the sum of its shots, so clipping is only seen
    when (nearly) every shot clipped; a line that clipped in 1 of 5 shots goes unnoticed.

    Returns:
        Boolean array aligned with the wavelength axis.
    '''
    columns = columns or [None] * len(files)
    mask = np.zeros(len(files[0].wavelengths), dtype=bool)
    for f, cols in zip(files, columns):
        spectra = f.spectra if cols is None else f.spectra[cols]
        n_shots = float(f.params['Number Of Shots']) if f.accumulated else 1.0
        mask |= (spectra.max(axis=1).to_numpy(dtype=float) / n_shots) >= level
    if pad_px:
        mask = np.convolve(mask, np.ones(2 * pad_px + 1), mode='same') > 0
    return mask


def total_area(wl: np.ndarray, corrected: np.ndarray, exclude=None) -> float:
    '''
    Integrated area (counts*nm per shot) of a continuum-corrected spectrum.

    Args:
        exclude: pixels to drop before integrating, e.g. saturated lines; either a boolean
            array (see `saturated_mask`) or a list of (lo, hi) nm ranges. Masked pixels
            are set to zero so the integration doesn't bridge across them.
    '''
    y = np.asarray(corrected, dtype=float).copy()
    y[_exclusion_mask(wl, exclude)] = 0.0
    return float(np.trapezoid(y, wl))


# ------------------------------------------------------------- 4. local peak
def peak_bounds(wl: np.ndarray, y: np.ndarray, center: float, search: float = 0.1,
                max_half_width: float = 0.35) -> tuple[int, int, int]:
    '''
    Locates one emission line in a measured spectrum.

    The apex is the highest pixel within `search` nm of `center` (this absorbs small
    wavelength-calibration offsets). From the apex the window walks downhill on each side
    until the signal starts rising again (the valley where the line meets its neighbour
    or the continuum), but never more than `max_half_width` nm from the apex.

    Returns:
        (left, apex, right) pixel indices; the line occupies y[left:right + 1].
    '''
    near = np.flatnonzero(np.abs(wl - center) <= search)
    if len(near) == 0:
        raise ValueError(f'{center} nm is outside the wavelength range')
    apex = near[np.argmax(y[near])]
    left = apex
    while left > 0 and y[left - 1] <= y[left] and wl[apex] - wl[left - 1] <= max_half_width:
        left -= 1
    right = apex
    while right < len(y) - 1 and y[right + 1] <= y[right] and wl[right + 1] - wl[apex] <= max_half_width:
        right += 1
    return left, apex, right


def integrate_local_peak(wl: np.ndarray, y: np.ndarray, center: float,
                         half_width: float | None = None, edge_pts: int = 1,
                         search: float = 0.1, max_half_width: float = 0.35,
                         bounds: tuple[int, int] | None = None) -> dict:
    '''
    Area of one emission line above a straight local baseline.

    By default (half_width=None) the window is found from the data with `peak_bounds`:
    the apex nearest `center`, out to the valley on each side, so the baseline is the line
    "across the bottom where the peak begins". With a number for `half_width` a fixed
    window [center - half_width, center + half_width] is used instead. `bounds`
    = (left, right) pixel indices overrides both; use it to apply the window found on a
    reference spectrum to other spectra so every temperature is integrated over the same pixels.

    The baseline joins the mean of the first and last `edge_pts` pixels of the window and
    the area above it is integrated. Pass the per-shot spectrum (not the continuum-corrected
    one) so two baselines aren't subtracted.

    Returns:
        dict with 'area', 'apex_nm', 'wl', 'y', 'local_baseline', 'net' (arrays for plotting).
    '''
    if bounds is not None:
        sl = slice(bounds[0], bounds[1] + 1)
        apex = sl.start + int(np.argmax(y[sl]))
    elif half_width is None:
        left, apex, right = peak_bounds(wl, y, center, search, max_half_width)
        sl = slice(left, right + 1)
    else:
        idx = np.flatnonzero((wl >= center - half_width) & (wl <= center + half_width))
        sl = slice(idx[0], idx[-1] + 1)
        apex = sl.start + int(np.argmax(y[sl]))
    w, s = wl[sl], y[sl]
    if len(w) < max(3, 2 * edge_pts + 1):
        raise ValueError(f'window around {center} nm has only {len(w)} pixels')
    x1, x2 = w[:edge_pts].mean(), w[-edge_pts:].mean()
    y1, y2 = s[:edge_pts].mean(), s[-edge_pts:].mean()
    local = y1 + (y2 - y1) * (w - x1) / (x2 - x1)  # straight line, extrapolated to the edges
    net = s - local
    return {'area': float(np.trapezoid(net, w)), 'apex_nm': float(wl[apex]), 'wl': w, 'y': s,
            'local_baseline': local, 'net': net}


# ------------------------------------------------------------ 5. sample level
def temperature_from_name(name: str) -> int:
    '''Parses the temperature from names like 'Ce_Sample2_Temp_300C.tdms' or 'Gd_Sample5_300_3.5delay.tdms'.'''
    m = re.search(r'_(\d{3})C?(?=[_.])', Path(name).name)
    if not m:
        raise ValueError(f'no temperature found in {name!r}')
    return int(m.group(1))


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
    n_shots = float(f.params['Number Of Shots']) if f.accumulated else 1.0
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


# ------------------------------------------------------------ line selection
def strong_lines(db_path: str | Path, element: str, species: tuple[str, ...] = ('I', 'II'),
                 wl_range: tuple[float, float] = (186.0, 1050.0)) -> pd.DataFrame:
    '''
    Strong lines of one element from the scraped NIST handbook database (strong_lines table).

    Returns a DataFrame with columns species, wavelength_nm, intensity, sorted by intensity.
    '''
    names = [f'{element} {s}' for s in species]
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            f"SELECT species, wavelength_A, intensity FROM strong_lines "
            f"WHERE element = ? AND species IN ({','.join('?' * len(names))})",
            con, params=[element, *names])
    df['wavelength_nm'] = df.pop('wavelength_A') / 10.0
    df = df[df.wavelength_nm.between(*wl_range)]
    return df.sort_values('intensity', ascending=False).reset_index(drop=True)


def select_lines(db_path: str | Path, element: str, wl: np.ndarray,
                 y_refs: dict[str, np.ndarray], n: int = 4, interferents: list[str] | None = None,
                 interf_species: tuple[str, ...] = ('I',), interf_tol: float = 0.15,
                 exclude=None, exclude_pad: float = 0.5,
                 peak_tol: float = 0.1, min_sep: float = 1.0, min_prominence: float = 2.0,
                 min_side_px: int = 2, candidates: int = 40) -> pd.DataFrame:
    '''
    Picks the n strongest NIST lines of an element that are usable in measured spectra.

    Args:
        y_refs: {sample label: reference spectrum} for every sample of this element.

    A candidate is rejected outright if it is within `exclude_pad` nm of an excluded
    (saturated) pixel, or within `interf_tol` nm of a strong line (of `interf_species`,
    neutral by default) of one of `interferents`.

    It is then tested in each reference spectrum, where it counts as *detected* if there is
    a peak (prominence >= min_prominence x median of that spectrum) within `peak_tol` nm,
    and the peak is not blended (the valley-to-valley window from `peak_bounds` is at least
    `min_side_px` pixels wide on both sides of the apex).

    Lines detected in more samples are preferred, then stronger NIST intensity; lines
    within `min_sep` nm of one already picked are skipped so windows don't overlap.

    Returns:
        DataFrame of every candidate with 'label' (e.g. 'Ce II 418.66'), 'status'
        ('picked' or the reason it was not) and 'detected_in' (comma-separated samples).
    '''
    cands = strong_lines(db_path, element, wl_range=(wl.min(), wl.max())).head(candidates)
    others = pd.concat([strong_lines(db_path, el, interf_species, (wl.min(), wl.max())).assign(element=el)
                        for el in interferents or []], ignore_index=True) if interferents else None
    excl_wl = wl[_exclusion_mask(wl, exclude)]
    peak_wl = {name: wl[find_peaks(y, prominence=min_prominence * np.median(y))[0]]
               for name, y in y_refs.items()}

    status, detected_in = [], []
    for c in cands.wavelength_nm:
        reason, found, why = '', [], set()
        if len(excl_wl) and np.abs(excl_wl - c).min() <= exclude_pad:
            reason = 'near saturated region'
        elif others is not None and (hit := others[(others.wavelength_nm - c).abs() <= interf_tol]).shape[0]:
            reason = 'interference: ' + ', '.join(f'{s} {w:.2f}' for s, w in zip(hit.species, hit.wavelength_nm))
        else:
            for name, y in y_refs.items():
                if not len(peak_wl[name]) or np.abs(peak_wl[name] - c).min() > peak_tol:
                    why.add('no peak in spectrum')
                elif min(np.diff(peak_bounds(wl, y, c, peak_tol))) < min_side_px:
                    why.add('blended (neighbour line too close)')
                else:
                    found.append(name)
            if not found:
                reason = ' / '.join(sorted(why))
        status.append(reason)
        detected_in.append(found)

    # rank the usable candidates: detected in most samples first, then NIST intensity
    cands = cands.assign(status=status, n_detected=[len(d) for d in detected_in],
                         detected_in=[', '.join(d) for d in detected_in])
    picked = []
    order = cands[cands.status == ''].sort_values(['n_detected', 'intensity'], ascending=False, kind='stable')
    for i, c in order.wavelength_nm.items():
        if any(abs(c - p) < min_sep for p in picked):
            cands.loc[i, 'status'] = 'too close to a picked line'
        elif len(picked) >= n:
            cands.loc[i, 'status'] = 'not needed'
        else:
            cands.loc[i, 'status'] = 'picked'
            picked.append(c)
    cands['label'] = cands.species + ' ' + cands.wavelength_nm.map('{:.2f}'.format)
    return cands
