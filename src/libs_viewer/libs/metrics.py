'''
Numbers measured on one spectrum: peaks, total area, line windows, line areas.

All functions take arrays and return numbers / arrays / DataFrames; no plotting, no file I/O.
'''

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .processing import exclusion_mask

# Pixels averaged at each window edge to place a line's local baseline. With 1, a single noisy
# edge pixel sets the baseline, and the reference window's edges tend to land on noise dips,
# biasing ratios ~1-2% low; 2 cuts that to <0.6% and still fits the narrowest window that
# select_lines accepts (2 pixels + apex + 2 pixels).
EDGE_PTS = 2


def detect_peaks(wl: np.ndarray, y: np.ndarray, min_prominence: float = 1.0,
                 exclude=None) -> pd.DataFrame:
    '''
    The peaks of one spectrum: every local maximum that stands out from its surroundings by
    at least `min_prominence` x the spectrum's median. On a raw (not continuum-corrected)
    per-shot spectrum the median is about the continuum level, so 1 finds the clear lines,
    0.25 also weak lines on a high continuum. This is the peak test used by `select_lines`
    and the peak labels of `plotting.plot_spectrum`.

    Args:
        wl, y: wavelength axis (nm) and intensity.
        min_prominence: threshold, as a multiple of the median of `y`.
        exclude: pixels to flag, e.g. `processing.saturated_mask`, or (lo, hi) nm ranges.
            Peaks there are kept but marked, since their apex is the detector's ceiling.

    Returns:
        DataFrame, one row per peak, sorted by wavelength:
            wavelength_nm, intensity   at the apex
            prominence                 height above the higher of the two flanking valleys
            pixel                      index of the apex in `wl`
            excluded                   True if the apex is an excluded pixel

    Raises:
        ValueError: the median of `y` is not positive (e.g. a continuum-corrected spectrum),
            so a relative threshold means nothing.
    '''
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    median = float(np.median(y))
    if not median > 0:
        raise ValueError(f'the median intensity is {median:g}; detect_peaks needs a spectrum '
                         'with a positive continuum (use the raw, not the corrected, spectrum)')
    idx, props = find_peaks(y, prominence=min_prominence * median)
    return pd.DataFrame({'wavelength_nm': wl[idx], 'intensity': y[idx],
                         'prominence': props['prominences'], 'pixel': idx,
                         'excluded': exclusion_mask(wl, exclude)[idx]})


def total_area(wl: np.ndarray, corrected: np.ndarray, exclude=None) -> float:
    '''
    Integrated area (counts*nm per shot) of a continuum-corrected spectrum.

    Args:
        exclude: pixels to drop before integrating, e.g. saturated lines; either a boolean
            array (see `saturated_mask`) or a list of (lo, hi) nm ranges. Masked pixels
            are set to zero so the integration doesn't bridge across them.
    '''
    y = np.where(exclusion_mask(wl, exclude), 0.0, np.asarray(corrected, dtype=float))
    return float(np.trapezoid(y, wl))


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


def line_window(wl: np.ndarray, y: np.ndarray, center: float, half_width: float | None = None,
                search: float = 0.1, max_half_width: float = 0.35,
                min_px: int = 3) -> tuple[int, int]:
    '''
    Where one emission line sits in a spectrum, as a (left, right) pixel range.

    half_width=None (default): found from the data with `peak_bounds`, from the apex near
    `center` down to the valley on each side. A number in nm: the fixed window
    center ± half_width instead.

    Returns:
        (left, right) pixel indices. Both ends are included, so the line is y[left:right + 1].

    Raises:
        ValueError: the window is outside the wavelength range or has fewer than `min_px` pixels.
    '''
    if half_width is None:
        left, _, right = peak_bounds(wl, y, center, search, max_half_width)

    else:
        idx = np.flatnonzero((wl >= center - half_width) & (wl <= center + half_width))
        if len(idx) == 0:
            raise ValueError(f'{center} ± {half_width} nm is outside the wavelength range')
        left, right = idx[0], idx[-1]

    if right - left + 1 < min_px:
        raise ValueError(f'window around {center} nm has only {right - left + 1} pixels')
    
    return int(left), int(right)


def integrate_local_peak(wl: np.ndarray, y: np.ndarray, center: float,
                         half_width: float | None = None, edge_pts: int = EDGE_PTS,
                         search: float = 0.1, max_half_width: float = 0.35,
                         bounds: tuple[int, int] | None = None) -> dict:
    '''
    Area of one emission line above a straight local baseline.

    Steps:
        1. Window: `bounds` = (left, right) if given, otherwise found with `line_window`
           (peak to valley by default, or center ± `half_width`). Pass the `bounds` found
           on a reference spectrum to measure every temperature over the same pixels.
        2. Baseline: a straight line through the mean of the first `edge_pts` pixels and
           the mean of the last `edge_pts` pixels of the window.
        3. Area: integrate what is above that line.

    Pass the per-shot spectrum, not the continuum-corrected one: the local baseline already
    removes the background under the line.

    Returns:
        dict with 'area' (counts·nm per shot), 'apex_nm', and for plotting the window's
        'wl', 'y', 'local_baseline' and 'net' (= y - local_baseline).
    '''
    # 1. finding window for integration
    if bounds is None:
        bounds = line_window(wl, y, center, half_width, search, max_half_width)
    left, right = bounds
    x = wl[left:right + 1]
    signal = y[left:right + 1]
    if len(x) < max(3, 2 * edge_pts + 1):
        raise ValueError(f'window around {center} nm has only {len(x)} pixels')

    # 2. straight baseline through the two edges
    x_left, x_right = x[:edge_pts].mean(), x[-edge_pts:].mean()
    y_left, y_right = signal[:edge_pts].mean(), signal[-edge_pts:].mean()
    slope = (y_right - y_left) / (x_right - x_left)
    baseline = y_left + slope * (x - x_left)

    # 3. area above it
    net = signal - baseline
    return {'area': float(np.trapezoid(net, x)),
            'apex_nm': float(x[np.argmax(signal)]),
            'wl': x, 'y': signal, 'local_baseline': baseline, 'net': net}
