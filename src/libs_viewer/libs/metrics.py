'''
Numbers measured on one spectrum: total area, line windows, line areas.

All functions take arrays and return numbers / arrays; no plotting, no file I/O.
'''

import numpy as np

from .processing import exclusion_mask


def total_area(wl: np.ndarray, corrected: np.ndarray, exclude=None) -> float:
    '''
    Integrated area (counts*nm per shot) of a continuum-corrected spectrum.

    Args:
        exclude: pixels to drop before integrating, e.g. saturated lines; either a boolean
            array (see `saturated_mask`) or a list of (lo, hi) nm ranges. Masked pixels
            are set to zero so the integration doesn't bridge across them.
    '''
    y = np.asarray(corrected, dtype=float).copy()
    y[exclusion_mask(wl, exclude)] = 0.0
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
