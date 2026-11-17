'''
Spectrum-level processing: shot scaling, saturation, continuum (baseline) removal.

All functions take arrays / LIBSFile objects and return arrays; no plotting, no file I/O.
'''

from typing import Sequence
import numpy as np
from pybaselines import Baseline

from .loading import LIBSFile

BASELINE_METHODS = ('snip', 'airpls')


def shots_per_column(f: LIBSFile) -> float:
    '''
    Number of laser shots summed into each spectrum column.

    Accumulated files (Accumulate = 1) store the sum of `Number Of Shots` shots per column;
    otherwise each column is one shot.
    '''
    return float(f.params['Number Of Shots']) if f.accumulated else 1.0


def per_shot_spectrum(f: LIBSFile, columns: list[str] | None = None) -> np.ndarray:
    '''
    Mean single-shot spectrum of one file.

    Each column is divided by `shots_per_column(f)` and the selected columns are averaged.

    Args:
        f: a loaded LIBSFile.
        columns: column names to use (default: all). Use this to drop columns that were
            acquired with different settings than the first column, since `f.params`
            only describes the first column.

    Returns:
        1-D array of intensities (counts per shot), aligned with `f.wavelengths`.
    '''
    spectra = f.spectra if columns is None else f.spectra[columns]
    return (spectra / shots_per_column(f)).mean(axis=1).to_numpy(dtype=float)


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
        raise ValueError(f'method must be one of {BASELINE_METHODS}, got {method!r}')
    return y - baseline, baseline


def exclusion_mask(wl: np.ndarray, exclude) -> np.ndarray:
    '''
    Boolean pixel mask from `exclude`, which may be None (nothing excluded), a boolean array
    aligned with `wl` (e.g. from `saturated_mask`), or a list of (lo, hi) nm ranges.
    '''
    if exclude is None:
        return np.zeros(len(wl), dtype=bool)
    if isinstance(exclude, np.ndarray) and exclude.dtype == bool:
        return exclude
    mask = np.zeros(len(wl), dtype=bool)
    for lo, hi in exclude:
        mask |= (wl >= lo) & (wl <= hi)
    return mask


def saturated_mask(files: list[LIBSFile], columns: Sequence[Sequence[str] | None] | None = None,
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
        mask |= (spectra.max(axis=1).to_numpy(dtype=float) / shots_per_column(f)) >= level
    if pad_px:
        mask = np.convolve(mask, np.ones(2 * pad_px + 1), mode='same') > 0
    return mask
