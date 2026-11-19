'''
Spectrum-level processing: per-shot scaling, saturation, continuum (baseline) removal.

Per-shot scaling uses `LIBSFile.acquisitions['shots_summed']` of every stored spectrum, so
accumulated spectra, single shots and files mixing both all end up in counts per shot. To use
only some spectra or locations, pass `f.subset(...)`.

All functions take arrays / LIBSFile objects and return arrays; no plotting, no file I/O.
'''

import numpy as np
import pandas as pd
from pybaselines import Baseline

from .loading import LIBSFile

BASELINE_METHODS = ('snip', 'airpls')


def per_shot_columns(f: LIBSFile) -> pd.DataFrame:
    '''Every stored spectrum in counts per shot (each column divided by its shots summed).'''
    return f.spectra.div(f.acquisitions['shots_summed'], axis=1)


def per_shot_spectrum(f: LIBSFile) -> np.ndarray:
    '''
    Mean single-shot spectrum of one file: all counts divided by all shots, so every shot
    weighs the same whether it was stored on its own or accumulated.

    Returns:
        1-D array of intensities (counts per shot), aligned with `f.wavelengths`.
    '''
    total = f.spectra.sum(axis=1) / f.acquisitions['shots_summed'].sum()
    return total.to_numpy(dtype=float)


def location_spectra(f: LIBSFile) -> pd.DataFrame:
    '''Mean single-shot spectrum of every location (columns = location numbers).'''
    loc = f.acquisitions['location']
    summed = f.spectra.T.groupby(loc).sum().T
    out = summed / f.acquisitions.groupby('location')['shots_summed'].sum()
    out.columns.name = 'location'
    return out


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


def saturated_mask(files: list[LIBSFile], level: float = 64900.0,
                   accumulated_level: float = 52000.0, pad_px: int = 2) -> np.ndarray:
    '''
    Pixels where the detector clipped in any stored spectrum of any of the given files.

    Single shots: flagged where the intensity reaches `level` counts (the detector clips
    at 64,980-65,500 counts per shot, depending on the spectrometer channel).

    Accumulated spectra: the stored sum of N shots only reaches N x `level` where every shot
    clipped; where only some did, the per-shot mean sits below `level`. So these are flagged
    from the lower `accumulated_level` (about 80% of full scale) up: with the 15-50%
    shot-to-shot scatter seen in LIBS, a pixel that bright on average clipped on some shots.

    Each flag is widened by `pad_px` pixels on each side. Pass all temperatures of one sample
    together so the same pixels are excluded at every temperature.

    Returns:
        Boolean array aligned with the wavelength axis.
    '''
    mask = np.zeros(len(files[0].wavelengths), dtype=bool)
    for f in files:
        accumulated = f.acquisitions['shots_summed'].to_numpy() > 1
        limit = np.where(accumulated, accumulated_level, level)        # one limit per spectrum
        mask |= (per_shot_columns(f).to_numpy(dtype=float) >= limit).any(axis=1)
    if pad_px:
        mask = np.convolve(mask, np.ones(2 * pad_px + 1), mode='same') > 0
    return mask
