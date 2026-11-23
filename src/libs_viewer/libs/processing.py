'''
Spectrum-level processing: per-shot scaling, saturation, continuum (baseline) removal.

Per-shot scaling uses `LIBSFile.acquisitions['shots_summed']` of every stored spectrum, so
accumulated spectra, single shots and files mixing both all end up in counts per shot. To use
only some spectra or locations, pass `f.subset(...)`.

All functions take arrays / LIBSFile objects and return arrays; no plotting, no file I/O.
'''

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pybaselines import Baseline

from .loading import LIBSFile


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


# ------------------------------------------------------------------ continuum (baseline) removal
@dataclass(frozen=True)
class BaselineMethod:
    '''One continuum-removal method (see `register_baseline`).'''
    name: str
    fit: Callable[..., np.ndarray]     # fit(wl, y, **params) -> baseline, same length as y
    defaults: dict                     # every parameter the method takes, with its default
    description: str                   # one plain sentence, e.g. for help text in a UI


BASELINES: dict[str, BaselineMethod] = {}


def register_baseline(name: str, description: str, **defaults):
    '''
    Decorator that adds a continuum-removal method under `name`. The decorated function
    takes (wl, y, **params) and returns the baseline (same length as y); `defaults` lists
    every parameter it takes. Once registered, the method works everywhere a baseline is
    chosen (`global_baseline`, `areas_table`, `analyze_sample`, `run_study`, `plot_baseline`):

        @register_baseline('asls', 'Asymmetric least squares.', lam=1e6, p=0.01)
        def _asls(wl, y, lam, p):
            return Baseline(x_data=wl).asls(y, lam=lam, p=p)[0]
    '''
    def add(fit: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        if name in BASELINES:
            raise ValueError(f'baseline method {name!r} is already registered')
        BASELINES[name] = BaselineMethod(name, fit, dict(defaults), description)
        return fit
    return add


@register_baseline('snip', 'SNIP: repeatedly clips peaks narrower than the window, leaving the '
                   'broad continuum. Good for sharp lines on a broad background (default).',
                   max_half_window=40)
def _snip(wl, y, max_half_window):
    # max_half_window: clipping half-window in pixels (~0.07 nm/pixel here, so 40 px ~ 2.8 nm).
    # Too small and SNIP eats into the bases of broad lines.
    return Baseline(x_data=wl).snip(y, max_half_window=int(max_half_window), decreasing=True)[0]


@register_baseline('airpls', 'airPLS: a smooth curve fitted under the spectrum, pushed down '
                   'wherever peaks rise above it. lam sets the smoothness (1e5-1e7 typical).',
                   lam=1e6)
def _airpls(wl, y, lam):
    return Baseline(x_data=wl).airpls(y, lam=float(lam))[0]

@register_baseline("asls", "Asymmetric least squares: ...", lam=1e6, p=0.01)
def _asls(wl, y, lam, p):
    return Baseline(x_data=wl).asls(y, lam=lam, p=p)[0]

def baseline_settings(method: str = 'snip', params: dict | None = None) -> dict:
    '''
    All parameters of a baseline method: its defaults, overridden by `params`.

    Raises:
        ValueError: unknown method, or a parameter the method does not take.
    '''
    if method not in BASELINES:
        raise ValueError(f'baseline method must be one of {list(BASELINES)}, got {method!r}')
    defaults = BASELINES[method].defaults
    unknown = sorted(set(params or {}) - set(defaults))
    if unknown:
        raise ValueError(f'baseline {method!r} takes {list(defaults) or "no parameters"}, '
                         f'not {unknown}')
    return {**defaults, **(params or {})}


def global_baseline(wl: np.ndarray, y: np.ndarray, method: str = 'snip',
                    **params) -> tuple[np.ndarray, np.ndarray]:
    '''
    Estimates the broad plasma continuum and subtracts it.

    Args:
        wl, y: wavelength axis (nm) and per-shot intensity.
        method: a name in `BASELINES`: 'snip' (default) or 'airpls', plus any added with
            `register_baseline`.
        **params: that method's parameters; any left out take their defaults
            (`BASELINES[method].defaults`), e.g. max_half_window=60 for SNIP, lam=1e7 for airPLS.

    Returns:
        (corrected, baseline) arrays.
    '''
    settings = baseline_settings(method, params)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(BASELINES[method].fit(wl, y, **settings), dtype=float)
    return y - baseline, baseline


def exclusion_mask(wl: np.ndarray, exclude) -> np.ndarray:
    '''
    Boolean pixel mask from `exclude`, which may be None (nothing excluded), a boolean
    array/sequence aligned with `wl` (e.g. from `saturated_mask`), or an iterable of
    (lo, hi) nm ranges.
    '''
    if exclude is None:
        return np.zeros(len(wl), dtype=bool)

    arr = np.asarray(exclude)
    if arr.dtype == bool:
        if arr.shape != wl.shape:
            raise ValueError(f"Boolean exclude mask shape {arr.shape} does not match wl shape {wl.shape}")
        return arr.copy()

    mask = np.zeros(len(wl), dtype=bool)
    
    for item in exclude:
        try:
            lo, hi = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected a boolean mask or (lo, hi) wavelength pairs, got element {item!r}"
            ) from exc
        mask |= (wl >= lo) & (wl <= hi)

    return mask


SATURATION_LEVEL = 64900.0          # counts per shot where a single shot is taken as clipped
ACCUMULATED_SATURATION_LEVEL = 52000.0   # per-shot mean where an accumulated spectrum is


def saturated_pixels(f: LIBSFile, level: float = SATURATION_LEVEL,
                     accumulated_level: float = ACCUMULATED_SATURATION_LEVEL) -> pd.DataFrame:
    '''
    Which pixels clipped in each stored spectrum of one file.

    Single shots are flagged where the intensity reaches `level` counts. Accumulated spectra
    are flagged from the lower `accumulated_level` (per-shot mean), because their sum only
    reaches N x `level` where every shot clipped (see `saturated_mask`). No padding.

    Returns:
        Boolean DataFrame shaped like `f.spectra` (index = wavelength in nm, one column per
        stored spectrum). `.any()` gives the clipped spectra, `.any(axis=1)` the clipped pixels.
    '''
    accumulated = f.acquisitions['shots_summed'] > 1
    limit = accumulated.map({True: accumulated_level, False: level}).astype(float)
    return per_shot_columns(f).ge(limit, axis=1)


def saturation_summary(f: LIBSFile, level: float = SATURATION_LEVEL,
                       accumulated_level: float = ACCUMULATED_SATURATION_LEVEL) -> pd.DataFrame:
    '''
    One row per stored spectrum: how close it came to the detector limit.

    Returns:
        DataFrame indexed like `f.acquisitions` with columns
            location, shot, shots_summed   from `f.acquisitions`
            peak             highest intensity, counts per shot
            limit            the threshold used (`level`, or `accumulated_level` if accumulated)
            saturated_px     number of flagged pixels
            saturated        True if any pixel is flagged
    '''
    sat = saturated_pixels(f, level, accumulated_level)
    acq = f.acquisitions
    out = acq[['location', 'shot', 'shots_summed']].copy()
    out['peak'] = per_shot_columns(f).max()
    out['limit'] = np.where(acq['shots_summed'] > 1, accumulated_level, level)
    out['saturated_px'] = sat.sum().astype(int)
    out['saturated'] = out['saturated_px'] > 0
    return out


def mask_ranges(wl: np.ndarray, mask) -> list[tuple[float, float]]:
    '''
    The (lo, hi) nm ranges where `mask` is True, one per run of adjacent pixels; the
    inverse of `exclusion_mask`. Handy for shading saturated regions on a plot.
    '''
    wl = np.asarray(wl, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if m.shape != wl.shape:
        raise ValueError(f'mask shape {m.shape} does not match wl shape {wl.shape}')
    edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(float(wl[a]), float(wl[b])) for a, b in zip(starts, stops)]


def saturated_mask(files: list[LIBSFile], level: float = SATURATION_LEVEL,
                   accumulated_level: float = ACCUMULATED_SATURATION_LEVEL,
                   pad_px: int = 2) -> np.ndarray:
    '''
    Pixels where the detector clipped in any stored spectrum of any of the given files.

    Single shots: flagged where the intensity reaches `level` counts (the detector clips
    at 64,980-65,500 counts per shot, depending on the spectrometer channel).

    Accumulated spectra: the stored sum of N shots only reaches N x `level` where every shot
    clipped; where only some did, the per-shot mean sits below `level`. So these are flagged
    from the lower `accumulated_level` (about 80% of full scale) up: with the 15-50%
    shot-to-shot scatter seen in LIBS, a pixel that bright on average clipped on some shots.

    Each flag is widened by `pad_px` pixels on each side. Pass all temperatures of one sample
    together so the same pixels are excluded at every temperature. For one file, spectrum
    by spectrum, see `saturated_pixels` / `saturation_summary`.

    Returns:
        Boolean array aligned with the wavelength axis.
    '''
    mask = np.zeros(len(files[0].wavelengths), dtype=bool)
    for f in files:
        mask |= saturated_pixels(f, level, accumulated_level).to_numpy().any(axis=1)
    if pad_px:
        mask = np.convolve(mask, np.ones(2 * pad_px + 1), mode='same') > 0
    return mask
