'''
Choosing which emission lines to measure, using the NIST Handbook database.
'''

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..nist.query import strong_lines
from .metrics import peak_bounds
from .processing import exclusion_mask


def select_lines(element: str, wl: np.ndarray, y_refs: dict[str, np.ndarray], n: int = 4,
                 interferents: list[str] | None = None,
                 interf_species: tuple[str, ...] = ('I', 'II'), interf_tol: float = 0.15,
                 exclude=None, exclude_pad: float = 0.5,
                 peak_tol: float = 0.1, min_sep: float = 1.0, min_prominence: float = 2.0,
                 min_side_px: int = 2, candidates: int = 40,
                 db_path: str | Path | None = None) -> pd.DataFrame:
    '''
    Pick the `n` best NIST lines of one element to measure in the given spectra.

    The candidates are the `candidates` strongest NIST lines of `element` inside the
    wavelength range. Each goes through three checks in order; the first it fails becomes
    its status:

        1. Saturation    no excluded (e.g. saturated) pixel within `exclude_pad` nm.
        2. Interference  no strong line of an `interferents` element (ion stages
                         `interf_species`) within `interf_tol` nm.
        3. Detection     tested in every reference spectrum: there must be a peak within
                         `peak_tol` nm with a prominence of at least `min_prominence` x the
                         spectrum's median, and at least `min_side_px` pixels between its
                         apex and the valley on each side (fewer means a neighbouring line
                         is blended in). A line passes if it is detected in at least one
                         spectrum.

    The lines that pass are ranked by how many spectra they were detected in, then by NIST
    intensity, and picked in that order until there are `n`. A line within `min_sep` nm of
    one already picked is skipped, so the integration windows don't overlap.

    Args:
        element: chemical symbol, e.g. 'Ce'.
        wl: wavelength axis (nm) shared by the reference spectra.
        y_refs: {sample name: per-shot reference spectrum}, one per sample of this element.
        n: how many lines to pick.
        interferents: other elements in the samples whose lines must be avoided.
        interf_species: ion stages of the interferents to check ('I' neutral, 'II' singly
            ionized).
        interf_tol: how close (nm) an interferent line may be.
        exclude: pixels to stay away from: a boolean mask (e.g. `saturated_mask`) or a list
            of (lo, hi) nm ranges.
        exclude_pad: how close (nm) a line may be to an excluded pixel.
        peak_tol: how far (nm) the measured peak may be from the NIST wavelength.
        min_prominence: peak threshold, as a multiple of the spectrum's median.
        min_side_px: pixels needed between the apex and the valley on each side.
        min_sep: smallest distance (nm) allowed between two picked lines.
        candidates: how many of the strongest NIST lines to consider.
        db_path: NIST database (default: the libs_viewer data folder).

    Returns:
        DataFrame, one row per candidate, strongest NIST line first:
            species, intensity, wavelength_nm   from NIST
            status        'picked', or why not: 'near saturated region',
                          'interference: <lines>', 'no peak in spectrum',
                          'blended (neighbour line too close)' (both joined with ' / ' when
                          the spectra disagree), 'too close to a picked line', 'not needed'
            n_detected    number of spectra the line was detected in
            detected_in   names of those spectra (keys of `y_refs`)
            label         e.g. 'Ce II 418.66'
    '''
    wl_range = (wl.min(), wl.max())
    cands = strong_lines(element, wl_range=wl_range, db_path=db_path).head(candidates)

    others = (
        pd.concat(
            [strong_lines(el, interf_species, wl_range, db_path=db_path).assign(element=el)
             for el in interferents],
            ignore_index=True,
        )
        if interferents else None
    )

    excl_wl = wl[exclusion_mask(wl, exclude)]
    peak_wl = {
        name: wl[find_peaks(y, prominence=min_prominence * np.median(y))[0]]
        for name, y in y_refs.items()
    }

    # Screen every candidate line
    screened = [
        _screen_candidate(
            c, wl, y_refs, excl_wl, others, peak_wl,
            exclude_pad=exclude_pad,
            interf_tol=interf_tol,
            peak_tol=peak_tol,
            min_side_px=min_side_px,
        )
        for c in cands.wavelength_nm
    ]
    status, detected_in = zip(*screened) if screened else ([], [])

    cands = cands.assign(
        status=list(status),
        n_detected=[len(d) for d in detected_in],
        detected_in=list(detected_in),
    )

    # Rank usable candidates (detected in most samples first, then NIST intensity)
    picked: list[float] = []
    usable = cands[cands.status == ''].sort_values(
        ['n_detected', 'intensity'], ascending=False, kind='stable'
    )
    for i, c in usable.wavelength_nm.items():
        if any(abs(c - p) < min_sep for p in picked):
            cands.loc[i, 'status'] = 'too close to a picked line'
        elif len(picked) >= n:
            cands.loc[i, 'status'] = 'not needed'
        else:
            cands.loc[i, 'status'] = 'picked'
            picked.append(c)

    cands['label'] = cands.species + ' ' + cands.wavelength_nm.map('{:.2f}'.format)
    return cands


def _screen_candidate(
    c: float,
    wl: np.ndarray,
    y_refs: dict[str, np.ndarray],
    excl_wl: np.ndarray,
    others: pd.DataFrame | None,
    peak_wl: dict[str, np.ndarray],
    exclude_pad: float,
    interf_tol: float,
    peak_tol: float,
    min_side_px: int,
) -> tuple[str, list[str]]:
    '''Return (rejection_reason_or_empty, list_of_samples_where_detected) for one line.'''
    if len(excl_wl) and np.abs(excl_wl - c).min() <= exclude_pad:
        return 'near saturated region', []

    if others is not None:
        hits = others[(others.wavelength_nm - c).abs() <= interf_tol]
        if not hits.empty:
            details = ', '.join(f'{s} {w:.2f}' for s, w in zip(hits.species, hits.wavelength_nm))
            return f'interference: {details}', []

    found: list[str] = []
    reasons: set[str] = set()

    for name, y in y_refs.items():
        peaks = peak_wl[name]
        if not len(peaks) or np.abs(peaks - c).min() > peak_tol:
            reasons.add('no peak in spectrum')
            continue

        left, apex, right = peak_bounds(wl, y, c, peak_tol)
        if min(apex - left, right - apex) < min_side_px:
            reasons.add('blended (neighbour line too close)')
            continue

        found.append(name)

    status = '' if found else ' / '.join(sorted(reasons))
    return status, found