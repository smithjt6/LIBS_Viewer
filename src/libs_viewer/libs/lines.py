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
                 interf_species: tuple[str, ...] = ('I',), interf_tol: float = 0.15,
                 exclude=None, exclude_pad: float = 0.5,
                 peak_tol: float = 0.1, min_sep: float = 1.0, min_prominence: float = 2.0,
                 min_side_px: int = 2, candidates: int = 40,
                 db_path: str | Path | None = None) -> pd.DataFrame:
    '''
    Picks the n strongest NIST lines of an element that are usable in measured spectra.

    Args:
        element: symbol, e.g. 'Ce'.
        wl: wavelength axis (nm) shared by the reference spectra.
        y_refs: {sample label: reference spectrum} for every sample of this element.
        db_path: NIST database (default: the libs_viewer data folder).

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
        ('picked' or the reason it was not) and 'detected_in' (list of sample labels).
    '''

    wl_range = (wl.min(), wl.max())

    cands = strong_lines(element, wl_range=wl_range, db_path=db_path).head(candidates)

    others = pd.concat([strong_lines(el, interf_species, wl_range, db_path=db_path).assign(element=el)
                        for el in interferents], ignore_index=True) if interferents else None
    
    excl_wl = wl[exclusion_mask(wl, exclude)]
    
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
                         detected_in=detected_in)
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
