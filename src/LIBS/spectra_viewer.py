'''
This deals with plotting LIBS Spectra from a DF or TDMS File directly
'''

import matplotlib.pyplot as plt
import warnings
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import find_peaks
import pandas as pd
import numpy as np

class PlotLIBSData:
    '''
    This class plots LIBS data from a DataFrame.
    '''
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_tdms(self, wavelengths, intensities, log_y=True, n_labels=15, min_label_gap=12, title='Ce Sample',
              ref_lines=None, peak_tol=0.2, wl_range=None):
        wl = np.asarray(wavelengths, dtype=float)
        counts = np.asarray(intensities, dtype=float)
        if wl.ndim != 1 or counts.ndim != 1 or len(wl) != len(counts):
            raise ValueError('wavelengths and intensities must be one-dimensional and have equal length')
        if len(wl) == 0:
            raise ValueError('wavelengths and intensities cannot be empty')

        # A mapping gives reference labels; a sequence uses the wavelength as its label.
        if hasattr(ref_lines, 'items'):
            references = [(str(label), self._to_nm(value)) for label, value in ref_lines.items()]
        else:
            references = [
                (f'{self._to_nm(value):.2f} nm', self._to_nm(value))
                for value in (ref_lines or [])
            ]

        # zoom window: whole spectrum, or each end snapped to the nearest measured wavelength
        if wl_range is None:
            lo, hi = wl.min(), wl.max()
        else:
            lo, hi = sorted(wl[np.argmin(np.abs(wl - self._to_nm(x)))] for x in wl_range)
        in_view = (wl >= lo) & (wl <= hi)
        span = hi - lo

        # peaks are found on the full spectrum so zooming doesn't change what counts as a peak
        peaks, props = find_peaks(counts, prominence=np.median(counts))
        peaks = peaks[np.argsort(props['prominences'])[::-1]]
        # label spacing shrinks with the window so a zoom still gets labels
        label_gap = min_label_gap * span / (wl.max() - wl.min())
        labelled = []
        for p in peaks[in_view[peaks]]:
            if all(abs(wl[p] - wl[q]) >= label_gap for q in labelled):
                labelled.append(p)
            if len(labelled) == n_labels:
                break

        fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
        ax.plot(wl[in_view], counts[in_view], color='#1f5fa8', linewidth=0.6 if span > 100 else 1)

        decimals = 1 if span > 50 else 2
        for p in labelled:
            ax.annotate(f'{wl[p]:.{decimals}f}', (wl[p], counts[p]), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', rotation=90, fontsize=8, color='#444444',
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

        view_counts = counts[in_view]
        visible_refs = [(label, x) for label, x in references if lo <= x <= hi]
        for i, (label, x) in enumerate(visible_refs):
            nearest_index = np.argmin(np.abs(wl - x))
            actual_x = wl[nearest_index]
            ax.axvline(actual_x, color='#d62728', linewidth=0.8, linestyle='--',
                       label='Reference lines' if i == 0 else None)
            ax.annotate(label, xy=(actual_x, counts[nearest_index]),
                        xytext=(actual_x, view_counts.max() * 0.88),
                        ha='center', va='top', rotation=90, fontsize=8,
                        color='#d62728',
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))
        if visible_refs:
            ax.legend(loc='upper right', frameon=False)

        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(view_counts.min() * 0.8, view_counts.max() * 3)
        else:
            ax.set_ylim(0, view_counts.max() * 1.15)

        ax.set_xlim(lo, hi)
        if span > 200:
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='major', color='#e6e6e6', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (counts)')
        ax.set_title(title)
        plt.show()

        # warn after the figure so the message sits below it, not on it
        peak_wl = wl[peaks]
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            for _, x in references:
                nearest = peak_wl[np.argmin(np.abs(peak_wl - x))]
                if abs(nearest - x) > peak_tol:
                    warnings.warn(f'no peak within {peak_tol} nm of {x} nm (nearest peak: {nearest:.2f} nm)', stacklevel=2)

    def _to_nm(self, x: float | str) -> float:
        '''Converts a value to nanometers. Accepts numbers or strings like '394.22nm' '''
        return float(str(x).lower().replace('nm', ''))