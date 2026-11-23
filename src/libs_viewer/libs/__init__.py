'''
LIBS (Applied Spectra .tdms) loading, processing and analysis.

    loading     LIBSFile (one file: spectra + one row per stored spectrum), LoadLIBSData
    plan        read_manifest, study_plan: which file is which (manifest / pattern)
    processing  per-shot scaling, location spectra, saturation per spectrum / mask, continuum (baseline) removal
    metrics     peak detection, total area, line windows, local-baseline line area
    lines       NIST-based line selection
    analysis    area tables and ratio tables
    study       run_study: a whole temperature study in one call
    plotting    figures (import it directly: `from libs_viewer.libs.plotting import plot_spectrum`)

The common names are re-exported here. Plotting is not, so the analysis code can be imported
without matplotlib.
'''

from .loading import LayoutWarning, LIBSFile, LoadLIBSData, SettingsWarning, overview_table
from .plan import read_manifest, study_plan, temperature_from_name
from .processing import (BASELINES, baseline_settings, exclusion_mask, global_baseline,
                         location_spectra, mask_ranges, per_shot_columns, per_shot_spectrum,
                         register_baseline, saturated_mask, saturated_pixels, saturation_summary)
from .metrics import detect_peaks, integrate_local_peak, line_window, peak_bounds, total_area
from .lines import select_lines
from .analysis import (TOTAL, TOTAL_EXCL, RatioWarning, analyze_sample, areas_table, line_windows,
                       ratio_tables, saturated_share)
from .study import Sample, StudyResult, run_study

__all__ = [
    'LIBSFile', 'LoadLIBSData', 'overview_table', 'LayoutWarning', 'SettingsWarning',
    'read_manifest', 'study_plan', 'temperature_from_name',
    'BASELINES', 'baseline_settings', 'register_baseline',
    'exclusion_mask', 'global_baseline', 'location_spectra', 'per_shot_columns', 'per_shot_spectrum',
    'mask_ranges', 'saturated_mask', 'saturated_pixels', 'saturation_summary',
    'detect_peaks', 'integrate_local_peak', 'line_window', 'peak_bounds', 'total_area',
    'select_lines',
    'TOTAL', 'TOTAL_EXCL', 'RatioWarning', 'analyze_sample', 'areas_table', 'line_windows',
    'ratio_tables', 'saturated_share',
    'Sample', 'StudyResult', 'run_study',
]
