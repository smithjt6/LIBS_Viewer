'''
LIBS (Applied Spectra .tdms) loading, processing and analysis.

    loading     LIBSFile, LoadLIBSData, study_plan (pattern / manifest), file-name parsing
    processing  per-shot scaling, saturation mask, continuum (baseline) removal
    metrics     total area, line windows, local-baseline line area
    lines       NIST-based line selection
    analysis    per-sample area tables and ratio tables
    study       run_study: a whole temperature study in one call
    plotting    figures (import it directly: `from libs_viewer.libs.plotting import plot_spectrum`)

The common names are re-exported here. Plotting is not, so the analysis code can be imported
without matplotlib.
'''

from .loading import LIBSFile, LoadLIBSData, study_plan, temperature_from_name
from .processing import (exclusion_mask, global_baseline, per_shot_spectrum, saturated_mask,
                         shots_per_column)
from .metrics import integrate_local_peak, peak_bounds, total_area
from .lines import select_lines
from .analysis import TOTAL, TOTAL_EXCL, analyze_sample, line_windows, per_column_areas, ratio_tables
from .study import StudyResult, run_study

__all__ = [
    'LIBSFile', 'LoadLIBSData', 'study_plan', 'temperature_from_name',
    'exclusion_mask', 'global_baseline', 'per_shot_spectrum', 'saturated_mask', 'shots_per_column',
    'integrate_local_peak', 'peak_bounds', 'total_area',
    'select_lines',
    'TOTAL', 'TOTAL_EXCL', 'analyze_sample', 'line_windows', 'per_column_areas', 'ratio_tables',
    'StudyResult', 'run_study',
]
