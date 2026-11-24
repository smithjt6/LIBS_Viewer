'''The Study page's plotly figures, built from a real (synthetic) study result.'''

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('plotly')
sys.path.insert(0, str(Path(__file__).parents[1] / 'src' / 'ui'))

from libs_viewer.libs import run_study  # noqa: E402
from study_figures import baseline_figure, lines_figure, ratio_figure  # noqa: E402
from conftest import gaussian, write_tdms  # noqa: E402

WL = np.round(np.arange(390.0, 420.0, 0.02), 4)
CE_LINES = [395.2545, 399.9237, 418.6596]


@pytest.fixture
def result(tmp_path, nist_db):
    for sample, scales in {'s1': (1, 2, 3), 's2': (1, 1, 1)}.items():
        for t, k in zip((300, 350, 400), scales):
            y = 100.0 + sum(gaussian(WL, c, 20.0 * k, sigma=0.03) for c in CE_LINES)
            write_tdms(tmp_path / 'Ce' / sample / f'Ce_{sample}_{t}C.tdms', y, wl=WL)
    return run_study(tmp_path, pattern='{element}/{sample}/*{temperature}*', n_lines=3,
                     db_path=nist_db, scatter=False)


def test_ratio_figure_has_a_panel_per_sample(result):
    fig = ratio_figure(result)
    assert [a.text for a in fig.layout.annotations] == ['Ce s1', 'Ce s2']
    assert sum(t.name.startswith('R_total') for t in fig.data) == 2
    assert fig.layout.yaxis.type == 'log' and ratio_figure(result, log_y=False).layout.yaxis.type == 'linear'


def test_baseline_figure_has_spectrum_baseline_and_corrected(result):
    s = result.settings
    fig = baseline_figure(result.wl, result.spectra['Ce s1'][350], s['baseline'], s['baseline_params'])
    assert [t.name.split(' ')[0] for t in fig.data] == ['spectrum', s['baseline'], 'corrected']


def test_lines_figure_marks_only_lines_in_the_window(result):
    spectrum = result.spectra['Ce s1'][350]
    lines = result.lines['Ce s1']
    assert len(lines_figure(result.wl, spectrum, lines, (390, 420)).layout.shapes) == 3
    fig = lines_figure(result.wl, spectrum, lines, (394, 400))
    assert len(fig.layout.shapes) == 2 and fig.layout.xaxis.range == (394, 400)
