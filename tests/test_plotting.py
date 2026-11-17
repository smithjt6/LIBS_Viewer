import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from libs_viewer.libs.plotting import plot_spectrum, to_nm  # noqa: E402
from conftest import WL, synth_spectrum  # noqa: E402


def test_returns_figure_without_pyplot_state():
    plt.close('all')
    fig = plot_spectrum(WL, synth_spectrum({'B 400.00': 100.0}))
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == []                 # nothing left open in pyplot
    assert fig.axes[0].get_title() == ''           # no hard-coded title


def test_title_zoom_and_reference_lines():
    y = synth_spectrum({'A 390.00': 50.0, 'B 400.00': 100.0})
    fig = plot_spectrum(WL, y, title='Ce sample 1', wl_range=('395nm', 405), ref_lines={'Ce II': 400.0})
    ax = fig.axes[0]
    assert ax.get_title() == 'Ce sample 1'
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(395.0) and hi == pytest.approx(405.0)
    assert ax.get_legend() is not None


def test_warns_when_reference_line_has_no_peak():
    with pytest.warns(UserWarning, match='no peak within'):
        plot_spectrum(WL, synth_spectrum({'B 400.00': 100.0}), ref_lines=[405.0])


def test_draws_into_given_axes():
    fig = Figure()
    ax = fig.subplots()
    assert plot_spectrum(WL, synth_spectrum({'B 400.00': 100.0}), ax=ax) is fig


def test_bad_input():
    with pytest.raises(ValueError):
        plot_spectrum(WL, synth_spectrum({})[:-1])
    assert to_nm('394.22nm') == 394.22


def test_figure_renders_as_png_without_pyplot():
    """Jupyter shows `_repr_png_` even when pyplot's inline backend was never activated."""
    fig = plot_spectrum(WL, synth_spectrum({'B 400.00': 100.0}))
    assert fig._repr_png_().startswith(b'\x89PNG')
