import numpy as np
import pandas as pd
import pytest

from libs_viewer.libs import (LIBSFile, TOTAL, TOTAL_EXCL, analyze_sample, integrate_local_peak,
                              line_windows, peak_bounds, per_column_areas, ratio_tables, total_area)
from conftest import LINES, WL, gaussian, synth_spectrum


# ------------------------------------------------------------------ metrics
def test_peak_bounds_finds_apex_and_valleys():
    y = synth_spectrum({'B 400.00': 100.0})
    left, apex, right = peak_bounds(WL, y, 400.03)       # small calibration offset
    assert WL[apex] == pytest.approx(400.0)
    assert left < apex < right
    assert WL[apex] - WL[left] <= 0.35 + 1e-9 and WL[right] - WL[apex] <= 0.35 + 1e-9


def test_peak_bounds_outside_range():
    with pytest.raises(ValueError, match='outside'):
        peak_bounds(WL, synth_spectrum({}), 500.0)


def test_integrate_local_peak_recovers_area_on_sloped_continuum():
    y = synth_spectrum({'B 400.00': 100.0, 'C 410.00': 30.0})
    r = integrate_local_peak(WL, y, 400.0)
    assert r['area'] == pytest.approx(100.0, rel=0.02)
    assert r['apex_nm'] == pytest.approx(400.0)
    assert integrate_local_peak(WL, y, 410.0)['area'] == pytest.approx(30.0, rel=0.02)


def test_integrate_local_peak_fixed_window_and_bounds_agree():
    y = synth_spectrum({'B 400.00': 100.0})
    fixed = integrate_local_peak(WL, y, 400.0, half_width=0.3)
    i0, i1 = np.searchsorted(WL, fixed['wl'][0]), np.searchsorted(WL, fixed['wl'][-1])
    by_bounds = integrate_local_peak(WL, y, 400.0, bounds=(i0, i1))
    assert fixed['area'] == pytest.approx(by_bounds['area'])


def test_total_area_and_exclusion():
    flat = np.zeros(len(WL))
    flat[200:220] = 1.0
    full = total_area(WL, flat)
    assert full == pytest.approx(np.trapezoid(flat, WL))
    assert total_area(WL, flat, exclude=[(WL[195], WL[225])]) == 0.0


# ------------------------------------------------------------------ analysis
@pytest.fixture
def temperature_series():
    '''Line B doubles at 350 C and triples at 400 C; A and C stay put.'''
    base = {'A 390.00': 50.0, 'C 410.00': 30.0}
    return {300: synth_spectrum({**base, 'B 400.00': 100.0}),
            350: synth_spectrum({**base, 'B 400.00': 200.0}),
            400: synth_spectrum({**base, 'B 400.00': 300.0})}


def test_analyze_sample_and_ratio_tables(temperature_series):
    areas = analyze_sample(WL, temperature_series, LINES)
    assert list(areas.columns) == [300, 350, 400]
    assert list(areas.index) == [*LINES, TOTAL, TOTAL_EXCL]
    assert areas.loc['B 400.00', 300] == pytest.approx(100.0, rel=0.03)

    r = ratio_tables(areas)
    direct, rel = r['R_direct'], r['R_rel']
    np.testing.assert_allclose(direct[300], 1.0)
    assert direct.loc['B 400.00', 350] == pytest.approx(2.0, rel=0.02)
    assert direct.loc['B 400.00', 400] == pytest.approx(3.0, rel=0.02)
    assert direct.loc['A 390.00', 400] == pytest.approx(1.0, rel=0.02)
    assert direct.loc[TOTAL, 400] > 1.0                     # more light overall
    # R_rel = R_direct(line) / R_direct(total): A didn't change, so its share drops
    for t in (350, 400):
        assert rel.loc['A 390.00', t] == pytest.approx(direct.loc['A 390.00', t] / direct.loc[TOTAL, t])
        assert rel.loc['A 390.00', t] < 1.0
    pd.testing.assert_series_equal(rel.loc[TOTAL], direct.loc[TOTAL])


def test_windows_come_from_reference_temperature(temperature_series):
    w300 = line_windows(WL, temperature_series[300], LINES)
    areas = analyze_sample(WL, temperature_series, LINES, ref=300)
    for label, (i0, i1) in w300.items():
        expected = integrate_local_peak(WL, temperature_series[400], LINES[label], bounds=(i0, i1))['area']
        assert areas.loc[label, 400] == pytest.approx(expected)


def test_per_column_areas_one_column_per_location(tdms_file):
    f = LIBSFile.from_tdms(tdms_file[0])
    windows = line_windows(WL, tdms_file[1], LINES)
    table = per_column_areas(f, LINES, windows)
    assert list(table.columns) == ['Location 1', 'Location 2', 'Location 3']
    # columns were scaled 0.99 / 1.00 / 1.01
    np.testing.assert_allclose(table.loc['B 400.00'] / table.loc['B 400.00', 'Location 2'],
                               [0.99, 1.0, 1.01], rtol=1e-9)
