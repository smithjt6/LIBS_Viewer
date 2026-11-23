import warnings

import numpy as np
import pandas as pd
import pytest

from libs_viewer.libs.metrics import EDGE_PTS, detect_peaks
from libs_viewer.libs import (LIBSFile, TOTAL, TOTAL_EXCL, RatioWarning, saturated_share,
                              analyze_sample, areas_table,
                              integrate_local_peak, line_window, line_windows, location_spectra,
                              peak_bounds,
                              ratio_tables, total_area)
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


def test_line_window_peak_and_fixed():
    y = synth_spectrum({'B 400.00': 100.0})
    left, _, right = peak_bounds(WL, y, 400.0)
    assert line_window(WL, y, 400.0) == (left, right)
    i0, i1 = line_window(WL, y, 400.0, half_width=0.3)
    assert WL[i0] == pytest.approx(399.7) and WL[i1] == pytest.approx(400.3)
    assert integrate_local_peak(WL, y, 400.0)['area'] == pytest.approx(
        integrate_local_peak(WL, y, 400.0, bounds=(left, right))['area'])


def test_line_window_errors():
    y = synth_spectrum({'B 400.00': 100.0})
    with pytest.raises(ValueError, match='outside'):
        line_window(WL, y, 500.0, half_width=0.2)
    with pytest.raises(ValueError, match='only'):
        line_window(WL, y, 400.0, half_width=0.01)     # one pixel
    with pytest.raises(ValueError, match='only'):
        line_windows(WL, y, {'B': 400.0}, half_width=0.05, edge_pts=2)   # 3 px < 5 needed


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

    r = ratio_tables(areas, ref=300)
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


def _areas(line_ref, total=(100.0, 120.0), total_excl=(100.0, 120.0)):
    return pd.DataFrame({300: [line_ref, total[0], total_excl[0]], 400: [5.0, total[1], total_excl[1]]},
                        index=['L', TOTAL, TOTAL_EXCL])


def test_ratio_tables_warns_when_total_is_saturated():
    areas = _areas(10.0, total=(100.0, 120.0), total_excl=(60.0, 90.0))   # 40% saturated at 300
    np.testing.assert_allclose(saturated_share(areas), [0.4, 0.25])
    with pytest.warns(RatioWarning, match='40% of the Total light'):
        ratio_tables(areas, ref=300, total_row=TOTAL)                        # opted in: warns
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        r = ratio_tables(areas, ref=300)                    # default leaves saturation out: silent
    assert r['R_rel'].loc['L', 400] == pytest.approx((5 / 90) / (10 / 60))
    with pytest.raises(ValueError, match='total_row'):
        ratio_tables(areas, ref=300, total_row='L')


def test_areas_table_warns_on_saturated_line_window(temperature_series):
    y = temperature_series[300]
    windows = line_windows(WL, y, LINES)
    on_b = [(399.95, 400.05)]                                # excluded pixels inside line B
    with pytest.warns(RatioWarning, match=r"\['B 400.00'\]"):
        areas_table(WL, {300: y}, LINES, windows, exclude=on_b)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        areas_table(WL, {300: y}, LINES, windows, exclude=[(395.0, 396.0)])   # away from lines


def test_ratio_tables_nonpositive_reference_gives_nan():
    for bad in (0.0, -1.0):
        with pytest.warns(RatioWarning, match="zero, negative or missing for \\['L'\\]"):
            r = ratio_tables(_areas(bad), ref=300)
        assert np.isnan(r['R_direct'].loc['L', 400]) and np.isnan(r['R_rel'].loc['L', 400])
        assert r['R_direct'].loc[TOTAL, 400] == pytest.approx(1.2)


def test_edge_pts_default_reduces_reference_window_bias():
    '''Windows found on a noisy reference spectrum must not bias ratios (edge_pts=1 gave ~2% low).'''
    rng = np.random.default_rng(1)
    y = synth_spectrum({'B 400.00': 100.0})
    r = [ratio_tables(analyze_sample(WL, {300: y + rng.normal(0, 5, len(y)), 400: y + rng.normal(0, 5, len(y))},
                                     {'B 400.00': 400.0}), ref=300)['R_direct'].loc['B 400.00', 400]
         for _ in range(100)]
    assert EDGE_PTS == 2
    assert np.mean(r) == pytest.approx(1.0, abs=0.01)


def test_windows_come_from_reference_temperature(temperature_series):
    w300 = line_windows(WL, temperature_series[300], LINES)
    areas = analyze_sample(WL, temperature_series, LINES, ref=300)
    for label, (i0, i1) in w300.items():
        expected = integrate_local_peak(WL, temperature_series[400], LINES[label], bounds=(i0, i1))['area']
        assert areas.loc[label, 400] == pytest.approx(expected)


def test_areas_table_one_column_per_location(tdms_file):
    f = LIBSFile.from_tdms(tdms_file[0])
    windows = line_windows(WL, tdms_file[1], LINES)
    table = areas_table(WL, location_spectra(f), LINES, windows)
    assert list(table.columns) == [1, 2, 3]
    assert list(table.index) == [*LINES, TOTAL, TOTAL_EXCL]
    # columns were scaled 0.99 / 1.00 / 1.01
    np.testing.assert_allclose(table.loc['B 400.00'] / table.loc['B 400.00', 2],
                               [0.99, 1.0, 1.01], rtol=1e-9)


def test_detect_peaks_threshold_and_exclusion():
    y = synth_spectrum({'A 390.00': 200.0, 'B 400.00': 100.0, 'C 410.00': 30.0})
    p = detect_peaks(WL, y)                                    # 1 x median: the two clear lines
    assert list(p.columns) == ['wavelength_nm', 'intensity', 'prominence', 'pixel', 'excluded']
    assert p.wavelength_nm.tolist() == pytest.approx([390.0, 400.0])
    assert (WL[p.pixel] == p.wavelength_nm).all() and not p.excluded.any()
    weak = detect_peaks(WL, y, min_prominence=0.25)            # also the weak line on the continuum
    assert weak.wavelength_nm.tolist() == pytest.approx([390.0, 400.0, 410.0])
    assert weak.prominence.is_monotonic_decreasing             # areas 200 > 100 > 30
    flagged = detect_peaks(WL, y, exclude=[(399.5, 400.5)])
    assert flagged.excluded.tolist() == [False, True]          # kept, but marked
    with pytest.raises(ValueError, match='median'):
        detect_peaks(WL, y - np.median(y) - 1)                 # e.g. a corrected spectrum
