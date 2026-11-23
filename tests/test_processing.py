import numpy as np
import pytest

from libs_viewer.libs import (BASELINES, LIBSFile, baseline_settings, exclusion_mask,
                              global_baseline, location_spectra, register_baseline,
                              mask_ranges, per_shot_columns, per_shot_spectrum, saturated_mask,
                              saturated_pixels, saturation_summary)
from conftest import LINES, WL, continuum, synth_spectrum, write_tdms


def test_per_shot_divides_accumulated_columns(tdms_file):
    path, y = tdms_file
    f = LIBSFile.from_tdms(path)
    np.testing.assert_allclose(per_shot_columns(f)['Location 2'], y, rtol=1e-12)
    np.testing.assert_allclose(per_shot_spectrum(f), y, rtol=1e-12)


def test_per_shot_leaves_single_shot_files(tmp_path):
    y = synth_spectrum({'B 400.00': 100.0})
    f = LIBSFile.from_tdms(write_tdms(tmp_path / 'x.tdms', y, accumulate=0, shots=10))
    np.testing.assert_allclose(per_shot_columns(f)['Location 2'], y, rtol=1e-12)
    np.testing.assert_allclose(per_shot_spectrum(f), y, rtol=1e-12)


def test_per_shot_subset(tdms_file):
    f = LIBSFile.from_tdms(tdms_file[0])
    one = per_shot_spectrum(f.subset(columns=['Location 3']))
    np.testing.assert_allclose(one, tdms_file[1] * 1.01, rtol=1e-12)


def test_mixed_file_weights_every_shot_equally(tmp_path):
    '''Location 1: 5 accumulated shots (x0.99); location 2: 10 single shots (x1.00 ... x1.09).'''
    y = synth_spectrum({'B 400.00': 100.0})
    with pytest.warns(UserWarning):
        f = LIBSFile.from_tdms(write_tdms(tmp_path / 'm.tdms', y, n_cols=11, shots=5,
                                          locations=[1] + [2] * 10, accumulates=[1] + [0] * 10))
    np.testing.assert_allclose(per_shot_spectrum(f), y * (5 * 0.99 + 10.45) / 15, rtol=1e-12)
    loc = location_spectra(f)
    assert list(loc.columns) == [1, 2]
    np.testing.assert_allclose(loc[1], y * 0.99, rtol=1e-12)
    np.testing.assert_allclose(loc[2], y * 1.045, rtol=1e-12)


@pytest.mark.parametrize('method', ['snip', 'airpls'])
def test_global_baseline_recovers_continuum_away_from_lines(method):
    y = synth_spectrum({'A 390.00': 50.0, 'B 400.00': 100.0, 'C 410.00': 30.0})
    corrected, baseline = global_baseline(WL, y, method=method)
    np.testing.assert_allclose(corrected + baseline, y)
    # judge the continuum estimate away from the lines and the array ends
    far = np.ones(len(WL), bool)
    for c in LINES.values():
        far &= np.abs(WL - c) > 1.0
    far &= (WL > WL[0] + 3) & (WL < WL[-1] - 3)
    rel_err = np.abs(baseline[far] - continuum(WL)[far]) / continuum(WL)[far]
    assert np.median(rel_err) < 0.03


def test_global_baseline_rejects_unknown_method():
    with pytest.raises(ValueError, match='method'):
        global_baseline(WL, synth_spectrum({}), method='magic')


def test_exclusion_mask_forms():
    assert not exclusion_mask(WL, None).any()
    m = exclusion_mask(WL, [(390.0, 391.0), (400.0, 400.1)])
    assert m.sum() == 21 + 3
    boolean = np.zeros(len(WL), bool)
    boolean[5] = True
    out = exclusion_mask(WL, boolean)
    np.testing.assert_array_equal(out, boolean)
    out[0] = True
    assert not boolean[0]                                    # a copy: the caller's mask is untouched
    np.testing.assert_array_equal(exclusion_mask(WL, list(boolean)), boolean)   # plain list works
    with pytest.raises(ValueError, match='shape'):
        exclusion_mask(WL, boolean[:-1])
    with pytest.raises(ValueError, match='pairs'):
        exclusion_mask(WL, [390.0, 391.0])                   # flat list, not (lo, hi) pairs


def test_saturated_mask_flags_and_pads(tmp_path):
    y = synth_spectrum({})
    y[400] = 70000.0                                   # one clipped pixel (per shot)
    f = LIBSFile.from_tdms(write_tdms(tmp_path / 'sat.tdms', y))
    mask = saturated_mask([f], pad_px=2)
    assert np.flatnonzero(mask).tolist() == [398, 399, 400, 401, 402]
    clean = LIBSFile.from_tdms(write_tdms(tmp_path / 'ok.tdms', synth_spectrum({})))
    assert not saturated_mask([clean]).any()


def test_saturated_mask_partial_clipping_in_accumulated_spectra(tmp_path):
    '''Some of 10 shots clipped: the per-shot mean (55000) is below full scale but still flagged.'''
    y = synth_spectrum({})
    y[400] = 55000.0
    acc = LIBSFile.from_tdms(write_tdms(tmp_path / 'acc.tdms', y, accumulate=1))
    assert np.flatnonzero(saturated_mask([acc], pad_px=0)).tolist() == [400]
    single = LIBSFile.from_tdms(write_tdms(tmp_path / 'one.tdms', y, accumulate=0))
    assert not saturated_mask([single]).any()          # a single shot at 55000 did not clip


def test_saturated_pixels_per_spectrum(tmp_path):
    '''Location 1 single shot clipped at pixel 400; location 2 accumulated, partly clipped at 500.'''
    y = synth_spectrum({})
    y[400], y[500] = 70000.0, 55000.0
    with pytest.warns(UserWarning, match='settings differ'):
        f = LIBSFile.from_tdms(write_tdms(tmp_path / 'mix.tdms', y, n_cols=2, accumulates=[0, 1]))
    sat = saturated_pixels(f)
    assert sat.shape == f.spectra.shape and sat.index.equals(f.spectra.index)
    first, second = f.spectra.columns
    assert np.flatnonzero(sat[first]).tolist() == [400]              # 55000 in one shot: not clipped
    assert np.flatnonzero(sat[second]).tolist() == [400, 500]
    # the file-wide mask is the union of the per-spectrum flags
    np.testing.assert_array_equal(saturated_mask([f], pad_px=0), sat.to_numpy().any(axis=1))


def test_saturation_summary(tmp_path):
    y = synth_spectrum({})
    y[400] = 70000.0
    f = LIBSFile.from_tdms(write_tdms(tmp_path / 'sat.tdms', y, n_cols=3))
    s = saturation_summary(f)
    assert list(s.columns) == ['location', 'shot', 'shots_summed', 'peak', 'limit',
                               'saturated_px', 'saturated']
    assert s.index.equals(f.acquisitions.index)
    assert s['saturated'].all() and (s['saturated_px'] == 1).all()
    assert (s['limit'] == 52000.0).all()                          # accumulated spectra
    np.testing.assert_allclose(s['peak'], 70000.0, rtol=0.02)     # columns scaled 0.99-1.01
    clean = saturation_summary(LIBSFile.from_tdms(write_tdms(tmp_path / 'ok.tdms', synth_spectrum({}))))
    assert not clean['saturated'].any() and (clean['saturated_px'] == 0).all()


def test_mask_ranges_inverts_exclusion_mask():
    mask = np.zeros(len(WL), dtype=bool)
    mask[[0, 1, 2, 100, 799]] = True
    ranges = mask_ranges(WL, mask)
    assert ranges == [(WL[0], WL[2]), (WL[100], WL[100]), (WL[799], WL[799])]
    np.testing.assert_array_equal(exclusion_mask(WL, ranges), mask)
    assert mask_ranges(WL, np.zeros(len(WL), dtype=bool)) == []
    with pytest.raises(ValueError):
        mask_ranges(WL, mask[:10])


def test_baseline_settings_fill_defaults_and_reject_unknown():
    assert baseline_settings('snip') == {'max_half_window': 40}
    assert baseline_settings('snip', {'max_half_window': 60}) == {'max_half_window': 60}
    assert baseline_settings('airpls', {}) == {'lam': 1e6}
    with pytest.raises(ValueError, match=r"takes \['max_half_window'\], not \['lam'\]"):
        baseline_settings('snip', {'lam': 1e7})            # the other method's parameter
    with pytest.raises(ValueError, match='must be one of'):
        baseline_settings('magic')


def test_baseline_params_change_the_fit():
    y = synth_spectrum({'B 400.00': 100.0})
    _, narrow = global_baseline(WL, y, 'snip', max_half_window=5)
    _, default = global_baseline(WL, y, 'snip')
    _, same = global_baseline(WL, y, 'snip', max_half_window=40)
    np.testing.assert_array_equal(default, same)
    assert not np.allclose(narrow, default)       # a 5 px window cuts into the 400 nm line


def test_register_baseline_makes_a_method_usable_everywhere():
    from libs_viewer.libs import analyze_sample
    try:
        @register_baseline('flat_test', 'A constant level, for this test only.', level=50.0)
        def _flat(wl, y, level):
            return np.full(len(y), level)

        corrected, base = global_baseline(WL, np.full(len(WL), 80.0), 'flat_test')
        np.testing.assert_allclose(base, 50.0)
        np.testing.assert_allclose(corrected, 30.0)
        _, base = global_baseline(WL, np.full(len(WL), 80.0), 'flat_test', level=10.0)
        np.testing.assert_allclose(base, 10.0)
        assert BASELINES['flat_test'].defaults == {'level': 50.0}
        y = synth_spectrum({'B 400.00': 100.0})
        a = analyze_sample(WL, {300: y}, {'B': 400.0}, method='flat_test', baseline_params={'level': 0.0})
        b = analyze_sample(WL, {300: y}, {'B': 400.0}, method='flat_test', baseline_params={'level': 10.0})
        assert a.loc['B', 300] == b.loc['B', 300]                  # line areas use their own local baseline
        assert a.loc['Total', 300] > b.loc['Total', 300]           # the total depends on the baseline
        with pytest.raises(ValueError, match='already registered'):
            register_baseline('flat_test', 'again')(_flat)
    finally:
        BASELINES.pop('flat_test', None)
