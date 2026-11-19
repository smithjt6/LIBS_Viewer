import numpy as np
import pytest

from libs_viewer.libs import (LIBSFile, exclusion_mask, global_baseline, location_spectra,
                              per_shot_columns, per_shot_spectrum, saturated_mask)
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
    assert exclusion_mask(WL, boolean) is boolean


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
