import warnings

import numpy as np
import pandas as pd
import pytest
from nptdms import ChannelObject, TdmsWriter

from libs_viewer.libs import (LayoutWarning, LIBSFile, LoadLIBSData, SettingsWarning, location_spectra,
                              per_shot_columns, per_shot_spectrum, temperature_from_name)
from conftest import WL, synth_spectrum, write_tdms


def layout_warnings(path) -> tuple[LIBSFile, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f = LIBSFile.from_tdms(path)
    return f, [str(w.message) for w in caught if issubclass(w.category, LayoutWarning)]


# ------------------------------------------------------------------ location / shot checks
def test_single_shots_of_one_location_at_two_positions_warn(tmp_path):
    # names say location 1 has 4 shots, but the stage moved after the second one
    path = write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), n_cols=4, accumulate=0,
                      locations=[1, 1, 1, 1], x_positions=[0, 0, 5, 5])
    f, msgs = layout_warnings(path)
    assert len(msgs) == 1 and '4 single shots named as location 1' in msgs[0] and '2 different' in msgs[0]
    assert f.acquisitions.location.tolist() == [1, 1, 2, 2]       # taken from the positions


def test_two_locations_at_one_position_warn(tmp_path):
    path = write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), n_cols=4, accumulate=0,
                      locations=[1, 1, 2, 2], x_positions=[0, 0, 0, 0])
    f, msgs = layout_warnings(path)
    assert any('locations [1, 2] are at the same stage position' in m for m in msgs)
    assert f.n_locations == 2                                     # names are kept


def test_repeated_accumulations_at_one_location_warn(tmp_path):
    path = write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), n_cols=3, accumulate=1, locations=[1, 1, 2])
    f, msgs = layout_warnings(path)
    assert msgs == ['x.tdms: location 1 holds 2 accumulated spectra; they are treated as '
                    'repeated measurements at one spot']
    assert f.shots_per_location().to_dict() == {1: 20, 2: 10}


def test_consistent_files_raise_no_layout_warning(tmp_path):
    single = write_tdms(tmp_path / 's.tdms', synth_spectrum({}), n_cols=6, accumulate=0, locations=[1] * 3 + [2] * 3)
    mixed = write_tdms(tmp_path / 'm.tdms', synth_spectrum({}), n_cols=4, shots=5,
                       locations=[1, 2, 2, 2], accumulates=[1, 0, 0, 0])   # like Gd_Sample5_300
    assert layout_warnings(single)[1] == [] and layout_warnings(mixed)[1] == []


# ------------------------------------------------------------------ accumulate
def test_accumulate_single_shots(tmp_path):
    y = synth_spectrum({'B 400.00': 100.0})
    f = LIBSFile.from_tdms(write_tdms(tmp_path / 's.tdms', y, n_cols=6, shots=3, accumulate=0,
                                      locations=[1] * 3 + [2] * 3))
    acc = f.accumulate()
    assert list(acc.spectra.columns) == ['Location 1', 'Location 2']
    assert acc.acquisitions.shots_summed.tolist() == [3, 3] and acc.storage == 'accumulated'
    np.testing.assert_allclose(acc.spectra['Location 1'], f.spectra.iloc[:, :3].sum(axis=1))
    # same per-shot average; per-location spectra = the accumulated spectra per shot
    np.testing.assert_allclose(per_shot_spectrum(acc), per_shot_spectrum(f), rtol=1e-12)
    np.testing.assert_allclose(per_shot_columns(acc).to_numpy(), location_spectra(f).to_numpy(), rtol=1e-12)


def test_accumulate_warns_when_already_accumulated(tdms_file):
    f = LIBSFile.from_tdms(tdms_file[0])                 # 3 locations x 1 accumulated spectrum
    with pytest.warns(LayoutWarning, match='nothing to accumulate'):
        assert f.accumulate() is f


def test_accumulate_warns_on_accumulated_or_mixed_spectra_at_a_location(tmp_path):
    with pytest.warns(UserWarning):                      # SettingsWarning + LayoutWarning
        f = LIBSFile.from_tdms(write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), n_cols=3, shots=5,
                                          locations=[1, 1, 2], accumulates=[1, 0, 0]))
    with pytest.warns(LayoutWarning) as rec:
        acc = f.accumulate()
    text = ' | '.join(str(w.message) for w in rec)
    assert 'already hold accumulated spectra' in text and 'mix acquisition settings' in text
    assert acc.acquisitions.shots_summed.tolist() == [6, 1]


def test_from_tdms_splits_params_positions_spectra(tdms_file):
    path, _ = tdms_file
    f = LIBSFile.from_tdms(path)
    assert f.spectra.shape == (len(WL), 3)
    np.testing.assert_allclose(f.wavelengths, WL)
    assert f.spectra.index.name == 'wavelength_nm'
    assert f.storage == 'accumulated'
    assert f.acquisitions[['location', 'shot', 'shots_summed']].values.tolist() == [[1, 1, 10], [2, 1, 10], [3, 1, 10]]
    assert f.params['Number Of Shots'] == 10
    # repeated names are suffixed, positions are split out
    assert f.params['Gas Type (1)'] == 1 and f.params['Gas Type (2)'] == 2
    assert 'X Position' not in f.params
    assert list(f.positions['X Position']) == [0.0, 10.0, 20.0]


def test_param_table_returns_dataframe(tdms_file):
    table = LIBSFile.from_tdms(tdms_file[0]).param_table()
    assert isinstance(table, pd.DataFrame)
    assert table.loc['Energy', 'Value'] == '80'          # :g formatting, no trailing .0
    assert table.loc['Locations', 'Value'] == '3'
    assert table.loc['Stored as', 'Value'] == 'accumulated'


def test_differing_parameters_warn(tmp_path):
    path = write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), energies=[80, 80, 90])
    with pytest.warns(SettingsWarning, match='settings differ'):
        f = LIBSFile.from_tdms(path)
    assert f.params['Energy'] == 80                      # params shows the first spectrum
    # every spectrum's values are kept, so the odd one can be found and dropped
    assert list(f.varying_params().index) == ['Energy']
    assert list(f.acquisitions.index[f.acquisitions['Energy'] == 80]) == ['Location 1', 'Location 2']


def test_single_shots_grouped_by_location_from_channel_names(tmp_path):
    '''Accumulate = 0: one stored spectrum per shot, 'Grid Location # L-S' names.'''
    path = write_tdms(tmp_path / 's.tdms', synth_spectrum({}), n_cols=6, shots=3, accumulate=0,
                      locations=[1, 1, 1, 2, 2, 2])
    f = LIBSFile.from_tdms(path)
    assert f.spectra.columns[3] == 'Grid Location # 2-1'
    a = f.acquisitions
    assert a.location.tolist() == [1, 1, 1, 2, 2, 2] and a.shot.tolist() == [1, 2, 3, 1, 2, 3]
    assert (a.shots_summed == 1).all()
    assert f.n_locations == 2 and f.storage == 'single shots'
    assert f.describe() == '2 locations x 3 shots (single shots)'


def test_mixed_accumulated_and_single_shots(tmp_path):
    '''Like Gd_Sample5_300: location 1 accumulated (5 shots), location 2 as 10 single shots.'''
    path = write_tdms(tmp_path / 'm.tdms', synth_spectrum({}), n_cols=11, shots=5,
                      locations=[1] + [2] * 10, accumulates=[1] + [0] * 10)
    with pytest.warns(UserWarning, match='Accumulate'):
        f = LIBSFile.from_tdms(path)
    assert f.storage == 'mixed'
    assert f.shots_per_location().to_dict() == {1: 5, 2: 10}
    assert f.acquisitions.shots_summed.tolist() == [5] + [1] * 10


def test_locations_from_positions_when_names_have_no_numbers(tdms_file):
    f = LIBSFile.from_tdms(tdms_file[0])                 # 'Location 1..3', 3 stage positions
    assert f.acquisitions.location.tolist() == [1, 2, 3]


def test_subset(tmp_path):
    path = write_tdms(tmp_path / 's.tdms', synth_spectrum({}), n_cols=4, accumulate=0, locations=[1, 1, 2, 2])
    f = LIBSFile.from_tdms(path)
    two = f.subset(locations=[2])
    assert list(two.spectra.columns) == ['Grid Location # 2-1', 'Grid Location # 2-2']
    assert list(two.acquisitions.index) == list(two.spectra.columns)
    assert f.subset(columns=['Grid Location # 1-2']).spectra.shape[1] == 1
    with pytest.raises(ValueError, match='unknown'):
        f.subset(columns=['nope'])
    with pytest.raises(ValueError, match='unknown location'):
        f.subset(locations=[9])
    with pytest.raises(ValueError, match='keeps no spectra'):
        f.subset(columns=['Grid Location # 1-1'], locations=[2])


def test_varying_params_empty_when_consistent(tdms_file):
    assert LIBSFile.from_tdms(tdms_file[0]).varying_params().empty


def test_overview_one_row_per_file(tmp_path):
    y = synth_spectrum({})
    write_tdms(tmp_path / 'a.tdms', y, shots=5)
    write_tdms(tmp_path / 'b.tdms', y, accumulate=0, energies=[80, 80, 90])
    with pytest.warns(UserWarning):
        table = LoadLIBSData(tmp_path).overview()
    assert list(table.index) == ['a.tdms', 'b.tdms']
    assert table.loc['a.tdms', 'shots per location'] == '5' and table.loc['a.tdms', 'stored as'] == 'accumulated'
    assert table.loc['b.tdms', 'stored as'] == 'single shots'
    assert table.loc['a.tdms', 'locations'] == 3 and table.loc['a.tdms', 'spectra'] == 3
    assert table.loc['a.tdms', 'gate delay (us)'] == 3.5
    assert table.loc['a.tdms', 'mixed settings'] == '' and table.loc['b.tdms', 'mixed settings'] == 'Energy'


def test_not_a_libs_file_raises(tmp_path):
    path = tmp_path / 'bad.tdms'
    with TdmsWriter(str(path)) as w:
        w.write_segment([ChannelObject('G', 'only', np.arange(5.0))])
    with pytest.raises(ValueError):
        LIBSFile.from_tdms(path)


def test_decreasing_wavelengths_raise(tmp_path):
    path = write_tdms(tmp_path / 'rev.tdms', synth_spectrum({})[::-1], wl=WL[::-1])
    with pytest.raises(ValueError, match='wavelength axis'):
        LIBSFile.from_tdms(path)


def test_load_folder_file_and_recursive(tmp_path):
    y = synth_spectrum({})
    write_tdms(tmp_path / 'a_300C.tdms', y)
    write_tdms(tmp_path / 'b_350C.TDMS', y)                  # extension matched case-insensitively
    (tmp_path / 'sub').mkdir()
    write_tdms(tmp_path / 'sub' / 'c_400C.tdms', y)
    (tmp_path / 'notes.txt').write_text('not data')

    flat = LoadLIBSData(tmp_path)
    assert sorted(flat.keys()) == ['a_300C.tdms', 'b_350C.TDMS']
    deep = LoadLIBSData(tmp_path, recursive=True)
    assert 'sub/c_400C.tdms' in deep.keys() and len(deep) == 3
    assert isinstance(deep[0], LIBSFile)                    # positional access

    single = LoadLIBSData(tmp_path / 'a_300C.tdms')
    assert list(single) == ['a_300C.tdms']


def test_load_skips_unreadable_with_warning(tmp_path):
    write_tdms(tmp_path / 'good.tdms', synth_spectrum({}))
    (tmp_path / 'broken.tdms').write_bytes(b'not a tdms file')
    with pytest.warns(UserWarning, match='Could not read'):
        loaded = LoadLIBSData(tmp_path)
    assert list(loaded) == ['good.tdms']


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LoadLIBSData(tmp_path / 'nope')


# ------------------------------------------------------------------ manifests
@pytest.fixture
def labelled(tmp_path):
    """Two .tdms files in a sub-folder, an unlisted third one, and the folder they sit in."""
    y = synth_spectrum({})
    write_tdms(tmp_path / 's1' / 'a.tdms', y)
    write_tdms(tmp_path / 's1' / 'b.tdms', y)
    write_tdms(tmp_path / 'unlisted.tdms', y)
    return tmp_path


@pytest.mark.parametrize('suffix', ['.csv', '.xlsx'])
def test_load_manifest_only_listed_files_with_labels(labelled, suffix):
    # headers in any case / spacing; only 'file' is required; extra label columns kept
    table = pd.DataFrame({' File ': ['s1/b.tdms', 's1/a.tdms'], 'Sample': ['one', 'one'],
                          'operator': ['JS', 'AB']})
    path = labelled / f'manifest{suffix}'
    table.to_csv(path, index=False) if suffix == '.csv' else table.to_excel(path, index=False)

    data = LoadLIBSData(manifest=path)                     # folder defaults to the manifest's
    assert list(data) == ['s1/b.tdms', 's1/a.tdms']        # manifest order, unlisted file left out
    assert data.labels.loc['s1/a.tdms', 'operator'] == 'AB'
    overview = data.overview()
    assert list(overview.columns[:2]) == ['sample', 'operator']
    assert overview.loc['s1/b.tdms', 'locations'] == 3


def test_load_manifest_dataframe_needs_folder(labelled):
    table = pd.DataFrame({'file': ['s1/a.tdms'], 'use_columns': ['Location 1; Location 2']})
    with pytest.raises(ValueError, match='folder'):
        LoadLIBSData(manifest=table)
    data = LoadLIBSData(labelled, manifest=table)
    assert data.labels.loc['s1/a.tdms', 'use_columns'] == ['Location 1', 'Location 2']
    assert data.overview().loc['s1/a.tdms', 'use_columns'] == 'Location 1; Location 2'


@pytest.mark.parametrize('table, error, match', [
    ({'path': ['s1/a.tdms']}, ValueError, "'file' column"),
    ({'file': ['s1/a.tdms', 's1/a.tdms']}, ValueError, 'more than once'),
    ({'file': ['s1/a.tdms', None], 'sample': ['x', 'y']}, ValueError, r'without a file: \[3\]'),
    ({'file': ['notes.txt']}, ValueError, 'not .tdms'),
    ({'file': ['s1/a.tdms', 's1/missing.tdms']}, FileNotFoundError, 'missing.tdms'),
])
def test_load_manifest_rejects_bad_tables(labelled, table, error, match):
    with pytest.raises(error, match=match):
        LoadLIBSData(labelled, manifest=pd.DataFrame(table))


def test_load_manifest_unreadable_file_raises_listing_all(labelled):
    (labelled / 'bad1.tdms').write_bytes(b'not a tdms file')
    (labelled / 'bad2.tdms').write_bytes(b'not a tdms file')
    table = pd.DataFrame({'file': ['s1/a.tdms', 'bad1.tdms', 'bad2.tdms']})
    with pytest.raises(ValueError, match='2 manifest file') as e:
        LoadLIBSData(labelled, manifest=table)
    assert 'bad1.tdms' in str(e.value) and 'bad2.tdms' in str(e.value)


def test_non_tdms_path_raises_and_points_to_manifest(labelled):
    csv = labelled / 'manifest.csv'
    pd.DataFrame({'file': ['s1/a.tdms']}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='manifest='):
        LoadLIBSData(csv)
    with pytest.raises(ValueError):
        LoadLIBSData()


@pytest.mark.parametrize('name, temp', [
    ('Ce_Sample2_Temp_300C.tdms', 300),
    ('Gd_Sample5_350_3.5delay.tdms', 350),
    ('folder/Ce_Sample1_Temp_400C.tdms', 400),
])
def test_temperature_from_name(name, temp):
    assert temperature_from_name(name) == temp


def test_temperature_from_name_missing():
    with pytest.raises(ValueError):
        temperature_from_name('Ce_Sample2.tdms')
