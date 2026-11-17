import numpy as np
import pandas as pd
import pytest
from nptdms import ChannelObject, TdmsWriter

from libs_viewer.libs import LIBSFile, LoadLIBSData, temperature_from_name
from conftest import WL, synth_spectrum, write_tdms


def test_from_tdms_splits_params_positions_spectra(tdms_file):
    path, _ = tdms_file
    f = LIBSFile.from_tdms(path)
    assert f.spectra.shape == (len(WL), 3)
    np.testing.assert_allclose(f.wavelengths, WL)
    assert f.spectra.index.name == 'wavelength_nm'
    assert f.accumulated
    assert f.params['Number Of Shots'] == 10
    # repeated names are suffixed, positions are split out
    assert f.params['Gas Type (1)'] == 1 and f.params['Gas Type (2)'] == 2
    assert 'X Position' not in f.params
    assert list(f.positions['X Position']) == [0.0, 10.0, 20.0]


def test_param_table_returns_dataframe(tdms_file):
    table = LIBSFile.from_tdms(tdms_file[0]).param_table()
    assert isinstance(table, pd.DataFrame)
    assert table.loc['Energy', 'Value'] == '80'          # :g formatting, no trailing .0
    assert table.loc['Grid locations', 'Value'] == '3'


def test_differing_parameters_warn(tmp_path):
    path = write_tdms(tmp_path / 'x.tdms', synth_spectrum({}), energies=[80, 80, 90])
    with pytest.warns(UserWarning, match='parameters differ'):
        f = LIBSFile.from_tdms(path)
    assert f.params['Energy'] == 80                      # first location wins
    # the per-location values are kept, so the odd column can be found and dropped
    assert list(f.varying_params().index) == ['Energy']
    assert list(f.column_params.columns[f.column_params.loc['Energy'] == 80]) == ['Location 1', 'Location 2']


def test_varying_params_empty_when_consistent(tdms_file):
    assert LIBSFile.from_tdms(tdms_file[0]).varying_params().empty


def test_overview_one_row_per_file(tmp_path):
    y = synth_spectrum({})
    write_tdms(tmp_path / 'a.tdms', y, shots=5)
    write_tdms(tmp_path / 'b.tdms', y, accumulate=0, energies=[80, 80, 90])
    with pytest.warns(UserWarning):
        table = LoadLIBSData(tmp_path).overview()
    assert list(table.index) == ['a.tdms', 'b.tdms']
    assert table.loc['a.tdms', 'shots'] == 5 and table.loc['a.tdms', 'accumulated']
    assert not table.loc['b.tdms', 'accumulated']
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
