'''
NIST database: builder (run offline on real Handbook pages in tests/fixtures/nist),
read-only queries, paths and CLI.
'''

import shutil
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from libs_viewer import cli
from libs_viewer.core import paths
from libs_viewer.libs import select_lines
from libs_viewer.nist import build, query
from conftest import gaussian

def test_build_row_counts(nist_db):
    info = query.database_info(nist_db)
    assert info['exists'] and info['elements'] == 2
    assert info['rows'] == {'atomic_data': 4, 'strong_lines': 447, 'persistent_lines': 62,
                            'energy_levels': 358, 'bibliographic_refs': 13}


def _row(db, sql, *params):
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        return dict(con.execute(sql, params).fetchone())


def test_table1_atomic_data(nist_db):
    ce1 = _row(nist_db, "SELECT * FROM atomic_data WHERE species = 'Ce I'")
    assert ce1['atomic_number'] == 58 and ce1['atomic_weight'] == pytest.approx(140.115)
    assert ce1['ground_j'] == '4' and ce1['ground_g'] == 9
    assert ce1['ionization_energy_eV'] == pytest.approx(5.5387)
    assert ce1['ionization_energy_cm1'] == pytest.approx(44672.0)


def test_table3_4_persistent_line_energies_match_wavelength(nist_db):
    r = _row(nist_db, "SELECT * FROM persistent_lines WHERE species = 'Ce II' AND wavelength_A = 3942.157")
    assert (r['j_i'], r['j_k'], r['term_k']) == ('7/2', '5/2', '4G')
    assert r['aki_10_8_s1'] == pytest.approx(0.43)
    assert r['medium'] == 'air'
    # Ek - Ei = 1 / lambda_vac (lambda in air above 2000 A)
    lam_vac = r['wavelength_A'] * build._air_index(r['wavelength_A'])
    assert r['ek_cm1'] - r['ei_cm1'] == pytest.approx(1e8 / lam_vac, abs=0.5)


def test_jj_coupled_term_is_split_from_j(nist_db):
    r = _row(nist_db, "SELECT * FROM persistent_lines WHERE species = 'Gd II' AND wavelength_A = 4049.86")
    assert r['term_k'].startswith('(6,1/2)') and r['j_k'] == '13/2'


def test_table5_6_ionization_limit(nist_db):
    r = _row(nist_db, "SELECT * FROM energy_levels WHERE species = 'Ce II' AND is_ionization_limit = 1")
    assert r['term'] == 'Limit' and r['energy_cm1'] == pytest.approx(87500.0)


def test_strong_lines_query_in_nm(nist_db):
    df = query.strong_lines('Ce', db_path=nist_db)
    top = df.iloc[0]
    assert top.species in ('Ce I', 'Ce II') and top.intensity == 1000
    assert df.wavelength_nm.between(186, 1050).all()
    assert query.strong_lines('Ce', species=('II',), wl_range=(418, 419), db_path=nist_db) \
        .wavelength_nm.tolist() == pytest.approx([418.6596])


def test_connection_is_read_only(nist_db):
    con = query.connect(nist_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM strong_lines")
    finally:
        con.close()


def test_missing_database(tmp_path):
    with pytest.raises(query.NISTDatabaseMissing, match='libs-viewer nist build'):
        query.strong_lines('Ce', db_path=tmp_path / 'none.sqlite')
    assert query.database_info(tmp_path / 'none.sqlite') == {'path': str(tmp_path / 'none.sqlite'),
                                                             'exists': False}


def test_air_vacuum_conversion():
    assert build.vacuum_to_air(1500.0) is None
    air = build.vacuum_to_air(5000.0)
    assert 4998.5 < air < 4998.7                       # n ~ 1.00028 in the visible
    assert build.resolve_medium(2500.0, None) == 'air'
    assert build.resolve_medium(1500.0, None) == 'vacuum'
    assert build.resolve_medium(2500.0, 'vacuum') == 'vacuum'


def test_select_lines_on_synthetic_spectrum(nist_db):
    wl = np.round(np.arange(390.0, 420.0, 0.02), 4)
    present = [395.2545, 394.2746, 399.9237, 418.6596]     # strong Ce II lines (air, nm)
    y = 100.0 + sum(gaussian(wl, c, 20.0, sigma=0.03) for c in present)
    table = select_lines('Ce', wl, {'s1': y}, n=3, db_path=nist_db)
    picked = sorted(table.loc[table.status == 'picked', 'wavelength_nm'])
    assert picked == pytest.approx([395.2545, 399.9237, 418.6596])
    # 394.27 is detected but < 1 nm from 395.25, which has the higher NIST intensity
    assert table.loc[np.isclose(table.wavelength_nm, 394.2746), 'status'].item() == 'too close to a picked line'
    assert (table.status == 'no peak in spectrum').any()


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))
    assert paths.nist_db_path() == tmp_path / 'nist_handbook.sqlite'
    assert paths.nist_cache_dir() == tmp_path / 'nist_cache'


def test_cli_status(nist_db, tmp_path, capsys):
    assert cli.main(['nist', 'status', '--db', str(nist_db)]) == 0
    assert 'strong_lines' in capsys.readouterr().out
    assert cli.main(['nist', 'status', '--db', str(tmp_path / 'none.sqlite')]) == 1


def test_elements_in_atomic_number_order(nist_db):
    assert query.elements(nist_db) == ['Ce', 'Gd']


def test_lines_near(nist_db):
    df = query.lines_near([394.27, 394.0], tol=0.02, db_path=nist_db)
    assert list(df.columns) == ['peak_nm', 'element', 'species', 'wavelength_nm', 'intensity', 'delta_nm']
    at = df[df.peak_nm == 394.27]
    assert at.species.tolist() == ['Ce II', 'Gd I']            # Ce II 394.2746 is closer than Gd I 394.263
    assert at.delta_nm.abs().is_monotonic_increasing and (df.delta_nm.abs() <= 0.02).all()
    assert 394.0 not in set(df.peak_nm)                         # nothing within 0.02 nm: no rows
    assert query.lines_near(394.27, tol=0.02, elements=['Gd'], db_path=nist_db).species.tolist() == ['Gd I']
    assert query.lines_near(394.27, tol=0.02, species=('II',), db_path=nist_db).species.tolist() == ['Ce II']
    assert query.lines_near([], db_path=nist_db).empty


def test_queries_return_air_wavelengths(nist_db, tmp_path):
    '''Below 200 nm the Handbook lists vacuum wavelengths; the spectrometer measures in air.'''
    db = tmp_path / 'vac.sqlite'
    shutil.copy(nist_db, db)
    with closing(sqlite3.connect(db)) as con, con:
        con.execute("INSERT INTO strong_lines (element, species, wavelength_A, medium, wavelength_air_A, "
                    "intensity) VALUES ('Ce', 'Ce II', 1930.905, 'vacuum', 1930.27, 999)")
    assert query.strong_lines('Ce', wl_range=(186, 200), db_path=db).wavelength_nm.tolist()         == pytest.approx([193.027])
    assert query.lines_near(193.03, tol=0.01, db_path=db).wavelength_nm.tolist() == pytest.approx([193.027])
    assert query.lines_near(193.09, tol=0.01, db_path=db).empty    # the vacuum value is not matched
