'''
Study plan (pattern / manifest) and the one-call run_study, on synthetic Ce spectra with real
Ce II lines and the offline NIST fixture database.
'''

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from libs_viewer import cli  # noqa: E402
from libs_viewer.libs import TOTAL, run_study, study_plan  # noqa: E402
from libs_viewer.libs.plotting import plot_baseline, plot_study  # noqa: E402
from conftest import gaussian, write_tdms  # noqa: E402

WL = np.round(np.arange(390.0, 420.0, 0.02), 4)
CE_LINES = [395.2545, 399.9237, 418.6596]          # strong Ce II lines (air, nm)
PATTERN = '{element}/{sample}/*{temperature}*'


def ce_spectrum(scale: float) -> np.ndarray:
    '''Continuum + the three Ce II lines, each with area 20 * scale.'''
    return 100.0 + sum(gaussian(WL, c, 20.0 * scale, sigma=0.03) for c in CE_LINES)


@pytest.fixture
def study(tmp_path):
    '''Two Ce samples at 300/350/400 C; in "s1" the lines double and triple with temperature.'''
    root = tmp_path / 'study'
    for sample, scales in {'s1': (1, 2, 3), 's2': (1, 1, 1)}.items():
        for t, k in zip((300, 350, 400), scales):
            write_tdms(root / 'Ce' / sample / f'Ce_{sample}_{t}C.tdms', ce_spectrum(k), wl=WL)
    return root


# ------------------------------------------------------------------ study_plan
def test_plan_from_pattern(study):
    plan = study_plan(study, pattern=PATTERN)
    assert len(plan) == 6
    assert plan.loc[0].to_dict() == {'file': 'Ce/s1/Ce_s1_300C.tdms', 'element': 'Ce',
                                     'sample': 'Ce s1', 'temperature': 300, 'use_columns': None}


def test_plan_needs_exactly_one_source(study):
    with pytest.raises(ValueError, match='exactly one'):
        study_plan(study)
    with pytest.raises(ValueError, match='exactly one'):
        study_plan(study, pattern=PATTERN, manifest='m.csv')


def test_plan_rejects_unmatched_and_duplicate_files(study):
    write_tdms(study / 'Ce' / 's1' / 'notes.tdms', ce_spectrum(1), wl=WL)
    with pytest.raises(ValueError, match='do not match the pattern'):
        study_plan(study, pattern=PATTERN)
    (study / 'Ce' / 's1' / 'notes.tdms').unlink()
    write_tdms(study / 'Ce' / 's1' / 'Ce_s1_300C_repeat.tdms', ce_spectrum(1), wl=WL)
    with pytest.raises(ValueError, match='same sample and temperature'):
        study_plan(study, pattern=PATTERN)


def test_plan_temperatures_list_handles_glued_digits(tmp_path):
    write_tdms(tmp_path / 'Ce' / 'a' / 'Ce1400.tdms', ce_spectrum(1), wl=WL)
    write_tdms(tmp_path / 'Ce' / 'a' / 'Ce1_300C.tdms', ce_spectrum(1), wl=WL)
    with pytest.raises(ValueError, match='temperatures='):
        study_plan(tmp_path, pattern=PATTERN)
    plan = study_plan(tmp_path, pattern=PATTERN, temperatures=[300, 350, 400])
    assert plan.temperature.tolist() == [300, 400]


def test_plan_rejects_two_temperatures_in_one_name(tmp_path):
    write_tdms(tmp_path / 'Ce' / 'a' / 'Ce_300_350.tdms', ce_spectrum(1), wl=WL)
    with pytest.raises(ValueError, match='more than one temperature'):
        study_plan(tmp_path, pattern=PATTERN, temperatures=[300, 350])


def test_plan_from_manifest_csv(study, tmp_path):
    csv = tmp_path / 'manifest.csv'
    pd.DataFrame({'file': ['Ce/s1/Ce_s1_300C.tdms', 'Ce/s1/Ce_s1_350C.tdms'],
                  'element': 'Ce', 'sample': 'one', 'temperature': [300, 350],
                  'use_columns': ['Location 1;Location 2', '']}).to_csv(csv, index=False)
    plan = study_plan(study, manifest=csv)
    assert plan.use_columns.tolist() == [['Location 1', 'Location 2'], None]
    with pytest.raises(FileNotFoundError):
        study_plan(study, manifest=pd.DataFrame({'file': ['x.tdms'], 'element': ['Ce'],
                                                 'sample': ['a'], 'temperature': [300]}))


# ------------------------------------------------------------------ run_study
def test_run_study_end_to_end(study, nist_db):
    r = run_study(study, elements=['Ce'], pattern=PATTERN, n_lines=3, db_path=nist_db)
    assert r.samples == ['Ce s1', 'Ce s2']
    assert sorted(r.lines['Ce s1'].values()) == pytest.approx(CE_LINES)
    # s1 lines grow 2x and 3x; s2 stays flat
    d = r.r_direct.loc['Ce s1']
    for label in r.lines['Ce s1']:
        assert d.loc[label, 350] == pytest.approx(2.0, rel=0.03)
        assert d.loc[label, 400] == pytest.approx(3.0, rel=0.03)
    flat = r.r_direct.loc['Ce s2'].drop(index=[TOTAL])
    np.testing.assert_allclose(flat[[350, 400]].to_numpy(dtype=float), 1.0, rtol=0.02)
    assert r.notes == []
    # accumulated files: 3 locations, no single shots -> location CV only
    assert len(r.scatter) == 6 and (r.scatter.locations == 3).all()
    assert r.scatter['location CV, Total (%)'].notna().all() and r.scatter['shot CV, Total (%)'].isna().all()
    assert r.settings['baseline'] == 'snip' and r.settings['pattern'] == PATTERN


def test_run_study_drops_spectra_with_other_settings(study, nist_db):
    # replace one file with a mixed one: location 3 was acquired with a different energy
    write_tdms(study / 'Ce' / 's2' / 'Ce_s2_300C.tdms', ce_spectrum(1), wl=WL, energies=[80, 80, 10])
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    row = r.plan.set_index('file').loc['Ce/s2/Ce_s2_300C.tdms']
    assert row.use_columns == ['Location 1', 'Location 2']
    assert any('used 2 locations x 10 shots' in n and 'dropped 1 location' in n for n in r.notes)
    assert r.scatter.empty


def test_run_study_single_shots_give_shot_scatter(study, nist_db):
    # s2 at 300 C stored as single shots: 2 locations x 4 shots
    write_tdms(study / 'Ce' / 's2' / 'Ce_s2_300C.tdms', ce_spectrum(1), wl=WL, n_cols=8, shots=4,
               accumulate=0, locations=[1] * 4 + [2] * 4)
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db)
    row = r.scatter.loc[('Ce s2', 300)]
    assert row.locations == 2 and row.spectra == 8
    assert np.isfinite(row['shot CV, Total (%)']) and np.isfinite(row['location CV, Total (%)'])
    assert any(n.startswith('Ce s2: acquired differently') for n in r.notes)
    # the per-shot average is unaffected by how the shots were stored
    flat = r.r_direct.loc['Ce s2'].drop(index=[TOTAL])
    np.testing.assert_allclose(flat[[350, 400]].to_numpy(dtype=float), 1.0, rtol=0.03)


def test_run_study_reports_layout_warnings_in_notes(study, nist_db):
    write_tdms(study / 'Ce' / 's2' / 'Ce_s2_300C.tdms', ce_spectrum(1), wl=WL, locations=[1, 1, 2])
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    assert any('location 1 holds 2 accumulated spectra' in n for n in r.notes)


def test_run_study_errors(study, nist_db, tmp_path):
    with pytest.raises(ValueError, match='no files for element'):
        run_study(study, elements=['Gd'], pattern=PATTERN, db_path=nist_db)
    with pytest.raises(ValueError, match='reference temperature'):
        run_study(study, pattern=PATTERN, ref=250, db_path=nist_db)
    with pytest.raises(ValueError, match='baseline'):
        run_study(study, pattern=PATTERN, baseline='magic', db_path=nist_db)
    with pytest.raises(FileNotFoundError, match='nist build'):
        run_study(study, pattern=PATTERN, db_path=tmp_path / 'none.sqlite')


def test_save_writes_tables_and_settings(study, nist_db, tmp_path):
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    written = r.save(tmp_path / 'out')
    names = {p.name for p in written}
    assert {'areas.csv', 'R_rel.csv', 'R_direct.csv', 'plan.csv', 'lines_Ce.csv',
            'settings.json', 'summary.txt'} <= names
    meta = json.loads((tmp_path / 'out' / 'settings.json').read_text())
    assert meta['elements'] == ['Ce'] and 'Ce s1' in meta['lines']
    back = pd.read_csv(tmp_path / 'out' / 'R_rel.csv', index_col=[0, 1])
    assert back.loc[('Ce s1', TOTAL), '300'] == 1.0
    with pytest.raises(ValueError, match='inside the data folder'):
        r.save(study / 'results')


def test_plots(study, nist_db):
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    fig = plot_study(r)
    assert [ax.get_title() for ax in fig.axes] == ['Ce s1', 'Ce s2']
    fig = plot_baseline(r.wl, r.spectra['Ce s1'][300], method='airpls', wl_range=(395, 400))
    assert len(fig.axes) == 2


def test_cli_analyze(study, nist_db, tmp_path, capsys):
    out = tmp_path / 'cli_out'
    code = cli.main(['analyze', str(study), '--pattern', PATTERN, '--lines', '3',
                     '--db', str(nist_db), '--out', str(out), '--no-scatter'])
    assert code == 0
    assert 'R_rel' in capsys.readouterr().out
    assert (out / 'R_rel.csv').is_file()
    assert cli.main(['analyze', str(study), '--pattern', '{element}/*', '--db', str(nist_db)]) == 2
