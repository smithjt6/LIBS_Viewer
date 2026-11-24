'''
Draft manifests: label loaded files in memory (what the Inspect page does) and hand the table to
run_study, with no manifest file and no typed paths.
'''

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from libs_viewer.core.manifest import (clean_manifest, draft_manifest, labelled_rows, manifest_csv,
                                       missing_labels)
from libs_viewer.libs import LoadLIBSData, guess_labels, run_study, study_plan
from conftest import gaussian, write_tdms

WL = np.round(np.arange(390.0, 420.0, 0.02), 4)
PATTERN = '{element}/{sample}/*{temperature}*'


@pytest.fixture
def study(tmp_path):
    root = tmp_path / 'study'
    for sample, scales in {'s1': (1, 2, 3), 's2': (1, 1, 1)}.items():
        for t, k in zip((300, 350, 400), scales):
            spectrum = 100.0 + sum(gaussian(WL, c, 20.0 * k, sigma=0.03)
                                   for c in (395.2545, 399.9237, 418.6596))
            write_tdms(root / 'Ce' / sample / f'Ce_{sample}_{t}C.tdms', spectrum, wl=WL)
    return root


def test_draft_is_one_empty_row_per_file():
    d = draft_manifest(['a/x.tdms', 'b/y.tdms'])
    assert list(d.columns) == ['file', 'element', 'sample', 'temperature']
    assert d['file'].tolist() == ['a/x.tdms', 'b/y.tdms'] and d.iloc[:, 1:].isna().all().all()
    assert set(missing_labels(d)) == {'element', 'sample', 'temperature'}


def test_clean_and_missing_labels():
    d = pd.DataFrame({'file': ['a', 'b', 'c'], 'element': [' Ce ', '', None],
                      'sample': ['s', 's', None], 'temperature': ['300', 'x', None]})
    c = clean_manifest(d)
    assert c['element'].tolist()[0] == 'Ce' and c['element'].isna().tolist() == [False, True, True]
    assert c['temperature'].tolist()[0] == 300 and c['temperature'].isna().tolist() == [False, True, True]
    assert labelled_rows(d)['file'].tolist() == ['a', 'b']          # 'c' is blank: not in the study
    assert missing_labels(d) == {'element': ['b'], 'temperature': ['b']}
    assert missing_labels(d.iloc[[2]]) == {'element': [], 'sample': [], 'temperature': []}


def test_guess_labels_is_lenient(study):
    files = ['Ce/s1/Ce_s1_300C.tdms', 'notes.tdms']
    g = guess_labels(files, PATTERN)
    assert g.loc[files[0]].tolist() == ['Ce', 'Ce s1', 300]
    assert g.loc['notes.tdms'].isna().all()                         # study_plan would raise here
    with pytest.raises(ValueError, match='must contain'):
        guess_labels(files, '{element}/*')


def test_draft_handed_to_run_study_matches_pattern_study(study, nist_db):
    data = LoadLIBSData(study, recursive=True)
    assert data.root == study
    draft = clean_manifest(draft_manifest(data, prefill=guess_labels(list(data), PATTERN)))
    assert not missing_labels(draft)
    via_draft = run_study(data.root, manifest=draft, n_lines=3, db_path=nist_db, scatter=False)
    via_pattern = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    pd.testing.assert_frame_equal(via_draft.r_rel, via_pattern.r_rel)


def test_unlabelled_files_stay_out_of_the_study(study, nist_db):
    write_tdms(study / 'extra.tdms', 100.0 + np.zeros_like(WL), wl=WL)
    data = LoadLIBSData(study, recursive=True)
    guess = guess_labels(list(data), PATTERN)
    draft = clean_manifest(draft_manifest(data, prefill=guess))
    assert len(draft) == 7 and not missing_labels(draft)
    plan = study_plan(study, manifest=labelled_rows(draft))
    assert len(plan) == 6 and 'extra.tdms' not in set(plan.file)


def test_single_file_root_is_its_folder(study):
    data = LoadLIBSData(study / 'Ce' / 's1' / 'Ce_s1_300C.tdms')
    assert data.root == study / 'Ce' / 's1' and list(data) == ['Ce_s1_300C.tdms']


def test_manifest_csv_round_trips(study, tmp_path):
    data = LoadLIBSData(study, recursive=True)
    draft = clean_manifest(draft_manifest(data, prefill=guess_labels(list(data), PATTERN)))
    out = tmp_path / 'manifest.csv'
    assert manifest_csv(draft, out) == out.read_bytes()
    assert len(study_plan(study, manifest=out)) == 6 and (study / 'manifest.csv').exists() is False


def test_results_zip_holds_a_reusable_manifest(study, nist_db, tmp_path):
    r = run_study(study, pattern=PATTERN, n_lines=3, db_path=nist_db, scatter=False)
    with zipfile.ZipFile(io.BytesIO(r.to_zip())) as z:
        assert {'plan.csv', 'settings.json', 'R_rel.csv'} <= set(z.namelist())
        z.extract('plan.csv', tmp_path)
    again = study_plan(study, manifest=tmp_path / 'plan.csv')
    pd.testing.assert_frame_equal(again, r.plan, check_dtype=False)
