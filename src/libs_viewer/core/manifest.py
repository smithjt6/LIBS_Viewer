'''
Draft manifests: build the file/label table from files that are already loaded, so nobody has to
type paths. Instrument-neutral (LIBS studies today, ICP-MS later): which label columns a study
needs is the caller's business, e.g. `libs.plan.PLAN_COLUMNS`.

    draft = draft_manifest(data)                      # one row per loaded file, empty labels
    draft = draft_manifest(data, prefill=guess)       # labels guessed elsewhere, e.g. from names
    problems = missing_labels(draft, ['element', 'sample', 'temperature'])
    run_study(data.root, manifest=clean_manifest(draft), ...)   # a DataFrame is a manifest
    manifest_csv(draft, 'manifest.csv')               # keep it for next time

The drafted `file` column holds the keys of the loaded data (paths relative to its root), so the
table reads back with `read_manifest(table, folder=root)` or from the saved CSV next to the data.
'''

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_LABELS = ('element', 'sample', 'temperature')


def draft_manifest(files: Iterable[str], labels: Iterable[str] = DEFAULT_LABELS,
                   prefill: pd.DataFrame | None = None) -> pd.DataFrame:
    '''
    One row per file, a `file` column and one empty column per label.

    Args:
        files: file keys as loaded (paths relative to the data folder), e.g. a `LoadLIBSData`.
        labels: label column names to create.
        prefill: optional table indexed by file (or with a `file` column); its values fill the
            matching columns, and files or columns it lacks stay empty.

    Returns:
        DataFrame in the order of `files`; label columns are object dtype (temperature-like
        columns become numeric once `clean_manifest` runs).
    '''
    files = list(files)
    table = pd.DataFrame({'file': files})
    for label in labels:
        table[label] = pd.Series([None] * len(files), dtype=object)
    if prefill is not None and len(prefill):
        guess = prefill.set_index('file') if 'file' in prefill.columns else prefill
        for label in labels:
            if label in guess.columns:
                table[label] = [guess[label].get(f) for f in files]
    return table


def clean_manifest(table: pd.DataFrame, numeric: Iterable[str] = ('temperature',)) -> pd.DataFrame:
    '''
    A copy of an edited draft that reads as a manifest: text labels stripped, blank text turned
    into missing values, and the `numeric` columns converted to numbers (unparseable text
    becomes missing, so `missing_labels` reports it).
    '''
    out = table.copy()
    numeric = [c for c in numeric if c in out.columns]
    for col in out.columns:
        if col == 'file' or col in numeric:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            text = out[col].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            out[col] = text.where(text.map(lambda v: not (isinstance(v, str) and v == '')), None)
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def labelled_rows(table: pd.DataFrame, required: Iterable[str] = DEFAULT_LABELS) -> pd.DataFrame:
    '''
    The cleaned rows that have at least one label filled in. A file left completely blank is
    taken to be "not part of this study", so a folder can hold extra files.
    '''
    table = clean_manifest(table)
    cols = [c for c in required if c in table.columns]
    return table[table[cols].notna().any(axis=1)].reset_index(drop=True) if cols else table.iloc[0:0]


def missing_labels(table: pd.DataFrame, required: Iterable[str] = DEFAULT_LABELS) -> dict[str, list[str]]:
    '''
    {column: files with no value there} for every required column that has gaps (empty =
    complete). Files with no label at all are ignored (see `labelled_rows`); if no file is
    labelled, every required column is reported with an empty list.
    '''
    table = labelled_rows(table, required)
    if table.empty:
        return {col: [] for col in required}
    gaps = {}
    for col in required:
        if col not in table.columns:
            gaps[col] = table['file'].tolist()
        elif table[col].isna().any():
            gaps[col] = table.loc[table[col].isna(), 'file'].tolist()
    return gaps


def manifest_csv(table: pd.DataFrame, path: str | Path | None = None) -> bytes:
    '''
    The manifest as CSV: returned as UTF-8 bytes (e.g. for a download button) and, if `path` is
    given, written there. A `use_columns` column of lists is written ';'-separated, as
    `read_manifest` reads it. Never point `path` into a folder of raw data files.
    '''
    out = table.copy()
    if 'use_columns' in out.columns:
        out['use_columns'] = [';'.join(v) if isinstance(v, (list, tuple)) and v else ''
                              for v in out['use_columns']]
    data = out.to_csv(index=False).encode('utf-8')
    if path is not None:
        Path(path).write_bytes(data)
    return data
