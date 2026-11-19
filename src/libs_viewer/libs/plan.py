'''
Which file is which: map every file of a study to element / sample / temperature.

`study_plan` takes either a manifest (CSV or DataFrame) or a file-name pattern, and raises if
neither (or both) is given, if a file does not match, or if the mapping is ambiguous. There is
no guessing from file names beyond what the pattern states.
'''

import re
from pathlib import Path

import numpy as np
import pandas as pd

PLAN_COLUMNS = ['file', 'element', 'sample', 'temperature', 'use_columns']
_PLACEHOLDERS = ('{element}', '{sample}', '{temperature}')


def study_plan(folder: str | Path, pattern: str | None = None, manifest=None,
               temperatures=None) -> pd.DataFrame:
    """
    One row per file with element, sample, temperature and the stored spectra to use.
    Exactly one of `pattern` or `manifest` must be given.

    Args:
        folder: study folder; file paths are relative to it.
        pattern: file-name template matched against every .tdms file under `folder`
            (sub-folders included, `/` between folders, extension optional). Placeholders:
            `{element}` (a chemical symbol, required), `{temperature}` (required),
            `{sample}` (optional), and `*` for any text within one folder or file name.
            Example for `Ce/1_wt%/Ce1_300C.tdms`: `'{element}/{sample}/*{temperature}*'`.
            Every .tdms file must match, or this raises.
        manifest: CSV path or DataFrame with columns file, element, sample, temperature and
            optionally use_columns (stored-spectrum names such as 'Grid Location # 1-1',
            separated by ';'; empty = choose automatically).
        temperatures: the temperatures used in the study, e.g. [300, 350, 400]. Without it,
            `{temperature}` matches a stand-alone 3-digit number; with it, only these values,
            even when glued to other digits ('Ce1400' -> 400).

    Returns:
        DataFrame with columns file, element, sample, temperature, use_columns
        (None = choose automatically), sorted by element, sample and temperature. With a
        pattern, the sample is labelled '<element> <sample>' (or just the element if the
        pattern has no {sample}).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'No such folder: {folder}')
    if (pattern is None) == (manifest is None):
        raise ValueError('give exactly one of pattern= (e.g. "{element}/{sample}/*{temperature}*") '
                         'or manifest= (CSV with file, element, sample, temperature)')
    plan = (_plan_from_pattern(folder, pattern, temperatures) if pattern is not None
            else _plan_from_manifest(folder, manifest))

    dup = plan[plan.duplicated(['sample', 'temperature'], keep=False)]
    if len(dup):
        raise ValueError('more than one file for the same sample and temperature:\n'
                         + dup[['file', 'sample', 'temperature']].to_string(index=False))
    mixed = plan.groupby('sample')['element'].nunique()
    if (mixed > 1).any():
        raise ValueError(f'sample(s) with more than one element: {list(mixed[mixed > 1].index)}')
    return plan.sort_values(['element', 'sample', 'temperature'], kind='stable').reset_index(drop=True)


def temperature_from_name(name: str) -> int:
    '''Parses the temperature from names like 'Ce_Sample2_Temp_300C.tdms' or 'Gd_Sample5_300_3.5delay.tdms'.'''
    m = re.search(r'_(\d{3})C?(?=[_.])', Path(name).name)
    if not m:
        raise ValueError(f'no temperature found in {name!r}')
    return int(m.group(1))


# ------------------------------------------------------------------ pattern
def _temperature_regex(temperatures) -> str:
    if temperatures:
        # longest first so 1000 is tried before 100
        return '|'.join(sorted({f'{t:g}' for t in temperatures}, key=len, reverse=True))
    return r'(?<!\d)\d{3}(?!\d)'          # a stand-alone 3-digit number


def _pattern_regex(pattern: str, temperatures) -> re.Pattern:
    for ph in ('{element}', '{temperature}'):
        if ph not in pattern:
            raise ValueError(f'pattern {pattern!r} must contain {ph}')
    out = []
    for part in re.split(r'(\{element\}|\{sample\}|\{temperature\}|\*)', pattern):
        if part in _PLACEHOLDERS and pattern.count(part) > 1:
            raise ValueError(f'{part} appears more than once in the pattern')
        if part == '{element}':
            out.append(r'(?P<element>[A-Z][a-z]?)')
        elif part == '{sample}':
            out.append(r'(?P<sample>[^/]+?)')
        elif part == '{temperature}':
            out.append(f'(?P<temperature>{_temperature_regex(temperatures)})')
        elif part == '*':
            out.append(r'[^/]*')
        else:
            out.append(re.escape(part))
    return re.compile(''.join(out))


def _plan_from_pattern(folder: Path, pattern: str, temperatures) -> pd.DataFrame:
    if pattern.lower().endswith('.tdms'):
        pattern = pattern[:-5]
    regex = _pattern_regex(pattern, temperatures)
    any_temp = re.compile(_temperature_regex(temperatures))
    files = sorted(f.relative_to(folder).as_posix() for f in folder.rglob('*')
                   if f.is_file() and f.suffix.lower() == '.tdms')
    if not files:
        raise ValueError(f'no .tdms files under {folder}')
    rows, unmatched = [], []
    for rel in files:
        stem = rel[:-5]                                   # matched without the extension
        m = regex.fullmatch(stem)
        if not m:
            unmatched.append(rel)
            continue
        # the folder / file name holding the temperature must not contain a second one
        start = stem.rfind('/', 0, m.start('temperature')) + 1
        end = stem.find('/', m.end('temperature'))
        segment = stem[start:end if end >= 0 else len(stem)]
        found = {float(x) for x in any_temp.findall(segment)}
        if len(found) > 1:
            raise ValueError(f'{rel}: more than one temperature in {segment!r} {sorted(found)}; '
                             'use a manifest for this study')
        t = float(m['temperature'])
        sample = m.groupdict().get('sample')
        rows.append((rel, m['element'], f"{m['element']} {sample}" if sample else m['element'],
                     int(t) if t.is_integer() else t, None))
    if unmatched:
        shown = ', '.join(unmatched[:10]) + (' ...' if len(unmatched) > 10 else '')
        hint = ('' if temperatures else
                '. If a temperature is glued to other digits (e.g. Ce1400), pass temperatures=[...]')
        raise ValueError(f'{len(unmatched)} file(s) do not match the pattern {pattern!r}: {shown}{hint}')
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


# ------------------------------------------------------------------ manifest
def _split_columns(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, str):
        return [c.strip() for c in v.split(';') if c.strip()]
    return list(v)


def _plan_from_manifest(folder: Path, manifest) -> pd.DataFrame:
    df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    missing = [c for c in PLAN_COLUMNS[:4] if c not in df.columns]
    if missing:
        raise ValueError(f'manifest is missing column(s) {missing}; needs {PLAN_COLUMNS[:4]} '
                         '(+ optional use_columns, ";"-separated)')
    df = df.reindex(columns=PLAN_COLUMNS).astype({'use_columns': object})
    df['use_columns'] = [_split_columns(v) for v in df['use_columns']]
    df['file'] = [Path(f).as_posix() for f in df['file']]
    absent = [f for f in df['file'] if not (folder / f).is_file()]
    if absent:
        raise FileNotFoundError(f'manifest file(s) not found under {folder}: {absent[:10]}')
    return df
