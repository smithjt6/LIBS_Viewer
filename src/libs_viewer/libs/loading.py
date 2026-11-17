'''
Loading Applied Spectra LIBS .tdms files.

Each file is split into parameters, wavelengths and spectra so a UI (or tool) can query them
directly. Raw files are only ever read.

`study_plan` maps every file of a study to element / sample / temperature, from either a
manifest (CSV or DataFrame) or a file-name pattern, and raises if neither is given.
'''

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from nptdms import TdmsFile

# Applied Spectra file layout: the first N_PARAMS rows of every channel are the
# parameter block, followed by the spectrum (header channel = wavelengths).
# Change these if your data follows a different format.

N_PARAMS = 31
POSITION_PARAMS = ('X Position', 'Y Position', 'Z Position')  # vary per grid location

@dataclass
class LIBSFile:
    '''
    One loaded LIBS TDMS file.

    Attributes:
        params: Series of the instrument parameters shared by every grid location
            (index = parameter name; repeated names get a ' (1)', ' (2)' suffix).
        positions: DataFrame of X/Y/Z stage position, one row per grid location.
        spectra: DataFrame of intensities (counts); index = wavelength (nm),
            one column per grid location.
        column_params: DataFrame of every non-position parameter for every grid location
            (rows = parameters, columns = grid locations). Normally all columns agree and
            `params` is enough; when they don't, use this to pick the columns to keep.
    '''
    params: pd.Series
    positions: pd.DataFrame
    spectra: pd.DataFrame
    column_params: pd.DataFrame | None = None

    @property
    def wavelengths(self)-> np.ndarray:
        return self.spectra.index.to_numpy()

    @property
    def accumulated(self) -> bool:
        '''True if the file was acquired with Accumulate = 1.'''
        return bool(self.params['Accumulate'])

    def param_table(self) -> pd.DataFrame:
        '''
        The parameters plus the number of grid locations, as a display table
        (index 'Param', one 'Value' column of text). Notebooks and the UI render it directly;
        use `self.params` for the numeric values.
        '''
        values = [f'{v:g}' for v in self.params]  # :g drops trailing zeros (80.0 -> 80)
        values.append(str(self.spectra.shape[1]))
        index = pd.Index([*self.params.index, 'Grid locations'], name='Param')
        return pd.DataFrame({'Value': values}, index=index)

    def varying_params(self) -> pd.DataFrame:
        '''
        The parameters that differ between grid locations (rows) for every location (columns).
        Empty if all locations were acquired with the same settings.
        '''
        if self.column_params is None:
            return pd.DataFrame()
        return self.column_params[self.column_params.nunique(axis=1) > 1]

    @staticmethod
    def _unique_names(names: list[str]) -> list[str]:
        '''Suffixes repeated names with (1), (2), ... so they can index a Series.'''
        # How many times each name appears; only names appearing more than once get a suffix.
        counts = {n: names.count(n) for n in names}
        seen: dict[str, int] = {}  # running count per repeated name
        out = []
        for n in names:
            if counts[n] > 1:
                seen[n] = seen.get(n, 0) + 1
                n = f'{n} ({seen[n]})'  # e.g. 'Gas Type' -> 'Gas Type (1)', 'Gas Type (2)'
            out.append(n)
        return out

    @classmethod
    def from_tdms(cls, file_path: str | Path) -> 'LIBSFile':
        '''
        Reads one Applied Spectra TDMS file into a LIBSFile.

        Raises ValueError if the file does not match the expected layout.
        '''
        # TDMS files are organised as groups, and each
        # group holds channels (columns). We flatten every channel from every group into
        # one dict of {channel_name: data}.

        tdms = TdmsFile.read(file_path)
        columns = {ch.name: ch[:] for group in tdms.groups() for ch in group.channels()}

        # We need one header channel (parameter names, then wavelengths) plus at least one spectrum channel (one per grid location).
        if len(columns) < 2:
            raise ValueError('expected a header channel and at least one spectrum channel')
        # The first channel is the header; the rest are the grid locations.
        header_name, *loc_names = columns

        #turn the dict into a DataFrame so we can slice it by row and column.
        df = pd.DataFrame(columns)

        if len(df) <= N_PARAMS:
            raise ValueError(f'only {len(df)} rows; expected more than {N_PARAMS}')

        # wavelengths. Below the parameter block the header channel holds the
        # wavelength axis (nm). Convert to numbers and check it really is one: all values
        # numeric and increasing. This catches files that don't follow the expected layout.
        wavelengths = pd.to_numeric(df[header_name].iloc[N_PARAMS:], errors='coerce')
        if wavelengths.isna().any() or not wavelengths.is_monotonic_increasing:
            raise ValueError('rows after the parameter block are not an increasing wavelength axis')

        # The names live in the first N_PARAMS rows of the header
        # channel (with a leading ' that we strip); the values are the first N_PARAMS rows
        # of each grid-location channel. Repeated names are suffixed to make them unique.
        raw_names = [str(n).lstrip("'") for n in df[header_name].iloc[:N_PARAMS]]
        param_values = df[loc_names].iloc[:N_PARAMS].astype(float)
        param_values.index = LIBSFile._unique_names(raw_names)

        #  X/Y/Z position is the only parameter that changes between grid
        # locations, so it gets its own table (one row per location, transposed with .T).
        # Everything else is shared across locations.
        positions = param_values.loc[list(POSITION_PARAMS)].T
        shared = param_values.drop(index=list(POSITION_PARAMS))

        # verify that "shared" really is shared. If any non-position parameter
        # differs between locations, warn and fall back to the first location's values.
        if (shared.nunique(axis=1) > 1).any():
            varying = list(shared.index[shared.nunique(axis=1) > 1])
            warnings.warn(f'{Path(file_path).name}: parameters differ between grid locations, '
                          f'using the first: {varying}', stacklevel=3)

        # Everything below the parameter block is intensity data;
        # index it by the wavelength axis so spectra.loc[wavelength] works.
        spectra = df[loc_names].iloc[N_PARAMS:].astype(float)
        spectra.index = pd.Index(wavelengths.to_numpy(), name='wavelength_nm')

        # build the LIBSFile. params is the first location's column of the
        # shared parameters, as a Series indexed by parameter name.
        return cls(params=shared.iloc[:, 0].rename('value'), positions=positions, spectra=spectra,
                   column_params=shared)


class LoadLIBSData:
    '''
    Loads LIBS TDMS data from a single file or a folder into a dict of LIBSFile.

    Args:
        path (str | Path): A .tdms file, or a folder containing .tdms files
            (matched case-insensitively).
        recursive (bool): If True, also search subfolders of a folder. Default False.

    The loaded data lives in `self.data` as {key: LIBSFile}. The key is the file's
    path relative to the folder (just the file name for a flat folder or a single
    file), so keys stay unique across subfolders. The object also behaves like a
    read-only dict (`loader['Ce1400.tdms']`, `loader.keys()`, `len(loader)`, iteration).
    Files that fail to read are skipped with a warning rather than aborting the load.
    '''
    def __init__(self, path: str | Path, recursive: bool = False):
        self.path = Path(path)
        self.recursive = recursive
        self.data: dict[str, LIBSFile] = {}  # filled by _load: {key: LIBSFile}
        self._load()

    def _load(self):
        if self.path.is_file():
            files, root = [self.path], self.path.parent
        elif self.path.is_dir():
            candidates = self.path.rglob('*') if self.recursive else self.path.glob('*')
            files = sorted(f for f in candidates if f.is_file() and f.suffix.lower() == '.tdms')
            root = self.path
        else:
            raise FileNotFoundError(f'No such file or folder: {self.path}')

        # parse each file and store it under its relative path. A file that
        # fails is skipped with a warning so the rest still load.
        for f in files:
            try:
                self.data[f.relative_to(root).as_posix()] = LIBSFile.from_tdms(f)
            except Exception as e:
                warnings.warn(f'Could not read {f}: {e}', stacklevel=3)

    def overview(self) -> pd.DataFrame:
        '''
        One row per file with the settings that matter when comparing files: grid locations,
        accumulation, shots, laser energy, gate delay, and which parameters (if any) differ
        between the file's grid locations.
        '''
        return overview_table(self.data)

    # dict-like access so a UI can query keys directly
    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __getitem__(self, key: str | int) -> LIBSFile:
        if isinstance(key, int):
            return list(self.data.values())[key]  # positional access
        return self.data[key]                    # filename access

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def temperature_from_name(name: str) -> int:
    '''Parses the temperature from names like 'Ce_Sample2_Temp_300C.tdms' or 'Gd_Sample5_300_3.5delay.tdms'.'''
    m = re.search(r'_(\d{3})C?(?=[_.])', Path(name).name)
    if not m:
        raise ValueError(f'no temperature found in {name!r}')
    return int(m.group(1))


def overview_table(files: dict[str, LIBSFile]) -> pd.DataFrame:
    """One row of comparison settings per file (see `LoadLIBSData.overview`)."""
    rows = {}
    for key, f in files.items():
        rows[key] = {'grid locations': f.spectra.shape[1],
                     'accumulated': f.accumulated,
                     'shots': int(f.params['Number Of Shots']),
                     'energy (mJ)': f.params.get('Energy', np.nan),
                     'gate delay (us)': f.params.get('Gate Delay (usec)', np.nan),
                     'mixed settings': ', '.join(f.varying_params().index)}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'file'
    return df


# ------------------------------------------------------------------ study plan
PLAN_COLUMNS = ['file', 'element', 'sample', 'temperature', 'use_columns']
_PLACEHOLDERS = ('{element}', '{sample}', '{temperature}')


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


def study_plan(folder: str | Path, pattern: str | None = None, manifest=None,
               temperatures=None) -> pd.DataFrame:
    """
    Which file is which: one row per file with element, sample, temperature and the grid
    locations to use. Exactly one of `pattern` or `manifest` must be given; there is no guessing.

    Args:
        folder: study folder; file paths are relative to it.
        pattern: file-name template matched against every .tdms file under `folder`
            (sub-folders included, `/` between folders, extension optional). Placeholders:
            `{element}` (a chemical symbol, required), `{temperature}` (required),
            `{sample}` (optional), and `*` for any text within one folder or file name.
            Example for `Ce/1_wt%/Ce1_300C.tdms`: `'{element}/{sample}/*{temperature}*'`.
            Every .tdms file must match, or this raises.
        manifest: CSV path or DataFrame with columns file, element, sample, temperature and
            optionally use_columns (grid-location names separated by ';', empty = all).
        temperatures: the temperatures used in the study, e.g. [300, 350, 400]. Without it,
            `{temperature}` matches a stand-alone 3-digit number; with it, only these values,
            even when glued to other digits ('Ce1400' -> 400).

    Returns:
        DataFrame with columns file, element, sample, temperature, use_columns
        (None = all grid locations), sorted by element, sample and temperature. With a
        pattern, the sample is labelled '<element> <sample>' (or just the element if
        the pattern has no {sample}).
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
