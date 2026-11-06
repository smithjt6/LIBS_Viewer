'''
TDMS loader module.

Loads Applied Spectra LIBS .tdms files and splits each into parameters,
wavelengths and spectra so a UI (or tool) can query them directly.
'''

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
    '''
    params: pd.Series
    positions: pd.DataFrame
    spectra: pd.DataFrame

    @property
    def wavelengths(self)-> np.ndarray:
        return self.spectra.index.to_numpy()

    @property
    def accumulated(self) -> bool:
        '''True if the file was acquired with Accumulate = 1.'''
        return bool(self.params['Accumulate'])

    def param_table(self) -> None:
        '''Prints the parameters (and number of grid locations) as an aligned table.'''
        names = list(self.params.index)
        values = [f'{v:g}' for v in self.params]  # :g drops trailing zeros (80.0 -> 80)

        # Column widths = longest entry, but never narrower than the header word.
        name_w = max(len('Param'), *map(len, names))
        value_w = max(len('Value'), *map(len, values))

        # Header row, divider, then one row per parameter (names left, values right aligned).
        print(f"{'Param':<{name_w}} | {'Value':>{value_w}}")
        print(f"{'-' * name_w}-+-{'-' * value_w}")
        for name, value in zip(names, values):
            print(f'{name:<{name_w}} | {value:>{value_w}}')
        print(f"{'Grid locations':<{name_w}} | {self.spectra.shape[1]:>{value_w}}")

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
        return cls(params=shared.iloc[:, 0].rename('value'), positions=positions, spectra=spectra)


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

    # dict-like access so a UI can query keys directly
    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __getitem__(self, key: str) -> LIBSFile:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
