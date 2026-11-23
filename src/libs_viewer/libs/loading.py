'''
Read Applied Spectra LIBS .tdms files into Python. The raw files are only ever read, never changed.

What is inside a .tdms file
---------------------------
The file is a table of columns (TDMS "channels"):

    first column      the header: 31 parameter names, then the wavelengths (nm)
    every other one   one stored spectrum: 31 parameter values, then the intensities

Spectrum columns are named 'Grid Location # L-S', meaning location L, shot S.
'Grid Location # 2-7' is the 7th shot at location 2.

What one spectrum holds depends on the instrument's Accumulate setting:

    Accumulate = 1   the instrument adds up `Number Of Shots` shots and stores one spectrum
    Accumulate = 0   every shot is stored as its own spectrum

A file can contain both kinds. `LIBSFile.acquisitions` has one row per spectrum recording
its location, its shot and how many shots it holds (`shots_summed`). Code further down
reads that table, so it does not need to know which kind the file was.

Words used in this package
--------------------------
    spectrum / column   one stored spectrum, i.e. one column of `LIBSFile.spectra`
    location            one spot on the sample (one stage position)
    shot                one laser pulse
'''

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Mapping, Iterator
import numpy as np
import pandas as pd
from nptdms import TdmsFile

from .plan import MANIFEST_SUFFIXES, read_manifest

# ---------------------------------------------------------------- file format constants
# Change these if your files use a different layout.

N_PARAMS = 31                                              # parameter rows at the top of each column
POSITION_PARAMS = ('X Position', 'Y Position', 'Z Position')  # stage position parameters
LAYOUT = ('location', 'shot', 'shots_summed')              # columns this module adds to `acquisitions`
_CHANNEL_NAME = re.compile(r'(\d+)-(\d+)\s*$')          # 'Grid Location # 2-7' -> 2, 7


class SettingsWarning(UserWarning):
    '''
    The spectra in one file were not all taken with the same instrument settings.

    Example: two spectra used 80 mJ and one used 10 mJ. They should usually not be averaged
    together. `LIBSFile.varying_params()` lists which settings differ.
    '''

class LayoutWarning(UserWarning):
    '''
    The locations or shots in a file don't add up.

    Examples: two spectra named "location 1" were taken at different stage positions, two
    different locations are at the same stage position, or one location holds several
    already-summed spectra. The message says what was found and what the loader did about it.
    '''


def _warn(source: str, msg: str, category: type[Warning] = LayoutWarning, *, stacklevel: int = 2) -> None:
    '''
    Raise a warning that reads '<source>: <msg>', e.g. 'Ce_300C.tdms: location 1 holds ...'.

    Args:
        source: what the warning is about, usually the file name.
        msg: what went wrong.
        category: the warning class. Defaults to LayoutWarning.
        stacklevel: which line the warning points at. 2 means "the line that called the
            function that called _warn", the same as for `warnings.warn`. This helper adds
            1 for its own frame, so you don't need to.
    '''
    warnings.warn(f'{source}: {msg}', category, stacklevel=stacklevel + 1)


@dataclass
class LIBSFile:
    '''
    One .tdms file, loaded. Create it with `LIBSFile.from_tdms(path)`.

    Attributes:
        spectra: the raw intensities (counts). Rows are wavelengths (nm) and there is one
            column per stored spectrum, named like the file's channels
            ('Grid Location # 1-1', ...).
        acquisitions: one row per stored spectrum, in the same order as the `spectra`
            columns. It says what each spectrum is:
                location        which spot on the sample (1, 2, 3, ...)
                shot            which shot at that spot (1, 2, 3, ...)
                shots_summed    how many laser shots the spectrum holds: `Number Of Shots`
                                if Accumulate = 1, otherwise 1. Divide the spectrum by this
                                to get counts per shot.
                X/Y/Z Position  where the stage was
                ...             then every other instrument parameter (Energy, Gate Delay, ...)
        path: the file it came from (None if the object was built some other way).
    '''
    spectra: pd.DataFrame
    acquisitions: pd.DataFrame
    path: Path | None = None

    # ---------------------------------------------------------------- views
    @property
    def wavelengths(self) -> np.ndarray:
        '''The wavelength axis in nm, as a numpy array (the row labels of `spectra`).'''
        return self.spectra.index.to_numpy()

    @property
    def params(self) -> pd.Series:
        '''
        The instrument settings, read from the first spectrum, e.g. params['Energy'] -> 80.0.

        Usually every spectrum in a file has the same settings. If they don't, the file
        raised a SettingsWarning and `varying_params()` shows the differences.
        When the file repeats a parameter name, the copies are called 'Name (1)', 'Name (2)'.
        Stage positions are not included; see `positions`.
        '''
        return self._param_values().iloc[:, 0].rename('value')

    @property
    def positions(self) -> pd.DataFrame:
        '''Where the stage was for each spectrum: one row per spectrum, columns X/Y/Z Position.'''
        return self.acquisitions[list(POSITION_PARAMS)]

    @property
    def n_locations(self) -> int:
        '''How many different spots on the sample were measured.'''
        return int(self.acquisitions['location'].nunique())

    @property
    def storage(self) -> str:
        '''
        How the shots were saved, from the Accumulate setting:
            'accumulated'    every spectrum is a sum of several shots (Accumulate = 1)
            'single shots'   every spectrum is one shot (Accumulate = 0)
            'mixed'          the file has both
        '''
        acc = set(self.acquisitions['Accumulate'].astype(bool))
        return 'mixed' if len(acc) > 1 else ('accumulated' if acc.pop() else 'single shots')

    def shots_per_location(self) -> pd.Series:
        '''
        Total laser shots fired at each location, indexed by location number.

        Example: 10 single-shot spectra at location 1 -> 10. One spectrum that sums 10
        shots -> also 10.
        '''
        return self.acquisitions.groupby('location')['shots_summed'].sum()

    def describe(self) -> str:
        '''A one-line summary, e.g. '2 locations x 10 shots (single shots)'.'''
        shots = self.shots_per_location()
        n = len(shots)
        per = f'{shots.iloc[0]:g}' if shots.nunique() == 1 else f'{shots.min():g}-{shots.max():g}'
        return f"{n} location{'s' if n > 1 else ''} x {per} shots ({self.storage})"

    def varying_params(self) -> pd.DataFrame:
        '''
        The settings that are not the same for every spectrum in the file.

        Returns a table with one row per differing setting and one column per spectrum.
        It is empty if every spectrum used the same settings, which is the normal case.
        Stage positions are left out because they are supposed to change.
        '''
        values = self._param_values()
        return values[values.nunique(axis=1) > 1]

    def param_table(self) -> pd.DataFrame:
        '''
        The settings as a text table, ready to display (e.g. in a UI or notebook).

        The rows are every parameter, then 'Locations', 'Spectra stored' and 'Stored as'.
        There is one 'Value' column, as text: 80.0 is shown as '80'. For numbers you can
        calculate with, use `params` or `acquisitions`.
        '''
        values = [f'{v:g}' for v in self.params]  # :g drops trailing zeros (80.0 -> 80)
        values += [str(self.n_locations), str(self.spectra.shape[1]), self.storage]
        index = pd.Index([*self.params.index, 'Locations', 'Spectra stored', 'Stored as'], name='Param')
        return pd.DataFrame({'Value': values}, index=index)

    def subset(self, columns=None, locations=None) -> "LIBSFile":
        '''
        A copy of the file that keeps only some of its spectra.

        Args:
            columns: spectrum column names to keep, e.g. ['Grid Location # 1-1'].
                None keeps all of them.
            locations: location numbers to keep, e.g. [1, 3]. None keeps all of them.
            If you give both, a spectrum must match both to be kept.

        Raises:
            ValueError: a name or number you asked for is not in the file, or the
                selection would leave no spectra.

        Example:
            f.subset(locations=[1, 2])      # only locations 1 and 2
        '''

        def _filter_mask(items, valid_series, label: str) -> np.ndarray:
            '''True for each spectrum to keep. Raises if `items` names something unknown.'''
            if items is None:
                return np.ones(len(self.acquisitions), dtype=bool)
            item_list = list(items)
            unknown = sorted(set(item_list) - set(valid_series))
            if unknown:
                raise ValueError(f"{self._name}: unknown {label}(s) {unknown}")
            # Index.isin returns an ndarray, Series.isin a Series; np.asarray handles both
            return np.asarray(valid_series.isin(item_list))

        keep = _filter_mask(columns, self.spectra.columns, "spectrum column") & _filter_mask(
            locations, self.acquisitions["location"], "location"
        )

        if not keep.any():
            raise ValueError(f"{self._name}: the selection keeps no spectra")

        return LIBSFile(self.spectra.loc[:, keep], self.acquisitions[keep], self.path)

    def accumulate(self) -> 'LIBSFile':
        '''
        Add up all the spectra at each location, so each location has one spectrum.

        The result looks as if the instrument had been set to Accumulate = 1. Columns
        are renamed 'Location 1', 'Location 2', ..., and `shots_summed` tells you how many
        shots went into each one. Dividing by `shots_summed` gives the same per-shot
        average as before. You lose the shot-to-shot variation.

        Warns (LayoutWarning), but still works, when:
            - every location already has only one spectrum. Nothing is done and the
              file is returned unchanged.
            - some spectra at a location are already sums of several shots. They are
              added in like the rest.
            - spectra at the same location used different settings. They are added
              anyway.

        Careful: adding shots together can hide a single saturated shot. Check
        `processing.saturated_mask` first if that matters.
        '''
        a = self.acquisitions
        per_loc = a.groupby('location').size()
        if (per_loc == 1).all():
            _warn(self._name, f'already one spectrum per location ({self.storage}); '
                  'nothing to accumulate')
            return self
        multi = per_loc.index[per_loc > 1]
        acc = sorted(set(a.loc[a['location'].isin(multi) & (a['shots_summed'] > 1), 'location']))
        if acc:
            _warn(self._name, f'location(s) {acc} already hold accumulated spectra; '
                  'they are summed with the rest')
        params = a.drop(columns=[*LAYOUT, *POSITION_PARAMS])
        mixed = sorted({loc for loc, g in params.groupby(a['location']) if (g.nunique() > 1).any()})
        if mixed:
            _warn(self._name, f'location(s) {mixed} mix acquisition settings; summed anyway')

        # Sum the spectra of each location; keep the first spectrum's settings for the row
        loc = a['location']
        spectra = self.spectra.T.groupby(loc, sort=False).sum().T
        rows = a.groupby('location', sort=False).first()
        rows['shots_summed'] = a.groupby('location', sort=False)['shots_summed'].sum()
        rows['shot'] = 1
        rows['Accumulate'] = 1.0
        rows['Number Of Shots'] = rows['shots_summed']
        names = [f'Location {l}' for l in rows.index]
        spectra.columns = names
        rows = rows.reset_index()[list(a.columns)]
        rows.index = pd.Index(names, name='spectrum')
        return LIBSFile(spectra, rows, self.path)

    # ---------------------------------------------------------------- reading
    @classmethod
    def from_tdms(cls, file_path: str | Path) -> "LIBSFile":
        '''
        Load one Applied Spectra .tdms file. The file is only read.

        Steps:
            1. Read every channel into a table.
            2. The first column is the header: its first 31 rows are parameter names and
               the rest are wavelengths.
            3. Every other column is a spectrum: 31 parameter values, then intensities.
            4. Work out the location and shot of each spectrum (see `_layout`).

        Raises:
            ValueError: the file doesn't look like an Applied Spectra LIBS file. For
                example it has no spectrum columns, it is too short, the wavelengths don't
                increase, or a required parameter (Accumulate, Number Of Shots, X/Y/Z
                Position) is missing.

        Warns:
            SettingsWarning: the spectra were not all taken with the same settings.
            LayoutWarning: the locations or shots don't add up (see `_layout`).
        '''
        file_path = Path(file_path)
        header_name, spec_names, df = cls._read_tdms_table(file_path)

        wavelengths = pd.to_numeric(df[header_name].iloc[N_PARAMS:], errors="coerce")
        if wavelengths.isna().any() or not wavelengths.is_monotonic_increasing:
            raise ValueError("rows after the parameter block are not an increasing wavelength axis")

        param_names = _unique_names([str(n).lstrip("'") for n in df[header_name].iloc[:N_PARAMS]])
        params = df[spec_names].iloc[:N_PARAMS].astype(float).set_axis(param_names).T

        missing = {"Accumulate", "Number Of Shots", *POSITION_PARAMS} - set(params.columns)
        if missing:
            raise ValueError(f"missing required parameter(s): {sorted(missing)}")

        varying = [p for p in params.columns.drop(list(POSITION_PARAMS)) if params[p].nunique() > 1]
        if varying:
            _warn(file_path.name,
                  f"settings differ between stored spectra: {varying} (see varying_params())",
                  SettingsWarning)

        acquisitions = pd.concat([_layout(file_path.name, params), params], axis=1)
        acquisitions.index.name = "spectrum"

        spectra = df[spec_names].iloc[N_PARAMS:].astype(float)
        spectra.index = pd.Index(wavelengths.to_numpy(), name="wavelength_nm")
        return cls(spectra=spectra, acquisitions=acquisitions, path=file_path)

    @staticmethod
    def _read_tdms_table(file_path: Path) -> tuple[str, list[str], pd.DataFrame]:
        '''
        Read every channel of the .tdms file into one table, with no interpretation yet.

        Returns:
            (header column name, list of spectrum column names, the table)

        Raises:
            ValueError: there is no spectrum column, or the table is too short to hold
                the 31 parameter rows plus at least one wavelength.
        '''
        tdms = TdmsFile.read(file_path)
        channels = {ch.name: ch[:] for group in tdms.groups() for ch in group.channels()}
        if len(channels) < 2:
            raise ValueError("expected a header channel and at least one spectrum channel")
        header_name, *spec_names = channels
        df = pd.DataFrame(channels)
        if len(df) <= N_PARAMS:
            raise ValueError(f"only {len(df)} rows; expected more than {N_PARAMS}")
        return header_name, spec_names, df

    # ---------------------------------------------------------------- helpers
    @property
    def _name(self) -> str:
        '''The file name, used to start error and warning messages ('file' if unknown).'''
        return self.path.name if self.path else 'file'

    def _param_values(self) -> pd.DataFrame:
        '''
        Every instrument setting for every spectrum: one row per setting, one column per
        spectrum. Location, shot and stage positions are left out.
        '''
        return self.acquisitions.drop(columns=[*LAYOUT, *POSITION_PARAMS]).T


# ---------------------------------------------------------------- reading helpers
def _unique_names(names: list[str]) -> list[str]:
    '''
    Number the names that appear more than once, so every name is different.

    Example: ['Energy', 'Delay', 'Delay'] -> ['Energy', 'Delay (1)', 'Delay (2)']
    '''
    totals = Counter(names)
    seen: Counter[str] = Counter()
    out = []
    for name in names:
        if totals[name] > 1:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        out.append(name)
    return out

def _layout(file_name: str, params: pd.DataFrame) -> pd.DataFrame:
    '''
    Work out the location, shot and number of shots summed for every spectrum.

    Args:
        file_name: used in warning messages.
        params: the parameters, one row per spectrum, indexed by channel name.

    Returns:
        A table with one row per spectrum and the columns 'location', 'shot' and
        'shots_summed'.

    How it decides:
        - Location and shot come from the channel names ('Grid Location # 2-7' ->
          location 2, shot 7). If the names don't fit the stage positions, they come from
          the stage positions instead (see `_resolve_locations_and_shots`).
        - shots_summed is `Number Of Shots` when Accumulate = 1, otherwise 1.

    Warns (LayoutWarning) when two locations share a stage position, or when a location
    holds more than one spectrum and some of them are already sums of several shots.
    '''
    positions = [
        tuple(p) for p in params[list(POSITION_PARAMS)].round(6).itertuples(index=False)
    ]
    is_accumulated = params["Accumulate"].astype(bool).to_numpy()
    shots_summed = np.where(is_accumulated, params["Number Of Shots"].to_numpy(), 1.0)

    location, shot = _resolve_locations_and_shots(
        file_name, params.index, positions, is_accumulated
    )

    layout = pd.DataFrame(
        {"location": location, "shot": shot, "shots_summed": shots_summed},
        index=params.index,
    )

    _warn_overlapping_positions(file_name, layout["location"], positions)
    _warn_mixed_accumulations(file_name, layout)

    return layout

def _resolve_locations_and_shots(
    file_name: str,
    channel_names: pd.Index,
    positions: list[tuple],
    is_accumulated: np.ndarray,
) -> tuple[list[int], list[int]]:
    '''
    Give every spectrum a location number and a shot number.

    First choice: read them from the channel names ('Grid Location # 2-7' -> 2, 7).

    Fallback: if any name doesn't end in 'L-S', or the names disagree with the stage
    positions (see `_name_problems`), number the locations by stage position instead.
    The first position seen is location 1, the next new one is location 2, and so on.
    Shots are counted 1, 2, 3, ... within each location.
    If the names were usable but wrong, a LayoutWarning explains why they were ignored.

    Returns:
        (locations, shots): two lists with one number per spectrum.
    '''
    matches = [_CHANNEL_NAME.search(str(name)) for name in channel_names]

    if all(matches):
        # m is guaranteed not None inside `if all(matches)`, satisfying type checkers
        location = [int(m.group(1)) for m in matches if m is not None]
        shot = [int(m.group(2)) for m in matches if m is not None]

        problems = _name_problems(location, shot, positions, is_accumulated)
        if not problems:
            return location, shot

        # stacklevel 4: here -> _layout -> from_tdms -> caller of from_tdms
        _warn(file_name, f'{"; ".join(problems)}. Locations are taken from the '
              'stage positions instead', stacklevel=4)

    # Fallback: number locations by order of first appearance at each stage position
    first_seen = {pos: i + 1 for i, pos in enumerate(dict.fromkeys(positions))}
    location = [first_seen[pos] for pos in positions]
    shot = (pd.Series(location).groupby(location).cumcount() + 1).tolist()
    return location, shot

def _name_problems(location, shot, position, accumulate) -> list[str]:
    '''
    Check whether the location/shot numbers from the channel names make sense.

    Args:
        location, shot: the numbers read from the channel names, one per spectrum.
        position: the (X, Y, Z) stage position of each spectrum.
        accumulate: True/False per spectrum (Accumulate setting), used for wording only.

    Returns:
        A list of problems in plain words. An empty list means the names are fine.
        A problem is either:
            - spectra with the same location number were taken at different stage
              positions, so the "location" is really several spots, or
            - the same location-shot pair appears twice.
    '''
    problems = []
    indices_by_loc = defaultdict(list)
    for i, loc in enumerate(location):
        indices_by_loc[loc].append(i)

    for loc, idx in indices_by_loc.items():
        spots = {position[i] for i in idx}
        if len(spots) > 1:
            kind = "single shots" if not accumulate[idx].any() else "spectra"
            problems.append(
                f"the {len(idx)} {kind} named as location {loc} were taken at "
                f"{len(spots)} different stage positions"
            )

    if len(set(zip(location, shot))) < len(location):
        problems.append("some location-shot names repeat")
    return problems

def _warn_overlapping_positions(
    file_name: str, locations: pd.Series, positions: list[tuple]
) -> None:
    '''
    Warn (LayoutWarning) if two different location numbers are at the same stage position.

    Why it matters: spread between "locations" is meant to show how much the sample varies
    from spot to spot. If two locations are really the same spot, that spread is just
    shot-to-shot noise.
    '''
    locs_by_pos = defaultdict(list)
    for loc, pos in dict(zip(locations, positions)).items():
        locs_by_pos[pos].append(loc)

    for locs in locs_by_pos.values():
        if len(locs) > 1:
            _warn(file_name, f"locations {locs} are at the same stage position; "
                  "their scatter is not a location-to-location scatter", stacklevel=4)

def _warn_mixed_accumulations(file_name: str, layout: pd.DataFrame) -> None:
    '''
    Warn (LayoutWarning) if a location holds more than one spectrum and at least one of
    them is already a sum of several shots.

    Normally a location has either one summed spectrum (Accumulate = 1) or several single
    shots (Accumulate = 0). Several summed spectra at one spot usually means the spot was
    measured again. The loader keeps them all and treats them as repeat measurements.
    '''
    for loc, group in layout.groupby("location"):
        n_total = len(group)
        n_acc = (group["shots_summed"] > 1).sum()

        if n_total > 1 and n_acc > 0:
            kind = (
                "accumulated spectra"
                if n_acc == n_total
                else f"spectra ({n_acc} accumulated, {n_total - n_acc} single shots)"
            )
            _warn(file_name, f"location {loc} holds {n_total} {kind}; they are treated as "
                  "repeated measurements at one spot", stacklevel=4)


# ---------------------------------------------------------------- many files
class LoadLIBSData(Mapping[str, LIBSFile]):
    '''
    Load .tdms files and use them like a dict. Three ways in:

        LoadLIBSData('data/Ce/run1.tdms')                 one file
        LoadLIBSData('data/Ce', recursive=True)           every .tdms file in a folder
        LoadLIBSData(manifest='data/Ce/manifest.xlsx')    exactly the files a manifest lists

    Args:
        path: a .tdms file or a folder. With a manifest: the folder its file paths are
            relative to (default: the manifest's own folder).
        recursive: if True, also look in subfolders (folder loading only).
        manifest: a .csv / .xlsx file or DataFrame with a `file` column and any label columns
            (sample, temperature, ...). See `read_manifest`; only `file` is required here.

    The keys are file paths relative to the folder, e.g. 'Ce/s1/Ce_s1_300C.tdms'. You
    can also index by position: data[0] is the first file.

    Folder loading skips a file that can't be read, with a warning, so one bad file doesn't
    stop the rest. A manifest names its files on purpose, so there any unreadable file is an
    error (all of them are listed at once). The manifest's labels are kept in `labels`
    (one row per file, indexed like the keys) and shown first in `overview()`.

    Raises:
        FileNotFoundError: `path`, the manifest, or a listed file does not exist.
        ValueError: `path` is a file but not a .tdms file (e.g. a manifest passed as `path`),
            neither `path` nor `manifest` is given, the manifest is invalid, or a listed
            file can't be read.

    Example:
        data = LoadLIBSData('data/Ce', recursive=True)
        data.overview()                  # one summary row per file
        f = data['s1/Ce_s1_300C.tdms']   # a LIBSFile
    '''
    def __init__(self, path: str | Path | None = None, recursive: bool = False, manifest=None):
        self.path = None if path is None else Path(path)
        self.recursive = recursive
        self.manifest = manifest
        self.labels: pd.DataFrame | None = None
        self.data: dict[str, LIBSFile] = {}
        if manifest is not None:
            self._load_manifest()
        elif self.path is None:
            raise ValueError('give a .tdms file or folder as path=, or a manifest=')
        else:
            self._load()

    def _load(self) -> None:
        '''Find the .tdms files under `path` and read each one into `self.data`.'''
        if self.path.is_file():
            if self.path.suffix.lower() != '.tdms':
                hint = (f"; to load the files it lists, use LoadLIBSData(manifest={str(self.path)!r})"
                        if self.path.suffix.lower() in MANIFEST_SUFFIXES else '')
                raise ValueError(f'{self.path.name} is not a .tdms file{hint}')
            files, root = [self.path], self.path.parent
        elif self.path.is_dir():
            pattern = self.path.rglob("*") if self.recursive else self.path.glob("*")
            files = sorted(f for f in pattern if f.is_file() and f.suffix.lower() == ".tdms")
            root = self.path
        else:
            raise FileNotFoundError(f"No such file or folder: {self.path}")

        for f in files:
            try:
                self.data[f.relative_to(root).as_posix()] = LIBSFile.from_tdms(f)
            except Exception as e:
                _warn(f.name, f"Could not read {f}: {e}", UserWarning, stacklevel=3)

    def _load_manifest(self) -> None:
        '''Read exactly the files the manifest lists; any that can't be read is an error.'''
        table = read_manifest(self.manifest, self.path)
        root = self.path if self.path is not None else Path(self.manifest).parent
        failed = []
        for key in table['file']:
            try:
                self.data[key] = LIBSFile.from_tdms(root / key)
            except Exception as e:
                failed.append(f'{key}: {e}')
        if failed:
            raise ValueError(f'{len(failed)} manifest file(s) could not be read:\n' + '\n'.join(failed))
        self.labels = table.set_index('file')

    def overview(self) -> pd.DataFrame:
        '''
        A summary table with one row per loaded file. See `overview_table`. With a manifest,
        its label columns come first (a `use_columns` list is shown as ';'-separated text).
        '''
        table = overview_table(self.data)
        if self.labels is None:
            return table
        labels = self.labels.copy()
        if 'use_columns' in labels:
            labels['use_columns'] = ['; '.join(v) if v else '' for v in labels['use_columns']]
        return labels.join(table)

    def __getitem__(self, key: str | int) -> LIBSFile:
        '''data['file.tdms'] gets a file by name; data[0] gets the first file.'''
        return list(self.data.values())[key] if isinstance(key, int) else self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def overview_table(files: dict[str, LIBSFile]) -> pd.DataFrame:
    '''
    A summary table of several files, one row per file, for checking a data set at a glance.

    Columns:
        locations           how many spots were measured
        spectra             how many spectra are stored
        shots per location  text: '10', or '10, 10, 5' if the locations differ (always a
                            string, so the column has one type and displays in Streamlit / Arrow)
        stored as           'accumulated', 'single shots' or 'mixed'
        energy (mJ)         laser energy (NaN if the file doesn't record it)
        gate delay (us)     gate delay (NaN if the file doesn't record it)
        mixed settings      settings that differ between the file's spectra (blank if none)
    '''
    rows = {}
    for key, f in files.items():
        shots = f.shots_per_location()
        rows[key] = {'locations': f.n_locations,
                     'spectra': f.spectra.shape[1],
                     'shots per location': (f'{shots.iloc[0]:g}' if shots.nunique() == 1
                                            else ', '.join(f'{s:g}' for s in shots)),
                     'stored as': f.storage,
                     'energy (mJ)': f.params.get('Energy', np.nan),
                     'gate delay (us)': f.params.get('Gate Delay (usec)', np.nan),
                     'mixed settings': ', '.join(f.varying_params().index)}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'file'
    return df
