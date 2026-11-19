'''
Shared fixtures: synthetic spectra and synthetic Applied Spectra .tdms files.

The .tdms layout mirrors the real files: one header channel whose first 31 rows are
parameter names (with a leading ') followed by the wavelength axis, and one channel per stored
spectrum whose first 31 rows are that spectrum's parameter values, followed by intensities.
'''

import shutil
from pathlib import Path

import numpy as np
import pytest
from nptdms import ChannelObject, TdmsWriter

from libs_viewer.libs import loading
from libs_viewer.nist import build

NIST_FIXTURES = Path(__file__).parent / 'fixtures' / 'nist'

# 31 names, including the ones the code reads and a repeated name (as in the real files)
PARAM_NAMES = (['Accumulate', 'Number Of Shots', 'Energy', 'Gas Type', 'Gas Type',
                'X Position', 'Y Position', 'Z Position', 'Gate Delay (usec)']
               + [f'Param {i}' for i in range(10, loading.N_PARAMS + 1)])
assert len(PARAM_NAMES) == loading.N_PARAMS

WL = np.round(np.arange(380.0, 420.0, 0.05), 4)   # 800 pixels, 0.05 nm/pixel
LINES = {'A 390.00': 390.0, 'B 400.00': 400.0, 'C 410.00': 410.0}
SIGMA = 0.06                                       # nm


def gaussian(wl, center, area, sigma=SIGMA):
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def continuum(wl):
    '''Smooth, broad plasma-like continuum (counts per shot).'''
    return 200.0 + 3.0 * (wl - wl[0]) + 150.0 * np.exp(-0.5 * ((wl - 400.0) / 15.0) ** 2)


def synth_spectrum(areas: dict[str, float], wl=WL) -> np.ndarray:
    '''Per-shot spectrum: continuum plus one Gaussian per line, with the given areas (counts*nm).'''
    y = continuum(wl)
    for label, a in areas.items():
        y = y + gaussian(wl, LINES[label], a)
    return y


def write_tdms(path, per_shot: np.ndarray, n_cols: int = 3, shots: int = 10, accumulate: int = 1,
               energy: float = 80.0, energies: list[float] | None = None, wl=WL,
               locations: list[int] | None = None, accumulates: list[int] | None = None,
               x_positions: list[float] | None = None):
    '''
    Writes one synthetic Applied Spectra file. Columns hold `per_shot * shots` (accumulated)
    or `per_shot` (not accumulated), each scaled by a slightly different factor so the columns differ.
    `energies` / `accumulates` give per-column Energy / Accumulate values (mixed files).

    By default every column is its own location, named 'Location c'. With `locations` (the
    location number of every column) the columns are named like the instrument does,
    'Grid Location # L-S', and columns of one location share a stage position.
    `x_positions` overrides the X stage position of every column (to break that on purpose).
    '''
    header = np.array(["'" + n for n in PARAM_NAMES] + [repr(float(w)) for w in wl])
    channels = [ChannelObject('Group', 'Wavelength', header)]
    shot_no: dict[int, int] = {}
    for c in range(n_cols):
        e = energies[c] if energies else energy
        acc = accumulates[c] if accumulates else accumulate
        loc = locations[c] if locations else c
        shot_no[loc] = shot_no.get(loc, 0) + 1
        name = f'Grid Location # {loc}-{shot_no[loc]}' if locations else f'Location {c + 1}'
        # positional, in PARAM_NAMES order (the two 'Gas Type' rows get 1 and 2)
        x = x_positions[c] if x_positions else 10.0 * loc
        values = ([acc, shots, e, 1, 2, x, 5.0, 0.0, 3.5]
                  + [float(i) for i in range(10, loading.N_PARAMS + 1)])
        scale = (shots if acc else 1) * (1 + 0.01 * (c - 1))
        data = np.concatenate([np.asarray(values, dtype=float), per_shot * scale])
        channels.append(ChannelObject('Group', name, data))
    path.parent.mkdir(parents=True, exist_ok=True)
    with TdmsWriter(str(path)) as w:
        w.write_segment(channels)
    return path


@pytest.fixture
def wl():
    return WL.copy()


@pytest.fixture
def tdms_file(tmp_path):
    '''One accumulated file (3 columns x 10 shots) of a known per-shot spectrum.'''
    y = synth_spectrum({'A 390.00': 50.0, 'B 400.00': 100.0, 'C 410.00': 30.0})
    return write_tdms(tmp_path / 'Ce_Sample1_Temp_300C.tdms', y), y


@pytest.fixture(scope='session')
def nist_db(tmp_path_factory):
    '''Ce + Gd database built from the cached real pages; any network access fails the test.'''
    tmp = tmp_path_factory.mktemp('nist')
    cache = tmp / 'cache'
    shutil.copytree(NIST_FIXTURES, cache)

    def no_network(self, url):
        raise AssertionError(f'tried to download {url}')

    mp = pytest.MonkeyPatch()
    mp.setattr(build.PoliteNISTFetcher, '_fetch', no_network)
    try:
        build.build_database(db_path=tmp / 'db.sqlite', elements=['Ce', 'Gd'], cache_dir=cache)
    finally:
        mp.undo()
    return tmp / 'db.sqlite'
