'''
Read-only queries on the NIST Handbook database built by `libs_viewer.nist.build`.

Every function takes an optional `db_path`; by default the database in the libs_viewer data
folder is used (see `libs_viewer.core.paths`). Connections are opened read-only, so nothing
here can change the database.

Wavelengths are stored in Angstrom as published (vacuum below 2000 A, air above, see the
`medium` column) together with the standard air wavelength (`wavelength_air_A`). Functions
here return **air wavelengths in nm**, which is what a spectrometer in air measures; below
200 nm the published vacuum value is ~0.06 nm longer, too much for peak matching.
'''

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.paths import nist_db_path

TABLES = ('atomic_data', 'strong_lines', 'persistent_lines', 'energy_levels', 'bibliographic_refs')


class NISTDatabaseMissing(FileNotFoundError):
    '''The NIST database has not been built (or is not where it was expected).'''


def resolve_db(db_path: str | Path | None = None) -> Path:
    '''The database file to use: `db_path` if given, else the default location. Must exist.'''
    path = Path(db_path) if db_path else nist_db_path()
    if not path.is_file():
        raise NISTDatabaseMissing(
            f'NIST database not found at {path}. Build it with `libs-viewer nist build`, '
            f'or pass db_path= (--db on the command line) to use another file.')
    return path


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    '''Read-only connection to the NIST database.'''
    path = resolve_db(db_path)
    return sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)


def database_info(db_path: str | Path | None = None) -> dict:
    '''
    Status of the database, for the CLI / UI settings page. Never raises if it is missing.

    Returns:
        dict with 'path', 'exists', 'size_bytes', 'elements' (number crawled) and
        'rows' ({table: row count}); the last three are absent if the file doesn't exist.
    '''
    path = Path(db_path) if db_path else nist_db_path()
    info = {'path': str(path), 'exists': path.is_file()}
    if not info['exists']:
        return info
    with closing(connect(path)) as con:
        info['size_bytes'] = path.stat().st_size
        info['elements'] = con.execute('SELECT COUNT(*) FROM crawl_log').fetchone()[0]
        info['rows'] = {t: con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in TABLES}
    return info


def strong_lines(element: str, species: tuple[str, ...] = ('I', 'II'),
                 wl_range: tuple[float, float] = (186.0, 1050.0),
                 db_path: str | Path | None = None) -> pd.DataFrame:
    '''
    Strong lines of one element (Handbook Table 2).

    Args:
        element: symbol, e.g. 'Ce'.
        species: ion stages to include ('I' = neutral, 'II' = singly ionized).
        wl_range: (lo, hi) in nm.

    Returns:
        DataFrame with columns species, intensity, wavelength_nm, sorted by intensity
        (strongest first).
    '''
    names = [f'{element} {s}' for s in species]
    with closing(connect(db_path)) as con:
        df = pd.read_sql_query(
            f"SELECT species, {_AIR_NM} AS wavelength_nm, intensity FROM strong_lines "
            f"WHERE element = ? AND species IN ({','.join('?' * len(names))})",
            con, params=[element, *names])
    df = df[df.wavelength_nm.between(*wl_range)][['species', 'intensity', 'wavelength_nm']]
    return df.sort_values('intensity', ascending=False).reset_index(drop=True)


# air wavelength in nm; the published value where no air wavelength exists (below 170 nm)
_AIR_NM = 'COALESCE(wavelength_air_A, wavelength_A) / 10.0'


def elements(db_path: str | Path | None = None) -> list[str]:
    '''Symbols of every element in the database, in order of atomic number.'''
    with closing(connect(db_path)) as con:
        rows = con.execute(
            'SELECT c.element FROM crawl_log c LEFT JOIN '
            '(SELECT element, MIN(atomic_number) AS z FROM atomic_data GROUP BY element) a '
            'USING (element) ORDER BY a.z IS NULL, a.z, c.element').fetchall()
    return [r[0] for r in rows]


def lines_near(wavelengths, tol: float = 0.1, elements: list[str] | None = None,
               species: tuple[str, ...] = ('I', 'II'),
               db_path: str | Path | None = None) -> pd.DataFrame:
    '''
    NIST strong lines (Handbook Table 2) within `tol` nm of each given wavelength: the
    candidate identities of observed peaks.

    Args:
        wavelengths: one wavelength in nm, or many (e.g. the `wavelength_nm` column of
            `metrics.detect_peaks`); all are looked up in one query.
        tol: how far (nm) a line may be from the wavelength.
        elements: only these elements (e.g. the ones expected in the sample); default all.
        species: ion stages ('I' = neutral, 'II' = singly ionized).

    Returns:
        DataFrame, one row per (wavelength, line) match, columns
            peak_nm        the wavelength looked up
            element, species, wavelength_nm, intensity   the NIST line (air wavelength)
            delta_nm       line minus peak
        sorted by peak_nm, then closest line first (ties: stronger first). A wavelength
        with no line within `tol` has no rows.
    '''
    peaks = np.unique(np.atleast_1d(np.asarray(wavelengths, dtype=float)))
    cols = ['peak_nm', 'element', 'species', 'wavelength_nm', 'intensity', 'delta_nm']
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    stages = [s.strip() for s in species]
    where = [f'{_AIR_NM} BETWEEN ? AND ?']
    params: list = [peaks.min() - tol, peaks.max() + tol]
    if elements is not None:
        where.append(f"element IN ({','.join('?' * len(elements))})")
        params += list(elements)
    with closing(connect(db_path)) as con:
        lines = pd.read_sql_query(
            f"SELECT element, species, {_AIR_NM} AS wavelength_nm, intensity "
            f"FROM strong_lines WHERE {' AND '.join(where)}", con, params=params)
    lines = lines[lines.species.str.split().str[-1].isin(stages)].reset_index(drop=True)

    # every (peak, line) pair within tol; both sides are small (peaks x lines in range)
    delta = lines.wavelength_nm.to_numpy()[None, :] - peaks[:, None]
    p_idx, l_idx = np.nonzero(np.abs(delta) <= tol)
    out = lines.iloc[l_idx].reset_index(drop=True)
    out.insert(0, 'peak_nm', peaks[p_idx])
    out['delta_nm'] = delta[p_idx, l_idx]
    out = out.assign(_dist=out.delta_nm.abs(), _neg_int=-out.intensity.fillna(0))
    out = out.sort_values(['peak_nm', '_dist', '_neg_int'], kind='stable')
    return out[cols].reset_index(drop=True)
