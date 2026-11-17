'''
Read-only queries on the NIST Handbook database built by `libs_viewer.nist.build`.

Every function takes an optional `db_path`; by default the database in the libs_viewer data
folder is used (see `libs_viewer.core.paths`). Connections are opened read-only, so nothing
here can change the database.

Wavelengths are stored in Angstrom as published (vacuum below 2000 A, air above, see the
`medium` column); functions here return nm.
'''

import sqlite3
from contextlib import closing
from pathlib import Path
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
            f"SELECT species, wavelength_A, intensity FROM strong_lines "
            f"WHERE element = ? AND species IN ({','.join('?' * len(names))})",
            con, params=[element, *names])
    df['wavelength_nm'] = df.pop('wavelength_A') / 10.0
    df = df[df.wavelength_nm.between(*wl_range)]
    return df.sort_values('intensity', ascending=False).reset_index(drop=True)
