'''
Where libs_viewer keeps its own files (not your data): the NIST database and its page cache.

The folder is the per-user application data folder, or `$LIBS_VIEWER_DATA` if that is set:
    Windows  %LOCALAPPDATA%\\libs_viewer
    macOS    ~/Library/Application Support/libs_viewer
    Linux    ~/.local/share/libs_viewer

Nothing here creates folders; the code that writes (the NIST builder) does that.
'''

import os
from pathlib import Path
from platformdirs import user_data_dir

APP_NAME = 'libs_viewer'
ENV_DATA_DIR = 'LIBS_VIEWER_DATA'


def data_dir() -> Path:
    '''The libs_viewer data folder ($LIBS_VIEWER_DATA, else the per-user default).'''
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir(APP_NAME, appauthor=False))


def nist_db_path() -> Path:
    '''Default location of the NIST Handbook SQLite database.'''
    return data_dir() / 'nist_handbook.sqlite'


def nist_cache_dir() -> Path:
    '''Default folder for the raw NIST pages kept by the builder (lets rebuilds skip the network).'''
    return data_dir() / 'nist_cache'
