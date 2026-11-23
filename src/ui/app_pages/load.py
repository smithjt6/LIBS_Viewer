# ui/app_pages/load.py

'''
The Load page: read one .tdms file, a folder of them (optionally with subfolders), or a
manifest (.csv / .xlsx) listing the files.

Session state this page writes, for the other pages:
    source          the loaded LoadLIBSData (None until something is loaded)
    load_messages   warnings raised while loading `source`
'''

import os
import subprocess
import sys
import warnings
from pathlib import Path

import streamlit as st

from libs_viewer.libs import LoadLIBSData
from libs_viewer.libs.plan import MANIFEST_SUFFIXES


# ------------------------------------------------------------------ helpers
def load(path: Path, recursive: bool, folder: Path | None = None) -> tuple[LoadLIBSData, list[str]]:
    '''Load a file, folder or manifest; return the data and any warnings raised while loading.'''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if path.suffix.lower() in MANIFEST_SUFFIXES:
            data = LoadLIBSData(folder, manifest=path)
        else:
            data = LoadLIBSData(path, recursive=recursive)
    return data, [str(w.message) for w in caught]


def clean(text: str | None) -> Path | None:
    '''A path typed or pasted by the user ("Copy as path" adds quotes), or None if blank.'''
    text = (text or "").strip().strip('"')
    return Path(text) if text else None


# The Windows dialog runs in its own short-lived Python process: tkinter must own the main
# thread, and Streamlit runs pages in worker threads (Tk there can crash the server).
_DIALOG = r'''
import sys, tkinter as tk
from tkinter import filedialog
kind, start = sys.argv[1], sys.argv[2] or None
root = tk.Tk()
root.withdraw()                          # no empty Tk window
root.wm_attributes("-topmost", 1)        # dialog in front of the browser
if kind == "folder":
    chosen = filedialog.askdirectory(parent=root, initialdir=start)
else:
    chosen = filedialog.askopenfilename(parent=root, initialdir=start, filetypes=[
        ("TDMS files or manifests", "*.tdms *.csv *.xlsx *.xlsm"), ("All files", "*.*")])
root.destroy()
print(chosen or "", end="")
'''


def pick(kind: str, start: Path | None = None) -> str:
    '''
    Open the Windows file ("file") or folder ("folder") dialog, starting at `start`, and
    return the chosen path ("" if cancelled). The dialog opens on the computer running the
    app, so this only makes sense while the app runs locally.
    '''
    if start is not None and not start.is_dir():
        start = start.parent
    result = subprocess.run([sys.executable, "-c", _DIALOG, kind, str(start or "")],
                            capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return result.stdout.strip()


def browse(kind: str, key: str) -> None:
    '''Button callback: put the chosen path into the text box with this key.'''
    chosen = pick(kind, clean(st.session_state.get(key)))
    if chosen:
        st.session_state[key] = str(Path(chosen))     # Windows-style backslashes


# ------------------------------------------------------------------ header
left, right = st.columns([5, 1], vertical_alignment="bottom")
left.title("Load data")
with right.popover("How loading works", icon=":material/info:", type="tertiary"):
    st.markdown(
        "**Three ways to load**\n"
        "- **A .tdms file**: just that file.\n"
        "- **A folder**: every .tdms file in it. Tick *Include subfolders* to look deeper. "
        "Files that can't be read are skipped with a warning.\n"
        "- **A manifest** (.csv, .xlsx, .xlsm): exactly the files it lists, with your labels. "
        "A file that can't be read stops the load with an error.\n\n"
        "**Manifests**\n\n"
        "One row per file. Only the `file` column is required; every other column "
        "(sample, temperature, notes, ...) is kept as a label. Column names ignore case. "
        "Paths can be absolute, or relative to the *Data folder* (default: the manifest's folder). "
        "A study also needs `element`, `sample` and `temperature`."
    )
    st.dataframe(
        {"file": ["Ce/1_wt%/Ce1_300C.tdms", "Ce/1_wt%/Ce1_350C.tdms", r"D:\other\Gd_300.tdms"],
         "element": ["Ce", "Ce", "Gd"], "sample": ["1 wt%", "1 wt%", "1 wt%"],
         "temperature": [300, 350, 300]},
        hide_index=True,
    )
    st.caption("Your .tdms files are only read, never changed.")

st.caption("Load a .tdms file, a folder of them, or a manifest that lists them.")

# ------------------------------------------------------------------ inputs
# What the path box holds decides which of the other inputs apply.
path = clean(st.session_state.get("load_path"))
is_manifest = path is not None and path.suffix.lower() in MANIFEST_SUFFIXES
is_file = path is not None and path.suffix.lower() == ".tdms"

with st.container(border=True):
    c1, c2, c3 = st.columns([6, 1, 1], vertical_alignment="bottom")
    c1.text_input("File, folder or manifest", key="load_path", persist_state="session",
                  placeholder=r"C:\data\Ce   or   C:\data\Ce\manifest.xlsx")
    c2.button("File", icon=":material/description:", width="stretch",
              on_click=browse, args=("file", "load_path"), help="Pick a .tdms file or a manifest")
    c3.button("Folder", icon=":material/folder_open:", width="stretch",
              on_click=browse, args=("folder", "load_path"), help="Pick a folder of .tdms files")

    c1, c2, _ = st.columns([6, 1, 1], vertical_alignment="bottom")
    c1.text_input("Data folder (manifests only)", key="load_folder", persist_state="session",
                  disabled=not is_manifest, placeholder="Default: the manifest's folder",
                  help="The folder that relative paths in the manifest start from.")
    c2.button("Folder", key="browse_data_folder", icon=":material/folder_open:", width="stretch",
              on_click=browse, args=("folder", "load_folder"), disabled=not is_manifest)

    st.checkbox("Include subfolders", key="load_recursive", persist_state="session",
                disabled=is_manifest or is_file, help="Only used when loading a folder.")
    submitted = st.button("Load", type="primary", icon=":material/upload_file:")

# ------------------------------------------------------------------ load
if submitted:
    if path is None:
        st.warning("Enter or pick a file, folder or manifest first.")
    else:
        folder = clean(st.session_state.get("load_folder")) if is_manifest else None
        recursive = st.session_state.get("load_recursive", False)
        try:
            with st.spinner("Loading..."):
                data, messages = load(path, recursive, folder)
        except (OSError, ValueError) as e:
            st.error(str(e), title="Could not load")
        else:
            if len(data) == 0:
                st.warning(f"No .tdms files found in {path}."
                           + ("" if recursive else " Tick 'Include subfolders' to look deeper."))
            else:                                    # data and its warnings are saved together
                st.session_state["source"] = data
                st.session_state["load_messages"] = messages

# ------------------------------------------------------------------ what is loaded
source = st.session_state.get("source")
if source is not None:
    st.subheader("Currently loaded")
    if source.manifest is not None:
        where = f"manifest {source.manifest}" + (f" (data in {source.path})" if source.path else "")
    else:
        where = str(source.path)
    st.caption(f"{len(source)} file(s) from {where}.")

    messages = st.session_state.get("load_messages", [])
    if messages:
        with st.expander(f"{len(messages)} warning(s) while loading", icon=":material/warning:"):
            for m in messages:
                st.warning(m)

    st.dataframe(source.overview(), alt="One row per loaded file with its labels and key settings")
    st.page_link("app_pages/inspect.py", label="Look at the files on the Inspect page",
                 icon=":material/arrow_forward:")
