# ui/app.py

'''
This file contains the logic and code for running the streamlit front-end UI app. 

For now, this serves as the primary user interfacte; however, this will likely move into a separate TypeScript front-end in the near future.

This serves as a GUI for inspecting and interacting with LIBS data, and soon to be ICP-MS data. 
'''

import streamlit as st
import pandas as pd

# Don't need this block, holding though just in case. 
# from app_pages.load import load_page
# from app_pages.inspect import inspect_page
# from app_pages.study import study_page
# from app_pages.nist_tables import nist_tables_page
# from app_pages.settings import settings_page

# ------------------------------
# Creating the runtime/page navigation stuff
# ------------------------------

st.set_page_config(page_title="LIBS Data Explorer", page_icon=":bar_chart:", layout="wide")
st.session_state.setdefault("source", None)

pg = st.navigation(
    [
        st.Page(page="app_pages/load.py", title="Load"),
        st.Page(page="app_pages/inspect.py", title="Inspect"),
        st.Page(page="app_pages/study.py", title="Study"),
        st.Page(page="app_pages/nist_tables.py", title="NIST Tables"),
        st.Page(page="app_pages/settings.py", title="Settings"),
    ],
    position="top",
)

pg.run()