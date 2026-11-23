# ui/app_pages/study.py

'''
This is the study/quick study page for the streamlit app
'''

import streamlit as st

st.title("Study Page")
source = st.session_state.get("source")
st.write("This section allows you to perform studies on the data. We utilize the settings that you have chosen on the Inspect page." \
" If you want to change the settings, go back to the Inspect page and select a different baseline method or adjust the parameters.")

recipe = st.session_state.get("recipe")
if recipe is None:
    st.info("No study recipe found. Please go to the Inspect page and set up a study.", icon=":material/info:")
    st.stop()

st.write(f"Using the following recipe: {recipe}")

st.session_state["recipe"] = {"baseline": method, "baseline_params": params}
# Add functionality for studying data here