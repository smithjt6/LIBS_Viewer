# ui/app_pages/study.py

'''
The Study page: apply the Inspect recipe (baseline method + parameters, and optionally the
lines marked there) to the loaded manifest with `run_study`.

Reads st.session_state["source"] (Load page) and st.session_state["recipe"] (Inspect page).
Writes st.session_state["study_result"].
'''

import streamlit as st

from libs_viewer.core.manifest import labelled_rows, missing_labels
from libs_viewer.libs import run_study
from study_figures import baseline_figure, lines_figure, ratio_figure

PLOT_CONFIG = {"scrollZoom": True, "responsive": True}


st.title("Study")
st.caption("Applies the settings from the Inspect page to every file in the loaded manifest. "
           "To change them, go back to Inspect.")

source = st.session_state.get("source")
recipe = st.session_state.get("recipe")
if source is None:
    st.info("No data loaded yet.", icon=":material/info:")
    st.page_link("app_pages/load.py", label="Go to the Load page", icon=":material/upload_file:")
    st.stop()
if recipe is None:
    st.info("No recipe yet. Set the baseline on the Inspect page first.", icon=":material/info:")
    st.page_link("app_pages/inspect.py", label="Go to the Inspect page", icon=":material/search:")
    st.stop()

# Which file is which: the loaded manifest, or the table labelled on the Inspect page.
if source.manifest is not None:
    manifest = source.manifest
    study_elements = set(source.labels["element"]) if "element" in source.labels else set()
else:
    draft = st.session_state.get("draft_manifest")
    gaps = missing_labels(draft) if draft is not None else {"element": []}
    if gaps:
        st.warning("A study needs to know each file's element, sample and temperature. Label "
                   "the files on the Inspect page, or load a manifest on the Load page.",
                   icon=":material/warning:")
        st.page_link("app_pages/inspect.py", label="Label the files on the Inspect page",
                     icon=":material/label:")
        st.stop()
    manifest = labelled_rows(draft)
    study_elements = set(manifest["element"])
    st.caption(f"Using the labels from the Inspect page: {len(manifest)} file(s), "
               f"{manifest['sample'].nunique()} sample(s).")

# ------------------------------------------------------------------ the recipe
with st.container(border=True):
    st.subheader("Recipe")
    st.write(f"**Baseline:** {recipe['baseline']}  ·  "
             + ", ".join(f"{k} = {v}" for k, v in recipe["baseline_params"].items()))
    marked = recipe.get("lines", {})
    use_marked = st.checkbox(
        "Measure the lines marked on Inspect (otherwise NIST lines are picked automatically)",
        value=False, disabled=not marked, key="study_use_marked", persist_state="session")
    if marked:
        st.caption("Marked: " + "; ".join(f"{el} ({len(w)} lines)" for el, w in marked.items()))
    ref = st.number_input("Reference temperature", value=300.0, step=10.0, key="study_ref",
                          persist_state="session",
                          help="Every sample needs a file at this temperature.")

# ------------------------------------------------------------------ run
if st.button("Run study", type="primary", icon=":material/play_arrow:"):
    folder = source.root
    lines = ({el: w for el, w in marked.items() if el in study_elements}
             if use_marked and marked else None)
    bar = st.progress(0.0, text="Starting...")
    try:
        result = run_study(
            folder, manifest=manifest, baseline=recipe["baseline"],
            baseline_params=recipe["baseline_params"], lines=lines, ref=float(ref),
            progress=lambda done, total, msg: bar.progress(done / total, text=msg))
    except (OSError, ValueError) as e:
        bar.empty()
        st.error(str(e), title="The study could not be set up")
    else:
        bar.empty()
        st.session_state["study_result"] = result

# ------------------------------------------------------------------ results
result = st.session_state.get("study_result")
if result is not None:
    skipped = result.skipped_lines
    if skipped:
        st.warning(
            "**Not all of the marked lines were applied.** These had no clear peak in the "
            "sample's spectrum at the reference temperature, so they were left out of "
            "that sample (everything else was measured):\n\n"
            + "\n".join(f"- **{sample}**: {', '.join(labels)}" for sample, labels in skipped.items()),
            icon=":material/warning:")
    for note in result.notes:
        st.caption(f":material/info: {note}")
    st.download_button("Download results (.zip)", result.to_zip(), "libs_study.zip",
                       "application/zip", icon=":material/download:",
                       help="Tables, settings.json and plan.csv. plan.csv also works as a manifest.")
    tabs = st.tabs(["Figures", "R_rel", "R_direct", "Areas", "Scatter", "Lines used"])
    with tabs[0]:
        st.subheader("Ratios against temperature")
        st.caption("Each line's R_rel and the total's R_direct (grey dashed), relative to the "
                   "reference temperature.")
        log_y = st.toggle("Log scale", value=True, key="study_fig_log", persist_state="session")
        st.plotly_chart(ratio_figure(result, log_y=log_y), key="study_ratio_plot",
                        width="stretch", config=PLOT_CONFIG,
                        alt="Line ratios against temperature, one panel per sample")

        st.subheader("One spectrum")
        c1, c2 = st.columns(2)
        sample = c1.selectbox("Sample", result.samples, key="study_fig_sample")
        temps = sorted(result.spectra[sample])
        temp = c2.selectbox("Temperature", temps, key="study_fig_temp",
                            format_func=lambda t: f"{t:g} °C")
        spectrum, title = result.spectra[sample][temp], f"{sample}, {temp:g} °C"

        st.caption("**Baseline check:** the top panel shows the fitted continuum; the bottom is "
                   "what the Total area integrates.")
        st.plotly_chart(
            baseline_figure(result.wl, spectrum, result.settings["baseline"],
                            result.settings["baseline_params"], log_y=log_y, title=title),
            key="study_baseline_plot", width="stretch", config=PLOT_CONFIG,
            alt=f"Spectrum, fitted baseline and corrected spectrum for {title}")

        st.caption("**Lines measured**, marked on the spectrum. Narrow the window to read them.")
        wl_lo, wl_hi = float(result.wl.min()), float(result.wl.max())
        lo, hi = st.slider("Wavelength window (nm)", wl_lo, wl_hi, (wl_lo, wl_hi))
        st.plotly_chart(
            lines_figure(result.wl, spectrum, result.lines[sample], (lo, hi), log_y=log_y,
                         title=title),
            key="study_lines_plot", width="stretch", config=PLOT_CONFIG,
            alt=f"Spectrum for {title} with the measured lines marked")
    with tabs[1]:
        st.caption("(line ÷ total at T) ÷ (line ÷ total at reference): change relative to the whole spectrum.")
        st.dataframe(result.r_rel)
    with tabs[2]:
        st.caption("Area at T ÷ area at the reference temperature.")
        st.dataframe(result.r_direct)
    with tabs[3]:
        st.dataframe(result.areas)
    with tabs[4]:
        st.dataframe(result.scatter)
    with tabs[5]:
        for element, table in result.line_tables.items():
            st.markdown(f"**{element}**")
            st.dataframe(table)
