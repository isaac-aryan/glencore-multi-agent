import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Findings", layout="wide")
st.title("📋 Research Findings")
st.caption("Running log from reports/findings.md — updated after each analysis stage.")
st.divider()

findings_path = Path("reports/findings.md")
if findings_path.exists():
    st.markdown(findings_path.read_text())
else:
    st.warning("reports/findings.md not found. Run the analysis notebooks and add findings.")