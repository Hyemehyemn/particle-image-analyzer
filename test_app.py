import streamlit as st

from streamlit_app import (
    initialize_state,
    render_calibration_panel,
    render_formula_panel,
    render_decimal_panel,
)

st.set_page_config(page_title="Test", layout="wide")

initialize_state()

st.title("Test")

render_calibration_panel()
render_formula_panel()
render_decimal_panel()

st.write("Reached the end")
