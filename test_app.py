import streamlit as st
from PIL import Image
import io

from streamlit_app import analyze_image, dataframe_for_display

st.title("Dataframe test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    if st.button("Analyze"):
        result = analyze_image(
            image=image,
            minimum_area=20,
            length_scale=None,
            formulas=[],
        )

        st.write("Analysis finished")
        st.write("Particles:", result["particle_count"])

        df = dataframe_for_display(
            result["rows"],
            result["calibrated"],
            []
        )

        st.write("Dataframe created")

        st.dataframe(
            df,
            width="stretch",
            height=420,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        st.write("Dataframe finished")
