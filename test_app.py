import streamlit as st
from PIL import Image
import io

from streamlit_app import analyze_image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Interactive image test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    if st.button("Analyze"):
        st.session_state["result"] = analyze_image(
            image=image,
            minimum_area=20,
            length_scale=None,
            formulas=[],
        )
        st.session_state["image"] = image

    if "result" in st.session_state:
        result = st.session_state["result"]

        st.write("Analysis finished")
        st.write("Particles:", result["particle_count"])

        click = streamlit_image_coordinates(
            result["annotated_image"],
            key="test_coordinates",
        )

        st.write("Interactive image finished")
        st.write(click)
