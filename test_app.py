import streamlit as st
from PIL import Image
import io

from streamlit_app import analyze_image

st.title("Full analysis test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    st.image(image)

    if st.button("Analyze"):
        result = analyze_image(
            image=image,
            minimum_area=20,
            length_scale=None,
            formulas=[],
        )

        st.write("Analysis finished")
        st.write("Particles:", result["particle_count"])
        st.image(result["annotated_image"])
