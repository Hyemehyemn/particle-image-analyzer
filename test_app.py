import streamlit as st
from PIL import Image
import io

from analyzer_core import segment_particles

st.title("Segmentation test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    st.image(image)

    if st.button("Test segmentation"):
        st.write("Starting segmentation...")
        mask = segment_particles(image)
        st.write("Segmentation finished")
