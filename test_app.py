import streamlit as st
from PIL import Image
import io

from analyzer_core import segment_particles, find_particles

st.title("Memory test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

    if st.button("Analyze"):
        mask = segment_particles(image)
        particles = find_particles(mask, 20)

        st.write("Particles:", len(particles))
        st.write("Finished")
