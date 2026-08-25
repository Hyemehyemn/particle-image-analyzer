import streamlit as st
from PIL import Image
import io

from analyzer_core import segment_particles, find_particles

st.title("Particle finding test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    st.image(image)

    if st.button("Test find particles"):
        st.write("Starting...")
        mask = segment_particles(image)
        particles = find_particles(mask, 20)
        st.write("Finished")
        st.write("Particles found:", len(particles))
