import streamlit as st
from PIL import Image
import io

from analyzer_core import (
    segment_particles,
    find_particles,
    particle_measurements,
    annotate_particles,
)

st.title("Measurement test")

uploaded = st.file_uploader("Upload image")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    if st.button("Test measurements"):
        mask = segment_particles(image)
        particles = find_particles(mask, 20)

        st.write("Particles found:", len(particles))

        for particle in particles:
            particle_measurements(particle, None)

        st.write("Measurements finished")

        annotated = annotate_particles(image, particles)
        st.image(annotated)

        st.write("Annotation finished")
        
