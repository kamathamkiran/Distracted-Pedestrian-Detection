# app.py
import streamlit as st
import tempfile
import os
from inference import run_inference

st.title("📱 Pedestrian Distraction Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Run Detection"):
        with st.spinner("Processing...",show_time = True):
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_file.read())
            temp_input.close()

            output_path = run_inference(temp_input.name)

        st.success("✅ Detection complete!")

        if os.path.exists("output.mp4"):
            st.video("output.mp4")
        else:
            st.error("output.mp4 not found")
            
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="output.mp4",
                mime="video/mp4"
            )
