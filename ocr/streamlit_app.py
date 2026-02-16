import json
import os
import streamlit as st
import numpy as np
from PIL import Image

from main import run_ocr_on_image

st.set_page_config(
    page_title="Offline OCR for Industrial / Stenciled Text",
    layout="wide"
)

st.title("📦 Offline OCR for Industrial / Stenciled Text")

uploaded_file = st.file_uploader(
    "Upload an industrial / stenciled text image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=450)

    with col2:
        st.subheader("📝 Detected Text")

        with st.spinner("Running offline OCR..."):
            text_lines, latency, accuracy = run_ocr_on_image(image_np)

        if text_lines:
            for line in text_lines:
                st.write(f"• {line}")
        else:
            st.warning(
                "Text detected but confidence was too low for strict extraction."
            )

        st.markdown("---")
        st.subheader("📊 OCR Metrics")
        st.write(f"⏱ **Latency:** {latency} ms")
        st.write(f"🎯 **Extraction Accuracy:** {accuracy} %")

        # -------------------------------------------------
        # Save results (append to JSON)
        # -------------------------------------------------
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/ocr_results.json"

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.append({
            "image_name": uploaded_file.name,
            "detected_text": text_lines,
            "latency_ms": latency,
            "extraction_accuracy": accuracy
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
# kiro , anti gravity ,ml