import io
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tempfile

import tensorflow as tf

from utils.model_io import load_keras_model, get_model_input_size, preprocess_for_model
from utils.gradcam import compute_gradcam, overlay_heatmap
from utils.gemini import gemini_analyze

st.set_page_config(page_title="Skin Lesion Analyzer", layout="wide")

st.title("Skin Lesion Analyzer: Model + Grad-CAM + Gemini")

st.markdown(
    "Upload a Keras .h5 model and a dermoscopic image. We'll run prediction, visualize Grad-CAM hotspots, and optionally ask Gemini for a narrative using your symptoms + image."
)

with st.sidebar:
    st.header("Inputs")
    model_file = st.file_uploader("Upload .h5/.keras model", type=["h5", "keras"], accept_multiple_files=False)
    image_file = st.file_uploader("Upload lesion image", type=["png", "jpg", "jpeg"])
    symptoms = st.text_area("Symptoms / clinical notes (optional for Gemini)")
    use_gemini = st.checkbox("Use Gemini analysis", value=False)
    gemini_model_name = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    run_btn = st.button("Analyze")

col_left, col_right = st.columns([1, 1])

if run_btn:
    if not model_file or not image_file:
        st.error("Please upload both a model and an image.")
        st.stop()

    with st.spinner("Loading model..."):
        # Persist uploaded model to a temp path for tf.keras to load
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.name)[1]) as tmp:
            tmp.write(model_file.read())
            model_path = tmp.name
        try:
            model = load_keras_model(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
        finally:
            # Best-effort cleanup later by OS; keeping file for potential lazy loading assets
            pass

    # Load image
    try:
        pil_img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    # Determine expected input size
    try:
        target_h, target_w, channels = get_model_input_size(model)
    except Exception as e:
        st.error(f"Couldn't infer model input size: {e}")
        st.stop()

    with st.spinner("Preprocessing & predicting..."):
        x = preprocess_for_model(pil_img, (target_h, target_w))
        try:
            preds = model.predict(x, verbose=0)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Interpret predictions generically
    probs = preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else np.array(preds)
    if probs.ndim == 0:
        probs = np.array([probs])
    if probs.ndim == 1:
        top_idx = int(np.argmax(probs))
        top_score = float(np.max(probs))
    else:
        # If output is more complex, fallback to argmax over last axis
        top_idx = int(np.argmax(probs))
        top_score = float(np.max(probs))

    with st.spinner("Computing Grad-CAM..."):
        try:
            heatmap = compute_gradcam(model, x, class_index=top_idx)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
            heatmap = None

    with col_left:
        st.subheader("Original Image")
        st.image(pil_img, use_column_width=True)

    with col_right:
        st.subheader("Grad-CAM Overlay")
        if heatmap is not None:
            overlay = overlay_heatmap(pil_img, heatmap, alpha=0.45)
            st.image(overlay, use_column_width=True)
        else:
            st.info("No heatmap available.")

    st.markdown("---")
    st.subheader("Model Output")
    st.write({"top_class_index": top_idx, "top_score": round(top_score, 4), "raw_output_shape": list(np.array(preds).shape)})

    if use_gemini:
        api_key_set = os.getenv("GEMINI_API_KEY") is not None
        if not api_key_set:
            st.warning("Set GEMINI_API_KEY environment variable to enable Gemini analysis.")
        else:
            with st.spinner("Querying Gemini..."):
                try:
                    # Convert image to bytes for Gemini
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    report = gemini_analyze(img_bytes, symptoms=symptoms.strip(), model_name=gemini_model_name)
                    st.subheader("Gemini Analysis")
                    st.write(report)
                except Exception as e:
                    st.error(f"Gemini request failed: {e}")
