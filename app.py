import io
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tempfile

import tensorflow as tf

from utils.model_io import load_keras_model, get_model_input_size, preprocess_for_model
from utils.gradcam import compute_gradcam, overlay_heatmap, list_conv_like_layers
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
    def infer_input_size_from_model(model):
        """
        Try several Keras/TF attributes to extract (H, W, C).
        Returns tuple (H, W, C) or raises ValueError.
        """
        candidates = []

        # model.input_shape (legacy)
        if hasattr(model, "input_shape") and model.input_shape is not None:
            candidates.append(model.input_shape)

        # model.inputs -> tf.Tensor shapes
        if hasattr(model, "inputs"):
            try:
                for inp in model.inputs:
                    # TensorShape -> list
                    shape = getattr(inp, "shape", None)
                    if shape is not None:
                        try:
                            candidates.append(tuple(shape.as_list()))
                        except Exception:
                            # shape may already be tuple-like
                            candidates.append(tuple(shape))
            except Exception:
                pass

        # Search for InputLayer among layers
        if hasattr(model, "layers"):
            for layer in getattr(model, "layers", []):
                try:
                    if layer.__class__.__name__ == "InputLayer":
                        shp = getattr(layer, "input_shape", None)
                        if shp is not None:
                            candidates.append(shp)
                except Exception:
                    continue

        # Normalize candidates and pick first complete (H,W,C) with numeric dims
        for s in candidates:
            if s is None:
                continue
            s_list = list(s)
            # remove batch dim if present
            if len(s_list) == 4:
                _, h, w, c = s_list
            elif len(s_list) == 3:
                h, w, c = s_list
            else:
                continue
            if h is None or w is None:
                continue
            try:
                return int(h), int(w), int(c)
            except Exception:
                continue

        raise ValueError("Unable to infer (H,W,C) from model attributes")

    try:
        try:
            target_h, target_w, channels = get_model_input_size(model)
        except Exception:
            # fallback inference from model internals
            target_h, target_w, channels = infer_input_size_from_model(model)
    except Exception as e:
        # Ask user to provide input size manually instead of failing
        st.warning(f"Couldn't infer model input size automatically: {e}")
        st.info("Please enter the model's expected input height, width, and channels.")
        target_h = int(st.number_input("Input height (pixels)", min_value=1, value=224))
        target_w = int(st.number_input("Input width (pixels)", min_value=1, value=224))
        channels = int(st.selectbox("Channels", options=[1, 3], index=1))
        # Continue without st.stop()

    # Resize uploaded image to model input size so heatmap and image align
    try:
        pil_img = pil_img.resize((int(target_w), int(target_h)), resample=Image.BILINEAR)
    except Exception:
        # If resize fails for any reason, continue with the original image
        pass

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
            # If no conv layer found, offer user a list of candidate layers to pick from
            msg = str(e)
            if "No convolutional layer found" in msg or "list_conv_like_layers" in msg:
                candidates = list_conv_like_layers(model)
                if candidates:
                    sel = st.selectbox("Select layer to use for Grad-CAM", options=candidates)
                    try:
                        heatmap = compute_gradcam(model, x, class_index=top_idx, layer_name=sel)
                    except Exception as e2:
                        st.warning(f"Grad-CAM failed with selected layer: {e2}")
                        heatmap = None
                else:
                    st.warning("No candidate conv-like layers found in the model; skipping Grad-CAM.")
                    heatmap = None
            else:
                st.warning(f"Grad-CAM failed: {e}")
                heatmap = None

    with col_left:
        st.subheader("Original Image")
        st.image(pil_img, use_container_width=True)

    with col_right:
        st.subheader("Grad-CAM Overlay")
        if heatmap is not None:
            overlay = overlay_heatmap(pil_img, heatmap, alpha=0.45)
            st.image(overlay, use_container_width=True)
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
