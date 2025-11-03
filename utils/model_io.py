from __future__ import annotations
from typing import Tuple

import numpy as np
from PIL import Image
import tensorflow as tf


def load_keras_model(path: str) -> tf.keras.Model:
    # Load without compiling to avoid missing optimizer/loss issues
    model = tf.keras.models.load_model(path, compile=False)
    return model


def get_model_input_size(model: tf.keras.Model) -> Tuple[int, int, int]:
    """Return (H, W, C) expected by the model input."""
    shape = model.input_shape
    # If multiple inputs, take the first
    if isinstance(shape, (list, tuple)):
        shape = shape[0]
    if shape is None:
        raise ValueError("Model input_shape is None")
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input shape (None, H, W, C), got {shape}")
    _, H, W, C = shape
    if None in (H, W, C):
        raise ValueError(f"Dynamic input dims not supported for preprocessing: {shape}")
    return int(H), int(W), int(C)


def preprocess_for_model(pil_img: Image.Image, target_hw: Tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    img = pil_img.resize((W, H))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    # Simple normalization to [0,1]; adjust here if your model expects something else
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")
    return arr
