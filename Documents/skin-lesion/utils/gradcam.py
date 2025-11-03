from __future__ import annotations
import io
import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    # Try to find the last 2D conv-like layer
    for layer in reversed(model.layers):
        name = layer.name.lower()
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in name or "sepconv" in name or "mobilenet" in name:
            return layer
        # Fallback: any layer with 4D output (batch, h, w, c)
        try:
            out_shape = layer.output_shape
            if out_shape is not None and len(out_shape) == 4:
                return layer
        except Exception:
            pass
    raise ValueError("No suitable conv layer found for Grad-CAM.")


def compute_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the top or specified class.

    Args:
        model: Keras model.
        img_batch: Preprocessed batch of shape (1, H, W, C).
        layer_name: Specific layer name to use for CAM; if None, auto-detect last conv.
        class_index: Target class index; if None, uses argmax of model output.

    Returns:
        Heatmap as float32 array in [0, 1] with shape (H, W).
    """
    if img_batch.ndim != 4 or img_batch.shape[0] != 1:
        raise ValueError("img_batch must be shape (1, H, W, C)")

    # Identify conv layer
    conv_layer = model.get_layer(layer_name) if layer_name else _find_last_conv_layer(model)

    # Build a model mapping input to conv outputs + predictions
    grad_model = Model([model.inputs], [conv_layer.output, model.outputs])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    # Compute gradients of the class score wrt conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute gradients for Grad-CAM.")

    # Global average pooling the gradients over width and height
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)
    pooled_grads = pooled_grads[0]  # (C,)

    # Weight the channels by corresponding gradients
    conv_outputs = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-8)

    # Resize heatmap to input image size
    H, W = img_batch.shape[1:3]
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (H, W))
    heatmap = tf.squeeze(heatmap).numpy().astype("float32")
    return heatmap


def overlay_heatmap(pil_img, heatmap: np.ndarray, alpha: float = 0.4):
    import cv2
    from PIL import Image

    img = np.array(pil_img)
    if heatmap.ndim == 3 and heatmap.shape[-1] == 1:
        heatmap = heatmap[..., 0]
    heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype("uint8")
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colored + (1 - alpha) * img).astype("uint8")
    return Image.fromarray(overlay)
