from __future__ import annotations
import io
import os
from typing import List, Optional, Tuple

import tensorflow as tf
import numpy as np


def _is_conv_like(layer) -> bool:
    """Return True for conv-like layers or layers whose output has spatial dims."""
    try:
        clsname = layer.__class__.__name__.lower()
        if "conv" in clsname or "separableconv" in clsname or "depthwiseconv" in clsname:
            return True
        out_shape = getattr(layer, "output_shape", None) or getattr(layer, "output", None)
        if out_shape is None:
            return False
        # Normalize TensorShape -> tuple
        try:
            if hasattr(out_shape, "as_list"):
                out_shape = tuple(out_shape.as_list())
            else:
                out_shape = tuple(out_shape)
        except Exception:
            return False
        # Spatial dims exist if len >= 3 (batch, spatial..., channels)
        return len(out_shape) >= 3
    except Exception:
        return False


def list_conv_like_layers(model) -> List[str]:
    """
    Return list of layer names that are conv-like or have spatial outputs.
    Useful to let a user pick a layer for Grad-CAM when automatic selection fails.
    """
    names = []
    for layer in getattr(model, "layers", []):
        try:
            if _is_conv_like(layer):
                names.append(layer.name)
        except Exception:
            continue
    return names


def compute_gradcam(model, input_tensor, class_index: Optional[int] = None, layer_name: Optional[str] = None):
    """
    Compute Grad-CAM heatmap for a single input.
    - model: tf.keras.Model
    - input_tensor: np.ndarray or tf.Tensor shaped (1, H, W, C)
    - class_index: int index of target class. If None, uses model argmax.
    - layer_name: optional layer name to use as target conv layer.
    Returns: 2D numpy array heatmap normalized to [0,1] with shape (H, W)
    """
    if isinstance(input_tensor, np.ndarray):
        x = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    else:
        x = tf.cast(input_tensor, dtype=tf.float32)

    # Choose conv layer
    target_layer = None
    if layer_name:
        try:
            target_layer = model.get_layer(name=layer_name)
        except Exception as e:
            raise ValueError(f"Requested layer '{layer_name}' not found: {e}")
    else:
        # find last conv-like layer
        for layer in reversed(getattr(model, "layers", [])):
            if _is_conv_like(layer):
                target_layer = layer
                break

    if target_layer is None:
        raise ValueError("No convolutional layer found in model for Grad-CAM. "
                         "Call list_conv_like_layers(model) to see candidate layers.")

    try:
        conv_output = target_layer.output
    except Exception as e:
        raise ValueError(f"Couldn't access output of layer '{target_layer.name}': {e}")

    # Use first model output tensor if model.outputs is list/tuple
    model_output = model.outputs[0] if isinstance(model.outputs, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_output, model_output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x)
        preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

        preds_tensor = tf.convert_to_tensor(preds_tensor)

        if class_index is None:
            class_index = int(tf.argmax(preds_tensor[0]).numpy())

        # Select score for target class
        if preds_tensor.shape.rank == 0:
            score = preds_tensor
        elif preds_tensor.shape.rank == 1:
            # single example vector
            score = preds_tensor[class_index]
            score = tf.expand_dims(score, 0)
        else:
            # batch x classes
            score = preds_tensor[:, class_index]

        tape.watch(conv_outputs)
        loss = tf.reduce_sum(score)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None; check the model and inputs for compatibility.")

    # Pool gradients to get weights
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted sum of feature maps
    cam = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.expand_dims(weights, 1), 1), conv_outputs), axis=-1)
    cam = tf.nn.relu(cam)

    # Normalize
    cam_min = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
    cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    denom = cam_max - cam_min
    denom = tf.where(denom == 0, tf.ones_like(denom), denom)
    cam_norm = (cam - cam_min) / denom

    # Resize to input spatial dims
    input_shape = x.shape
    if len(input_shape) < 3:
        raise ValueError("Input tensor has unexpected shape for Grad-CAM.")
    target_h = int(input_shape[1])
    target_w = int(input_shape[2])
    cam_resized = tf.image.resize(tf.expand_dims(cam_norm, axis=-1), (target_h, target_w), method="bilinear")
    heatmap = tf.squeeze(cam_resized[0]).numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)
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
