from __future__ import annotations
import os
from typing import Optional

import google.generativeai as genai


def _configure():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=api_key)


def gemini_analyze(image_bytes: bytes, symptoms: str = "", model_name: str = "gemini-1.5-flash") -> str:
    """Send symptoms + image to Gemini and return the response text."""
    _configure()

    prompt = (
        "You are a helpful clinical assistant. The user provided a dermoscopic image of a skin lesion"
        " and some symptoms/notes. Summarize key visual findings, relate them to the symptoms,"
        " and suggest differential considerations and next steps. DO NOT provide a diagnosis;"
        " instead, discuss likelihoods and recommend seeing a dermatologist."
    )

    image_part = {"mime_type": "image/png", "data": image_bytes}

    model = genai.GenerativeModel(model_name)

    parts = [prompt]
    if symptoms:
        parts.append(f"Symptoms/notes: {symptoms}")
    parts.append(image_part)

    resp = model.generate_content(parts)
    return getattr(resp, "text", "(No response text)")
