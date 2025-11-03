# Skin Lesion Analyzer (Keras + Grad-CAM + Gemini)

This Streamlit app lets you:
- Upload a Keras `.h5/.keras` model and a dermoscopic image
- Run prediction and visualize Grad-CAM hotspots
- Optionally ask Gemini to synthesize a narrative using your symptoms + the image

## Setup

1) (Optional) Create and activate a virtual environment

PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Set Gemini API key
```
$env:GEMINI_API_KEY = "{{GEMINI_API_KEY}}"
```

4) Run
```
streamlit run app.py
```

Quick start:
1) pip install -r requirements.txt
2) In PowerShell: $env:GEMINI_API_KEY="{{GEMINI_API_KEY}}"
3) streamlit run app.py
If your model needs custom preprocessing or specific class labels, tell me and I’ll wire that in.

## Notes
- Your model's expected input size is inferred from `model.input_shape` and images are normalized to [0,1]. Adjust `preprocess_for_model` if your model needs different preprocessing.
- Grad-CAM auto-selects the last conv-like layer. You can customize in `utils/gradcam.py`.
- If your model uses custom layers/losses, ensure they are importable before loading.
