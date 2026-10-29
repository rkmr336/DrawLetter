"""
inference.py — Preprocessing + ONNX inference for DrawLetter air-writing.

Preprocessing:
1. Accept base64 PNG data URL (white strokes on black background; inverted if mostly bright).
2. Convert to grayscale, crop to bounding box of strokes with padding.
3. Resize so longer side is 20 px keeping aspect ratio, paste into 28x28 black canvas,
   center by center of mass.
4. Apply slight Gaussian blur (EMNIST-style anti-aliasing).
5. Normalize with (x - MEAN) / STD and apply the exact orientation fix used in training.
6. If empty, raise ValueError.
"""

import base64
import io
import json
import pathlib
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
MODEL_PATH = ROOT / "model" / "model.onnx"
LABELS_PATH = ROOT / "model" / "labels.json"

MEAN = 0.1736
STD = 0.3317
IMG_SIZE = 28

def _match_model_orientation(img: Image.Image) -> Image.Image:
    """
    EMNIST raw images in torchvision are rotated 90 deg and flipped horizontally.
    The training script applied `img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)`.
    Mathematically, this equals `img.transpose(Image.TRANSPOSE).rotate(180)`.
    Because user air-drawn strokes are already upright, mapping upright drawings to the
    exact tensor orientation expected by model.onnx requires rotating by 180 degrees.
    """
    return img.rotate(180)

_session: Optional[ort.InferenceSession] = None
_labels: Optional[list] = None

def _load():
    global _session, _labels
    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    if _labels is None:
        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
        with open(str(LABELS_PATH)) as f:
            _labels = json.load(f)
    return _session, _labels

def is_model_loaded() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False

def _decode_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGBA")

def _normalize_to_white_on_black(img_gray: Image.Image) -> Image.Image:
    arr = np.array(img_gray, dtype=np.float32)
    border_px = np.concatenate([
        arr[:5, :].ravel(), arr[-5:, :].ravel(),
        arr[:, :5].ravel(), arr[:, -5:].ravel(),
    ])
    if border_px.mean() > 128:
        arr = 255.0 - arr
    return Image.fromarray(arr.astype(np.uint8))

def _crop_to_bbox(img_gray: Image.Image, padding: int = 4) -> Image.Image:
    arr = np.array(img_gray)
    rows = np.any(arr > 15, axis=1)
    cols = np.any(arr > 15, axis=0)
    if not rows.any() or not cols.any():
        return img_gray

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - padding)
    rmax = min(arr.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(arr.shape[1] - 1, cmax + padding)

    return Image.fromarray(arr[rmin:rmax+1, cmin:cmax+1])

def _fit_into_20px(img_gray: Image.Image) -> Image.Image:
    w, h = img_gray.size
    max_side = max(w, h)
    if max_side <= 0:
        return img_gray
    scale = 20.0 / max_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img_gray.resize((new_w, new_h), Image.LANCZOS)

def _center_by_mass(small: Image.Image, canvas_size: int = 28) -> Image.Image:
    arr = np.array(small, dtype=np.float32)
    total_mass = arr.sum()
    if total_mass > 1e-4:
        cy = int(round(np.average(np.arange(arr.shape[0]), weights=arr.mean(axis=1) + 1e-8)))
        cx = int(round(np.average(np.arange(arr.shape[1]), weights=arr.mean(axis=0) + 1e-8)))
    else:
        cy = small.size[1] // 2
        cx = small.size[0] // 2

    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    offset_x = canvas_size // 2 - cx
    offset_y = canvas_size // 2 - cy
    canvas.paste(small, (offset_x, offset_y))
    return canvas

def preprocess(data_url: str) -> tuple[np.ndarray, Image.Image]:
    rgba = _decode_data_url(data_url)
    bg = Image.new("RGB", rgba.size, (0, 0, 0))
    bg.paste(rgba, mask=rgba.split()[3])
    grey = bg.convert("L")

    grey = _normalize_to_white_on_black(grey)
    arr_check = np.array(grey)
    if (arr_check > 25).sum() < 10:
        raise ValueError("Canvas is empty — please write a letter first.")

    grey = _crop_to_bbox(grey, padding=4)
    grey = _fit_into_20px(grey)
    grey = _center_by_mass(grey, canvas_size=IMG_SIZE)
    grey = grey.filter(ImageFilter.GaussianBlur(radius=0.5))

    preview = grey.copy()
    grey = _match_model_orientation(grey)

    arr = np.array(grey, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = arr[np.newaxis, np.newaxis, :, :]
    return tensor, preview

def _preview_to_base64(preview: Image.Image) -> str:
    arr = np.array(preview, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    img8 = Image.fromarray(arr.astype(np.uint8)).resize((112, 112), Image.NEAREST)
    buf = io.BytesIO()
    img8.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def predict(data_url: str, include_preview: bool = True) -> dict:
    tensor, preview = preprocess(data_url)
    session, labels = _load()

    logits = session.run(["output"], {"input": tensor})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    top_indices = np.argsort(probs)[::-1]
    top3 = [
        {"label": labels[i], "prob": round(float(probs[i]), 6)}
        for i in top_indices[:3]
    ]

    result = {
        "prediction": top3[0]["label"],
        "confidence": top3[0]["prob"],
        "top3": top3,
    }

    if include_preview:
        result["processed_preview"] = _preview_to_base64(preview)

    return result
