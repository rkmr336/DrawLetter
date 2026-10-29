"""
export_onnx.py — Standalone ONNX export script.

Can be used to re-export the ONNX from an existing checkpoint without retraining.

Usage:
    python train/export_onnx.py
"""

import sys
import pathlib
import json
import numpy as np
import torch

HERE      = pathlib.Path(__file__).parent
ROOT      = HERE.parent
MODEL_DIR = ROOT / "model"

sys.path.insert(0, str(HERE))
from train import (
    SmallCNN, NUM_CLASSES, LABEL_NAMES, CHECKPOINT, DEVICE, IMG_SIZE
)

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
print(f"[export] Loading checkpoint: {CHECKPOINT}")
model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
ckpt  = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------------------------------------------------------------------------
# Export ONNX
# ---------------------------------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
onnx_path = MODEL_DIR / "model.onnx"

dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    do_constant_folding=True,
)
print(f"[export] ONNX saved → {onnx_path}")

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
import onnxruntime as ort
test_input = torch.randn(4, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
with torch.no_grad():
    pt_out = model(test_input).cpu().numpy()

sess    = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ort_out = sess.run(["output"], {"input": test_input.cpu().numpy()})[0]

match = np.allclose(pt_out, ort_out, atol=1e-4)
print(f"[verify] PyTorch vs ONNX match: {match}")
print(f"[export] ONNX size: {onnx_path.stat().st_size / 1e6:.2f} MB")

# ---------------------------------------------------------------------------
# Save labels
# ---------------------------------------------------------------------------
labels_path = MODEL_DIR / "labels.json"
with open(labels_path, "w") as f:
    json.dump(LABEL_NAMES, f)
print(f"[export] Labels saved → {labels_path}")

