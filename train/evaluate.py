"""
evaluate.py — Evaluate the trained EMNIST model and save a confusion matrix PNG.

Usage (from project root):
    python train/evaluate.py
"""

import sys
import json
import pathlib

# Ensure UTF-8 output encoding for console on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so script can be run from anywhere
# ---------------------------------------------------------------------------
HERE      = pathlib.Path(__file__).parent
ROOT      = HERE.parent
MODEL_DIR = ROOT / "model"
DATA_DIR  = ROOT / "data"

sys.path.insert(0, str(HERE))
from train import (
    SmallCNN, LabelShiftDataset, emnist_orientation_fix,
    EMNIST_SPLIT, NUM_CLASSES, LABEL_OFFSET, LABEL_NAMES,
    BATCH_SIZE, CHECKPOINT, DEVICE, MEAN, STD,
)

# ---------------------------------------------------------------------------
# Load the best checkpoint
# ---------------------------------------------------------------------------
print(f"[eval] Loading checkpoint: {CHECKPOINT}")
model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
ckpt  = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------
test_transform = T.Compose([
    T.Lambda(emnist_orientation_fix),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

raw_test = torchvision.datasets.EMNIST(
    root=str(DATA_DIR), split=EMNIST_SPLIT, train=False,
    download=True, transform=test_transform,
)
test_ds = LabelShiftDataset(raw_test)
loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
all_preds  = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
acc = accuracy_score(all_labels, all_preds)
print(f"\n[eval] Test accuracy: {acc*100:.2f}%\n")
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
cm     = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(LABEL_NAMES, fontsize=9)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"EMNIST '{EMNIST_SPLIT}' Confusion Matrix  (acc={acc*100:.2f}%)", fontsize=13)
plt.tight_layout()
out_png = HERE / "confusion_matrix.png"
fig.savefig(str(out_png), dpi=120)
print(f"[eval] Confusion matrix saved → {out_png}")

# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------
print("\n[eval] Per-class accuracy:")
for i, name in enumerate(LABEL_NAMES):
    mask     = (all_labels == i)
    cls_acc  = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
    print(f"  {name}: {cls_acc*100:.1f}%  (n={mask.sum()})")

