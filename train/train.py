"""
train.py — Full EMNIST Letters training script.

CONFIG constants at the top let you switch between splits without touching the rest.
Trains a small CNN, saves the best checkpoint, exports to ONNX, verifies the
ONNX output matches PyTorch output, and writes model/labels.json and model/metrics.json.
"""

import os
import sys
import json
import random
import datetime
import pathlib

# Ensure UTF-8 output encoding for console on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — change EMNIST_SPLIT to "balanced" to switch datasets later
# ---------------------------------------------------------------------------
EMNIST_SPLIT = "letters"          # "letters" (26 classes) | "balanced" (47 classes)
DATA_DIR     = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR    = pathlib.Path(__file__).parent.parent / "model"
CHECKPOINT   = pathlib.Path(__file__).parent / "best_model.pt"

SEED         = 42
EPOCHS       = 18
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0          # 0 is safest on Windows
IMG_SIZE     = 28

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Derived label config
# ---------------------------------------------------------------------------
if EMNIST_SPLIT == "letters":
    NUM_CLASSES = 26
    # EMNIST letters labels are 1..26 — we shift to 0..25
    LABEL_OFFSET = 1
    LABEL_NAMES  = [chr(ord('A') + i) for i in range(26)]
elif EMNIST_SPLIT == "balanced":
    NUM_CLASSES  = 47
    LABEL_OFFSET = 0
    # balanced: 0-9 digits, then upper+lower merged letters
    digits = [str(i) for i in range(10)]
    upper  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    extra  = list("abdefghnqrt")   # 11 extra lower-case that differ from upper
    LABEL_NAMES = digits + upper + extra
else:
    raise ValueError(f"Unknown EMNIST split: {EMNIST_SPLIT}")

# ---------------------------------------------------------------------------
# EMNIST orientation fix:
# EMNIST images are stored transposed + flipped compared with normal MNIST
# orientation.  We undo that so letters appear upright.
# The same fix MUST be applied at inference time.
# ---------------------------------------------------------------------------
def emnist_orientation_fix(img):
    """Rotate 90° CCW then flip horizontally to make letters upright."""
    import PIL.Image
    img = img.rotate(90, expand=True)
    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
MEAN = (0.1307,)
STD  = (0.3081,)

train_transform = T.Compose([
    T.Lambda(emnist_orientation_fix),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

test_transform = T.Compose([
    T.Lambda(emnist_orientation_fix),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
class LabelShiftDataset(torch.utils.data.Dataset):
    """Wraps a torchvision EMNIST dataset and shifts labels by LABEL_OFFSET."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label - LABEL_OFFSET


def load_emnist(split, transform, train):
    """Download / load EMNIST and wrap with label shifting."""
    raw = torchvision.datasets.EMNIST(
        root=str(DATA_DIR),
        split=split,
        train=train,
        download=True,
        transform=transform,
    )
    return LabelShiftDataset(raw)


# ---------------------------------------------------------------------------
# Model — small CNN <10 MB
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),   # 14x14

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),   # 7x7

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),   # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# ONNX export + verification
# ---------------------------------------------------------------------------
def export_onnx(model, onnx_path):
    model.eval()
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


def verify_onnx(model, onnx_path):
    import onnxruntime as ort
    model.eval()
    dummy = torch.randn(4, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"input": dummy.cpu().numpy()})[0]

    match = np.allclose(pt_out, ort_out, atol=1e-4)
    print(f"[verify] PyTorch vs ONNX outputs match: {match}")
    if not match:
        max_diff = np.abs(pt_out - ort_out).max()
        print(f"[verify] WARNING — max diff = {max_diff:.6f}")
    return match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("[data] Loading EMNIST …")
    train_ds = load_emnist(EMNIST_SPLIT, train_transform, train=True)
    test_ds  = load_emnist(EMNIST_SPLIT, test_transform,  train=False)

    # Use a small validation split from the training set
    val_size   = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    print(f"[data] train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Model
    model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion)

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "va_loss": va_loss, "va_acc": va_acc})
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": va_acc}, str(CHECKPOINT))
            print(f"  [+] New best val acc {best_val_acc:.4f} - checkpoint saved")

    # Load best checkpoint for export
    ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n[final] Best checkpoint: epoch {ckpt['epoch']}, val acc={ckpt['val_acc']:.4f}")

    # Test accuracy
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"[final] Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Export ONNX
    onnx_path = MODEL_DIR / "model.onnx"
    export_onnx(model, onnx_path)
    verify_onnx(model, onnx_path)

    onnx_mb = onnx_path.stat().st_size / 1e6
    print(f"[export] ONNX size: {onnx_mb:.2f} MB")

    # Save label map
    labels_path = MODEL_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(LABEL_NAMES, f)
    print(f"[export] Labels saved → {labels_path}")

    # Save metrics
    metrics = {
        "emnist_split": EMNIST_SPLIT,
        "num_classes": NUM_CLASSES,
        "test_accuracy": round(test_acc, 6),
        "best_val_accuracy": round(best_val_acc, 6),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "onnx_size_mb": round(onnx_mb, 3),
        "total_params": total_params,
        "date": datetime.datetime.now().isoformat(),
        "history": history,
    }
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[export] Metrics saved → {metrics_path}")

    print("\n=== TRAINING COMPLETE ===")
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print(f"  ONNX model    : {onnx_path}")
    print(f"  Labels        : {labels_path}")
    print(f"  Metrics       : {metrics_path}")


if __name__ == "__main__":
    main()

