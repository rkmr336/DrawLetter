"""
per_letter_accuracy.py — Comprehensive per-letter evaluation on the full EMNIST Letters test set.

Compares:
1. TRAINING PATH: Raw EMNIST -> training orientation fix -> ToTensor -> Normalize -> model.onnx
2. APP PATH: Raw EMNIST -> upright PNG data URL (as sent by browser) -> app.inference.predict -> model.onnx

Outputs:
- model/per_letter_accuracy.csv
- model/per_letter_accuracy.png
- model/confusion_matrix_app.png
- Console table, top/bottom 5 letters, overall accuracy, metrics.json comparison, and verdict.
"""

import os
import sys
import io
import json
import base64
import csv
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import EMNIST
import onnxruntime as ort

# Ensure UTF-8 output encoding for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.inference import predict

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
ONNX_PATH = MODEL_DIR / "model.onnx"
METRICS_PATH = MODEL_DIR / "metrics.json"

CSV_PATH = MODEL_DIR / "per_letter_accuracy.csv"
BAR_CHART_PATH = MODEL_DIR / "per_letter_accuracy.png"
CONF_MATRIX_PATH = MODEL_DIR / "confusion_matrix_app.png"

# Normalization constants used during training
MEAN = 0.1736
STD = 0.3317

def run_evaluation():
    print("=" * 80)
    print("EMNIST LETTERS FULL TEST SET PER-LETTER EVALUATION")
    print("=" * 80)
    
    # Load dataset
    print(f"Loading EMNIST (split='letters', train=False) from {DATA_DIR}...")
    test_dataset = EMNIST(root=str(DATA_DIR), split="letters", train=False, download=False)
    total_samples = len(test_dataset)
    print(f"Total test samples loaded: {total_samples}")

    # Load ONNX session for training path
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    letters = [chr(ord('A') + i) for i in range(26)]
    num_classes = 26

    # Data structures for stats
    # Per class counts
    class_total = [0] * num_classes
    train_correct = [0] * num_classes
    app_correct = [0] * num_classes

    # App path confusion matrix [true_idx, pred_idx]
    confusion_app = np.zeros((num_classes, num_classes), dtype=np.int64)

    print("Evaluating all test samples across TRAINING PATH and APP PATH...")
    
    # Process in batches for training path, or sample by sample
    # To keep memory clean and reliable, iterate
    for idx in range(total_samples):
        if (idx + 1) % 2000 == 0 or idx == total_samples - 1:
            print(f"  Processed {idx + 1}/{total_samples} samples ({(idx+1)/total_samples*100:.1f}%)...")

        raw_img, label = test_dataset[idx]
        true_idx = label - 1  # 0 to 25
        class_total[true_idx] += 1

        # -------------------------------------------------------------
        # 1. TRAINING PATH
        # img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        # ToTensor -> Normalize
        # -------------------------------------------------------------
        t_img = raw_img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        arr_t = np.array(t_img, dtype=np.float32) / 255.0
        arr_t = (arr_t - MEAN) / STD
        tensor_t = arr_t[np.newaxis, np.newaxis, :, :]
        logits_t = session.run(None, {input_name: tensor_t})[0]
        pred_t_idx = int(np.argmax(logits_t, axis=1)[0])
        if pred_t_idx == true_idx:
            train_correct[true_idx] += 1

        # -------------------------------------------------------------
        # 2. APP PATH
        # The browser sends upright white-on-black PNG data URL.
        # In raw EMNIST, rotating 270 and FLIP_LR produces the visually upright character.
        # We put it onto a 400x400 canvas (standard DrawLetter export size)
        # and encode as PNG data URL.
        # -------------------------------------------------------------
        upright_img = raw_img.rotate(270, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        
        # Place onto 400x400 canvas with realistic scaling
        canvas = Image.new("L", (400, 400), 0)
        # EMNIST is 28x28, resize to 280x280 (70% of 400, leaving 60px margin)
        upright_resized = upright_img.resize((280, 280), Image.Resampling.BILINEAR)
        canvas.paste(upright_resized, (60, 60))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        # Run through real app/inference.py
        result = predict(data_url, include_preview=False)
        pred_char = result["prediction"]
        pred_app_idx = ord(pred_char) - ord('A')

        confusion_app[true_idx, pred_app_idx] += 1
        if pred_app_idx == true_idx:
            app_correct[true_idx] += 1

    # Overall Metrics
    total_train_correct = sum(train_correct)
    total_app_correct = sum(app_correct)
    train_overall_acc = (total_train_correct / total_samples) * 100.0
    app_overall_acc = (total_app_correct / total_samples) * 100.0

    # Compile Table Rows
    table_rows = []
    for i in range(num_classes):
        letter = letters[i]
        total = class_total[i]
        t_corr = train_correct[i]
        a_corr = app_correct[i]
        t_acc = (t_corr / total) * 100.0 if total > 0 else 0.0
        a_acc = (a_corr / total) * 100.0 if total > 0 else 0.0

        # Find most common wrong prediction for APP path
        row_conf = confusion_app[i].copy()
        row_conf[i] = 0  # ignore correct
        most_wrong_idx = int(np.argmax(row_conf))
        most_wrong_cnt = row_conf[most_wrong_idx]
        if most_wrong_cnt > 0:
            wrong_letter = letters[most_wrong_idx]
            wrong_pct = (most_wrong_cnt / total) * 100.0
            most_common_wrong = f"{letter} -> {wrong_letter}, {wrong_pct:.1f}%"
        else:
            most_common_wrong = "None"

        table_rows.append({
            "letter": letter,
            "samples": total,
            "train_correct": t_corr,
            "train_acc": t_acc,
            "app_correct": a_corr,
            "app_acc": a_acc,
            "most_common_wrong": most_common_wrong
        })

    # Print Table
    print("\n" + "=" * 88)
    print(f"{'Letter':<8} {'Samples':<9} {'Train Correct':<15} {'Train Acc %':<13} {'App Correct':<13} {'App Acc %':<11} {'Most Common Wrong (App Path)':<30}")
    print("-" * 88)
    for row in table_rows:
        print(f"{row['letter']:<8} {row['samples']:<9} {row['train_correct']:<15} {row['train_acc']:<13.2f} {row['app_correct']:<13} {row['app_acc']:<11.2f} {row['most_common_wrong']:<30}")
    print("=" * 88)

    print(f"\nOVERALL ACCURACY:")
    print(f"  * TRAINING PATH ACCURACY : {total_train_correct}/{total_samples} ({train_overall_acc:.2f}%)")
    print(f"  * APP PATH ACCURACY      : {total_app_correct}/{total_samples} ({app_overall_acc:.2f}%)")
    gap = train_overall_acc - app_overall_acc
    print(f"  * ACCURACY GAP           : {gap:.2f}%")

    # Metrics.json comparison
    metrics_acc_str = "N/A"
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            metrics_acc_str = f"{metrics_data.get('test_accuracy', 0)*100:.2f}%"
    print(f"  * model/metrics.json ACC : {metrics_acc_str}")

    # 5 Best and 5 Worst for APP path
    sorted_by_app = sorted(table_rows, key=lambda x: x["app_acc"], reverse=True)
    best_5 = sorted_by_app[:5]
    worst_5 = sorted_by_app[-5:][::-1]

    print("\nTOP 5 BEST LETTERS (APP PATH):")
    for r in best_5:
        print(f"  {r['letter']}: {r['app_acc']:.2f}% ({r['app_correct']}/{r['samples']})")

    print("\nTOP 5 WORST LETTERS (APP PATH):")
    for r in worst_5:
        print(f"  {r['letter']}: {r['app_acc']:.2f}% ({r['app_correct']}/{r['samples']}) | Most common: {r['most_common_wrong']}")

    # Save to CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "samples", "train_correct", "train_accuracy_pct", "app_correct", "app_accuracy_pct", "most_common_wrong_app"])
        for r in table_rows:
            writer.writerow([r["letter"], r["samples"], r["train_correct"], f"{r['train_acc']:.2f}", r["app_correct"], f"{r['app_acc']:.2f}", r["most_common_wrong"]])
    print(f"\nSaved CSV table to: {CSV_PATH}")

    # Generate and save Bar Chart
    plt.figure(figsize=(14, 6), dpi=150)
    x = np.arange(len(letters))
    width = 0.38

    train_accs = [r["train_acc"] for r in table_rows]
    app_accs = [r["app_acc"] for r in table_rows]

    plt.bar(x - width/2, train_accs, width, label=f"Training Path ({train_overall_acc:.1f}%)", color="#60A5FA", alpha=0.85)
    plt.bar(x + width/2, app_accs, width, label=f"App Path ({app_overall_acc:.1f}%)", color="#FFB547", alpha=0.9)

    plt.axhline(90, color="#EF4444", linestyle="--", linewidth=1.2, label="90% Benchmark")
    plt.title("DrawLetter — Per-Letter Accuracy on EMNIST Letters Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Letter (A–Z)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, letters, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(str(BAR_CHART_PATH))
    plt.close()
    print(f"Saved Bar Chart to: {BAR_CHART_PATH}")

    # Generate and save Confusion Matrix
    plt.figure(figsize=(11, 9), dpi=150)
    # Normalize confusion matrix by row (true class)
    row_sums = confusion_app.sum(axis=1, keepdims=True)
    norm_conf = np.divide(confusion_app, row_sums, where=row_sums!=0) * 100.0

    plt.imshow(norm_conf, interpolation="nearest", cmap="Blues")
    plt.title("DrawLetter — App Path Confusion Matrix (%)", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(label="Accuracy / Misclassification Rate (%)")
    tick_marks = np.arange(len(letters))
    plt.xticks(tick_marks, letters, fontsize=9)
    plt.yticks(tick_marks, letters, fontsize=9)
    plt.xlabel("Predicted Letter", fontsize=11, fontweight="bold")
    plt.ylabel("True Letter", fontsize=11, fontweight="bold")

    # Add text labels on cells with non-trivial values
    for r in range(num_classes):
        for c in range(num_classes):
            val = norm_conf[r, c]
            if val >= 5.0:
                color = "white" if val > 50 else "black"
                plt.text(c, r, f"{val:.0f}", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(str(CONF_MATRIX_PATH))
    plt.close()
    print(f"Saved Confusion Matrix to: {CONF_MATRIX_PATH}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    if train_overall_acc >= 90.0 and app_overall_acc >= 90.0 and abs(train_overall_acc - app_overall_acc) <= 5.0:
        print(f"  [PASS] Both paths are 90%+ and closely matched (Train: {train_overall_acc:.2f}%, App: {app_overall_acc:.2f}%).")
        print("         The model and preprocessing are working with high fidelity and verified pipeline parity.")
    elif train_overall_acc >= 90.0 and (train_overall_acc - app_overall_acc) > 5.0:
        print(f"  [WARNING] Training Path is high ({train_overall_acc:.2f}%) but App Path is lower ({app_overall_acc:.2f}%).")
        print("            A preprocessing mismatch (orientation, normalization, or stroke thickness) remains.")
    else:
        print(f"  [WARNING] Both paths are below 90% (Train: {train_overall_acc:.2f}%, App: {app_overall_acc:.2f}%).")
        print("            The model appears to be under-trained.")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()

