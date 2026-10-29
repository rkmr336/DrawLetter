# DrawLetter

A full-stack AI web app that recognizes handwritten letters (A–Z) drawn on a canvas. Built with a small CNN trained on EMNIST, exported to ONNX, and served via FastAPI.

## Architecture

```
Browser (Canvas) ──POST base64 PNG──► FastAPI ──► ONNX Runtime ──► JSON response
                                        │
                                   Static files
                                  (HTML/CSS/JS)
```

- **Training**: PyTorch CNN trained on EMNIST "letters" dataset, exported to ONNX (< 10 MB)
- **Backend**: FastAPI + uvicorn + onnxruntime (no PyTorch at runtime)
- **Frontend**: Plain HTML/CSS/Vanilla JS — no build step, works on mobile

## Folder Structure

```
drawletter/
├── README.md
├── requirements.txt          # Runtime: fastapi, uvicorn, onnxruntime, numpy, pillow
├── requirements-train.txt    # Training: torch, torchvision, sklearn, onnx, …
├── vercel.json               # Vercel routing config (prepared, not deployed)
├── .gitignore
├── .env.example
├── train/
│   ├── train.py              # Full training script (downloads EMNIST, trains, exports ONNX)
│   ├── evaluate.py           # Accuracy, confusion matrix PNG, per-class report
│   └── export_onnx.py        # Standalone ONNX re-export from checkpoint
├── model/
│   ├── model.onnx            # Trained model (ONNX, < 10 MB)
│   ├── labels.json           # ["A","B",…,"Z"]
│   └── metrics.json          # test_accuracy, epochs, date, …
├── app/
│   ├── main.py               # FastAPI app
│   └── inference.py          # Preprocessing + ONNX inference
├── api/
│   └── index.py              # Vercel entrypoint
├── public/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── tests/
    └── test_inference.py     # pytest tests
```

## Setup

### Windows (PowerShell)

```powershell
# Clone / enter the project
cd "drawletter"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) Install training dependencies
pip install -r requirements-train.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Linux / macOS

```bash
cd drawletter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional training deps
pip install -r requirements-train.txt
# For CUDA GPU:
pip install torch torchvision
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## How to Train

> ⚠️ Training downloads ~560 MB of EMNIST data. First run takes 5–10 min on CPU.

```powershell
# From project root (with venv activated)
python train/train.py
```

This will:
1. Download the EMNIST "letters" dataset to `data/`
2. Train 18 epochs with data augmentation
3. Save the best checkpoint to `train/best_model.pt`
4. Export `model/model.onnx` and verify it matches PyTorch output
5. Write `model/labels.json` and `model/metrics.json`

### Evaluate / Confusion Matrix

```powershell
python train/evaluate.py
```

Outputs a confusion matrix PNG to `train/confusion_matrix.png` and a per-class accuracy table.

### Re-export ONNX Only

```powershell
python train/export_onnx.py
```

## How to Run Locally

```powershell
# From project root (with venv activated)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000 in your browser, allow camera access, and write a letter in the air.

### "Write Inside the Box" Guide
- Draw letters inside the centered guide box for optimal spatial centering and accurate feature extraction.
- **Tip**: Draw letters large to fill most of the box for the highest recognition accuracy.
- If your fingertip moves outside the box while pointing, the app pauses stroke capture and prompts `"Move your finger inside the box"` until you return inside.
- Gestures:
  - ☝️ **Index finger up**: Pen down (draws stroke)
  - ✌️ **Peace sign**: Pen up (hover / move between strokes)
  - 🖐️ **Open palm (hold 1s)**: Clear canvas (works anywhere in frame)

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Enter` | Predict |
| `Escape` | Clear canvas |

## API Reference

### `GET /api/health`

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{"status": "ok", "model_loaded": true, "test_accuracy": 0.9342}
```

### `POST /api/predict`

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,<BASE64>", "include_preview": true}'
```

```json
{
  "prediction": "A",
  "confidence": 0.974231,
  "top3": [
    {"label": "A", "prob": 0.974231},
    {"label": "R", "prob": 0.018462},
    {"label": "V", "prob": 0.004012}
  ],
  "processed_preview": "data:image/png;base64,..."
}
```

**Error response (empty canvas):**
```json
{"detail": "Canvas is empty — please draw a letter first."}
```

## Switching to EMNIST "balanced" (Letters + Digits)

The split is controlled by a single constant at the top of `train/train.py`:

```python
EMNIST_SPLIT = "letters"   # ← change to "balanced" for 47 classes
```

The balanced split includes digits 0–9 and 26 upper-case letters plus 11 merged lower-case letters (47 classes total). After changing, re-run training and the rest of the pipeline adapts automatically.

## Known Limitations

- **Confusion between similar letters**: I/L, U/V, O/0 are the most common errors
- **Stroke thickness matters**: Very thin strokes may not be detected well
- **Single character only**: The app recognizes one letter per prediction
- **No GPU at inference time**: ONNX Runtime uses CPU; inference is ~10–30 ms
- **PIL font ≠ handwriting**: The test synthetic letters may not exactly match real handwriting, so test accuracy on EMNIST is higher than real-world accuracy

## Deploy to Vercel (Later)

> 📌 The app is Vercel-ready but **not yet deployed**. The ONNX model is committed in the repo.

### Steps

1. Push the project to GitHub:
   ```bash
   git init && git add . && git commit -m "initial"
   git remote add origin https://github.com/your-user/drawletter.git
   git push -u origin main
   ```

2. Import the repo at https://vercel.com/new

3. Set these settings in Vercel:
   - **Framework Preset**: Other
   - **Root Directory**: `.` (project root)
   - **Build Command**: *(leave empty)*
   - **Output Directory**: `public`

4. Or deploy via CLI:
   ```bash
   npm i -g vercel
   vercel --prod
   ```

### Common Troubleshooting

| Problem | Fix |
|---------|-----|
| `model.onnx` not found | Make sure it's committed (not in `.gitignore`) |
| Bundle too large | `requirements.txt` must NOT include `torch` or `torchvision` |
| 500 on `/api/predict` | Check Vercel function logs; ONNX Runtime needs `onnxruntime` not `onnxruntime-gpu` |
| CORS errors | Add your Vercel domain to `allow_origins` in `app/main.py` |
| Import errors | Ensure `api/index.py` correctly adds the project root to `sys.path` |

## Quick-start Commands

```powershell
# After cloning, if model already exists:
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# If you need to retrain from scratch:
.\.venv\Scripts\Activate.ps1
python train/train.py      # ~20-60 min on CPU, <5 min on GPU
python train/evaluate.py   # prints accuracy, saves confusion matrix
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Run tests
.\.venv\Scripts\pytest tests/ -v
```
