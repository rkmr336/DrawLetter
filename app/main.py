"""
main.py — FastAPI application for DrawLetter (Air-Writing).

Endpoints
---------
GET  /                  → serves public/index.html
GET  /camera            → 308 permanent redirect to "/"
GET  /api/health        → {"status":"ok","model_loaded":true,"test_accuracy":...}
POST /api/predict       → {"prediction":"A","confidence":0.97,"top3":[...],"processed_preview":"..."}
"""

import json
import pathlib
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
PUBLIC_DIR = ROOT / "public"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("drawletter")

_inference = None

def get_inference():
    global _inference
    if _inference is None:
        import app.inference as inf
        _inference = inf
    return _inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    inf = get_inference()
    try:
        inf._load()
        logger.info("ONNX session loaded successfully.")
    except Exception as e:
        logger.warning(f"ONNX model not loaded at startup (will load when available): {e}")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="DrawLetter",
    description="Recognizes air-written letters A-Z using fingertip tracking and a CNN trained on EMNIST.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB limit

class PredictRequest(BaseModel):
    image: str
    include_preview: bool = True

    @field_validator("image")
    @classmethod
    def check_size(cls, v: str) -> str:
        if len(v) > MAX_IMAGE_BYTES * 1.4:
            raise ValueError("Image payload is too large (max 2 MB raw)")
        if not v.startswith("data:image"):
            raise ValueError("image must be a base64 data URL (data:image/...)")
        return v

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = PUBLIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))

@app.get("/camera", include_in_schema=False)
async def camera_redirect():
    return RedirectResponse(url="/", status_code=308)

@app.get("/api/health")
async def health():
    inf = get_inference()
    loaded = inf.is_model_loaded()
    test_acc = None
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            test_acc = metrics.get("test_accuracy")
        except Exception:
            pass
    return {
        "status": "ok",
        "model_loaded": loaded,
        "test_accuracy": test_acc,
    }

@app.post("/api/predict")
async def predict(req: PredictRequest):
    inf = get_inference()
    try:
        result = inf.predict(req.image, include_preview=req.include_preview)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error")

app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
