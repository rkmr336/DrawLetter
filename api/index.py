"""
api/index.py — Vercel Serverless Function Entrypoint for DrawLetter.
Explicit top-level assignment for Vercel's AST parser: `app = FastAPI(...)`
"""

import sys
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# Add project root to sys.path
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Direct top-level assignment for Vercel
app = FastAPI(
    title="DrawLetter",
    description="Recognizes air-written letters A-Z using a CNN trained on EMNIST.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_IMAGE_BYTES = 2 * 1024 * 1024

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

@app.get("/api/health")
async def health():
    try:
        from app.inference import is_model_loaded
        loaded = is_model_loaded()
        return {
            "status": "ok",
            "model_loaded": loaded,
            "test_accuracy": 0.9342
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )

@app.post("/api/predict")
async def predict_route(req: PredictRequest):
    try:
        from app.inference import predict
        result = predict(req.image, include_preview=req.include_preview)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Inference error: {str(e)}"})
