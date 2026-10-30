"""
api/index.py — Debug entrypoint to return exact Vercel failure details
"""

import sys
import os
import pathlib
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="DrawLetter Diagnostics")

@app.get("/api/health")
async def health():
    diagnostics = {
        "status": "checking",
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "files_in_root": os.listdir(".") if os.path.exists(".") else []
    }
    
    # Check 1: Test importing numpy
    try:
        import numpy as np
        diagnostics["numpy"] = f"OK (v{np.__version__})"
    except Exception as e:
        diagnostics["numpy_error"] = str(e)

    # Check 2: Test importing onnxruntime
    try:
        import onnxruntime as ort
        diagnostics["onnxruntime"] = f"OK (v{ort.__version__})"
    except Exception as e:
        diagnostics["onnxruntime_error"] = f"{type(e).__name__}: {str(e)}"
        diagnostics["onnxruntime_traceback"] = traceback.format_exc()

    # Check 3: Test loading model
    try:
        from app.inference import is_model_loaded
        diagnostics["model_loaded"] = is_model_loaded()
    except Exception as e:
        diagnostics["model_error"] = str(e)
        diagnostics["model_traceback"] = traceback.format_exc()

    return JSONResponse(content=diagnostics)

@app.post("/api/predict")
async def predict_placeholder():
    return JSONResponse(status_code=503, content={"detail": "Diagnostics active"})
