"""
api/index.py — Robust Vercel serverless entrypoint for DrawLetter.
Provides debug fallback if model loading or imports encounter platform issues.
"""

import sys
import pathlib
import traceback

ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.main import app
except Exception as e:
    # If app failed to import at cold start, create a lightweight fallback FastAPI
    # app so we can see the exact error in /api/health instead of generic FUNCTION_INVOCATION_FAILED 500
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="DrawLetter Fallback")
    _err_msg = str(e)
    _err_tb = traceback.format_exc()
    
    @app.get("/api/health")
    async def fallback_health():
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": _err_msg,
                "traceback": _err_tb,
                "hint": "Check Vercel function dependencies or model path."
            }
        )

    @app.post("/api/predict")
    async def fallback_predict():
        return JSONResponse(
            status_code=500,
            content={"detail": f"Backend failed to load on Vercel: {_err_msg}"}
        )
