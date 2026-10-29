"""
api/index.py — Vercel serverless entrypoint.

Vercel's Python runtime looks for a callable named `app` (ASGI/WSGI).
We simply re-export the FastAPI app from app/main.py.
"""

import sys
import pathlib

# Make sure the project root is on the path so app.main can be imported
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app   # re-export for Vercel

__all__ = ["app"]

