"""
tests/test_api.py — API integration tests for DrawLetter air-writing app.
"""

import base64
import io
import pathlib
import sys
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app

client = TestClient(app)

def make_test_letter_b64(letter: str = "A", size: int = 400) -> str:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    # Draw simple lines simulating letter strokes
    if letter == "A":
        draw.line([(80, 320), (200, 80)], fill=(255, 255, 255, 255), width=24)
        draw.line([(200, 80), (320, 320)], fill=(255, 255, 255, 255), width=24)
        draw.line([(140, 220), (260, 220)], fill=(255, 255, 255, 255), width=24)
    elif letter == "L":
        draw.line([(100, 80), (100, 320)], fill=(255, 255, 255, 255), width=24)
        draw.line([(100, 320), (300, 320)], fill=(255, 255, 255, 255), width=24)
    else:
        draw.line([(120, 80), (280, 80)], fill=(255, 255, 255, 255), width=24)
        draw.line([(200, 80), (200, 320)], fill=(255, 255, 255, 255), width=24)
        draw.line([(120, 320), (280, 320)], fill=(255, 255, 255, 255), width=24)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

class TestRoutes:
    def test_root_serves_html_with_drawletter(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "DrawLetter" in resp.text
        assert "Air Writing" in resp.text
        assert "hand_landmarker" not in resp.text or "script" in resp.text

    def test_camera_redirects_to_root(self):
        resp = client.get("/camera", follow_redirects=False)
        assert resp.status_code == 308
        assert resp.headers["location"] == "/"

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_predict_rejects_invalid_body(self):
        resp = client.post("/api/predict", json={"image": "not_a_data_url"})
        assert resp.status_code == 422

    def test_predict_rejects_empty_canvas(self):
        # All black image
        blank = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        buf = io.BytesIO()
        blank.save(buf, format="PNG")
        b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        resp = client.post("/api/predict", json={"image": b64})
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    def test_predict_rejects_oversized_payload(self):
        huge = "data:image/png;base64," + "A" * (4 * 1024 * 1024)
        resp = client.post("/api/predict", json={"image": huge})
        assert resp.status_code == 422

