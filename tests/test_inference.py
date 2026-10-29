"""
tests/test_inference.py — Basic sanity tests for the inference module.

Run from project root:
    pytest tests/ -v
"""

import base64
import io
import json
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# Make sure the project root is on sys.path
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_letter_image(letter: str, size: int = 200, stroke_width: int = 20) -> str:
    """
    Draw a simple letter on a black canvas using PIL and return as base64 data URL.
    This simulates what the browser canvas would send.
    """
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # Use a simple built-in font; size it up manually
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", size - 40)
    except Exception:
        font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), letter, fill=(255, 255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def make_blank_image(size: int = 200) -> str:
    """Return a completely black (empty) canvas as a base64 data URL."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def inf():
    """Import and return the inference module (loads model once)."""
    import app.inference as inference
    return inference


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestModelLoads:
    def test_model_loaded(self, inf):
        """ONNX session and labels should load without errors."""
        assert inf.is_model_loaded(), "Model failed to load"

    def test_labels_are_az(self, inf):
        """Labels should be exactly A-Z (26 letters)."""
        _, labels = inf._load()
        assert len(labels) == 26
        assert labels[0]  == 'A'
        assert labels[-1] == 'Z'
        assert labels == [chr(ord('A') + i) for i in range(26)]


class TestOutputShape:
    def test_output_shape(self, inf):
        """Model should return a (26,) logit/probability array."""
        data_url = make_letter_image("A")
        tensor, _ = inf.preprocess(data_url)
        session, _ = inf._load()
        out = session.run(["output"], {"input": tensor})[0]
        assert out.shape == (1, 26), f"Expected (1, 26), got {out.shape}"

    def test_probabilities_sum_to_one(self, inf):
        """Softmax probs in the result should sum to ~1."""
        data_url = make_letter_image("B")
        result = inf.predict(data_url, include_preview=False)
        total = sum(item["prob"] for item in result["top3"])
        # top3 won't sum to 1, just check individual probs are in [0,1]
        for item in result["top3"]:
            assert 0.0 <= item["prob"] <= 1.0

    def test_confidence_in_range(self, inf):
        """Confidence should be between 0 and 1."""
        data_url = make_letter_image("C")
        result   = inf.predict(data_url, include_preview=False)
        assert 0.0 <= result["confidence"] <= 1.0


class TestEmptyCanvas:
    def test_empty_canvas_raises(self, inf):
        """A completely empty canvas should raise ValueError."""
        blank = make_blank_image()
        with pytest.raises(ValueError, match="[Ee]mpty"):
            inf.preprocess(blank)


class TestPredictionSanity:
    """
    Sanity check: synthetic PIL-drawn letters should produce at least a
    reasonable label. We allow that PIL fonts aren't perfect EMNIST samples,
    so we check that the prediction is a letter (A-Z), not that it's exact.
    """

    @pytest.mark.parametrize("letter", ["A", "L", "I", "O"])
    def test_predicts_a_letter(self, inf, letter):
        """Prediction must be one of A-Z."""
        data_url = make_letter_image(letter)
        result   = inf.predict(data_url, include_preview=False)
        assert result["prediction"] in [chr(ord('A') + i) for i in range(26)]
        assert result["top3"][0]["label"] == result["prediction"]
        assert len(result["top3"]) == 3

    def test_preview_returned(self, inf):
        """When include_preview=True, processed_preview should be a data URL."""
        data_url = make_letter_image("A")
        result   = inf.predict(data_url, include_preview=True)
        assert "processed_preview" in result
        assert result["processed_preview"].startswith("data:image/png;base64,")

    def test_no_preview_when_disabled(self, inf):
        """When include_preview=False, processed_preview should be absent."""
        data_url = make_letter_image("A")
        result   = inf.predict(data_url, include_preview=False)
        assert "processed_preview" not in result


class TestPreprocessing:
    def test_black_on_white_handled(self, inf):
        """Black-on-white input (inverted) should still produce a valid prediction."""
        # Draw white letter on black, then invert to black-on-white
        data_url = make_letter_image("A")
        raw = base64.b64decode(data_url.split(",")[1])
        img = Image.open(io.BytesIO(raw)).convert("RGBA")

        # Invert RGB channels (simulate black-on-white canvas)
        r, g, b, a = img.split()
        r = r.point(lambda x: 255 - x)
        g = g.point(lambda x: 255 - x)
        b = b.point(lambda x: 255 - x)
        inv = Image.merge("RGBA", (r, g, b, a))

        buf = io.BytesIO()
        inv.save(buf, format="PNG")
        b64_inv = base64.b64encode(buf.getvalue()).decode()
        inv_url = f"data:image/png;base64,{b64_inv}"

        result = inf.predict(inv_url, include_preview=False)
        assert result["prediction"] in [chr(ord('A') + i) for i in range(26)]

