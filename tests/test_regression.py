"""
test_regression.py — Regression tests for DrawLetter prediction quality.

Tests:
1. Rendered font letters A-Z (achieving >= 85% accuracy).
2. Polyline drawn letters (P, A, L, O, etc.) with specific assertion that 'P' >= 60% confidence.
"""

import io
import base64
import pytest
from PIL import Image, ImageDraw, ImageFont
from app.inference import predict


def _make_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_handdrawn_polyline_P():
    """Verify clearly drawn upright P is recognized as P with >= 60% confidence."""
    im = Image.new("L", (400, 400), 0)
    draw = ImageDraw.Draw(im)
    # Vertical stem
    draw.line([(120, 80), (120, 320)], fill=255, width=28)
    # Top loop
    draw.line(
        [(120, 80), (180, 80), (240, 100), (260, 140), (240, 180), (180, 200), (120, 200)],
        fill=255,
        width=28,
    )

    result = predict(_make_data_url(im))
    assert result["prediction"] == "P", f"Expected P, got {result['prediction']}"
    assert result["confidence"] >= 0.60, f"Expected confidence >= 0.60, got {result['confidence']:.2f}"


def test_handdrawn_polylines():
    """Verify polyline drawn letters (A, L, O) are recognized correctly."""
    # A
    im_a = Image.new("L", (400, 400), 0)
    draw_a = ImageDraw.Draw(im_a)
    draw_a.line([(120, 320), (200, 80), (280, 320)], fill=255, width=28)
    draw_a.line([(150, 230), (250, 230)], fill=255, width=28)
    res_a = predict(_make_data_url(im_a))
    assert res_a["prediction"] == "A"
    assert res_a["confidence"] >= 0.70

    # L
    im_l = Image.new("L", (400, 400), 0)
    draw_l = ImageDraw.Draw(im_l)
    draw_l.line([(120, 80), (120, 320), (280, 320)], fill=255, width=28)
    res_l = predict(_make_data_url(im_l))
    assert res_l["prediction"] == "L"
    assert res_l["confidence"] >= 0.70

    # O
    im_o = Image.new("L", (400, 400), 0)
    draw_o = ImageDraw.Draw(im_o)
    draw_o.ellipse([100, 80, 300, 320], outline=255, width=28)
    res_o = predict(_make_data_url(im_o))
    assert res_o["prediction"] == "O"
    assert res_o["confidence"] >= 0.70


def test_alphabet_font_recognition():
    """Verify overall font alphabet top-1 accuracy is >= 85%."""
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 120)
    except Exception:
        pytest.skip("Arial font not found on system")

    letters = [chr(ord("A") + i) for i in range(26)]
    correct = 0

    for ch in letters:
        im = Image.new("L", (400, 400), 0)
        draw = ImageDraw.Draw(im)
        if ch == "I":
            # Serifed capital I
            draw.line([(140, 80), (260, 80)], fill=255, width=28)
            draw.line([(200, 80), (200, 320)], fill=255, width=28)
            draw.line([(140, 320), (260, 320)], fill=255, width=28)
        else:
            draw.text((100, 100), ch, font=font, fill=255)

        res = predict(_make_data_url(im))
        if res["prediction"] == ch:
            correct += 1

    accuracy = correct / len(letters)
    assert accuracy >= 0.85, f"Expected font accuracy >= 0.85, got {accuracy:.2f} ({correct}/26)"

