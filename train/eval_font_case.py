"""
eval_font_case.py — Evaluates DejaVuSans font in Capital vs Small (lowercase) across multiple sizes.
"""
import io
import sys
import base64
import pathlib
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.inference import predict

# Locate DejaVuSans font
font_path = None
for f in fm.fontManager.ttflist:
    if 'dejavusans.ttf' in f.fname.lower():
        font_path = f.fname
        break
if not font_path:
    for f in fm.fontManager.ttflist:
        if 'dejavu' in f.name.lower() and 'sans' in f.name.lower() and 'oblique' not in f.name.lower():
            font_path = f.fname
            break

print(f"Using font: {font_path}")

sizes = [80, 110, 140]
letters = [chr(ord('A') + i) for i in range(26)]

cap_per_letter = {ch: 0 for ch in letters}
small_per_letter = {ch: 0 for ch in letters}
wrong_small_details = {}

for size in sizes:
    font = ImageFont.truetype(font_path, size)
    
    # Capital letters
    for ch in letters:
        im = Image.new('L', (400, 400), 0)
        draw = ImageDraw.Draw(im)
        draw.text((120, 100), ch, font=font, fill=255)
        
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        data_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        
        res = predict(data_url, include_preview=False)
        if res['prediction'] == ch:
            cap_per_letter[ch] += 1
            
    # Small (lowercase) letters
    for ch in letters:
        small_char = ch.lower()
        im = Image.new('L', (400, 400), 0)
        draw = ImageDraw.Draw(im)
        draw.text((120, 120), small_char, font=font, fill=255)
        
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        data_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        
        res = predict(data_url, include_preview=False)
        pred = res['prediction']
        if pred == ch:
            small_per_letter[ch] += 1
        else:
            if small_char not in wrong_small_details:
                wrong_small_details[small_char] = []
            wrong_small_details[small_char].append((size, pred, res['confidence']))

total_cap_trials = len(letters) * len(sizes)
total_cap_correct = sum(cap_per_letter.values())

total_small_trials = len(letters) * len(sizes)
total_small_correct = sum(small_per_letter.values())

print("\n" + "=" * 65)
print(f"{'Letter':<8} {'Capital Acc':<18} {'Small Acc':<18} {'Status':<15}")
print("-" * 65)
for ch in letters:
    c_acc = (cap_per_letter[ch] / len(sizes)) * 100
    s_acc = (small_per_letter[ch] / len(sizes)) * 100
    status = "Both OK" if (c_acc >= 66 and s_acc >= 66) else ("Cap Only" if c_acc >= 66 else "Weak")
    print(f"{ch} / {ch.lower():<4} {cap_per_letter[ch]}/{len(sizes)} ({c_acc:<5.1f}%)     {small_per_letter[ch]}/{len(sizes)} ({s_acc:<5.1f}%)     {status}")
print("=" * 65)

print(f"\nOVERALL FONT ACCURACY:")
print(f"  * Capital Letters : {total_cap_correct}/{total_cap_trials} ({total_cap_correct/total_cap_trials*100:.2f}%)")
print(f"  * Small Letters   : {total_small_correct}/{total_small_trials} ({total_small_correct/total_small_trials*100:.2f}%)")

print(f"\nMISPREDICTED SMALL LETTERS ({len(wrong_small_details)} letters failed at one or more sizes):")
for ch, fails in sorted(wrong_small_details.items()):
    details = ", ".join([f"size {sz} -> {pred} ({conf*100:.1f}%)" for sz, pred, conf in fails])
    print(f"  '{ch}' (expected {ch.upper()}) -> {details}")
