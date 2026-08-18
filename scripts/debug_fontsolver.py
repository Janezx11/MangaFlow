#!/usr/bin/env python3
"""Debug FontSizeSolver to understand why it returns font_size=12 for speech bubbles"""
import sys
sys.path.insert(0, "G:/trainslate/src")

import numpy as np
from loguru import logger
logger.remove()

from comic_translator.services.font_manager import FontManager
from comic_translator.services.bubble_layout_service import BubbleLayoutService
from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
from comic_translator.services.padding_solver import PaddingSolver
from comic_translator.services.font_size_solver import FontSizeSolver

fm = FontManager()
bl = BubbleLayoutService()
sl = ShapeLayoutSolver(bl)
brk = BalancedLineBreaker(fm)
ps = PaddingSolver()
fss = FontSizeSolver(fm, brk, sl, ps)

# Simulate a speech bubble from page 2 (bubble_yolo_0002, bbox ~221,116,555,203)
# Create a simple oval-ish polygon
poly = [
    (230, 120), (260, 116), (300, 116), (340, 117), (380, 119),
    (420, 122), (450, 128), (470, 136), (490, 148), (500, 160),
    (510, 172), (515, 184), (518, 196), (520, 203),  # bottom right
    (510, 203), (490, 203), (460, 203), (430, 203), (400, 203),
    (370, 203), (340, 203), (310, 203), (280, 203), (250, 203),
    (230, 200), (220, 195), (215, 188), (210, 180), (208, 170),
    (206, 160), (205, 150), (206, 140), (210, 130), (220, 124),
]

img_shape = (1200, 1600)
profile = fm.get_profile_by_name("speech")
font_path = profile.font_path
min_size = profile.min_size   # 12
max_size = profile.max_size   # 64
line_spacing = profile.line_spacing  # 1.3

# Compute safe area
safe = bl.safe_area(poly, img_shape, min_distance=0)
print(f"Safe area max_inscribed_rect: {safe.max_inscribed_rect}")
print(f"Safe area bbox: {safe.bbox}")

x1, y1, x2, y2 = safe.max_inscribed_rect
box_h = y2 - y1
box_w = x2 - x1
print(f"Box: {box_w}x{box_h}")

pad = ps.compute(safe, max_size)
print(f"Padding (at max_size={max_size}): x={pad.x}, y={pad.y}")
avail_h = box_h - pad.y * 2
avail_w = box_w - pad.x * 2
print(f"Available: {avail_w}x{avail_h}")

text = "噢宝贝！我简——直不敢相信这地方看起来有多棒！"
print(f"\nText: '{text}' ({len(text)} chars)")

# Simulate the solver loop
print("\n--- Solver iterations ---")
min_size_found = min_size
for size in range(max_size, min_size - 1, -2):
    _, line_h = fm.measure_text("A", font_path, size)
    if line_h <= 0:
        continue
    line_h_spaced = int(line_h * line_spacing)
    max_lines = max(1, avail_h // line_h_spaced)
    
    per_line_widths = sl.per_line_max_widths(
        poly, y1 + pad.y, max_lines, line_h_spaced, img_shape, pad.x
    )
    
    lines, cost = brk.break_text(text, per_line_widths, font_path, size)
    if not lines:
        print(f"  size={size}: NO LINES (line_h={line_h}, spaced={line_h_spaced}, max_lines={max_lines}, widths={per_line_widths})")
        continue
    
    total_h = len(lines) * line_h_spaced
    overflows = total_h > avail_h
    
    if overflows:
        print(f"  size={size}: OVERFLOW total_h={total_h} > avail_h={avail_h} (lines={len(lines)}, widths={per_line_widths})")
        print(f"    lines: {lines}")
        continue
    
    # Found a working size!
    size_score = (size - min_size) / max(max_size - min_size, 1)
    cost_score = max(0, 1.0 - cost / 100.0)
    score = size_score * 0.7 + cost_score * 0.3
    
    print(f"  size={size}: ✓ FITS! lines={len(lines)}, total_h={total_h}, score={score:.3f}, widths={per_line_widths}")
    for i, l in enumerate(lines):
        lw, _ = fm.measure_text(l, font_path, size)
        maxw = per_line_widths[i] if i < len(per_line_widths) else 0
        print(f"    line {i}: '{l}' ({len(l)} chars, w={lw}, max_w={maxw})")
    
    min_size_found = size
    break

print(f"\nResult: font_size={min_size_found}")

# Also test with the actual per_line_widths based on polygon shape
print("\n\n--- Testing per_line_widths at various Y positions ---")
for y_off in range(0, 80, 10):
    y = y1 + pad.y + y_off
    w = bl.line_width_at_y(poly, y, img_shape)
    print(f"  Y={y} (offset={y_off}): width={w}")