#!/usr/bin/env python3
"""Analyze trainslate rendering output"""
import sys, json
from PIL import Image
import numpy as np

# Source image
img = Image.open("data/comics/0002_GREECE0001.png")
print(f"Source: {img.size}, mode={img.mode}")

# Rendered page 1
rendered = Image.open("data/output/rendered/page_0001_rendered.png")
print(f"Rendered p1: {rendered.size}, mode={rendered.mode}")

# Mask page 1
mask = Image.open("data/output/rendered/page_0001_mask.png")
mask_np = np.array(mask)
print(f"Mask p1: {mask.size}, mode={mask.mode}")
print(f"Mask pixels set: {np.sum(mask_np > 0)} out of {mask_np.size} ({100*np.sum(mask_np > 0)/mask_np.size:.2f}%)")

# Compare source vs rendered p1
src = np.array(img.convert("RGB"))
ren = np.array(rendered.convert("RGB"))
diff = np.abs(src.astype(int) - ren.astype(int))
print(f"Total pixel diff p1: {np.sum(diff)}")
diff_mask = np.max(diff, axis=2) > 20
changed_pct = 100*np.sum(diff_mask)/diff_mask.size
print(f"Pixels with >20 diff p1: {np.sum(diff_mask)} ({changed_pct:.2f}%)")

# Check if diff is in the bubble area (top banner for p1)
# bbox from debug: [669, 0, 1600, 171]
bubble_region = diff_mask[0:171, 669:1600]
print(f"Bubble region (title) changed pixels: {np.sum(bubble_region)} out of {bubble_region.size} ({100*np.sum(bubble_region)/bubble_region.size:.2f}%)")

# Check page 3 rendered (has translation failures)
print("\n--- Page 3 Analysis ---")
p3 = Image.open("data/output/rendered/page_0003_rendered.png")
p3_src = Image.open("data/comics/0004_GREECE0003.png")
p3_a = np.array(p3.convert("RGB"))
p3_b = np.array(p3_src.convert("RGB"))
p3_diff = np.abs(p3_a.astype(int) - p3_b.astype(int))
p3_diff_mask = np.max(p3_diff, axis=2) > 20
p3_changed = 100*np.sum(p3_diff_mask)/p3_diff_mask.size
print(f"Page 3 vs source: {np.sum(p3_diff_mask)} pixels changed ({p3_changed:.2f}%)")

# Check bubbles on page 3
# bubble_yolo_0003: bbox [356, 139, 608, 199], region speech
# bubble_yolo_0002: bbox [1014, 371, 1493, 489], region speech (FAILED translation)
# bubble_yolo_0001: bbox [195, 614, 466, 694], region speech
b2_region = p3_diff_mask[371:489, 1014:1493]
b2_changed = 100*np.sum(b2_region)/b2_region.size if b2_region.size > 0 else 0
print(f"bubble_yolo_0002 (FAILED) region: {np.sum(b2_region)}/{b2_region.size} ({b2_changed:.2f}%)")

b1_region = p3_diff_mask[614:694, 195:466]
b1_changed = 100*np.sum(b1_region)/b1_region.size if b1_region.size > 0 else 0
print(f"bubble_yolo_0001 (OK) region: {np.sum(b1_region)}/{b1_region.size} ({b1_changed:.2f}%)")

# Check page 5
print("\n--- Page 5 Analysis ---")
p5 = Image.open("data/output/rendered/page_0005_rendered.png")
p5_src = Image.open("data/comics/0006_GREECE0005.png")
p5_a = np.array(p5.convert("RGB"))
p5_b = np.array(p5_src.convert("RGB"))
p5_diff = np.abs(p5_a.astype(int) - p5_b.astype(int))
p5_diff_mask = np.max(p5_diff, axis=2) > 20
p5_changed = 100*np.sum(p5_diff_mask)/p5_diff_mask.size
print(f"Page 5 vs source: {np.sum(p5_diff_mask)} pixels changed ({p5_changed:.2f}%)")

# bbox [1271, 272, 1585, 359] - FAILED
# bbox [1186, 624, 1567, 761] - OK
# bbox [33, 813, 339, 920] - FAILED
b_fail1 = p5_diff_mask[272:359, 1271:1585]
print(f"bubble_yolo_0001 (FAILED): {np.sum(b_fail1)}/{b_fail1.size} ({100*np.sum(b_fail1)/b_fail1.size:.2f}%)")
b_ok = p5_diff_mask[624:761, 1186:1567]
print(f"bubble_yolo_0003 (OK): {np.sum(b_ok)}/{b_ok.size} ({100*np.sum(b_ok)/b_ok.size:.2f}%)")
b_fail2 = p5_diff_mask[813:920, 33:339]
print(f"bubble_yolo_0004 (FAILED): {np.sum(b_fail2)}/{b_fail2.size} ({100*np.sum(b_fail2)/b_fail2.size:.2f}%)")

# Check the debug JSON
with open("data/output/rendered/page_0003_render_debug.json") as f:
    p3_data = json.load(f)
print(f"\nPage 3 blocks: {len(p3_data['blocks'])}")
for b in p3_data['blocks']:
    print(f"  {b['block_id']}: type={b['region_type']}, fs={b['font_size']}, lines={b['line_count']}, score={b['typography_score']}")
    print(f"    text: {b['translated_text'][:80]}")

with open("data/output/rendered/page_0005_render_debug.json") as f:
    p5_data = json.load(f)
print(f"\nPage 5 blocks: {len(p5_data['blocks'])}")
for b in p5_data['blocks']:
    print(f"  {b['block_id']}: type={b['region_type']}, fs={b['font_size']}, lines={b['line_count']}, score={b['typography_score']}")
    print(f"    text: {b['translated_text'][:80]}")