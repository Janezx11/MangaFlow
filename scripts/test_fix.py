#!/usr/bin/env python3
"""Test the max_inscribed_rect fix directly"""
import sys
sys.path.insert(0, "G:/trainslate/src")

import numpy as np
import cv2

# Import the fixed module
from comic_translator.services.bubble_layout_service import BubbleLayoutService
from comic_translator.models.typography_models import SafeArea

bl = BubbleLayoutService()

# Test 1: Simple rectangle polygon
print("=== Test 1: Rectangle ===")
poly = [(100, 100), (300, 100), (300, 200), (100, 200)]
img_shape = (500, 500)
safe = bl.safe_area(poly, img_shape)
print(f"Old boundingRect: (100, 100, 300, 200)")
print(f"New max_inscribed_rect: {safe.max_inscribed_rect}")
print(f"Safe bbox: {safe.bbox}")

# Test 2: Speech bubble with tail (L-shape)
print("\n=== Test 2: Bubble with tail ===")
poly2 = [(150, 100), (350, 100), (350, 250), (280, 250), (280, 300), (250, 250), (150, 250)]
img_shape = (500, 500)
safe2 = bl.safe_area(poly2, img_shape)
cnt = np.array(poly2, dtype=np.int32)
ox, oy, ow, oh = cv2.boundingRect(cnt)
old_rect = (ox, oy, ox + ow, oy + oh)
new_rect = safe2.max_inscribed_rect
print(f"Old (boundingRect): {old_rect}")
print(f"New (max_inscribed): {new_rect}")
old_area = (old_rect[2] - old_rect[0]) * (old_rect[3] - old_rect[1])
new_area = (new_rect[2] - new_rect[0]) * (new_rect[3] - new_rect[1])
print(f"Old area: {old_area}, New area: {new_area}, shrink: {(1-new_area/old_area)*100:.1f}%")
# The new rect should NOT include the tail area (y should be <= 250)
assert new_rect[3] <= 250, f"Max inscribed rect should avoid the tail! Got y2={new_rect[3]}"
print(f"✓ PASSED: Max inscribed rectangle avoids the tail (y2={new_rect[3]} <= 250)!")

# Test 3: Ellipse-like bubble
print("\n=== Test 3: Ellipse-like ===")
h, w = 400, 400
mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(mask, (200, 200), (120, 80), 0, 0, 360, 255, -1)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
poly3 = [(int(pt[0][0]), int(pt[0][1])) for pt in contours[0]]
img_shape = (h, w)
safe3 = bl.safe_area(poly3, img_shape)
cnt3 = np.array(poly3, dtype=np.int32)
ox3, oy3, ow3, oh3 = cv2.boundingRect(cnt3)
old_rect3 = (ox3, oy3, ox3 + ow3, oy3 + oh3)
new_rect3 = safe3.max_inscribed_rect
print(f"Old (boundingRect): {old_rect3}")
print(f"New (max_inscribed): {new_rect3}")
# Check that new rect is inside the ellipse
test_mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(test_mask, (200, 200), (120, 80), 0, 0, 360, 255, -1)
x1, y1, x2, y2 = new_rect3
inside = np.all(test_mask[y1:y2, x1:x2] > 0)
print(f"Rect fully inside ellipse: {inside}")
assert inside, "Max inscribed rect must be fully inside the ellipse!"
print("✓ PASSED: Rect is fully inside the ellipse!")

# Test 4: Auto min_distance
print("\n=== Test 4: Auto min_distance ===")
mask4 = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(mask4, (100, 100), 80, 255, -1)
dist = bl.distance_to_border(mask4)
max_dist = dist.max()
auto_md = bl._auto_min_distance(mask4)
print(f"Circle radius=80, max distance={max_dist:.1f}, auto min_distance={auto_md}")
assert auto_md >= 3 and auto_md <= 30, f"min_distance {auto_md} out of range [3, 30]"
print("✓ PASSED: min_distance auto-calculated correctly")

# Test 5: Singleton (no polygon)
print("\n=== Test 5: No polygon (bbox fallback) ===")
safe5 = bl.safe_area([], (500, 500))
print(f"Fallback rect: {safe5.max_inscribed_rect}")
print("✓ PASSED: Fallback works")

print("\n=== All tests passed! ===")