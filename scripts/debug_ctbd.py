#!/usr/bin/env python
"""Debug CTBD ONNX output format"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image


def debug_ctbd():
    from comic_translator.providers.text_detection.ctbd_text_detector import CTBDTextDetectorProvider

    provider = CTBDTextDetectorProvider(confidence=0.1, device="cpu")
    provider._ensure_model()

    img_path = Path("data/comics/0003_GREECE0002.png")
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Image: {img_path.name}, size=({orig_w}, {orig_h})")

    # Preprocess
    img_resized = img.resize((640, 640), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_nchw = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    print(f"Input shape: {img_nchw.shape}")

    # Test both orderings
    for label, sizes in [
        ("[H, W]", np.array([[orig_h, orig_w]], dtype=np.int64)),
        ("[W, H]", np.array([[orig_w, orig_h]], dtype=np.int64)),
    ]:
        print(f"\n{'='*60}")
        print(f"orig_target_sizes: {label} = {sizes}")
        print(f"{'='*60}")

        outputs = provider._session.run(
            None,
            {"images": img_nchw, "orig_target_sizes": sizes},
        )

        labels_arr, boxes, scores = outputs

        print(f"\nDetections with score>0.1:")
        print(f"{'idx':>4} {'label':>6} {'score':>8} {'box (xyxy)':>40}")
        print("-" * 70)

        for i in range(min(20, labels_arr.shape[-1])):
            label_id = int(labels_arr.flat[i])
            score = float(scores.flat[i])
            if score < 0.1:
                continue
            box_raw = boxes.flat[i * 4: (i + 1) * 4]
            print(f"{i:>4} {label_id:>6} {score:>8.4f} "
                  f"[{box_raw[0]:>8.1f}, {box_raw[1]:>8.1f}, {box_raw[2]:>8.1f}, {box_raw[3]:>8.1f}]")

        all_boxes = boxes.reshape(-1, 4)
        print(f"\nBox ranges: x=[{all_boxes[:, 0].min():.0f}, {all_boxes[:, 2].max():.0f}], "
              f"y=[{all_boxes[:, 1].min():.0f}, {all_boxes[:, 3].max():.0f}]")
        print(f"Expected:   x=[0, {orig_w}], y=[0, {orig_h}]")


if __name__ == "__main__":
    debug_ctbd()
