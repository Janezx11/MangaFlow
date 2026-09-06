# providers 包 __init__

from comic_translator.providers.base import *  # noqa: F403

# Bubble Detection Providers
from comic_translator.providers.bubble import BubbleProvider, YOLO11BubbleProvider

# Text Detection Providers
from comic_translator.providers.text_detection import TextDetectorProvider

# Vision Detection Providers
from comic_translator.providers.vision import (
    CTBDVisionAdapter,
    LocateAnythingProvider,
    VisionDetectionProvider,
    YOLOVisionAdapter,
)

__all__ = [
    "BubbleProvider",
    "CTBDVisionAdapter",
    "LocateAnythingProvider",
    "TextDetectorProvider",
    "VisionDetectionProvider",
    "YOLO11BubbleProvider",
    "YOLOVisionAdapter",
]
