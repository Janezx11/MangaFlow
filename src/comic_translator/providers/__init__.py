# providers 包 __init__

from comic_translator.providers.base import *

# Bubble Detection Providers
from comic_translator.providers.bubble import BubbleProvider, YOLO11BubbleProvider

# Text Detection Providers
from comic_translator.providers.text_detection import TextDetectorProvider

# Vision Detection Providers
from comic_translator.providers.vision import (
    VisionDetectionProvider,
    CTBDVisionAdapter,
    YOLOVisionAdapter,
    LocateAnythingProvider,
)

__all__ = [
    "BubbleProvider",
    "YOLO11BubbleProvider",
    "TextDetectorProvider",
    "VisionDetectionProvider",
    "CTBDVisionAdapter",
    "YOLOVisionAdapter",
    "LocateAnythingProvider",
]
