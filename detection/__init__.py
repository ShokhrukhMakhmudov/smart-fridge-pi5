"""Пакет детекции, трекинга и фиксации пересечения линии."""

from .crossing import CrossingBuffer, CrossingEvent, LineCrossingDetector
from .detector import Detection, Detector
from .tracker import ByteTracker, TrackedDetection

__all__ = [
    "ByteTracker",
    "CrossingBuffer",
    "CrossingEvent",
    "Detection",
    "Detector",
    "LineCrossingDetector",
    "TrackedDetection",
]
