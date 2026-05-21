"""
SVOD Data Models

Contains enum types and data classes used across the SVOD package.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class VideoOrientation(Enum):
    """Enum for video orientation states"""

    CORRECT = "CORRECT - Humans are upright"
    INCORRECT = "INCORRECT - Humans are sideways/rotated"
    UNCERTAIN = "UNCERTAIN - Cannot determine orientation"


class BatchResult:
    """Data class for batch processing results"""

    def __init__(
        self,
        filepath: str,
        orientation: VideoOrientation,
        confidence: float,
        detection_info: Dict,
        processing_time: float,
        error: Optional[str] = None,
    ):
        self.filepath = filepath
        self.filename = Path(filepath).name
        self.orientation = orientation
        self.confidence = confidence
        self.detection_info = detection_info
        self.processing_time = processing_time
        self.error = error
        self.filesize = self._get_file_size()

    def _get_file_size(self):
        try:
            return os.path.getsize(self.filepath) / (1024 * 1024)  # MB
        except Exception:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "orientation": self.orientation.name,
            "confidence": round(self.confidence, 4),
            "processing_time": round(self.processing_time, 2),
            "filesize_mb": round(self.filesize, 2),
            "error": self.error,
        }


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects for safe serialization."""
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    # numpy scalars
    type_name = type(obj).__name__
    if "int" in type_name and hasattr(obj, "item"):
        return int(obj)
    if "float" in type_name and hasattr(obj, "item"):
        return float(obj)
    if "ndarray" in type_name:
        return obj.tolist()  # type: ignore[union-attr]
    return obj


def results_to_json(results: Dict, indent: int = 2) -> str:
    """Serialize a single-video result dict to a JSON string."""
    clean = _sanitize_for_json(results)
    return json.dumps(clean, indent=indent, ensure_ascii=False)


def batch_results_to_json(results: list, indent: int = 2) -> str:
    """Serialize a list of BatchResult objects to a JSON string."""
    items = [r.to_dict() if isinstance(r, BatchResult) else _sanitize_for_json(r) for r in results]
    return json.dumps(items, indent=indent, ensure_ascii=False)


__all__ = ["VideoOrientation", "BatchResult", "results_to_json", "batch_results_to_json"]
