"""Types for one pipeline run."""

__all__ = [
    "InferencePayload",
    "PreprocessedPayload",
    "RunInput",
    "RunState",
]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .cli import ParsedArgs


@dataclass
class InferencePayload:
    """One source image for an inference."""

    image_path: Path


@dataclass
class PreprocessedPayload:
    """Session pixel values for one image."""

    pixel_values: np.ndarray


@dataclass
class RunInput:
    """Everything needed to run the pipeline."""

    parsed_args: ParsedArgs
    inference_payloads: list[InferencePayload]


@dataclass
class RunState:
    """State created by setup for one pipeline run."""

    session: ort.InferenceSession
    confidence_threshold: float
    nms_iou_threshold: float
    target_sizes: np.ndarray
    mean: np.ndarray | None
    std: np.ndarray | None
