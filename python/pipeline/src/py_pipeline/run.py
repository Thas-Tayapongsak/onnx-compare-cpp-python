"""Types for one pipeline run."""

__all__ = ["InferencePayload", "PreprocessedPayload", "RunInput"]

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
