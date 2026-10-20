"""Types for one pipeline run."""

__all__ = ["InferencePayload", "RunInput"]

from dataclasses import dataclass

import numpy as np

from .cli import ParsedArgs


@dataclass
class InferencePayload:
    """Session pixel values for one image."""

    pixel_values: np.ndarray


@dataclass
class RunInput:
    """Everything needed to run the pipeline."""

    parsed_args: ParsedArgs
    inference_payloads: list[InferencePayload]
