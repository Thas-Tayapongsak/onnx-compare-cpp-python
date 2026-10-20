"""Temporary stand-in for the pipeline inference stage."""

__all__ = ["run_stub"]

import numpy as np

from .results import InstanceSegmentationInferenceResults
from .run import InferencePayload


def run_stub(
    payload: InferencePayload,
) -> InstanceSegmentationInferenceResults:
    """Temporary stub inference for building the pipeline."""
    target_height, target_width = payload.pixel_values.shape[2:4]

    num_predictions = np.array([2], dtype=np.int64)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)
    boxes = np.array(
        [
            [0, 0, target_width, target_height],
            [
                target_width * 0.125,
                target_height * 0.125,
                target_width * 0.875,
                target_height * 0.875,
            ],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((2, target_height, target_width), dtype=np.bool)
    masks[0, :, : target_width // 2] = True
    masks[1, : target_height // 2, :] = True

    return InstanceSegmentationInferenceResults(
        num_predictions=num_predictions,
        scores=scores,
        labels=labels,
        boxes=boxes,
        masks=masks,
    )
