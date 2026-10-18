"""Temporary stand-in for the pipeline."""

__all__ = ["run_stub"]

import numpy as np

from .results import InstanceSegmentationInferenceResults


def run_stub() -> InstanceSegmentationInferenceResults:
    """Temporary stub function for building the pipeline"""
    num_predictions = np.array([2], dtype=np.int64)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)
    boxes = np.array([[0, 0, 64, 64], [8, 8, 56, 56]], dtype=np.float32)
    masks = np.zeros((2, 64, 64), dtype=np.bool)
    masks[0, :, :32] = True
    masks[1, :32, :] = True

    return InstanceSegmentationInferenceResults(
        num_predictions=num_predictions,
        scores=scores,
        labels=labels,
        boxes=boxes,
        masks=masks,
    )
