"""Temporary stand-in for the pipeline."""

__all__ = ["run_stub"]

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from .results import (
    InstanceSegmentationInferenceResults,
    MetaResults,
    RunOutput,
)
from .run import RunInput


def run_stub(run_input: RunInput) -> RunOutput:
    """Temporary stub function for building the pipeline."""
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = perf_counter()
    payload = run_input.inference_payloads[0]

    num_predictions = np.array([2], dtype=np.int64)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)
    boxes = np.array([[0, 0, 64, 64], [8, 8, 56, 56]], dtype=np.float32)
    masks = np.zeros((2, 64, 64), dtype=np.bool)
    masks[0, :, :32] = True
    masks[1, :32, :] = True

    results = InstanceSegmentationInferenceResults(
        num_predictions=num_predictions,
        scores=scores,
        labels=labels,
        boxes=boxes,
        masks=masks,
    )
    meta = MetaResults(
        pipeline="python",
        model=run_input.parsed_args.model_folder_path.name,
        input_image=run_input.parsed_args.image_path.name,
        confidence_threshold=payload.confidence_threshold,
        nms_iou_threshold=payload.nms_iou_threshold,
        target_sizes=payload.target_sizes.tolist(),
        created_utc=created_utc,
        duration_ms=round((perf_counter() - start) * 1000),
        execution_provider="stub",
    )
    return RunOutput(results=results, meta=meta)
