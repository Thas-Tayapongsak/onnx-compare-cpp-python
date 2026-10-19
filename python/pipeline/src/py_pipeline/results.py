"""Writers for inference results."""

__all__ = [
    "InstanceSegmentationInferenceResults",
    "MetaResults",
    "RunOutput",
    "RunTimings",
    "prepare_output_dir",
    "write_outputs",
    "write_meta",
]

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstanceSegmentationInferenceResults:
    """Session outputs for one instance-segmentation run."""

    num_predictions: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray


@dataclass
class MetaResults:
    """Run details for `meta.json`."""

    pipeline: str
    model: str
    input_image: str
    confidence_threshold: float
    nms_iou_threshold: float
    target_sizes: list[list[int]]
    created_utc: str
    setup_ms: int
    preprocess_ms: int
    inference_ms: int
    postprocess_ms: int
    duration_ms: int
    execution_provider: str


@dataclass
class RunTimings:
    """Timing results for one run.

    `setup_ms` is one setup measurement. The other fields are medians across
    the inference payloads.
    """

    setup_ms: int
    preprocess_ms: int
    inference_ms: int
    postprocess_ms: int
    duration_ms: int


@dataclass
class RunOutput:
    """Model results for each payload, run details, and timing results."""

    results: list[InstanceSegmentationInferenceResults]
    meta: MetaResults
    timings: RunTimings


def prepare_output_dir(path: Path) -> None:
    """Create the output folder if it does not exist."""
    if path.exists():
        logger.info(f"Output folder already exists: {path}")
        return
    logger.info(f"Creating output folder: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(
    output_dir: Path,
    results: list[InstanceSegmentationInferenceResults],
) -> None:
    """Write the single result's five arrays to `outputs.npz`."""
    if len(results) != 1:
        raise ValueError(
            "outputs.npz supports exactly one inference result; "
            f"got {len(results)}"
        )
    result = results[0]
    np.savez(
        output_dir / "outputs.npz",
        num_predictions=result.num_predictions,
        scores=result.scores,
        labels=result.labels,
        boxes=result.boxes,
        masks=result.masks,
    )
    logger.info("`outputs.npz` written")


def write_meta(
    output_dir: Path,
    meta: MetaResults,
) -> None:
    """Write `meta.json` from the meta results."""
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(meta), file)
    logger.info("`meta.json` written")
