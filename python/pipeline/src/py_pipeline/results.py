"""Writers for inference results."""

__all__ = [
    "InstanceSegmentationInferenceResults",
    "prepare_output_dir",
    "write_outputs",
    "write_meta",
]

import json
import logging
from dataclasses import dataclass
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


def prepare_output_dir(path: Path) -> None:
    """Create the output folder if it does not exist."""
    if path.exists():
        logger.info(f"Output folder already exists: {path}")
        return
    logger.info(f"Creating output folder: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(
    output_dir: Path,
    results: InstanceSegmentationInferenceResults,
) -> None:
    """Write the five result arrays to `outputs.npz`."""
    np.savez(
        output_dir / "outputs.npz",
        num_predictions=results.num_predictions,
        scores=results.scores,
        labels=results.labels,
        boxes=results.boxes,
        masks=results.masks,
    )
    logger.info("`outputs.npz` written")


def write_meta(
    output_dir: Path,
) -> None:
    """Write `meta.json` with the pipeline name."""
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump({"pipeline": "python"}, file)
    logger.info("`meta.json` written")
