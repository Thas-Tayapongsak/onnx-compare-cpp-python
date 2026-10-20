"""Packaging of the raw inference results."""

__all__ = ["package_results"]

import numpy as np

from .results import InstanceSegmentationInferenceResults
from .run import RawInferenceResults

OUTPUT_NAMES = ("num_predictions", "scores", "labels", "boxes", "masks")

OUTPUT_DTYPES = {
    "num_predictions": np.dtype(np.int64),
    "scores": np.dtype(np.float32),
    "labels": np.dtype(np.int64),
    "boxes": np.dtype(np.float32),
    "masks": np.dtype(np.bool_),
}


def _check_output_names(raw: RawInferenceResults) -> None:
    """Check that the raw outputs are exactly the five expected names."""
    missing = [name for name in OUTPUT_NAMES if name not in raw.outputs]
    extra = [name for name in raw.outputs if name not in OUTPUT_NAMES]
    if missing or extra:
        raise ValueError(
            f"Expected outputs {list(OUTPUT_NAMES)}; "
            f"missing {missing}; extra {extra}"
        )


def _convert_output(name: str, value: np.ndarray) -> np.ndarray:
    """Return the field dtype, or fail on an unexpected session dtype."""
    expected = OUTPUT_DTYPES[name]
    if value.dtype != expected:
        raise ValueError(
            f"Output {name!r} must have dtype {expected}; got {value.dtype}"
        )
    return value.astype(expected, copy=False)


def package_results(
    raw: RawInferenceResults,
) -> InstanceSegmentationInferenceResults:
    """Package the raw arrays by name and field dtype."""
    _check_output_names(raw)
    outputs = {
        name: _convert_output(name, raw.outputs[name]) for name in OUTPUT_NAMES
    }
    return InstanceSegmentationInferenceResults(
        num_predictions=outputs["num_predictions"],
        scores=outputs["scores"],
        labels=outputs["labels"],
        boxes=outputs["boxes"],
        masks=outputs["masks"],
    )
