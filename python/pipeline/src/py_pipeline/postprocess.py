"""Packaging of the raw inference results."""

__all__ = ["package_results"]

from .results import InstanceSegmentationInferenceResults
from .run import RawInferenceResults

OUTPUT_NAMES = ("num_predictions", "scores", "labels", "boxes", "masks")


def package_results(
    raw: RawInferenceResults,
) -> InstanceSegmentationInferenceResults:
    """Package the raw arrays by name; P6 adds the final rules."""
    missing = [name for name in OUTPUT_NAMES if name not in raw.outputs]
    extra = [name for name in raw.outputs if name not in OUTPUT_NAMES]
    if missing or extra:
        raise ValueError(
            f"Expected outputs {list(OUTPUT_NAMES)}; "
            f"missing {missing}; extra {extra}"
        )
    return InstanceSegmentationInferenceResults(
        num_predictions=raw.outputs["num_predictions"],
        scores=raw.outputs["scores"],
        labels=raw.outputs["labels"],
        boxes=raw.outputs["boxes"],
        masks=raw.outputs["masks"],
    )
