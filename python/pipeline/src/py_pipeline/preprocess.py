"""Image preprocessing for the inference pipeline."""

__all__ = ["build_pixel_values", "read_pixel_values_shape"]

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PIXEL_VALUES_INPUT_NAME = "pixel_values"


def read_pixel_values_shape(
    session: ort.InferenceSession,
) -> tuple[int, int, int, int]:
    """Return the static `pixel_values` input shape from the session."""
    inputs = session.get_inputs()
    matches = [item for item in inputs if item.name == PIXEL_VALUES_INPUT_NAME]
    if len(matches) != 1:
        names = [item.name for item in inputs]
        raise ValueError(
            f"Expected exactly one {PIXEL_VALUES_INPUT_NAME!r} input; "
            f"model inputs: {names}"
        )

    input_metadata = matches[0]
    if input_metadata.type != "tensor(float)":
        raise ValueError(
            f"Input {PIXEL_VALUES_INPUT_NAME!r} must have type tensor(float); "
            f"got {input_metadata.type}"
        )

    shape = input_metadata.shape
    if shape is None or len(shape) != 4:
        raise ValueError(
            f"Input {PIXEL_VALUES_INPUT_NAME!r} must have rank 4; got {shape}"
        )

    dimensions: list[int] = []
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"Input {PIXEL_VALUES_INPUT_NAME!r} must have static positive "
                f"dimensions; got {shape}"
            )
        dimensions.append(dimension)

    batch, channels, height, width = dimensions
    if batch != 1:
        raise ValueError(
            f"Input {PIXEL_VALUES_INPUT_NAME!r} must have batch 1; "
            f"got batch {batch} in shape {shape}"
        )
    if channels != 3:
        raise ValueError(
            f"Input {PIXEL_VALUES_INPUT_NAME!r} must have 3 channels; "
            f"got {channels} in shape {shape}"
        )
    return batch, channels, height, width


def build_pixel_values(
    image_path: Path,
    target_shape: tuple[int, int, int, int],
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    """Read an image and build the session pixel values."""
    _, _, height, width = target_shape

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    values = resized.astype(np.float32) / np.float32(255.0)
    if mean is not None and std is not None:
        values = (values - mean) / std

    channels_first = values.transpose(2, 0, 1)
    return np.ascontiguousarray(
        channels_first[np.newaxis, ...],
        dtype=np.float32,
    )
