"""ONNX Runtime inference for the pipeline."""

__all__ = ["run_session"]

import logging

import numpy as np
import onnxruntime as ort

from .preprocess import PIXEL_VALUES_INPUT_NAME, read_pixel_values_shape
from .run import PreprocessedPayload, RawInferenceResults, RunState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD_INPUT_NAME = "confidence_threshold"
NMS_IOU_THRESHOLD_INPUT_NAME = "nms_iou_threshold"
TARGET_SIZES_INPUT_NAME = "target_sizes"
SESSION_INPUT_NAMES = (
    PIXEL_VALUES_INPUT_NAME,
    CONFIDENCE_THRESHOLD_INPUT_NAME,
    NMS_IOU_THRESHOLD_INPUT_NAME,
    TARGET_SIZES_INPUT_NAME,
)

_FIXED_INPUT_CONTRACTS = {
    CONFIDENCE_THRESHOLD_INPUT_NAME: ("tensor(float)", [1]),
    NMS_IOU_THRESHOLD_INPUT_NAME: ("tensor(float)", [1]),
    TARGET_SIZES_INPUT_NAME: ("tensor(int64)", [1, 2]),
}


def _require_inputs(
    session: ort.InferenceSession,
) -> dict[str, ort.NodeArg]:
    """Return the session inputs keyed by name."""
    inputs = session.get_inputs()
    names = [item.name for item in inputs]
    if len(inputs) != len(SESSION_INPUT_NAMES) or set(names) != set(
        SESSION_INPUT_NAMES
    ):
        raise ValueError(
            f"Expected session inputs {list(SESSION_INPUT_NAMES)}; got {names}"
        )
    return {item.name: item for item in inputs}


def _require_fixed_input(
    metadata: ort.NodeArg,
    expected_type: str,
    expected_shape: list[int],
) -> None:
    """Check one setup-driven input against its fixed contract."""
    if metadata.type != expected_type:
        raise ValueError(
            f"Input {metadata.name!r} must have type {expected_type}; "
            f"got {metadata.type}"
        )
    if metadata.shape != expected_shape:
        raise ValueError(
            f"Input {metadata.name!r} must have shape {expected_shape}; "
            f"got {metadata.shape}"
        )


def _check_inputs(session: ort.InferenceSession) -> None:
    """Check the session input names, types, and fixed shapes."""
    inputs = _require_inputs(session)
    for name, contract in _FIXED_INPUT_CONTRACTS.items():
        _require_fixed_input(inputs[name], *contract)


def _build_feed(
    state: RunState,
    payload: PreprocessedPayload,
) -> dict[str, np.ndarray]:
    """Build the session feed from the run state and the payload."""
    expected_shape = read_pixel_values_shape(state.session)
    pixel_values = payload.pixel_values
    if pixel_values.dtype != np.float32:
        raise ValueError(
            f"Pixel values must have dtype float32; got {pixel_values.dtype}"
        )
    if pixel_values.shape != expected_shape:
        raise ValueError(
            f"Pixel values must have shape {expected_shape}; "
            f"got {pixel_values.shape}"
        )
    return {
        PIXEL_VALUES_INPUT_NAME: pixel_values,
        CONFIDENCE_THRESHOLD_INPUT_NAME: np.array(
            [state.confidence_threshold], dtype=np.float32
        ),
        NMS_IOU_THRESHOLD_INPUT_NAME: np.array(
            [state.nms_iou_threshold], dtype=np.float32
        ),
        TARGET_SIZES_INPUT_NAME: state.target_sizes,
    }


def _require_array(metadata: ort.NodeArg, value: object) -> np.ndarray:
    """Check one output array and return it as a NumPy array."""
    name = metadata.name
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"Output {name!r} must be a numpy.ndarray; "
            f"got {type(value).__name__}"
        )
    shape = metadata.shape
    if shape is None:
        raise ValueError(f"Output {name!r} has no declared shape")
    if value.ndim != len(shape):
        raise ValueError(
            f"Output {name!r} must have rank {len(shape)}; "
            f"declared {shape}; observed {value.shape}"
        )
    for dimension, actual in zip(shape, value.shape, strict=True):
        if isinstance(dimension, int) and dimension != actual:
            raise ValueError(
                f"Output {name!r} must match declared shape {shape}; "
                f"observed {value.shape}"
            )
    return value


def run_session(
    state: RunState,
    payload: PreprocessedPayload,
) -> RawInferenceResults:
    """Run the session and return the raw output arrays."""
    _check_inputs(state.session)
    feed = _build_feed(state, payload)

    output_metadata = state.session.get_outputs()
    output_details = [(item.name, item.shape) for item in output_metadata]
    logger.info(f"Session outputs: {output_details}")
    output_names = [item.name for item in output_metadata]
    arrays = state.session.run(output_names, feed)

    outputs: dict[str, np.ndarray] = {}
    for metadata, value in zip(output_metadata, arrays, strict=True):
        outputs[metadata.name] = _require_array(metadata, value)
    array_details = ", ".join(
        f"{name} {value.shape} {value.dtype}"
        for name, value in outputs.items()
    )
    logger.info(f"Raw outputs: {array_details}")
    return RawInferenceResults(outputs)
