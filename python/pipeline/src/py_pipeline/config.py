"""Read model-folder settings for pipeline setup."""

__all__ = [
    "read_confidence_threshold",
    "read_standardization",
]

import json
import math
from pathlib import Path

import numpy as np

EVAL_METRICS_FILENAME = "eval_metrics.json"
PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"
CONFIDENCE_THRESHOLD_KEY = "eval_optimal_threshold"


def _read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object from `path`."""
    if not path.exists():
        raise FileNotFoundError(f"Model settings file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Model settings path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(
            f"Model settings file must contain a JSON object: {path}"
        )
    return data


def _read_optional_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object, or return `None` when the file is absent."""
    if not path.exists():
        return None
    return _read_json_object(path)


def _read_number(value: object, path: Path, key: str) -> float:
    """Read one finite number from a JSON field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} field {key!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} field {key!r} must be finite")
    return number


def _select_standardization_pair(
    data: dict[str, object],
    path: Path,
) -> tuple[str, str] | None:
    """Select the mean/std key pair, or return `None` when no pair exists."""
    has_mean = "mean" in data
    has_std = "std" in data
    if has_mean or has_std:
        if not (has_mean and has_std):
            raise ValueError(f"{path} must contain both 'mean' and 'std'")
        return "mean", "std"

    has_image_mean = "image_mean" in data
    has_image_std = "image_std" in data
    if has_image_mean or has_image_std:
        if not (has_image_mean and has_image_std):
            raise ValueError(
                f"{path} must contain both 'image_mean' and 'image_std'"
            )
        return "image_mean", "image_std"

    return None


def _read_channel_values(value: object, path: Path, key: str) -> np.ndarray:
    """Read a scalar or three-value channel setting."""
    if isinstance(value, bool):
        raise ValueError(
            f"{path} field {key!r} must be a number or a list of three numbers"
        )
    if isinstance(value, (int, float)):
        values = np.asarray(value, dtype=np.float32)
    elif isinstance(value, list) and len(value) == 3:
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        ):
            raise ValueError(
                f"{path} field {key!r} must be a number or "
                "a list of three numbers"
            )
        values = np.asarray(value, dtype=np.float32)
    else:
        raise ValueError(
            f"{path} field {key!r} must be a number or a list of three numbers"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} field {key!r} must be finite")
    return values


def read_confidence_threshold(model_folder: Path) -> float:
    """Read `eval_optimal_threshold` from `eval_metrics.json`."""
    path = model_folder / EVAL_METRICS_FILENAME
    data = _read_json_object(path)
    if CONFIDENCE_THRESHOLD_KEY not in data:
        raise ValueError(f"{path} is missing {CONFIDENCE_THRESHOLD_KEY!r}")
    threshold = _read_number(
        data[CONFIDENCE_THRESHOLD_KEY],
        path,
        CONFIDENCE_THRESHOLD_KEY,
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{path} field {CONFIDENCE_THRESHOLD_KEY!r} must be in [0, 1]"
        )
    return threshold


def read_standardization(
    model_folder: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the optional mean/std pair from `preprocessor_config.json`."""
    path = model_folder / PREPROCESSOR_CONFIG_FILENAME
    data = _read_optional_json_object(path)
    if data is None:
        return None, None

    pair = _select_standardization_pair(data, path)
    if pair is None:
        return None, None

    mean_key, std_key = pair
    mean = _read_channel_values(data[mean_key], path, mean_key)
    std = _read_channel_values(data[std_key], path, std_key)
    if mean.shape != std.shape:
        raise ValueError(
            f"{path} fields {mean_key!r} and {std_key!r} must have "
            "the same shape"
        )
    if np.any(std <= 0.0):
        raise ValueError(f"{path} field {std_key!r} must be greater than zero")
    return mean, std
