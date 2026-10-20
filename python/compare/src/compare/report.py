"""Compare result files and write the report."""

__all__ = ["build_report", "write_report"]

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CONFIGURATION_FIELDS = (
    "model",
    "input_image",
    "confidence_threshold",
    "nms_iou_threshold",
    "target_sizes",
    "execution_provider",
    "cpu",
    "gpu",
)

TIMING_FIELDS = (
    ("setup", "setup_ms"),
    ("preprocess", "preprocess_ms"),
    ("inference", "inference_ms"),
    ("postprocess", "postprocess_ms"),
    ("duration", "duration_ms"),
)

MEMORY_FIELDS = (
    ("setup", "setup_ram_gb"),
    ("preprocess", "preprocess_ram_gb"),
    ("inference", "inference_ram_gb"),
    ("postprocess", "postprocess_ram_gb"),
    ("whole_run", "ram_gb"),
)

REQUIRED_META_FIELDS = (
    "repeats",
    *CONFIGURATION_FIELDS,
    *(meta_key for _, meta_key in TIMING_FIELDS),
    *(meta_key for _, meta_key in MEMORY_FIELDS),
)


@dataclass(frozen=True)
class MetaFile:
    """One `meta.json` file and its loaded values."""

    path: Path
    values: dict[str, object]


def read_meta(folder: Path) -> MetaFile:
    """Load `meta.json` and check that every compared field is present."""
    path = folder / "meta.json"
    with path.open("r", encoding="utf-8") as file:
        data: object = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    values = {str(key): value for key, value in data.items()}
    missing = [field for field in REQUIRED_META_FIELDS if field not in values]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return MetaFile(path=path, values=values)


def compare_outputs(python_folder: Path, cpp_folder: Path) -> dict[str, bool]:
    """Return exact equality for every array in both `outputs.npz` files."""
    python_path = python_folder / "outputs.npz"
    cpp_path = cpp_folder / "outputs.npz"
    with (
        np.load(python_path, allow_pickle=False) as python_outputs,
        np.load(cpp_path, allow_pickle=False) as cpp_outputs,
    ):
        python_keys = python_outputs.files
        cpp_keys = cpp_outputs.files
        if set(python_keys) != set(cpp_keys):
            only_python = sorted(set(python_keys) - set(cpp_keys))
            only_cpp = sorted(set(cpp_keys) - set(python_keys))
            raise ValueError(
                f"`outputs.npz` key sets differ; only in Python: {only_python}; "
                f"only in C++: {only_cpp}"
            )

        result: dict[str, bool] = {}
        for key in python_keys:
            python_array = python_outputs[key]
            cpp_array = cpp_outputs[key]
            if python_array.shape != cpp_array.shape:
                raise ValueError(
                    f"`outputs.npz` key `{key}` has different shapes: "
                    f"{python_array.shape} and {cpp_array.shape}"
                )
            if python_array.dtype != cpp_array.dtype:
                raise ValueError(
                    f"`outputs.npz` key `{key}` has different dtypes: "
                    f"{python_array.dtype} and {cpp_array.dtype}"
                )
            result[key] = np.array_equal(python_array, cpp_array)
        return result


def build_configuration(
    python_meta: MetaFile,
    cpp_meta: MetaFile,
) -> dict[str, object]:
    """Build the configuration object from both `meta.json` files."""
    configuration: dict[str, object] = {}
    for field in CONFIGURATION_FIELDS:
        python_value = python_meta.values[field]
        cpp_value = cpp_meta.values[field]
        configuration[field] = {
            "python": python_value,
            "cpp": cpp_value,
            "equal": python_value == cpp_value,
        }
    return configuration


def _read_timing(value: object, path: Path, field: str) -> int:
    """Return an integer timing value or fail."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} field `{field}` must be an integer")
    return value


def _read_memory(value: object, path: Path, field: str) -> int | float:
    """Return a numeric memory value or fail."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} field `{field}` must be a number")
    return value


def _build_group(
    fields: tuple[tuple[str, str], ...],
    python_meta: MetaFile,
    cpp_meta: MetaFile,
    read_value: Callable[[object, Path, str], int | float],
) -> dict[str, object]:
    """Build one measurement group with both values and the difference."""
    python_values = {
        key: read_value(
            python_meta.values[meta_key], python_meta.path, meta_key
        )
        for key, meta_key in fields
    }
    cpp_values = {
        key: read_value(cpp_meta.values[meta_key], cpp_meta.path, meta_key)
        for key, meta_key in fields
    }
    differences = {
        key: cpp_values[key] - python_values[key] for key, _ in fields
    }
    return {
        "python": python_values,
        "cpp": cpp_values,
        "cpp_minus_python": differences,
    }


def _read_repeat(value: object, path: Path) -> int:
    """Return the repeat count or fail."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} field `repeats` must be an integer")
    return value


def build_measurements(
    python_meta: MetaFile,
    cpp_meta: MetaFile,
) -> dict[str, object]:
    """Build the repeat count and the timing and memory measurement groups."""
    return {
        "repeats": {
            "python": _read_repeat(
                python_meta.values["repeats"], python_meta.path
            ),
            "cpp": _read_repeat(cpp_meta.values["repeats"], cpp_meta.path),
        },
        "timings_ms": _build_group(
            TIMING_FIELDS, python_meta, cpp_meta, _read_timing
        ),
        "memory_gb": _build_group(
            MEMORY_FIELDS, python_meta, cpp_meta, _read_memory
        ),
    }


def build_report(python_folder: Path, cpp_folder: Path) -> dict[str, object]:
    """Build the comparison report for both pipeline output folders."""
    python_meta = read_meta(python_folder)
    cpp_meta = read_meta(cpp_folder)
    return {
        "outputs": compare_outputs(python_folder, cpp_folder),
        "configuration": build_configuration(python_meta, cpp_meta),
        "measurements": build_measurements(python_meta, cpp_meta),
    }


def write_report(run_folder: Path, report: dict[str, object]) -> None:
    """Write `report.json` from the comparison report."""
    with (run_folder / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    logger.info("`report.json` written")
