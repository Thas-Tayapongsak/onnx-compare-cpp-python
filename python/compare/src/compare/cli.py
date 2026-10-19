"""Command-line arguments for the comparison program."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ONNX_FILENAME = "model.onnx"
DEFAULT_OUTPUT_ROOT = "runs"


@dataclass
class CompareArgs:
    """Validated command-line paths for one comparison run."""

    model_folder_path: Path
    image_path: Path
    output_root_path: Path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare the Python and C++ ONNX inference pipelines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model folder. Where `model.onnx` is located.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path to the folder that holds run folders.",
    )
    return parser.parse_args()


def build_compare_args(args: argparse.Namespace) -> CompareArgs | None:
    """Validate arguments and build the comparison paths."""
    model_folder_path = Path(args.model)
    model_onnx_path = model_folder_path / DEFAULT_MODEL_ONNX_FILENAME
    if not model_onnx_path.exists():
        logger.error(f"ONNX model file not found at {model_onnx_path}")
        return None
    if not model_onnx_path.is_file():
        logger.error(f"ONNX model path {model_onnx_path} is not a file")
        return None

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found at {image_path}")
        return None
    if not image_path.is_file():
        logger.error(f"Image path {image_path} is not a file")
        return None

    output_root_path = Path(args.output)
    if output_root_path.exists() and not output_root_path.is_dir():
        logger.error(f"Output path {output_root_path} is not a directory")
        return None

    return CompareArgs(
        model_folder_path=model_folder_path,
        image_path=image_path,
        output_root_path=output_root_path,
    )
