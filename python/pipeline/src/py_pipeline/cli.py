"""Command-line arguments and validation for the pipeline."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ONNX_FILENAME = "model.onnx"
DEFAULT_OUTPUT_FOLDER = "runs"
DEFAULT_REPEATS = 1


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Python ONNX inference pipeline"
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
        default=DEFAULT_OUTPUT_FOLDER,
        help="Path to the output folder.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=False,
        default=DEFAULT_REPEATS,
        help="Number of times to run the payload. Must be at least 1.",
    )
    args = parser.parse_args()
    return args


@dataclass
class ParsedArgs:
    """Validated command-line paths for one pipeline run."""

    model_folder_path: Path
    model_onnx_path: Path
    image_path: Path
    output_folder_path: Path
    repeats: int


def build_parsed_args(args: argparse.Namespace) -> ParsedArgs | None:
    """Validate arguments and build the run paths."""
    if not args.model:
        logger.error("Model folder path is required")
        return None

    model_folder_path = Path(args.model)
    model_onnx_path = model_folder_path / DEFAULT_MODEL_ONNX_FILENAME
    if not model_onnx_path.exists():
        logger.error(f"ONNX model file not found at {model_onnx_path}")
        return None
    if not model_onnx_path.is_file():
        logger.error(f"ONNX model path {model_onnx_path} is not a file")
        return None

    if not args.image:
        logger.error("Image path is required")
        return None

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found at {image_path}")
        return None
    if not image_path.is_file():
        logger.error(f"Image path {image_path} is not a file")
        return None

    output_folder_path = Path(args.output)
    if output_folder_path.exists() and not output_folder_path.is_dir():
        logger.error(f"Output path {output_folder_path} is not a directory")
        return None

    if args.repeat < 1:
        logger.error(f"Repeat count {args.repeat} must be at least 1")
        return None

    return ParsedArgs(
        model_folder_path=model_folder_path,
        model_onnx_path=model_onnx_path,
        image_path=image_path,
        output_folder_path=output_folder_path,
        repeats=args.repeat,
    )
