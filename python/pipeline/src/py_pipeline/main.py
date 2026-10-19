"""Main entrypoint for the inference pipeline."""

__all__ = ["pipeline_main"]

import logging
import sys

from .cli import build_parsed_args, parse_args
from .results import prepare_output_dir, write_meta, write_outputs
from .stub import run_stub

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging to stderr at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def pipeline_main() -> None:
    """Run the pipeline: parse arguments, run the stub, write results."""
    configure_logging()
    logger.info("Starting inference pipeline")
    logger.info("===========================")
    args = parse_args()
    parsed_args = build_parsed_args(args)
    if parsed_args is None:
        sys.exit(1)
    logger.info(f"Model folder path: {parsed_args.model_folder_path}")
    logger.info(f"Image path: {parsed_args.image_path}")
    logger.info(f"Output folder path: {parsed_args.output_folder_path}")

    logger.info("===========================")
    logger.info("Running inference pipeline")
    prepare_output_dir(parsed_args.output_folder_path)
    write_outputs(parsed_args.output_folder_path, run_stub())
    write_meta(parsed_args.output_folder_path)
    logger.info("===========================")
