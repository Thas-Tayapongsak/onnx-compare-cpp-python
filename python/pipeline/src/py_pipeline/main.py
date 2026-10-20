"""Main entrypoint for the inference pipeline."""

__all__ = ["pipeline_main"]

import logging
import sys

import numpy as np

from .cli import ParsedArgs, build_parsed_args, parse_args
from .memory import MemoryMonitor
from .pipeline import _run_pipeline
from .results import prepare_output_dir, write_meta, write_outputs
from .run import InferencePayload, RunInput

logger = logging.getLogger(__name__)

DEFAULT_PIXEL_VALUES_SHAPE = (1, 3, 64, 64)


def configure_logging() -> None:
    """Configure root logging to stderr at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def build_run_input(parsed_args: ParsedArgs) -> RunInput:
    """Build the run input with the temporary stub payload."""
    payload = InferencePayload(
        pixel_values=np.zeros(DEFAULT_PIXEL_VALUES_SHAPE, dtype=np.float32),
    )
    return RunInput(parsed_args=parsed_args, inference_payloads=[payload])


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

    memory = MemoryMonitor()
    try:
        memory.start()
        memory.enter_stage("startup")
        run_input = build_run_input(parsed_args)

        logger.info("===========================")
        logger.info("Running inference pipeline")
        prepare_output_dir(parsed_args.output_folder_path)
        run_output = _run_pipeline(run_input, memory)
        write_outputs(parsed_args.output_folder_path, run_output.results)
        memory.stop()
        run_output.meta.ram_gb = memory.peak_ram_gb
        write_meta(parsed_args.output_folder_path, run_output.meta)
    finally:
        memory.stop()
    logger.info("===========================")
