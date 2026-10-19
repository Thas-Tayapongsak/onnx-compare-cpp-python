"""Main entry point for the comparison program."""

__all__ = ["compare_main"]

import logging
import sys

from .cli import build_compare_args, parse_args
from .pipelines import (
    cpp_pipeline_path,
    create_run_paths,
    run_cpp_pipeline,
    run_python_pipeline,
)
from .report import compare_outputs, write_report

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging to stderr at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def compare_main() -> None:
    """Run both pipelines and write the comparison report."""
    configure_logging()
    logger.info("Starting comparison")
    args = build_compare_args(parse_args())
    if args is None:
        sys.exit(1)

    cpp_pipeline = cpp_pipeline_path()
    if not cpp_pipeline.is_file():
        logger.error(f"C++ pipeline not found at {cpp_pipeline}")
        logger.error("Run scripts/build.ps1 first")
        sys.exit(1)

    logger.info(f"Model folder path: {args.model_folder_path}")
    logger.info(f"Image path: {args.image_path}")
    logger.info(f"Output root path: {args.output_root_path}")

    try:
        paths = create_run_paths(args)
        logger.info(f"Run folder path: {paths.run_folder}")
        run_python_pipeline(args, paths.python_output)
        run_cpp_pipeline(args, paths.cpp_output)
        report = compare_outputs(paths.python_output, paths.cpp_output)
        write_report(paths.run_folder, report)
    except Exception as error:
        logger.error(f"Comparison failed: {error}")
        sys.exit(1)
