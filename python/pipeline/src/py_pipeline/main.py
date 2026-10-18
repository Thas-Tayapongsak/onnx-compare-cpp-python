"""Main entrypoint for the inference pipeline"""

__all__ = ["pipeline_main"]

import logging
import sys

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def pipeline_main() -> None:
    configure_logging()
    logger.info("Starting inference pipeline")
