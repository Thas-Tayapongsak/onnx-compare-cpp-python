"""Compare result files and write the report."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compare_outputs(python_folder: Path, cpp_folder: Path) -> dict[str, bool]:
    """Return exact equality for every array in both `outputs.npz` files."""
    python_path = python_folder / "outputs.npz"
    cpp_path = cpp_folder / "outputs.npz"
    with (
        np.load(python_path, allow_pickle=False) as python_outputs,
        np.load(cpp_path, allow_pickle=False) as cpp_outputs,
    ):
        return {
            key: np.array_equal(python_outputs[key], cpp_outputs[key])
            for key in python_outputs.files
        }


def write_report(run_folder: Path, report: dict[str, bool]) -> None:
    """Write `report.json` with one boolean per output key."""
    with (run_folder / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file)
    logger.info("`report.json` written")
