"""Run folders and pipeline subprocesses."""

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cli import CompareArgs

logger = logging.getLogger(__name__)

CPP_PIPELINE_PATH = Path("cpp/build/RelWithDebInfo/cpp_pipeline.exe")


@dataclass
class RunPaths:
    """Folders for one comparison run."""

    run_folder: Path
    python_output: Path
    cpp_output: Path


def create_run_paths(args: CompareArgs) -> RunPaths:
    """Create the run folder and both pipeline output folders."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = args.model_folder_path.name
    run_folder = args.output_root_path / f"{timestamp}-{model_name}"
    run_folder.mkdir(parents=True)
    python_output = run_folder / "python"
    cpp_output = run_folder / "cpp"
    python_output.mkdir()
    cpp_output.mkdir()
    return RunPaths(
        run_folder=run_folder,
        python_output=python_output,
        cpp_output=cpp_output,
    )


def cpp_pipeline_path() -> Path:
    """Return the built C++ pipeline path."""
    return CPP_PIPELINE_PATH


def run_python_pipeline(args: CompareArgs, output_folder: Path) -> None:
    """Run the Python pipeline into the given output folder."""
    command = [
        "uv",
        "run",
        "--project",
        "python/pipeline",
        "py-pipeline",
        "--model",
        str(args.model_folder_path),
        "--image",
        str(args.image_path),
        "--output",
        str(output_folder),
        "--repeat",
        str(args.repeats),
    ]
    _run_command(command, "Python")


def run_cpp_pipeline(args: CompareArgs, output_folder: Path) -> None:
    """Run the C++ pipeline into the given output folder."""
    command = [
        str(CPP_PIPELINE_PATH),
        "--model",
        str(args.model_folder_path),
        "--image",
        str(args.image_path),
        "--output",
        str(output_folder),
        "--repeat",
        str(args.repeats),
    ]
    _run_command(command, "C++")


def _run_command(command: list[str], name: str) -> None:
    """Run one pipeline and raise when it fails."""
    logger.info(f"Running the {name} pipeline")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(
            f"The {name} pipeline failed with exit code {result.returncode}"
        )
