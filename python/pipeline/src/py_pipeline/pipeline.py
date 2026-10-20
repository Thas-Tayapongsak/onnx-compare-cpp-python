"""Pipeline stages and orchestration."""

__all__ = [
    "RunState",
    "inference",
    "postprocess",
    "preprocess",
    "setup",
]

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from time import perf_counter

import numpy as np
import onnxruntime as ort

from .config import read_confidence_threshold, read_standardization
from .machine import read_machine_info
from .memory import MemoryMonitor
from .preprocess import build_pixel_values, read_pixel_values_shape
from .results import (
    InstanceSegmentationInferenceResults,
    MetaResults,
    RunOutput,
    RunTimings,
)
from .run import InferencePayload, PreprocessedPayload, RunInput
from .stub import run_stub

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """State created by setup for one pipeline run."""

    session: ort.InferenceSession
    confidence_threshold: float
    nms_iou_threshold: float
    target_sizes: np.ndarray
    mean: np.ndarray | None
    std: np.ndarray | None


_state: RunState | None = None


def _require_state() -> RunState:
    """Return the setup state or fail when setup has not run."""
    if _state is None:
        raise RuntimeError("Pipeline setup has not run")
    return _state


def setup(run_input: RunInput) -> None:
    """Load the model and store the run-wide setup values."""
    global _state

    model_folder = run_input.parsed_args.model_folder_path
    model_path = run_input.parsed_args.model_onnx_path
    confidence_threshold = read_confidence_threshold(model_folder)
    mean, std = read_standardization(model_folder)

    logger.info(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(str(model_path))
    _state = RunState(
        session=session,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=1.0,
        target_sizes=np.array([[0, 0]], dtype=np.int64),
        mean=mean,
        std=std,
    )
    logger.info(f"ONNX Runtime providers: {session.get_providers()}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(
        f"Standardization: {'enabled' if mean is not None else 'disabled'}"
    )


def preprocess(payload: InferencePayload) -> PreprocessedPayload:
    """Load the payload image and build the session pixel values."""
    state = _require_state()
    target_shape = read_pixel_values_shape(state.session)
    pixel_values = build_pixel_values(
        payload.image_path, target_shape, state.mean, state.std
    )
    logger.info(
        f"Pixel values: session shape {target_shape}, "
        f"preprocessed shape {pixel_values.shape}"
    )
    return PreprocessedPayload(pixel_values=pixel_values)


def inference(
    payload: PreprocessedPayload,
) -> InstanceSegmentationInferenceResults:
    """Run the stub inference; the session is deferred."""
    return run_stub(payload)


def postprocess(
    results: InstanceSegmentationInferenceResults,
) -> InstanceSegmentationInferenceResults:
    """Return the final results; postprocessing is deferred."""
    return results


def _elapsed_ms(start: float) -> int:
    """Return the elapsed integer milliseconds since `start`."""
    return round((perf_counter() - start) * 1000)


def _median_ms(samples: list[int]) -> int:
    """Return the integer median of the samples."""
    return int(median(samples))


def _run_pipeline(run_input: RunInput, memory: MemoryMonitor) -> RunOutput:
    """Run the stages and build the output for the caller to finalize."""
    if not run_input.inference_payloads:
        raise RuntimeError("No inference payloads were run")

    preprocess_samples: list[int] = []
    inference_samples: list[int] = []
    postprocess_samples: list[int] = []
    total_samples: list[int] = []
    results_list: list[InstanceSegmentationInferenceResults] = []

    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    machine_info = read_machine_info()
    memory.enter_stage("setup")
    setup_start = perf_counter()
    setup(run_input)
    state = _require_state()
    setup_ms = _elapsed_ms(setup_start)

    for payload in run_input.inference_payloads:
        memory.enter_stage("preprocess")
        total_start = perf_counter()

        start = perf_counter()
        processed_payload = preprocess(payload)
        preprocess_samples.append(_elapsed_ms(start))

        memory.enter_stage("inference")
        start = perf_counter()
        raw_results = inference(processed_payload)
        inference_samples.append(_elapsed_ms(start))

        memory.enter_stage("postprocess")
        start = perf_counter()
        results = postprocess(raw_results)
        postprocess_samples.append(_elapsed_ms(start))

        total_samples.append(_elapsed_ms(total_start))
        results_list.append(results)

    memory.enter_stage("output")

    timings = RunTimings(
        setup_ms=setup_ms,
        preprocess_ms=_median_ms(preprocess_samples),
        inference_ms=_median_ms(inference_samples),
        postprocess_ms=_median_ms(postprocess_samples),
        duration_ms=_median_ms(total_samples),
    )
    meta = MetaResults(
        pipeline="python",
        model=run_input.parsed_args.model_folder_path.name,
        input_image=run_input.parsed_args.image_path.name,
        confidence_threshold=state.confidence_threshold,
        nms_iou_threshold=state.nms_iou_threshold,
        target_sizes=state.target_sizes.tolist(),
        created_utc=created_utc,
        setup_ms=timings.setup_ms,
        preprocess_ms=timings.preprocess_ms,
        inference_ms=timings.inference_ms,
        postprocess_ms=timings.postprocess_ms,
        duration_ms=timings.duration_ms,
        execution_provider="stub",
        cpu=machine_info.cpu,
        gpu=machine_info.gpu,
        setup_ram_gb=memory.stage_ram_gb("setup"),
        preprocess_ram_gb=memory.stage_ram_gb("preprocess"),
        inference_ram_gb=memory.stage_ram_gb("inference"),
        postprocess_ram_gb=memory.stage_ram_gb("postprocess"),
        ram_gb=memory.peak_ram_gb,
        setup_vram_gb=0.0,
        preprocess_vram_gb=0.0,
        inference_vram_gb=0.0,
        postprocess_vram_gb=0.0,
        vram_gb=0.0,
    )
    return RunOutput(results=results_list, meta=meta, timings=timings)
