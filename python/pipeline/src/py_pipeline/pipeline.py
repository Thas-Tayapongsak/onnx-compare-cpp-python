"""Pipeline stages and orchestration."""

__all__ = [
    "RunState",
    "inference",
    "postprocess",
    "preprocess",
    "setup",
]

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from time import perf_counter

from .machine import read_machine_info
from .memory import MemoryMonitor
from .results import (
    InstanceSegmentationInferenceResults,
    MetaResults,
    RunOutput,
    RunTimings,
)
from .run import InferencePayload, RunInput
from .stub import run_stub


@dataclass
class RunState:
    """State created by setup for one pipeline run."""


def setup(run_input: RunInput) -> RunState:
    """Prepare the state for one run; the stub needs nothing."""
    return RunState()


def preprocess(payload: InferencePayload, state: RunState) -> InferencePayload:
    """Return the session inputs; preprocessing is deferred."""
    return payload


def inference(
    payload: InferencePayload,
    state: RunState,
) -> InstanceSegmentationInferenceResults:
    """Run the stub inference; the session is deferred."""
    return run_stub(payload)


def postprocess(
    results: InstanceSegmentationInferenceResults,
    payload: InferencePayload,
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
    first_payload = run_input.inference_payloads[0]

    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    machine_info = read_machine_info()
    memory.enter_stage("setup")
    setup_start = perf_counter()
    state = setup(run_input)
    setup_ms = _elapsed_ms(setup_start)

    for index, payload in enumerate(run_input.inference_payloads):
        memory.enter_stage("preprocess")
        total_start = perf_counter()

        start = perf_counter()
        processed_payload = preprocess(payload, state)
        preprocess_samples.append(_elapsed_ms(start))
        if index == 0:
            first_payload = processed_payload

        memory.enter_stage("inference")
        start = perf_counter()
        raw_results = inference(processed_payload, state)
        inference_samples.append(_elapsed_ms(start))

        memory.enter_stage("postprocess")
        start = perf_counter()
        results = postprocess(raw_results, processed_payload)
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
        confidence_threshold=first_payload.confidence_threshold,
        nms_iou_threshold=first_payload.nms_iou_threshold,
        target_sizes=first_payload.target_sizes.tolist(),
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
