"""Pipeline stages and orchestration."""

__all__ = [
    "RunState",
    "inference",
    "postprocess",
    "preprocess",
    "run_pipeline",
    "setup",
]

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from time import perf_counter

from .results import (
    InstanceSegmentationInferenceResults,
    MetaResults,
    RunOutput,
    RunTimings,
)
from .run import InferencePayload, RunInput
from .stub import run_stub

TIMING_RUNS = 3


@dataclass
class RunState:
    """State created by setup for one pipeline run."""


def setup(run_input: RunInput) -> RunState:
    """Prepare the state for one run; the stub needs nothing."""
    return RunState()


def preprocess(run_input: RunInput, state: RunState) -> InferencePayload:
    """Return the session inputs; preprocessing is deferred."""
    return run_input.inference_payloads[0]


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


def run_pipeline(run_input: RunInput) -> RunOutput:
    """Run the stages, take the median timings, and build the output."""
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    setup_samples: list[int] = []
    preprocess_samples: list[int] = []
    inference_samples: list[int] = []
    postprocess_samples: list[int] = []
    total_samples: list[int] = []
    first_payload: InferencePayload | None = None
    first_results: InstanceSegmentationInferenceResults | None = None

    for _ in range(TIMING_RUNS):
        total_start = perf_counter()

        start = perf_counter()
        state = setup(run_input)
        setup_samples.append(_elapsed_ms(start))

        start = perf_counter()
        payload = preprocess(run_input, state)
        preprocess_samples.append(_elapsed_ms(start))

        start = perf_counter()
        raw_results = inference(payload, state)
        inference_samples.append(_elapsed_ms(start))

        start = perf_counter()
        results = postprocess(raw_results, payload)
        postprocess_samples.append(_elapsed_ms(start))

        total_samples.append(_elapsed_ms(total_start))

        if first_results is None:
            first_results = results
            first_payload = payload

    if first_results is None or first_payload is None:
        raise RuntimeError("No pipeline run completed")

    timings = RunTimings(
        setup_ms=_median_ms(setup_samples),
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
        duration_ms=timings.duration_ms,
        execution_provider="stub",
    )
    return RunOutput(results=first_results, meta=meta, timings=timings)
