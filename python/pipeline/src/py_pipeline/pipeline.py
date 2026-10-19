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
from time import perf_counter

from .results import (
    InstanceSegmentationInferenceResults,
    MetaResults,
    RunOutput,
)
from .run import InferencePayload, RunInput
from .stub import run_stub


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


def run_pipeline(run_input: RunInput) -> RunOutput:
    """Run the stages once and build the run output."""
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = perf_counter()
    state = setup(run_input)
    payload = preprocess(run_input, state)
    raw_results = inference(payload, state)
    results = postprocess(raw_results, payload)
    meta = MetaResults(
        pipeline="python",
        model=run_input.parsed_args.model_folder_path.name,
        input_image=run_input.parsed_args.image_path.name,
        confidence_threshold=payload.confidence_threshold,
        nms_iou_threshold=payload.nms_iou_threshold,
        target_sizes=payload.target_sizes.tolist(),
        created_utc=created_utc,
        duration_ms=round((perf_counter() - start) * 1000),
        execution_provider="stub",
    )
    return RunOutput(results=results, meta=meta)
