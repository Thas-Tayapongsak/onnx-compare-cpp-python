#pragma once

#include "pipeline_core/memory.h"
#include "pipeline_core/results.h"
#include "pipeline_core/run.h"

namespace pipeline_core
{
    struct RunState
    {
    };

    RunState setup(const RunInput &run_input);

    InferencePayload preprocess(const InferencePayload &payload,
                                RunState &state);

    InstanceSegmentationInferenceResults
    inference(const InferencePayload &payload, RunState &state);

    InstanceSegmentationInferenceResults
    postprocess(InstanceSegmentationInferenceResults results,
                const InferencePayload &payload);

    RunOutput run_pipeline(const RunInput &run_input, MemoryMonitor &memory);
} // namespace pipeline_core
