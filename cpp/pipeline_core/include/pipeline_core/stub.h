#pragma once

#include "pipeline_core/results.h"
#include "pipeline_core/run.h"

namespace pipeline_core
{
    /// Temporary stand-in for the inference stage.
    InstanceSegmentationInferenceResults
    run_stub(const PreprocessedPayload &payload);
} // namespace pipeline_core
