#pragma once

#include "pipeline_core/run.h"
#include "pipeline_state.h"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace pipeline_core::detail
{
    struct RawOutput
    {
        std::string name;
        Ort::Value value;
    };

    struct RawInferenceResults
    {
        std::vector<RawOutput> outputs;
    };

    /// Run the session and return the raw output values in session order.
    RawInferenceResults run_session(PipelineState &state,
                                    const PreprocessedPayload &payload);
} // namespace pipeline_core::detail
