#pragma once

#include "inference.h"
#include "pipeline_core/results.h"

namespace pipeline_core::detail
{
    /// Copy the raw session outputs into the result arrays by name.
    InstanceSegmentationInferenceResults
    package_results(RawInferenceResults raw);
} // namespace pipeline_core::detail
