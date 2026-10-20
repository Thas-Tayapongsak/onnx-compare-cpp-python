#pragma once

#include "pipeline_core/results.h"

namespace pipeline_core::detail
{
    /// Check that the prediction count agrees with the result arrays.
    void
    check_result_counts(const InstanceSegmentationInferenceResults &results);

    /// Log the packaged fields in field order.
    void
    log_packaged_results(const InstanceSegmentationInferenceResults &results);
} // namespace pipeline_core::detail
