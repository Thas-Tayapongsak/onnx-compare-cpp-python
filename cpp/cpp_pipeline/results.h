#pragma once

#include "npz.h"
#include <filesystem>

namespace cpp_pipeline
{
    struct InstanceSegmentationInferenceResults
    {
        NDArray<std::int64_t> num_predictions;
        NDArray<float> scores;
        NDArray<std::int64_t> labels;
        NDArray<float> boxes;
        NDArray<std::uint8_t> masks;
    };

    /// Create the output folder when it does not exist yet.
    void prepare_output_dir(const std::filesystem::path &output_dir);

    /// Write the result arrays to `outputs.npz`.
    void write_outputs(const std::filesystem::path &output_dir,
                       const InstanceSegmentationInferenceResults &results);

    /// Write `meta.json` with the pipeline name.
    void write_meta(const std::filesystem::path &output_dir);
} // namespace cpp_pipeline
