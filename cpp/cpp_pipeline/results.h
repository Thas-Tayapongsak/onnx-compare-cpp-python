#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace cpp_pipeline
{
    struct InstanceSegmentationInferenceResults
    {
        std::vector<std::int64_t> num_predictions;
        std::vector<float> scores;
        std::vector<std::int64_t> labels;
        std::vector<float> boxes;
        std::vector<std::uint8_t> masks;
        std::array<std::size_t, 3> masks_shape{};
    };

    /// Create the output folder when it does not exist yet.
    void prepare_output_dir(const std::filesystem::path &output_dir);

    /// Write the result arrays to `outputs.npz`.
    void write_outputs(const std::filesystem::path &output_dir,
                       const InstanceSegmentationInferenceResults &results);

    /// Write `meta.json` with the pipeline name.
    void write_meta(const std::filesystem::path &output_dir);
} // namespace cpp_pipeline
