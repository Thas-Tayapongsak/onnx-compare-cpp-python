#pragma once

#include <pipeline_core/results.h>

#include <filesystem>
#include <vector>

namespace cpp_pipeline
{
    /// Create the output folder when it does not exist yet.
    void prepare_output_dir(const std::filesystem::path &output_dir);

    /// Write the single result's arrays to `outputs.npz`.
    void write_outputs(
        const std::filesystem::path &output_dir,
        const std::vector<pipeline_core::InstanceSegmentationInferenceResults>
            &results);

    /// Write `meta.json` from the meta results.
    void write_meta(const std::filesystem::path &output_dir,
                    const pipeline_core::MetaResults &meta);
} // namespace cpp_pipeline
