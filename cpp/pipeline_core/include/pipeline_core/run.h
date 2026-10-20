#pragma once

#include "pipeline_core/array.h"

#include <filesystem>
#include <vector>

namespace pipeline_core
{
    struct ParsedArgs
    {
        std::filesystem::path model_folder_path;
        std::filesystem::path model_onnx_path;
        std::filesystem::path image_path;
        std::filesystem::path output_folder_path;
        int repeats{1};
    };

    struct InferencePayload
    {
        std::filesystem::path image_path;
    };

    struct PreprocessedPayload
    {
        NDArray<float> pixel_values;
    };

    struct RunInput
    {
        ParsedArgs parsed_args;
        std::vector<InferencePayload> inference_payloads;
    };
} // namespace pipeline_core
