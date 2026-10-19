#pragma once

#include "pipeline_core/array.h"

#include <cstdint>
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
    };

    struct InferencePayload
    {
        NDArray<float> pixel_values;
        double confidence_threshold{0.0};
        double nms_iou_threshold{0.0};
        NDArray<std::int64_t> target_sizes;
    };

    struct RunInput
    {
        ParsedArgs parsed_args;
        std::vector<InferencePayload> inference_payloads;
    };
} // namespace pipeline_core
