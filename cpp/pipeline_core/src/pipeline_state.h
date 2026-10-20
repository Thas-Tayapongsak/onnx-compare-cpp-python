#pragma once

#include "config.h"
#include "pipeline_core/array.h"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace pipeline_core::detail
{
    inline constexpr const char *kExecutionProvider = "CPUExecutionProvider";

    struct PipelineState
    {
        explicit PipelineState(const std::filesystem::path &model_path)
            : session(environment, model_path.c_str(), session_options)
        {
        }

        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "cpp_pipeline"};
        Ort::SessionOptions session_options;
        Ort::Session session;
        std::string execution_provider{kExecutionProvider};
        double confidence_threshold{0.0};
        double nms_iou_threshold{1.0};
        NDArray<std::int64_t> target_sizes{{1, 2}, {0, 0}};
        Standardization standardization;
    };
} // namespace pipeline_core::detail
