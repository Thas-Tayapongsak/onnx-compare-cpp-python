#pragma once

#include "pipeline_core/array.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pipeline_core
{
    struct InstanceSegmentationInferenceResults
    {
        NDArray<std::int64_t> num_predictions;
        NDArray<float> scores;
        NDArray<std::int64_t> labels;
        NDArray<float> boxes;
        NDArray<std::uint8_t> masks;
    };

    struct MetaResults
    {
        std::string pipeline;
        std::string model;
        std::string input_image;
        double confidence_threshold{0.0};
        double nms_iou_threshold{0.0};
        std::vector<std::vector<std::int64_t>> target_sizes;
        std::string created_utc;
        std::int64_t setup_ms{0};
        std::int64_t preprocess_ms{0};
        std::int64_t inference_ms{0};
        std::int64_t postprocess_ms{0};
        std::int64_t duration_ms{0};
        std::string execution_provider;
        std::string cpu;
        std::optional<std::string> gpu;
        double setup_ram_gb{0.0};
        double preprocess_ram_gb{0.0};
        double inference_ram_gb{0.0};
        double postprocess_ram_gb{0.0};
        double ram_gb{0.0};
        double setup_vram_gb{0.0};
        double preprocess_vram_gb{0.0};
        double inference_vram_gb{0.0};
        double postprocess_vram_gb{0.0};
        double vram_gb{0.0};
    };

    struct RunTimings
    {
        int setup_ms{0};
        int preprocess_ms{0};
        int inference_ms{0};
        int postprocess_ms{0};
        int duration_ms{0};
    };

    struct RunOutput
    {
        std::vector<InstanceSegmentationInferenceResults> results;
        MetaResults meta;
        RunTimings timings;
    };
} // namespace pipeline_core
