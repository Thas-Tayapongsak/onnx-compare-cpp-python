#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pipeline_core::detail
{
    inline constexpr std::array<const char *, 4> kSessionInputNames{
        "pixel_values", "confidence_threshold", "nms_iou_threshold",
        "target_sizes"};

    struct OutputMetadata
    {
        std::string name;
        std::vector<std::int64_t> shape;
        std::vector<std::string> display_shape;
    };

    /// Check the session input names, types, and fixed shapes.
    void check_inputs(const Ort::Session &session);

    /// Read the declared output metadata in session order.
    std::vector<OutputMetadata>
    read_output_metadata(const Ort::Session &session);

    /// Check one runtime output against its declared metadata.
    void check_output(const OutputMetadata &metadata, const Ort::Value &value);

    /// Return the short display name for a tensor element type.
    std::string element_type_name(ONNXTensorElementDataType type);
} // namespace pipeline_core::detail
