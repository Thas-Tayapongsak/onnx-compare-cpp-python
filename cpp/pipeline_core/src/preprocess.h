#pragma once

#include "config.h"
#include "pipeline_core/array.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

namespace pipeline_core::detail
{
    using PixelValuesShape = std::array<std::size_t, 4>;

    /// Return the static `pixel_values` input shape from the session.
    PixelValuesShape read_pixel_values_shape(const Ort::Session &session);

    /// Read an image and build the session pixel values.
    NDArray<float> build_pixel_values(const std::filesystem::path &image_path,
                                      const PixelValuesShape &target_shape,
                                      const Standardization &standardization);

    /// Format dimensions as `(a, b, c, d)`.
    std::string format_shape(std::span<const std::size_t> shape);
} // namespace pipeline_core::detail
