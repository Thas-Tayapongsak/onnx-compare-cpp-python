#pragma once

#include "pipeline_core/array.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pipeline_core::detail
{
    using Clock = std::chrono::steady_clock;

    /// Return the elapsed integer milliseconds since `start`.
    int elapsed_ms(Clock::time_point start);

    /// Return the integer median of the samples.
    int median_ms(std::vector<int> samples);

    /// Return the current UTC time in ISO 8601 seconds precision.
    std::string utc_now();

    /// Write an informational log line in the executable log format.
    void log_info(std::string_view message);

    /// Return the target sizes as nested lists for `meta.json`.
    std::vector<std::vector<std::int64_t>>
    target_sizes_list(const NDArray<std::int64_t> &target_sizes);

    /// Return the last path component as UTF-8.
    std::string file_name(const std::filesystem::path &path);

    /// Format names as `['a', 'b']`.
    std::string format_names(std::span<const std::string> names);

    /// Format signed dimensions as `(a, b)`.
    std::string format_signed_shape(std::span<const std::int64_t> shape);

    /// Format display dimensions as `[a, b]`.
    std::string format_text_shape(std::span<const std::string> shape);
} // namespace pipeline_core::detail
