#pragma once

#include "pipeline_core/array.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
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

    /// Return the target sizes as nested lists for `meta.json`.
    std::vector<std::vector<std::int64_t>>
    target_sizes_list(const NDArray<std::int64_t> &target_sizes);

    /// Return the last path component as UTF-8.
    std::string file_name(const std::filesystem::path &path);
} // namespace pipeline_core::detail
