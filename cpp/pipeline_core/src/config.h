#pragma once

#include <array>
#include <filesystem>

namespace pipeline_core::detail
{
    struct Standardization
    {
        std::array<float, 3> mean{};
        std::array<float, 3> stddev{1.0F, 1.0F, 1.0F};
        bool enabled{false};
    };

    double read_confidence_threshold(const std::filesystem::path &folder);

    Standardization read_standardization(const std::filesystem::path &folder);
} // namespace pipeline_core::detail
