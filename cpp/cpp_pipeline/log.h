#pragma once

#include <string_view>

namespace cpp_pipeline
{
    void log_info(std::string_view message);
    void log_error(std::string_view message);
} // namespace cpp_pipeline
