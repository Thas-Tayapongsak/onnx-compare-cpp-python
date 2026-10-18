#include "log.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace
{
    void log(std::string_view level, std::string_view message)
    {
        const std::time_t time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm utc_time{};
        gmtime_s(&utc_time, &time);

        std::cerr << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ")
                  << " - cpp_pipeline - " << level << " - " << message << '\n';
    }
} // namespace

namespace cpp_pipeline
{
    void log_info(std::string_view message) { log("INFO", message); }
    void log_error(std::string_view message) { log("ERROR", message); }
} // namespace cpp_pipeline
