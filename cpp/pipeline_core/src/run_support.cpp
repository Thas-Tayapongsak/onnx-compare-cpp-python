#include "run_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace
{
    std::string utf8_string(const std::filesystem::path &path)
    {
        const std::u8string value = path.u8string();
        return std::string(reinterpret_cast<const char *>(value.data()),
                           value.size());
    }
} // namespace

namespace pipeline_core::detail
{
    int elapsed_ms(Clock::time_point start)
    {
        const auto elapsed = Clock::now() - start;
        return static_cast<int>(std::llround(
            std::chrono::duration<double, std::milli>(elapsed).count()));
    }

    int median_ms(std::vector<int> samples)
    {
        std::sort(samples.begin(), samples.end());
        const std::size_t count = samples.size();
        if (count % 2U == 1U)
        {
            return samples[count / 2U];
        }
        return (samples[count / 2U - 1U] + samples[count / 2U]) / 2;
    }

    std::string utc_now()
    {
        const std::time_t time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm utc_time{};
        gmtime_s(&utc_time, &time);
        std::ostringstream text;
        text << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
        return text.str();
    }

    void log_info(std::string_view message)
    {
        std::cerr << utc_now() << " - cpp_pipeline - INFO - " << message
                  << '\n';
    }

    std::vector<std::vector<std::int64_t>>
    target_sizes_list(const NDArray<std::int64_t> &target_sizes)
    {
        if (target_sizes.shape.size() != 2U)
        {
            throw std::runtime_error("target_sizes must have rank 2");
        }
        const std::size_t rows = target_sizes.shape[0];
        const std::size_t columns = target_sizes.shape[1];
        if (target_sizes.data.size() != rows * columns)
        {
            throw std::runtime_error(
                "target_sizes data does not match its shape");
        }

        std::vector<std::vector<std::int64_t>> values;
        for (std::size_t row = 0; row < rows; ++row)
        {
            std::vector<std::int64_t> value;
            value.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                value.push_back(target_sizes.data[row * columns + column]);
            }
            values.push_back(std::move(value));
        }
        return values;
    }

    std::string file_name(const std::filesystem::path &path)
    {
        const std::filesystem::path normalized = path.lexically_normal();
        std::filesystem::path name = normalized.filename();
        if (name.empty())
        {
            name = normalized.parent_path().filename();
        }
        return utf8_string(name);
    }

    std::string format_names(std::span<const std::string> names)
    {
        std::string text = "[";
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            if (index != 0)
            {
                text += ", ";
            }
            text += '\'';
            text += names[index];
            text += '\'';
        }
        return text + "]";
    }

    std::string format_signed_shape(std::span<const std::int64_t> shape)
    {
        std::string text = "(";
        for (std::size_t index = 0; index < shape.size(); ++index)
        {
            if (index != 0)
            {
                text += ", ";
            }
            text += std::to_string(shape[index]);
        }
        return text + ")";
    }

    std::string format_text_shape(std::span<const std::string> shape)
    {
        std::string text = "[";
        for (std::size_t index = 0; index < shape.size(); ++index)
        {
            if (index != 0)
            {
                text += ", ";
            }
            text += shape[index];
        }
        return text + "]";
    }
} // namespace pipeline_core::detail
