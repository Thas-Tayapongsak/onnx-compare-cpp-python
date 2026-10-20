#include "config.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    using Json = nlohmann::json;
    namespace fs = std::filesystem;
    constexpr std::size_t kChannelCount = 3;
    constexpr const char *kEvalMetricsFile = "eval_metrics.json";
    constexpr const char *kPreprocessorConfigFile = "preprocessor_config.json";
    constexpr const char *kThresholdKey = "eval_optimal_threshold";
    struct StandardizationKeys
    {
        const char *mean;
        const char *stddev;
    };

    [[noreturn]] void fail(const std::string &message)
    {
        throw std::runtime_error(message);
    }

    std::string field_error(const fs::path &path, std::string_view key,
                            std::string_view message)
    {
        return path.string() + " field '" + std::string(key) + "' " +
               std::string(message);
    }

    Json read_json_object(const fs::path &path)
    {
        if (!fs::is_regular_file(path))
        {
            fail("Model settings file not found: " + path.string());
        }
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            fail("Could not open model settings file: " + path.string());
        }
        Json data;
        try
        {
            data = Json::parse(file);
        }
        catch (const Json::exception &error)
        {
            fail("Invalid JSON in " + path.string() + ": " + error.what());
        }
        if (!data.is_object())
        {
            fail("Model settings file must contain a JSON object: " +
                 path.string());
        }
        return data;
    }

    double read_number(const Json &value, const fs::path &path,
                       std::string_view key)
    {
        if (!value.is_number())
        {
            fail(field_error(path, key, "must be a number"));
        }
        const double number = value.get<double>();
        if (!std::isfinite(number))
        {
            fail(field_error(path, key, "must be finite"));
        }
        return number;
    }

    std::array<float, 3> read_channel_values(const Json &value,
                                             const fs::path &path,
                                             std::string_view key)
    {
        std::array<float, 3> values{};
        if (value.is_number())
        {
            values.fill(static_cast<float>(read_number(value, path, key)));
        }
        else if (value.is_array() && value.size() == kChannelCount)
        {
            for (std::size_t index = 0; index < kChannelCount; ++index)
            {
                values[index] =
                    static_cast<float>(read_number(value.at(index), path, key));
            }
        }
        else
        {
            fail(field_error(path, key,
                             "must be a number or a list of three numbers"));
        }
        for (const float channel_value : values)
        {
            if (!std::isfinite(channel_value))
            {
                fail(field_error(path, key, "must be finite"));
            }
        }
        return values;
    }

    bool select_standardization_keys(const Json &data, const fs::path &path,
                                     StandardizationKeys &keys)
    {
        const bool has_mean = data.contains("mean");
        const bool has_std = data.contains("std");
        if (has_mean || has_std)
        {
            if (!(has_mean && has_std))
            {
                fail(path.string() + " must contain both 'mean' and 'std'");
            }
            keys = StandardizationKeys{"mean", "std"};
            return true;
        }
        const bool has_image_mean = data.contains("image_mean");
        const bool has_image_std = data.contains("image_std");
        if (has_image_mean || has_image_std)
        {
            if (!(has_image_mean && has_image_std))
            {
                fail(path.string() +
                     " must contain both 'image_mean' and 'image_std'");
            }
            keys = StandardizationKeys{"image_mean", "image_std"};
            return true;
        }
        return false;
    }
} // namespace

namespace pipeline_core::detail
{
    double read_confidence_threshold(const fs::path &folder)
    {
        const fs::path path = folder / kEvalMetricsFile;
        const Json data = read_json_object(path);
        if (!data.contains(kThresholdKey))
        {
            fail(path.string() + " is missing '" + kThresholdKey + "'");
        }
        const double threshold =
            read_number(data.at(kThresholdKey), path, kThresholdKey);
        if (threshold < 0.0 || threshold > 1.0)
        {
            fail(field_error(path, kThresholdKey, "must be in [0, 1]"));
        }
        return threshold;
    }

    Standardization read_standardization(const fs::path &folder)
    {
        const fs::path path = folder / kPreprocessorConfigFile;
        if (!fs::exists(path))
        {
            return {};
        }
        const Json data = read_json_object(path);
        StandardizationKeys keys{};
        if (!select_standardization_keys(data, path, keys))
        {
            return {};
        }
        const Json &mean_value = data.at(keys.mean);
        const Json &stddev_value = data.at(keys.stddev);
        if (mean_value.is_number() != stddev_value.is_number())
        {
            fail(path.string() + " fields '" + keys.mean + "' and '" +
                 keys.stddev + "' must have the same shape");
        }
        const std::array<float, 3> mean =
            read_channel_values(mean_value, path, keys.mean);
        const std::array<float, 3> stddev =
            read_channel_values(stddev_value, path, keys.stddev);
        for (const float channel_value : stddev)
        {
            if (channel_value <= 0.0F)
            {
                fail(field_error(path, keys.stddev,
                                 "must be greater than zero"));
            }
        }
        Standardization standardization;
        standardization.mean = mean;
        standardization.stddev = stddev;
        standardization.enabled = true;
        return standardization;
    }
} // namespace pipeline_core::detail
