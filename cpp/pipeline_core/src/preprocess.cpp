#include "preprocess.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr const char *kInputName = "pixel_values";
    constexpr std::size_t kChannelCount = 3;
    constexpr float kPixelScale = 255.0F;

    [[noreturn]] void fail(const std::string &message)
    {
        throw std::runtime_error(message);
    }

    std::string format_names(const std::vector<std::string> &names)
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

    std::string format_signed_shape(const std::vector<std::int64_t> &shape)
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
} // namespace

namespace pipeline_core::detail
{
    std::string format_shape(std::span<const std::size_t> shape)
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

    PixelValuesShape read_pixel_values_shape(const Ort::Session &session)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> names;
        std::vector<std::size_t> matches;
        const std::size_t input_count = session.GetInputCount();
        names.reserve(input_count);
        for (std::size_t index = 0; index < input_count; ++index)
        {
            Ort::AllocatedStringPtr name =
                session.GetInputNameAllocated(index, allocator);
            names.emplace_back(name.get());
            if (names.back() == kInputName)
            {
                matches.push_back(index);
            }
        }
        if (matches.size() != 1U)
        {
            fail("Expected exactly one '" + std::string(kInputName) +
                 "' input; model inputs: " + format_names(names));
        }

        Ort::TypeInfo type_info = session.GetInputTypeInfo(matches.front());
        Ort::ConstTensorTypeAndShapeInfo tensor_info =
            type_info.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType element_type =
            tensor_info.GetElementType();
        if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            fail("Input '" + std::string(kInputName) +
                 "' must have type tensor(float); got element type " +
                 std::to_string(static_cast<int>(element_type)));
        }

        const std::vector<std::int64_t> shape = tensor_info.GetShape();
        if (shape.size() != 4U)
        {
            fail("Input '" + std::string(kInputName) +
                 "' must have rank 4; got " + format_signed_shape(shape));
        }
        for (const std::int64_t dimension : shape)
        {
            if (dimension <= 0)
            {
                fail("Input '" + std::string(kInputName) +
                     "' must have static positive dimensions; got " +
                     format_signed_shape(shape));
            }
        }

        const std::size_t batch = static_cast<std::size_t>(shape[0]);
        const std::size_t channels = static_cast<std::size_t>(shape[1]);
        const std::size_t height = static_cast<std::size_t>(shape[2]);
        const std::size_t width = static_cast<std::size_t>(shape[3]);
        if (batch != 1U)
        {
            fail("Input '" + std::string(kInputName) +
                 "' must have batch 1; got batch " + std::to_string(batch) +
                 " in shape " + format_signed_shape(shape));
        }
        if (channels != kChannelCount)
        {
            fail("Input '" + std::string(kInputName) +
                 "' must have 3 channels; got " + std::to_string(channels) +
                 " in shape " + format_signed_shape(shape));
        }
        return {batch, channels, height, width};
    }

    NDArray<float> build_pixel_values(const std::filesystem::path &image_path,
                                      const PixelValuesShape &target_shape,
                                      const Standardization &standardization)
    {
        const std::size_t height = target_shape[2];
        const std::size_t width = target_shape[3];
        const std::size_t max_dimension =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (height > max_dimension || width > max_dimension)
        {
            fail("Session pixel_values dimensions are too large for OpenCV");
        }

        const cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty())
        {
            fail("Could not decode image: " + image_path.string());
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat resized;
        cv::resize(rgb, resized,
                   cv::Size(static_cast<int>(width), static_cast<int>(height)),
                   0.0, 0.0, cv::INTER_LINEAR);

        const std::size_t plane = height * width;
        NDArray<float> pixel_values{{1, 3, height, width},
                                    std::vector<float>(kChannelCount * plane)};
        for (std::size_t row = 0; row < height; ++row)
        {
            const cv::Vec3b *line =
                resized.ptr<cv::Vec3b>(static_cast<int>(row));
            for (std::size_t column = 0; column < width; ++column)
            {
                for (std::size_t channel = 0; channel < kChannelCount;
                     ++channel)
                {
                    float value =
                        static_cast<float>(line[column][channel]) / kPixelScale;
                    if (standardization.enabled)
                    {
                        value = (value - standardization.mean[channel]) /
                                standardization.stddev[channel];
                    }
                    pixel_values.data[channel * plane + row * width + column] =
                        value;
                }
            }
        }
        return pixel_values;
    }
} // namespace pipeline_core::detail
