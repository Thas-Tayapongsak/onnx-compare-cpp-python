#include "pipeline_core/stub.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
    constexpr std::size_t kPredictionCount = 2;
} // namespace

namespace pipeline_core
{
    InstanceSegmentationInferenceResults
    run_stub(const InferencePayload &payload)
    {
        std::int64_t height = payload.target_sizes.data.at(0);
        std::int64_t width = payload.target_sizes.data.at(1);
        if (height == 0 || width == 0)
        {
            height =
                static_cast<std::int64_t>(payload.pixel_values.shape.at(2));
            width = static_cast<std::int64_t>(payload.pixel_values.shape.at(3));
        }

        const std::size_t mask_height = static_cast<std::size_t>(height);
        const std::size_t mask_width = static_cast<std::size_t>(width);
        const double width_value = static_cast<double>(width);
        const double height_value = static_cast<double>(height);

        InstanceSegmentationInferenceResults results;
        results.num_predictions =
            NDArray<std::int64_t>{{1}, {kPredictionCount}};
        results.scores = NDArray<float>{{kPredictionCount}, {0.9F, 0.8F}};
        results.labels = NDArray<std::int64_t>{{kPredictionCount}, {0, 0}};
        results.boxes = NDArray<float>{
            {kPredictionCount, 4},
            {0.0F, 0.0F, static_cast<float>(width), static_cast<float>(height),
             static_cast<float>(width_value * 0.125),
             static_cast<float>(height_value * 0.125),
             static_cast<float>(width_value * 0.875),
             static_cast<float>(height_value * 0.875)}};
        results.masks = NDArray<std::uint8_t>{
            {kPredictionCount, mask_height, mask_width},
            std::vector<std::uint8_t>(
                kPredictionCount * mask_height * mask_width, 0)};

        for (std::size_t y = 0; y < mask_height; ++y)
        {
            for (std::size_t x = 0; x < mask_width / 2U; ++x)
            {
                results.masks.data[y * mask_width + x] = 1;
            }
        }
        for (std::size_t y = 0; y < mask_height / 2U; ++y)
        {
            for (std::size_t x = 0; x < mask_width; ++x)
            {
                results.masks
                    .data[mask_height * mask_width + y * mask_width + x] = 1;
            }
        }

        return results;
    }
} // namespace pipeline_core
