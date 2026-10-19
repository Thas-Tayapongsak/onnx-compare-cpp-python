#include "stub.h"

#include <cstddef>

namespace
{
    constexpr std::size_t kMaskCount = 2;
    constexpr std::size_t kMaskSide = 64;
} // namespace

namespace cpp_pipeline
{
    InstanceSegmentationInferenceResults run_stub()
    {
        InstanceSegmentationInferenceResults results;
        results.num_predictions = {2};
        results.scores = {0.9F, 0.8F};
        results.labels = {0, 0};
        results.boxes = {0.0F, 0.0F, 64.0F, 64.0F, 8.0F, 8.0F, 56.0F, 56.0F};
        results.masks_shape = {kMaskCount, kMaskSide, kMaskSide};
        results.masks.assign(kMaskCount * kMaskSide * kMaskSide, 0);

        for (std::size_t y = 0; y < kMaskSide; ++y)
        {
            for (std::size_t x = 0; x < kMaskSide / 2; ++x)
            {
                results.masks[y * kMaskSide + x] = 1;
            }
        }
        for (std::size_t y = 0; y < kMaskSide / 2; ++y)
        {
            for (std::size_t x = 0; x < kMaskSide; ++x)
            {
                results.masks[kMaskSide * kMaskSide + y * kMaskSide + x] = 1;
            }
        }

        return results;
    }
} // namespace cpp_pipeline
