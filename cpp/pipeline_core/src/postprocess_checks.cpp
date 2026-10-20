#include "postprocess_checks.h"

#include "run_support.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipeline_core::detail
{
    namespace
    {
        [[noreturn]] void fail(const std::string &message)
        {
            throw std::runtime_error(message);
        }

        template <typename T>
        void require_shape(const char *name, const NDArray<T> &array,
                           std::span<const std::size_t> expected)
        {
            if (array.shape.size() != expected.size() ||
                !std::equal(array.shape.begin(), array.shape.end(),
                            expected.begin()))
            {
                fail("Output '" + std::string(name) + "' must have shape " +
                     format_size_shape(expected) + "; got " +
                     format_size_shape(array.shape));
            }
        }
    } // namespace

    void
    check_result_counts(const InstanceSegmentationInferenceResults &results)
    {
        constexpr std::array<std::size_t, 1> count_shape{1};
        require_shape("num_predictions", results.num_predictions, count_shape);

        if (results.num_predictions.data.size() != 1U)
        {
            fail("Output 'num_predictions' must hold one element; got " +
                 std::to_string(results.num_predictions.data.size()));
        }

        const std::int64_t count = results.num_predictions.data.front();
        if (count < 0)
        {
            fail("Output 'num_predictions' must not be negative; got " +
                 std::to_string(count));
        }

        const std::size_t rows = static_cast<std::size_t>(count);
        const std::array<std::size_t, 1> row_shape{rows};
        const std::array<std::size_t, 2> box_shape{rows, 4U};
        require_shape("scores", results.scores, row_shape);
        require_shape("labels", results.labels, row_shape);
        require_shape("boxes", results.boxes, box_shape);

        const std::vector<std::size_t> &masks = results.masks.shape;
        if (masks.size() != 3U || masks[0] != rows)
        {
            fail("Output 'masks' must have shape (" + std::to_string(rows) +
                 ", H, W); got " + format_size_shape(masks));
        }
    }
} // namespace pipeline_core::detail
