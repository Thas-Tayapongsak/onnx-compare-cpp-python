#include "postprocess.h"

#include "postprocess_checks.h"
#include "run_support.h"
#include "session_contract.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pipeline_core::detail
{
    namespace
    {
        constexpr std::array<const char *, 5> kOutputNames{
            "num_predictions", "scores", "labels", "boxes", "masks"};

        [[noreturn]] void fail(const std::string &message)
        {
            throw std::runtime_error(message);
        }

        std::vector<std::string> expected_output_names()
        {
            std::vector<std::string> names;
            names.reserve(kOutputNames.size());
            for (const char *name : kOutputNames)
            {
                names.emplace_back(name);
            }
            return names;
        }

        struct CopyPlan
        {
            std::vector<std::size_t> shape;
            std::size_t element_count{0};
            std::size_t byte_count{0};
        };

        CopyPlan make_copy_plan(const RawOutput &output,
                                ONNXTensorElementDataType expected_type,
                                std::size_t element_size)
        {
            Ort::TensorTypeAndShapeInfo tensor_info =
                output.value.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType element_type =
                tensor_info.GetElementType();
            if (element_type != expected_type)
            {
                fail("Output '" + output.name + "' must have type " +
                     element_type_name(expected_type) + "; got " +
                     element_type_name(element_type));
            }

            const std::vector<std::int64_t> shape = tensor_info.GetShape();
            CopyPlan plan;
            plan.element_count = 1;
            plan.shape.reserve(shape.size());
            for (const std::int64_t dimension : shape)
            {
                if (dimension < 0)
                {
                    fail("Output '" + output.name +
                         "' has a negative runtime dimension; observed " +
                         format_signed_shape(shape));
                }
                const std::size_t value = static_cast<std::size_t>(dimension);
                plan.shape.push_back(value);
                plan.element_count *= value;
            }

            plan.byte_count = output.value.GetTensorSizeInBytes();
            const std::size_t expected_bytes =
                plan.element_count * element_size;
            if (plan.byte_count != expected_bytes)
            {
                fail("Output '" + output.name + "' has " +
                     std::to_string(plan.byte_count) + " bytes; expected " +
                     std::to_string(expected_bytes) + " for shape " +
                     format_signed_shape(shape));
            }
            return plan;
        }

        template <typename T>
        NDArray<T> copy_output(const RawOutput &output,
                               ONNXTensorElementDataType expected_type)
        {
            CopyPlan plan = make_copy_plan(output, expected_type, sizeof(T));
            std::vector<T> data(plan.element_count);
            if (plan.byte_count != 0)
            {
                std::memcpy(data.data(), output.value.GetTensorRawData(),
                            plan.byte_count);
            }
            return NDArray<T>{std::move(plan.shape), std::move(data)};
        }

        const RawOutput &require_output(const RawInferenceResults &raw,
                                        const char *name)
        {
            const RawOutput *match = nullptr;
            std::size_t count = 0;
            for (const RawOutput &output : raw.outputs)
            {
                if (output.name == name)
                {
                    match = &output;
                    ++count;
                }
            }
            if (count != 1)
            {
                fail("Output '" + std::string(name) + "' must occur once");
            }
            return *match;
        }

        void check_output_names(const RawInferenceResults &raw)
        {
            std::vector<std::string> observed;
            observed.reserve(raw.outputs.size());
            for (const RawOutput &output : raw.outputs)
            {
                observed.push_back(output.name);
            }

            bool valid = raw.outputs.size() == kOutputNames.size();
            for (const char *name : kOutputNames)
            {
                valid = valid &&
                        std::count(observed.begin(), observed.end(), name) == 1;
            }
            if (!valid)
            {
                fail("Expected outputs " +
                     format_names(expected_output_names()) + "; got " +
                     format_names(observed));
            }
        }
    } // namespace

    InstanceSegmentationInferenceResults
    package_results(RawInferenceResults raw)
    {
        check_output_names(raw);

        InstanceSegmentationInferenceResults results;
        results.num_predictions =
            copy_output<std::int64_t>(require_output(raw, "num_predictions"),
                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        results.scores = copy_output<float>(
            require_output(raw, "scores"), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        results.labels = copy_output<std::int64_t>(
            require_output(raw, "labels"), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        results.boxes = copy_output<float>(require_output(raw, "boxes"),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        results.masks = copy_output<std::uint8_t>(
            require_output(raw, "masks"), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        check_result_counts(results);
        return results;
    }
} // namespace pipeline_core::detail
