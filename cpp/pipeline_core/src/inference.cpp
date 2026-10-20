#include "inference.h"

#include "preprocess.h"
#include "run_support.h"
#include "session_contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pipeline_core::detail
{
    namespace
    {
        [[noreturn]] void fail(const std::string &message)
        {
            throw std::runtime_error(message);
        }

        void check_pixel_payload(const PreprocessedPayload &payload,
                                 const PixelValuesShape &session_shape)
        {
            const std::vector<std::size_t> &shape = payload.pixel_values.shape;
            bool valid = shape.size() == session_shape.size();
            for (std::size_t index = 0; valid && index < shape.size(); ++index)
            {
                valid = shape[index] == session_shape[index];
            }
            if (!valid)
            {
                fail("Pixel values must have session shape " +
                     format_size_shape(session_shape) + "; got " +
                     format_size_shape(shape));
            }

            std::size_t count = 1;
            for (const std::size_t dimension : shape)
            {
                count *= dimension;
            }
            if (count != payload.pixel_values.data.size())
            {
                fail("Pixel values hold " +
                     std::to_string(payload.pixel_values.data.size()) +
                     " elements; expected " + std::to_string(count) +
                     " for shape " + format_size_shape(shape));
            }
        }

        std::string
        format_session_outputs(const std::vector<OutputMetadata> &metadata)
        {
            std::string text = "[";
            for (std::size_t index = 0; index < metadata.size(); ++index)
            {
                if (index != 0)
                {
                    text += ", ";
                }
                text += metadata[index].name;
                text += ' ';
                text += format_text_shape(metadata[index].display_shape);
            }
            return text + "]";
        }

        std::string format_raw_outputs(const std::vector<RawOutput> &outputs)
        {
            std::string text;
            for (std::size_t index = 0; index < outputs.size(); ++index)
            {
                if (index != 0)
                {
                    text += ", ";
                }
                Ort::TensorTypeAndShapeInfo tensor_info =
                    outputs[index].value.GetTensorTypeAndShapeInfo();
                text += outputs[index].name;
                text += ' ';
                text += format_signed_shape(tensor_info.GetShape());
                text += ' ';
                text += element_type_name(tensor_info.GetElementType());
            }
            return text;
        }
    } // namespace

    RawInferenceResults run_session(PipelineState &state,
                                    const PreprocessedPayload &payload)
    {
        check_inputs(state.session);
        const PixelValuesShape session_shape =
            read_pixel_values_shape(state.session);
        check_pixel_payload(payload, session_shape);

        const Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<float, 1> confidence{
            static_cast<float>(state.confidence_threshold)};
        std::array<float, 1> nms_iou{
            static_cast<float>(state.nms_iou_threshold)};
        std::array<std::int64_t, 4> pixel_shape{
            static_cast<std::int64_t>(session_shape[0]),
            static_cast<std::int64_t>(session_shape[1]),
            static_cast<std::int64_t>(session_shape[2]),
            static_cast<std::int64_t>(session_shape[3])};
        std::array<std::int64_t, 2> target_sizes_shape{1, 2};
        std::array<std::int64_t, 1> scalar_shape{1};

        std::vector<Ort::Value> inputs;
        inputs.reserve(kSessionInputNames.size());
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float *>(payload.pixel_values.data.data()),
            payload.pixel_values.data.size(), pixel_shape.data(),
            pixel_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, confidence.data(), confidence.size(),
            scalar_shape.data(), scalar_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, nms_iou.data(), nms_iou.size(), scalar_shape.data(),
            scalar_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info, state.target_sizes.data.data(),
            state.target_sizes.data.size(), target_sizes_shape.data(),
            target_sizes_shape.size()));

        const std::vector<OutputMetadata> metadata =
            read_output_metadata(state.session);
        log_info("Session outputs: " + format_session_outputs(metadata));

        std::vector<const char *> output_names;
        output_names.reserve(metadata.size());
        for (const OutputMetadata &item : metadata)
        {
            output_names.push_back(item.name.c_str());
        }

        Ort::RunOptions run_options;
        std::vector<Ort::Value> values = state.session.Run(
            run_options, kSessionInputNames.data(), inputs.data(),
            inputs.size(), output_names.data(), output_names.size());
        if (values.size() != metadata.size())
        {
            fail("Session returned " + std::to_string(values.size()) +
                 " outputs; expected " + std::to_string(metadata.size()));
        }

        RawInferenceResults raw;
        raw.outputs.reserve(values.size());
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            check_output(metadata[index], values[index]);
            raw.outputs.push_back(
                RawOutput{metadata[index].name, std::move(values[index])});
        }
        log_info("Raw outputs: " + format_raw_outputs(raw.outputs));
        return raw;
    }
} // namespace pipeline_core::detail
