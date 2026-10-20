#include "session_contract.h"

#include "run_support.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
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

        std::vector<std::string> read_input_names(const Ort::Session &session)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            const std::size_t count = session.GetInputCount();
            std::vector<std::string> names;
            names.reserve(count);
            for (std::size_t index = 0; index < count; ++index)
            {
                Ort::AllocatedStringPtr name =
                    session.GetInputNameAllocated(index, allocator);
                names.emplace_back(name.get());
            }
            return names;
        }

        std::vector<std::string> expected_input_names()
        {
            std::vector<std::string> names;
            names.reserve(kSessionInputNames.size());
            for (const char *name : kSessionInputNames)
            {
                names.emplace_back(name);
            }
            return names;
        }

        std::size_t require_input_index(const std::vector<std::string> &names,
                                        const char *name)
        {
            const auto match = std::find(names.begin(), names.end(), name);
            return static_cast<std::size_t>(
                std::distance(names.begin(), match));
        }

        void require_fixed_input(const Ort::Session &session, std::size_t index,
                                 const char *name,
                                 ONNXTensorElementDataType expected_type,
                                 std::span<const std::int64_t> expected_shape)
        {
            Ort::TypeInfo type_info = session.GetInputTypeInfo(index);
            Ort::ConstTensorTypeAndShapeInfo tensor_info =
                type_info.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType element_type =
                tensor_info.GetElementType();
            if (element_type != expected_type)
            {
                fail("Input '" + std::string(name) + "' must have type " +
                     element_type_name(expected_type) + "; got " +
                     element_type_name(element_type));
            }
            const std::vector<std::int64_t> shape = tensor_info.GetShape();
            if (shape.size() != expected_shape.size() ||
                !std::equal(shape.begin(), shape.end(), expected_shape.begin()))
            {
                fail("Input '" + std::string(name) + "' must have shape " +
                     format_signed_shape(expected_shape) + "; got " +
                     format_signed_shape(shape));
            }
        }
    } // namespace

    std::string element_type_name(ONNXTensorElementDataType type)
    {
        switch (type)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        default:
            return "type_" + std::to_string(static_cast<int>(type));
        }
    }

    void check_inputs(const Ort::Session &session)
    {
        const std::vector<std::string> names = read_input_names(session);
        bool valid = names.size() == kSessionInputNames.size();
        for (const char *name : kSessionInputNames)
        {
            valid = valid && std::count(names.begin(), names.end(), name) == 1;
        }
        if (!valid)
        {
            fail("Expected session inputs " +
                 format_names(expected_input_names()) + "; got " +
                 format_names(names));
        }

        const std::array<std::int64_t, 1> scalar_shape{1};
        const std::array<std::int64_t, 2> target_sizes_shape{1, 2};
        require_fixed_input(session,
                            require_input_index(names, "confidence_threshold"),
                            "confidence_threshold",
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, scalar_shape);
        require_fixed_input(session,
                            require_input_index(names, "nms_iou_threshold"),
                            "nms_iou_threshold",
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, scalar_shape);
        require_fixed_input(session, require_input_index(names, "target_sizes"),
                            "target_sizes", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                            target_sizes_shape);
    }

    std::vector<OutputMetadata>
    read_output_metadata(const Ort::Session &session)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t count = session.GetOutputCount();
        std::vector<OutputMetadata> metadata;
        metadata.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            Ort::AllocatedStringPtr name =
                session.GetOutputNameAllocated(index, allocator);
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(index);
            Ort::ConstTensorTypeAndShapeInfo tensor_info =
                type_info.GetTensorTypeAndShapeInfo();
            const std::vector<const char *> symbols =
                tensor_info.GetSymbolicDimensions();

            OutputMetadata item;
            item.name = name.get();
            item.shape = tensor_info.GetShape();
            item.display_shape.reserve(item.shape.size());
            for (std::size_t dimension = 0; dimension < item.shape.size();
                 ++dimension)
            {
                if (item.shape[dimension] >= 0)
                {
                    item.display_shape.push_back(
                        std::to_string(item.shape[dimension]));
                    continue;
                }
                const char *symbol =
                    dimension < symbols.size() ? symbols[dimension] : nullptr;
                if (symbol != nullptr && symbol[0] != '\0')
                {
                    item.display_shape.push_back("'" + std::string(symbol) +
                                                 "'");
                }
                else
                {
                    item.display_shape.push_back("?");
                }
            }
            metadata.push_back(std::move(item));
        }
        return metadata;
    }

    void check_output(const OutputMetadata &metadata, const Ort::Value &value)
    {
        Ort::TensorTypeAndShapeInfo tensor_info =
            value.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> shape = tensor_info.GetShape();
        bool valid = shape.size() == metadata.shape.size();
        for (std::size_t index = 0; valid && index < shape.size(); ++index)
        {
            valid =
                shape[index] >= 0 && (metadata.shape[index] < 0 ||
                                      metadata.shape[index] == shape[index]);
        }
        if (!valid)
        {
            fail("Output '" + metadata.name +
                 "' does not match its declared shape; declared " +
                 format_text_shape(metadata.display_shape) + "; observed " +
                 format_signed_shape(shape));
        }
    }
} // namespace pipeline_core::detail
