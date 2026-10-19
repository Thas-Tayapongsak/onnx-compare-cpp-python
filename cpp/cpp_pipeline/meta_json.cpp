#include "log.h"
#include "results.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    void write_json_string(std::ostream &out, std::string_view value)
    {
        out << '"';
        for (const char character : value)
        {
            const unsigned char byte = static_cast<unsigned char>(character);
            switch (byte)
            {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (byte < 0x20U)
                {
                    out << std::format("\\u{:04x}",
                                       static_cast<unsigned int>(byte));
                }
                else
                {
                    out << static_cast<char>(byte);
                }
                break;
            }
        }
        out << '"';
    }

    void write_json_number(std::ostream &out, double value)
    {
        std::string text = std::format("{}", value);
        if (text.find_first_of(".eE") == std::string::npos)
        {
            text += ".0";
        }
        out << text;
    }

    void write_target_sizes(
        std::ostream &out,
        const std::vector<std::vector<std::int64_t>> &target_sizes)
    {
        out << '[';
        for (std::size_t row = 0; row < target_sizes.size(); ++row)
        {
            if (row != 0U)
            {
                out << ", ";
            }
            out << '[';
            for (std::size_t column = 0; column < target_sizes[row].size();
                 ++column)
            {
                if (column != 0U)
                {
                    out << ", ";
                }
                out << target_sizes[row][column];
            }
            out << ']';
        }
        out << ']';
    }

    void write_meta_fields(std::ostream &out,
                           const pipeline_core::MetaResults &meta)
    {
        out << "{\"pipeline\": ";
        write_json_string(out, meta.pipeline);
        out << ", \"model\": ";
        write_json_string(out, meta.model);
        out << ", \"input_image\": ";
        write_json_string(out, meta.input_image);
        out << ", \"confidence_threshold\": ";
        write_json_number(out, meta.confidence_threshold);
        out << ", \"nms_iou_threshold\": ";
        write_json_number(out, meta.nms_iou_threshold);
        out << ", \"target_sizes\": ";
        write_target_sizes(out, meta.target_sizes);
        out << ", \"created_utc\": ";
        write_json_string(out, meta.created_utc);
        out << ", \"setup_ms\": " << meta.setup_ms;
        out << ", \"preprocess_ms\": " << meta.preprocess_ms;
        out << ", \"inference_ms\": " << meta.inference_ms;
        out << ", \"postprocess_ms\": " << meta.postprocess_ms;
        out << ", \"duration_ms\": " << meta.duration_ms;
        out << ", \"execution_provider\": ";
        write_json_string(out, meta.execution_provider);
        out << ", \"cpu\": ";
        write_json_string(out, meta.cpu);
        out << ", \"gpu\": ";
        if (meta.gpu.has_value())
        {
            write_json_string(out, *meta.gpu);
        }
        else
        {
            out << "null";
        }
        out << '}';
    }
} // namespace

namespace cpp_pipeline
{
    void write_meta(const std::filesystem::path &output_dir,
                    const pipeline_core::MetaResults &meta)
    {
        const std::filesystem::path path = output_dir / "meta.json";
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not open {} for writing", path.string()));
        }
        write_meta_fields(file, meta);
        file.close();
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not write {}", path.string()));
        }
        log_info("`meta.json` written");
    }
} // namespace cpp_pipeline
