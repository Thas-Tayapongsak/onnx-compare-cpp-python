#include "results.h"
#include "log.h"
#include "npz.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ostream>
#include <stdexcept>
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
        out << ", \"confidence_threshold\": "
            << std::format("{}", meta.confidence_threshold);
        out << ", \"nms_iou_threshold\": "
            << std::format("{}", meta.nms_iou_threshold);
        out << ", \"target_sizes\": ";
        write_target_sizes(out, meta.target_sizes);
        out << ", \"created_utc\": ";
        write_json_string(out, meta.created_utc);
        out << ", \"duration_ms\": " << meta.duration_ms;
        out << ", \"execution_provider\": ";
        write_json_string(out, meta.execution_provider);
        out << '}';
    }
} // namespace

namespace cpp_pipeline
{
    void prepare_output_dir(const std::filesystem::path &output_dir)
    {
        if (std::filesystem::exists(output_dir))
        {
            if (!std::filesystem::is_directory(output_dir))
            {
                throw std::runtime_error(std::format(
                    "Output path {} is not a directory", output_dir.string()));
            }
            log_info(std::format("Output folder already exists: {}",
                                 output_dir.string()));
            return;
        }

        log_info(
            std::format("Creating output folder: {}", output_dir.string()));
        std::filesystem::create_directories(output_dir);
    }

    void write_outputs(
        const std::filesystem::path &output_dir,
        const std::vector<pipeline_core::InstanceSegmentationInferenceResults>
            &results)
    {
        if (results.size() != 1U)
        {
            throw std::runtime_error(std::format(
                "outputs.npz supports exactly one inference result; got {}",
                results.size()));
        }
        const pipeline_core::InstanceSegmentationInferenceResults &result =
            results.front();
        const std::vector<NpzEntry> entries{
            {"num_predictions.npy",
             make_npy("<i8", result.num_predictions.shape,
                      result.num_predictions.data)},
            {"scores.npy",
             make_npy("<f4", result.scores.shape, result.scores.data)},
            {"labels.npy",
             make_npy("<i8", result.labels.shape, result.labels.data)},
            {"boxes.npy",
             make_npy("<f4", result.boxes.shape, result.boxes.data)},
            {"masks.npy",
             make_npy("|b1", result.masks.shape, result.masks.data)},
        };
        write_npz(output_dir / "outputs.npz", entries);
        log_info("`outputs.npz` written");
    }

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
