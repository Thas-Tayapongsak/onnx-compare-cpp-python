#include "results.h"
#include "log.h"
#include "npz.h"

#include <filesystem>
#include <format>
#include <stdexcept>
#include <vector>

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
} // namespace cpp_pipeline
