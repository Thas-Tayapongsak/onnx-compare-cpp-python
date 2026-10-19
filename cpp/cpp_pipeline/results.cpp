#include "results.h"
#include "log.h"

#include "npz.h"
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string_view>
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

    void write_outputs(const std::filesystem::path &output_dir,
                       const InstanceSegmentationInferenceResults &results)
    {
        const std::vector<NpzEntry> entries{
            {"num_predictions.npy",
             make_npy("<i8", results.num_predictions.shape,
                      results.num_predictions.data)},
            {"scores.npy",
             make_npy("<f4", results.scores.shape, results.scores.data)},
            {"labels.npy",
             make_npy("<i8", results.labels.shape, results.labels.data)},
            {"boxes.npy",
             make_npy("<f4", results.boxes.shape, results.boxes.data)},
            {"masks.npy",
             make_npy("|b1", results.masks.shape, results.masks.data)},
        };
        write_npz(output_dir / "outputs.npz", entries);
        log_info("`outputs.npz` written");
    }

    void write_meta(const std::filesystem::path &output_dir)
    {
        std::ofstream file(output_dir / "meta.json",
                           std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not open {} for writing",
                            (output_dir / "meta.json").string()));
        }
        file << "{\"pipeline\": \"cpp\"}";
        if (!file)
        {
            throw std::runtime_error(std::format(
                "Could not write {}", (output_dir / "meta.json").string()));
        }
        log_info("`meta.json` written");
    }
} // namespace cpp_pipeline
