#include "results.h"
#include "log.h"

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <miniz.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
    static_assert(std::endian::native == std::endian::little,
                  "NPY output assumes a little-endian machine");

    constexpr std::array<std::uint8_t, 6> kNpyMagic{0x93, 'N', 'U',
                                                    'M',  'P', 'Y'};

    void append_u16(std::vector<std::uint8_t> &out, std::uint16_t value)
    {
        out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    }

    std::string make_npy_header(std::string_view descr,
                                std::span<const std::size_t> shape)
    {
        std::string header = std::format(
            "{{'descr': '{}', 'fortran_order': False, 'shape': (", descr);
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i != 0)
            {
                header += ", ";
            }
            header += std::to_string(shape[i]);
        }
        if (shape.size() == 1)
        {
            header += ",";
        }
        header += "), }";

        // Pad so the header ends on a 16-byte boundary. numpy uses the same
        // rule, and numpy reads any amount of trailing space.
        header.append(16 - ((header.size() + 10 + 1) % 16), ' ');
        header.push_back('\n');

        if (header.size() > 0xFFFFU)
        {
            throw std::runtime_error("NPY header is too large");
        }
        return header;
    }

    template <typename T>
    std::vector<std::uint8_t> make_npy(std::string_view descr,
                                       std::span<const std::size_t> shape,
                                       const std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable_v<T>);

        const std::string header = make_npy_header(descr, shape);

        std::vector<std::uint8_t> output;
        output.insert(output.end(), kNpyMagic.begin(), kNpyMagic.end());
        output.push_back(1); // NPY major version
        output.push_back(0); // NPY minor version
        append_u16(output, static_cast<std::uint16_t>(header.size()));
        output.insert(output.end(), header.begin(), header.end());

        const auto *begin =
            reinterpret_cast<const std::uint8_t *>(values.data());
        output.insert(output.end(), begin, begin + values.size() * sizeof(T));
        return output;
    }

    class ZipWriter
    {
      public:
        explicit ZipWriter(const std::filesystem::path &path)
        {
            if (mz_zip_writer_init_file(&archive_, path.string().c_str(), 0) ==
                MZ_FALSE)
            {
                throw std::runtime_error(std::format(
                    "Could not open {} for writing", path.string()));
            }
            open_ = true;
        }

        ~ZipWriter()
        {
            if (open_)
            {
                mz_zip_writer_end(&archive_);
            }
        }

        ZipWriter(const ZipWriter &) = delete;
        ZipWriter &operator=(const ZipWriter &) = delete;

        void add(std::string_view name, const std::vector<std::uint8_t> &data)
        {
            const std::string entry_name(name);
            if (mz_zip_writer_add_mem(&archive_, entry_name.c_str(),
                                      data.data(), data.size(),
                                      MZ_NO_COMPRESSION) == MZ_FALSE)
            {
                throw std::runtime_error(
                    std::format("Could not add {} to the ZIP archive", name));
            }
        }

        void finish()
        {
            if (mz_zip_writer_finalize_archive(&archive_) == MZ_FALSE)
            {
                throw std::runtime_error("Could not finalize the ZIP archive");
            }
            const mz_bool closed = mz_zip_writer_end(&archive_);
            open_ = false;
            if (closed == MZ_FALSE)
            {
                throw std::runtime_error("Could not close the ZIP archive");
            }
        }

      private:
        mz_zip_archive archive_{};
        bool open_{false};
    };

    void validate_results(
        const cpp_pipeline::InstanceSegmentationInferenceResults &results)
    {
        if (results.num_predictions.size() != 1)
        {
            throw std::runtime_error("`num_predictions` must have shape [1]");
        }
        if (results.num_predictions.front() < 0)
        {
            throw std::runtime_error("`num_predictions` must not be negative");
        }

        const auto count =
            static_cast<std::size_t>(results.num_predictions.front());
        if (results.scores.size() != count)
        {
            throw std::runtime_error("`scores` size does not match N");
        }
        if (results.labels.size() != count)
        {
            throw std::runtime_error("`labels` size does not match N");
        }
        if (results.boxes.size() != count * 4)
        {
            throw std::runtime_error("`boxes` size does not match [N, 4]");
        }
        if (results.masks_shape[0] != count)
        {
            throw std::runtime_error("`masks` shape does not match N");
        }
        if (results.masks.size() != results.masks_shape[0] *
                                        results.masks_shape[1] *
                                        results.masks_shape[2])
        {
            throw std::runtime_error("`masks` size does not match its shape");
        }
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

    void write_outputs(const std::filesystem::path &output_dir,
                       const InstanceSegmentationInferenceResults &results)
    {
        validate_results(results);

        const auto count =
            static_cast<std::size_t>(results.num_predictions.front());
        const std::array<std::size_t, 1> num_predictions_shape{1};
        const std::array<std::size_t, 1> count_shape{count};
        const std::array<std::size_t, 2> boxes_shape{count, 4};

        ZipWriter archive(output_dir / "outputs.npz");
        archive.add(
            "num_predictions.npy",
            make_npy("<i8", num_predictions_shape, results.num_predictions));
        archive.add("scores.npy", make_npy("<f4", count_shape, results.scores));
        archive.add("labels.npy", make_npy("<i8", count_shape, results.labels));
        archive.add("boxes.npy", make_npy("<f4", boxes_shape, results.boxes));
        archive.add("masks.npy",
                    make_npy("|b1", results.masks_shape, results.masks));
        archive.finish();

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
