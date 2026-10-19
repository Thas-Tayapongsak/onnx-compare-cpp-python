#include "results.h"
#include "log.h"

#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <fstream>
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

    void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value)
    {
        out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
    }

    std::uint32_t crc32(const std::uint8_t *data, std::size_t size)
    {
        static const std::array<std::uint32_t, 256> table = []
        {
            std::array<std::uint32_t, 256> values{};
            for (std::uint32_t i = 0; i < values.size(); ++i)
            {
                std::uint32_t value = i;
                for (int bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1U) ? (0xEDB88320U ^ (value >> 1U))
                                         : (value >> 1U);
                }
                values[i] = value;
            }
            return values;
        }();

        std::uint32_t crc = 0xFFFFFFFFU;
        for (std::size_t i = 0; i < size; ++i)
        {
            crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
        }
        return crc ^ 0xFFFFFFFFU;
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

    struct ZipEntry
    {
        std::string name;
        std::vector<std::uint8_t> data;
        std::uint32_t crc{};
        std::uint32_t offset{};
    };

    void write_zip(const std::filesystem::path &path,
                   std::vector<ZipEntry> &entries)
    {
        if (entries.size() > 0xFFFFU)
        {
            throw std::runtime_error("ZIP archive has too many entries");
        }

        std::vector<std::uint8_t> archive;
        for (ZipEntry &entry : entries)
        {
            if (entry.name.size() > 0xFFFFU || entry.data.size() > 0xFFFFFFFFU)
            {
                throw std::runtime_error("ZIP entry is too large");
            }

            entry.crc = crc32(entry.data.data(), entry.data.size());
            entry.offset = static_cast<std::uint32_t>(archive.size());

            // Local file header.
            append_u32(archive, 0x04034B50U);
            append_u16(archive, 20); // version needed to extract
            append_u16(archive, 0);  // general purpose flags
            append_u16(archive, 0);  // compression method: stored
            append_u16(archive, 0);  // modification time
            append_u16(archive, 0);  // modification date
            append_u32(archive, entry.crc);
            append_u32(archive, static_cast<std::uint32_t>(entry.data.size()));
            append_u32(archive, static_cast<std::uint32_t>(entry.data.size()));
            append_u16(archive, static_cast<std::uint16_t>(entry.name.size()));
            append_u16(archive, 0); // extra field length
            archive.insert(archive.end(), entry.name.begin(), entry.name.end());
            archive.insert(archive.end(), entry.data.begin(), entry.data.end());
        }

        const std::size_t central_directory_offset = archive.size();
        for (const ZipEntry &entry : entries)
        {
            // Central directory file header.
            append_u32(archive, 0x02014B50U);
            append_u16(archive, 20); // version made by
            append_u16(archive, 20); // version needed to extract
            append_u16(archive, 0);  // general purpose flags
            append_u16(archive, 0);  // compression method: stored
            append_u16(archive, 0);  // modification time
            append_u16(archive, 0);  // modification date
            append_u32(archive, entry.crc);
            append_u32(archive, static_cast<std::uint32_t>(entry.data.size()));
            append_u32(archive, static_cast<std::uint32_t>(entry.data.size()));
            append_u16(archive, static_cast<std::uint16_t>(entry.name.size()));
            append_u16(archive, 0); // extra field length
            append_u16(archive, 0); // file comment length
            append_u16(archive, 0); // disk number start
            append_u16(archive, 0); // internal file attributes
            append_u32(archive, 0); // external file attributes
            append_u32(archive, entry.offset);
            archive.insert(archive.end(), entry.name.begin(), entry.name.end());
        }
        const std::size_t central_directory_size =
            archive.size() - central_directory_offset;

        // End of central directory record.
        append_u32(archive, 0x06054B50U);
        append_u16(archive, 0); // number of this disk
        append_u16(archive, 0); // disk with the central directory
        append_u16(archive, static_cast<std::uint16_t>(entries.size()));
        append_u16(archive, static_cast<std::uint16_t>(entries.size()));
        append_u32(archive, static_cast<std::uint32_t>(central_directory_size));
        append_u32(archive,
                   static_cast<std::uint32_t>(central_directory_offset));
        append_u16(archive, 0); // archive comment length

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not open {} for writing", path.string()));
        }
        file.write(reinterpret_cast<const char *>(archive.data()),
                   static_cast<std::streamsize>(archive.size()));
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not write {}", path.string()));
        }
    }

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

        std::vector<ZipEntry> entries;
        entries.push_back(
            {"num_predictions.npy",
             make_npy("<i8", num_predictions_shape, results.num_predictions)});
        entries.push_back(
            {"scores.npy", make_npy("<f4", count_shape, results.scores)});
        entries.push_back(
            {"labels.npy", make_npy("<i8", count_shape, results.labels)});
        entries.push_back(
            {"boxes.npy", make_npy("<f4", boxes_shape, results.boxes)});
        entries.push_back(
            {"masks.npy", make_npy("|b1", results.masks_shape, results.masks)});

        write_zip(output_dir / "outputs.npz", entries);
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
