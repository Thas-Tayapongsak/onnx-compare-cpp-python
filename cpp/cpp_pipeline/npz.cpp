#include "npz.h"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <miniz.h>
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
                                const std::vector<std::size_t> &shape)
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

        // Pad so the header ends on a 16-byte boundary.

        header.append(16 - ((header.size() + 10 + 1) % 16), ' ');
        header.push_back('\n');

        if (header.size() > 0xFFFFU)
        {
            throw std::runtime_error("NPY header is too large");
        }
        return header;
    }
} // namespace

namespace cpp_pipeline
{
    template <typename T>
    std::vector<std::uint8_t> make_npy(std::string_view descr,
                                       const std::vector<std::size_t> &shape,
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
    template std::vector<std::uint8_t>
    make_npy<std::int64_t>(std::string_view, const std::vector<std::size_t> &,
                           const std::vector<std::int64_t> &);
    template std::vector<std::uint8_t>
    make_npy<float>(std::string_view, const std::vector<std::size_t> &,
                    const std::vector<float> &);
    template std::vector<std::uint8_t>
    make_npy<std::uint8_t>(std::string_view, const std::vector<std::size_t> &,
                           const std::vector<std::uint8_t> &);

    class ZipWriter
    {
      public:
        explicit ZipWriter(const std::filesystem::path &path) : path_(path)
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
            if (!finished_)
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }
        }

        ZipWriter(const ZipWriter &) = delete;
        ZipWriter &operator=(const ZipWriter &) = delete;

        void add(std::string_view name, const std::vector<std::uint8_t> &data)
        {
            if (!open_)
            {
                throw std::logic_error(
                    "ZipWriter::add called on a closed archive");
            }
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
            if (!open_)
            {
                throw std::logic_error(
                    "ZipWriter::finish called on a closed archive");
            }
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
            finished_ = true;
        }

      private:
        std::filesystem::path path_;
        mz_zip_archive archive_{};
        bool open_{false};
        bool finished_{false};
    };

    void write_npz(const std::filesystem::path &path,
                   const std::vector<NpzEntry> &entries)
    {
        ZipWriter archive(path);
        for (const NpzEntry &entry : entries)
        {
            archive.add(entry.name, entry.data);
        }
        archive.finish();
    }
} // namespace cpp_pipeline
