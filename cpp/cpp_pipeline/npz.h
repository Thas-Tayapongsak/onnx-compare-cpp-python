#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace cpp_pipeline
{
    template <typename T> class NDArray
    {
      public:
        std::vector<std::size_t> shape;
        std::vector<T> data;

        NDArray() = default;
        NDArray(std::vector<std::size_t> shape_, std::vector<T> data_)
            : shape(std::move(shape_)), data(std::move(data_))
        {
        }

        operator std::vector<T> &() { return data; }
        operator const std::vector<T> &() const { return data; }
    };

    template <typename T>
    std::vector<std::uint8_t> make_npy(std::string_view descr,
                                       const std::vector<std::size_t> &shape,
                                       const std::vector<T> &values);

    struct NpzEntry
    {
        std::string_view name;
        std::vector<std::uint8_t> data;
    };

    void write_npz(const std::filesystem::path &path,
                   const std::vector<NpzEntry> &entries);
} // namespace cpp_pipeline
