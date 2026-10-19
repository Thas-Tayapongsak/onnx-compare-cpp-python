#pragma once

#include <pipeline_core/array.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace cpp_pipeline
{
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
