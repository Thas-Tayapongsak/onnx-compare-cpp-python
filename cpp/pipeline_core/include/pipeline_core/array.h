#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace pipeline_core
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
} // namespace pipeline_core
