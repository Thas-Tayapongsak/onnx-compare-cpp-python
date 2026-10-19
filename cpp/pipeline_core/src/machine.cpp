#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "pipeline_core/machine.h"

#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{
    constexpr wchar_t kCpuKey[] =
        L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
    constexpr wchar_t kCpuValue[] = L"ProcessorNameString";

    std::wstring read_registry_string(const wchar_t *key, const wchar_t *value)
    {
        DWORD type = 0;
        DWORD size = 0;
        LSTATUS status = RegGetValueW(HKEY_LOCAL_MACHINE, key, value,
                                      RRF_RT_REG_SZ, &type, nullptr, &size);
        if (status != ERROR_SUCCESS || size < sizeof(wchar_t))
        {
            throw std::runtime_error(std::format(
                "Could not read ProcessorNameString (error {})", status));
        }

        std::wstring text(size / sizeof(wchar_t), L'\0');
        status = RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_SZ,
                              &type, text.data(), &size);
        if (status != ERROR_SUCCESS)
        {
            throw std::runtime_error(std::format(
                "Could not read ProcessorNameString (error {})", status));
        }

        text.resize(size / sizeof(wchar_t));
        if (!text.empty() && text.back() == L'\0')
        {
            text.pop_back();
        }
        return text;
    }

    std::wstring trim(std::wstring text)
    {
        const auto is_space = [](wchar_t character)
        {
            return character == L' ' || character == L'\t' ||
                   character == L'\r' || character == L'\n';
        };
        const auto first = std::find_if_not(text.begin(), text.end(), is_space);
        const auto last =
            std::find_if_not(text.rbegin(), text.rend(), is_space).base();
        if (first >= last)
        {
            return {};
        }
        return std::wstring(first, last);
    }

    std::string to_utf8(const std::wstring &text)
    {
        if (text.empty())
        {
            return {};
        }
        const int size = WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()),
                                             nullptr, 0, nullptr, nullptr);
        if (size <= 0)
        {
            throw std::runtime_error(
                "Could not convert ProcessorNameString to UTF-8");
        }

        std::string result(static_cast<std::size_t>(size), '\0');
        if (WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                static_cast<int>(text.size()), result.data(),
                                size, nullptr, nullptr) <= 0)
        {
            throw std::runtime_error(
                "Could not convert ProcessorNameString to UTF-8");
        }
        return result;
    }
} // namespace

namespace pipeline_core
{
    MachineInfo read_machine_info()
    {
        const std::wstring cpu = trim(read_registry_string(kCpuKey, kCpuValue));
        if (cpu.empty())
        {
            throw std::runtime_error("ProcessorNameString is empty");
        }
        return MachineInfo{to_utf8(cpu), std::nullopt};
    }
} // namespace pipeline_core
