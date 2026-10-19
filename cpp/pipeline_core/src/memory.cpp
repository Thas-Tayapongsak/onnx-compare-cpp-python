#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "pipeline_core/memory.h"

#include <windows.h>

#include <psapi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    constexpr double kGib = 1024.0 * 1024.0 * 1024.0;
    constexpr std::chrono::milliseconds kSampleInterval{50};
    constexpr std::array<std::string_view, 6> kStages{
        "startup", "setup", "preprocess", "inference", "postprocess", "output"};

    bool is_stage(std::string_view stage)
    {
        return std::find(kStages.begin(), kStages.end(), stage) !=
               kStages.end();
    }

    [[noreturn]] void throw_win32_error(unsigned long error)
    {
        throw std::runtime_error(
            std::format("GetProcessMemoryInfo failed with error {}", error));
    }
} // namespace

namespace pipeline_core
{
    MemoryMonitor::MemoryMonitor()
    {
        for (const std::string_view stage : kStages)
        {
            _stage_peaks.emplace(std::string(stage), 0.0);
        }
        _stage = "startup";
    }

    MemoryMonitor::~MemoryMonitor()
    {
        if (_thread.joinable())
        {
            _thread.request_stop();
            _condition.notify_all();
            _thread.join();
        }
    }

    void MemoryMonitor::start()
    {
        if (_thread.joinable())
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (!sample_locked())
            {
                throw_win32_error(_last_error);
            }
        }
        _thread = std::jthread([this](std::stop_token stop_token)
                               { run(stop_token); });
    }

    void MemoryMonitor::enter_stage(std::string_view stage)
    {
        if (!is_stage(stage))
        {
            throw std::invalid_argument(
                std::format("Unknown memory stage: {}", stage));
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (!sample_locked())
        {
            throw_win32_error(_last_error);
        }
        _stage = std::string(stage);
    }

    void MemoryMonitor::stop()
    {
        if (!_thread.joinable())
        {
            return;
        }
        _thread.request_stop();
        _condition.notify_all();
        _thread.join();

        std::lock_guard<std::mutex> lock(_mutex);
        if (!sample_locked())
        {
            throw_win32_error(_last_error);
        }
        if (_failed)
        {
            throw_win32_error(_last_error);
        }
    }

    double MemoryMonitor::peak_ram_gb() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _run_peak;
    }

    double MemoryMonitor::stage_ram_gb(std::string_view stage) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto entry = _stage_peaks.find(std::string(stage));
        return entry == _stage_peaks.end() ? 0.0 : entry->second;
    }

    void MemoryMonitor::run(std::stop_token stop_token)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while (!stop_token.stop_requested())
        {
            _condition.wait_for(lock, stop_token, kSampleInterval,
                                [] { return false; });
            if (stop_token.stop_requested())
            {
                break;
            }
            sample_locked();
        }
    }

    bool MemoryMonitor::sample_locked()
    {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = static_cast<DWORD>(sizeof(counters));
        if (!GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                static_cast<DWORD>(sizeof(counters))))
        {
            _failed = true;
            _last_error = GetLastError();
            return false;
        }

        const double value =
            static_cast<double>(counters.WorkingSetSize) / kGib;
        if (value > _run_peak)
        {
            _run_peak = value;
        }
        const auto entry = _stage_peaks.find(_stage);
        if (entry != _stage_peaks.end() && value > entry->second)
        {
            entry->second = value;
        }
        return true;
    }
} // namespace pipeline_core
