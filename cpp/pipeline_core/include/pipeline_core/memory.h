#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace pipeline_core
{
    class MemoryMonitor
    {
      public:
        MemoryMonitor();
        ~MemoryMonitor();

        MemoryMonitor(const MemoryMonitor &) = delete;
        MemoryMonitor &operator=(const MemoryMonitor &) = delete;
        MemoryMonitor(MemoryMonitor &&) = delete;
        MemoryMonitor &operator=(MemoryMonitor &&) = delete;

        /// Start sampling the process working set.
        void start();

        /// Record the current stage and start tracking the next one.
        void enter_stage(std::string_view stage);

        /// Stop sampling and take the final sample.
        void stop();

        /// Return the highest working set seen during this run.
        double peak_ram_gb() const;

        /// Return the highest working set seen during one stage.
        double stage_ram_gb(std::string_view stage) const;

      private:
        void run(std::stop_token stop_token);
        bool sample_locked();

        mutable std::mutex _mutex;
        std::condition_variable_any _condition;
        std::jthread _thread;
        std::string _stage;
        std::unordered_map<std::string, double> _stage_peaks;
        double _run_peak{0.0};
        bool _failed{false};
        unsigned long _last_error{0};
    };
} // namespace pipeline_core
