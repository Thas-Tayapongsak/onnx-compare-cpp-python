#pragma once

#include <optional>
#include <string>

namespace pipeline_core
{
    struct MachineInfo
    {
        std::string cpu;
        std::optional<std::string> gpu;
    };

    /// Read the names of the machine devices used by the run.
    MachineInfo read_machine_info();
} // namespace pipeline_core
