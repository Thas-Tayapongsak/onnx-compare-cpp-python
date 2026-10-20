#pragma once

#include "pipeline_core/memory.h"
#include "pipeline_core/results.h"
#include "pipeline_core/run.h"

namespace pipeline_core
{
    RunOutput run_pipeline(const RunInput &run_input, MemoryMonitor &memory);
} // namespace pipeline_core
