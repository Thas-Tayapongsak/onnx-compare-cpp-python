#pragma once

#include <pipeline_core/run.h>

namespace cpp_pipeline
{
    using ParsedArgs = pipeline_core::ParsedArgs;

    enum class ParseStatus
    {
        Success,
        Error,
        Help
    };

    /// Parse the command line and validate the model folder.
    ParseStatus parse_args(int argc, char *argv[], ParsedArgs &parsed_args);
} // namespace cpp_pipeline
