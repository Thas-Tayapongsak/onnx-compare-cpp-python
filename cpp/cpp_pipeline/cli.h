#pragma once

#include <filesystem>

namespace cpp_pipeline
{
    struct InferenceArgs
    {
        std::filesystem::path model_folder_path;
        std::filesystem::path model_onnx_path;
    };

    enum class ParseStatus
    {
        Success,
        Error,
        Help
    };

    /// Parse the command line and validate the model folder.
    ParseStatus parse_args(int argc, char *argv[], InferenceArgs &args);
} // namespace cpp_pipeline
