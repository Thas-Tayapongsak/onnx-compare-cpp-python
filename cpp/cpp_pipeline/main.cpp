#include "cli.h"
#include "log.h"

#include <format>
#include <pipeline_core/core.h>

int main(int argc, char *argv[])
{
    cpp_pipeline::InferenceArgs args;
    cpp_pipeline::ParseStatus status =
        cpp_pipeline::parse_args(argc, argv, args);
    if (status == cpp_pipeline::ParseStatus::Error)
    {
        return 1;
    }
    else if (status == cpp_pipeline::ParseStatus::Help)
    {
        return 0;
    }

    cpp_pipeline::log_info("Starting inference pipeline");
    cpp_pipeline::log_info("===========================");
    cpp_pipeline::log_info(
        std::format("Model folder path: {}", args.model_folder_path.string()));

    cpp_pipeline::log_info("===========================");

    int result = pipeline_core::answer();
    return result == 42 ? 0 : 1;
}
