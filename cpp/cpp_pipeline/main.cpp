#include "cli.h"
#include "log.h"
#include "results.h"
#include "stub.h"

#include <exception>
#include <format>

int main(int argc, char *argv[])
{
    cpp_pipeline::ParsedArgs parsed_args;
    cpp_pipeline::ParseStatus status =
        cpp_pipeline::parse_args(argc, argv, parsed_args);
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
    cpp_pipeline::log_info(std::format("Model folder path: {}",
                                       parsed_args.model_folder_path.string()));
    cpp_pipeline::log_info(
        std::format("Image path: {}", parsed_args.image_path.string()));
    cpp_pipeline::log_info(std::format(
        "Output folder path: {}", parsed_args.output_folder_path.string()));
    cpp_pipeline::log_info("===========================");
    cpp_pipeline::log_info("Running inference pipeline");

    try
    {
        cpp_pipeline::prepare_output_dir(parsed_args.output_folder_path);
        cpp_pipeline::write_outputs(parsed_args.output_folder_path,
                                    cpp_pipeline::run_stub());
        cpp_pipeline::write_meta(parsed_args.output_folder_path);
    }
    catch (const std::exception &error)
    {
        cpp_pipeline::log_error(std::format("Error: {}", error.what()));
        return 1;
    }

    cpp_pipeline::log_info("===========================");
    return 0;
}
