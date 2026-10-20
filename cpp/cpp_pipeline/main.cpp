#include "cli.h"
#include "log.h"
#include "results.h"

#include <pipeline_core/core.h>

#include <cstddef>
#include <exception>
#include <format>
#include <utility>

namespace
{
    pipeline_core::RunInput
    build_run_input(const pipeline_core::ParsedArgs &parsed_args)
    {
        pipeline_core::RunInput run_input;
        run_input.parsed_args = parsed_args;
        run_input.inference_payloads.reserve(
            static_cast<std::size_t>(parsed_args.repeats));
        for (int repeat = 0; repeat < parsed_args.repeats; ++repeat)
        {
            pipeline_core::InferencePayload payload{parsed_args.image_path};
            run_input.inference_payloads.push_back(std::move(payload));
        }
        return run_input;
    }
} // namespace

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
    cpp_pipeline::log_info(
        std::format("Repeat count: {}", parsed_args.repeats));
    cpp_pipeline::log_info("===========================");
    cpp_pipeline::log_info("Running inference pipeline");

    pipeline_core::MemoryMonitor memory;
    try
    {
        memory.start();
        memory.enter_stage("startup");

        const pipeline_core::RunInput run_input = build_run_input(parsed_args);
        cpp_pipeline::prepare_output_dir(parsed_args.output_folder_path);
        pipeline_core::RunOutput run_output =
            pipeline_core::run_pipeline(run_input, memory);
        cpp_pipeline::write_outputs(parsed_args.output_folder_path,
                                    run_output.results);
        memory.stop();
        run_output.meta.ram_gb = memory.peak_ram_gb();
        cpp_pipeline::write_meta(parsed_args.output_folder_path,
                                 run_output.meta);
    }
    catch (const std::exception &error)
    {
        cpp_pipeline::log_error(std::format("Error: {}", error.what()));
        return 1;
    }

    cpp_pipeline::log_info("===========================");
    return 0;
}
