#include "cli.h"
#include "log.h"

#include <charconv>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>

namespace
{
    constexpr std::string_view kDefaultModelOnnxFileName = "model.onnx";
    constexpr std::string_view kDefaultOutputFolder = "runs";

    void print_help(char *argv[])
    {
        std::cout
            << "Usage: " << argv[0]
            << " [-h] --model MODEL --image IMAGE [--output OUTPUT] "
               "[--repeat N]\n"
            << "\n"
            << "Run the C++ ONNX inference pipeline.\n"
            << "\n"
            << "Options:\n"
            << "  -h, --help       Show this help and exit.\n"
            << "  --model MODEL    Path to the ONNX model folder. Where "
               "`model.onnx` is located.\n"
            << "  --image IMAGE    Path to the image to run inference on.\n"
            << "  --output OUTPUT  Path to the output folder.\n"
            << "  --repeat N       Number of times to run the payload. Must "
               "be at least 1.\n";
    }

    /// Validate argument present on assignment
    cpp_pipeline::ParseStatus
    parse_argument(int &i, int argc, char *argv[], std::string_view flag,
                   std::optional<std::filesystem::path> &arg)
    {
        if (i + 1 >= argc) // check value after flag
        {
            cpp_pipeline::log_error(
                std::format("Error: {} requires a value.", flag));
            return cpp_pipeline::ParseStatus::Error;
        }
        const std::string_view value = argv[i + 1];
        if (value.empty() ||
            value.starts_with('-')) // check value if empty/not given
        {
            cpp_pipeline::log_error(
                std::format("Error: {} requires a value.", flag));
            return cpp_pipeline::ParseStatus::Error;
        }
        if (arg.has_value()) // check if value is assigned once already
        {
            cpp_pipeline::log_error(
                std::format("Error: {} can only be provided once.", flag));
            return cpp_pipeline::ParseStatus::Error;
        }
        arg = value;
        ++i;
        return cpp_pipeline::ParseStatus::Success;
    }

    bool validate_required_arg(std::string_view flag,
                               const std::optional<std::filesystem::path> &arg)
    {
        if (!arg.has_value())
        {
            cpp_pipeline::log_error(
                std::format("Error: {} is required.", flag));
            return false;
        }
        return true;
    }

    /// Parse a repeat count on assignment.
    cpp_pipeline::ParseStatus parse_repeat_argument(int &i, int argc,
                                                    char *argv[], int &repeats,
                                                    bool &assigned)
    {
        if (i + 1 >= argc) // check value after flag
        {
            cpp_pipeline::log_error("Error: --repeat requires a value.");
            return cpp_pipeline::ParseStatus::Error;
        }
        const std::string_view value = argv[i + 1];
        int parsed_value = 0;
        const char *value_begin = value.data();
        const char *value_end = value_begin + value.size();
        const auto [pointer, error] =
            std::from_chars(value_begin, value_end, parsed_value);
        if (error != std::errc{} || pointer != value_end)
        {
            cpp_pipeline::log_error(std::format(
                "Error: --repeat must be an integer, got: {}", value));
            return cpp_pipeline::ParseStatus::Error;
        }
        if (assigned) // check if value is assigned once already
        {
            cpp_pipeline::log_error(
                "Error: --repeat can only be provided once.");
            return cpp_pipeline::ParseStatus::Error;
        }
        if (parsed_value < 1)
        {
            cpp_pipeline::log_error("Error: --repeat must be at least 1.");
            return cpp_pipeline::ParseStatus::Error;
        }
        repeats = parsed_value;
        assigned = true;
        ++i;
        return cpp_pipeline::ParseStatus::Success;
    }

    bool
    validate_model_folder_path(const std::filesystem::path &model_folder_path)
    {
        if (!std::filesystem::is_directory(model_folder_path))
        {
            cpp_pipeline::log_error(
                std::format("Error: Model folder not found: {}",
                            model_folder_path.string()));
            return false;
        }

        return true;
    }

    bool validate_model_onnx_path(const std::filesystem::path &model_onnx_path)
    {
        if (!std::filesystem::is_regular_file(model_onnx_path))
        {
            cpp_pipeline::log_error(std::format(
                "Error: Model file not found: {}", model_onnx_path.string()));
            return false;
        }

        return true;
    }

    bool validate_image_path(const std::filesystem::path &image_path)
    {
        if (!std::filesystem::is_regular_file(image_path))
        {
            cpp_pipeline::log_error(std::format(
                "Error: Image file not found at {}", image_path.string()));
            return false;
        }

        return true;
    }

    bool
    validate_output_folder_path(const std::filesystem::path &output_folder_path)
    {
        if (std::filesystem::exists(output_folder_path) &&
            !std::filesystem::is_directory(output_folder_path))
        {
            cpp_pipeline::log_error(
                std::format("Error: Output path {} is not a directory",
                            output_folder_path.string()));
            return false;
        }
        return true;
    }
} // namespace

namespace cpp_pipeline
{
    ParseStatus parse_args(int argc, char *argv[], ParsedArgs &parsed_args)
    {
        std::optional<std::filesystem::path> model_folder_path;
        std::optional<std::filesystem::path> model_onnx_path;
        std::optional<std::filesystem::path> image_path;
        std::optional<std::filesystem::path> output_folder_path;
        int repeats = 1;
        bool repeats_assigned = false;

        // Discover cli args
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view flag = argv[i];
            ParseStatus status = ParseStatus::Success;

            if (flag == "-h" || flag == "--help")
            {
                print_help(argv);
                return ParseStatus::Help;
            }

            if (flag == "--model")
            {
                status = parse_argument(i, argc, argv, flag, model_folder_path);
            }
            else if (flag == "--image")
            {
                status = parse_argument(i, argc, argv, flag, image_path);
            }
            else if (flag == "--output")
            {
                status =
                    parse_argument(i, argc, argv, flag, output_folder_path);
            }
            else if (flag == "--repeat")
            {
                status = parse_repeat_argument(i, argc, argv, repeats,
                                               repeats_assigned);
            }
            else
            {
                log_error(std::format("Error: Unknown argument: {}", flag));
                status = ParseStatus::Error;
            }

            if (status != ParseStatus::Success)
            {
                return status;
            }
        }

        // Check required args
        if (!validate_required_arg("--model", model_folder_path) ||
            !validate_required_arg("--image", image_path))
        {
            return ParseStatus::Error;
        }

        // Validate args
        model_onnx_path = model_folder_path.value() / kDefaultModelOnnxFileName;
        if (!output_folder_path)
        {
            output_folder_path = kDefaultOutputFolder;
        }
        if (!validate_model_folder_path(model_folder_path.value()) ||
            !validate_model_onnx_path(model_onnx_path.value()) ||
            !validate_image_path(image_path.value()) ||
            !validate_output_folder_path(output_folder_path.value()))
        {
            return ParseStatus::Error;
        }

        parsed_args =
            ParsedArgs{model_folder_path.value(), model_onnx_path.value(),
                       image_path.value(), output_folder_path.value(), repeats};
        return ParseStatus::Success;
    }
} // namespace cpp_pipeline
