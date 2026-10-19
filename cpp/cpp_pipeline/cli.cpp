#include "cli.h"
#include "log.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <optional>

namespace
{
    constexpr std::string_view kDefaultOutputFolder = "runs";

    void print_help(char *argv[])
    {
        std::cout
            << "Usage: " << argv[0]
            << " [-h] --model MODEL --image IMAGE [--output OUTPUT]\n"
            << "\n"
            << "Run the C++ ONNX inference pipeline.\n"
            << "\n"
            << "Options:\n"
            << "  -h, --help       Show this help and exit.\n"
            << "  --model MODEL    Path to the ONNX model folder. Where "
               "`model.onnx` is located.\n"
            << "  --image IMAGE    Path to the image to run inference on.\n"
            << "  --output OUTPUT  Path to the output folder.\n";
    }

    /// Validate argument present on assignment
    bool arg_is_given(int &i, int argc, char *argv[], std::string_view flag,
                      std::string_view &argvar)
    {
        if (i + 1 >= argc) // flag with no value at end
        {
            cpp_pipeline::log_error(
                std::format("Error: {} requires a value.", flag));
            return false;
        }
        const std::string_view value = argv[i + 1];
        if (value.empty() ||
            value.starts_with('-')) // case --model --<flag> ...
        {
            cpp_pipeline::log_error(
                std::format("Error: {} requires a value.", flag));
            return false;
        }
        argvar = value;
        ++i;
        return true;
    }

    template <typename T>
    bool arg_set_once(std::optional<T> &argvar, std::string_view flag,
                      std::string_view value)
    {
        if (argvar.has_value())
        {
            cpp_pipeline::log_error(
                std::format("Error: {} can only be provided once.", flag));
            return false;
        }
        argvar = value;
        return true;
    }

    cpp_pipeline::ParseStatus
    validate_model_folder_path(std::filesystem::path model_folder_path)
    {
        if (!std::filesystem::is_directory(model_folder_path))
        {
            cpp_pipeline::log_error(
                std::format("Error: Model folder not found: {}",
                            model_folder_path.string()));
            return cpp_pipeline::ParseStatus::Error;
        }

        return cpp_pipeline::ParseStatus::Success;
    }

    cpp_pipeline::ParseStatus
    validate_model_onnx_path(std::filesystem::path model_onnx_path)
    {
        if (!std::filesystem::is_regular_file(model_onnx_path))
        {
            cpp_pipeline::log_error(
                std::format("Error: `model.onnx` not found in {}",
                            model_onnx_path.string()));
            return cpp_pipeline::ParseStatus::Error;
        }

        return cpp_pipeline::ParseStatus::Success;
    }

    cpp_pipeline::ParseStatus
    validate_image_path(std::filesystem::path image_path)
    {
        if (!std::filesystem::is_regular_file(image_path))
        {
            cpp_pipeline::log_error(std::format(
                "Error: Image file not found at {}", image_path.string()));
            return cpp_pipeline::ParseStatus::Error;
        }

        return cpp_pipeline::ParseStatus::Success;
    }

    cpp_pipeline::ParseStatus
    validate_output_folder_path(std::filesystem::path output_folder_path)
    {
        if (std::filesystem::exists(output_folder_path) &&
            !std::filesystem::is_directory(output_folder_path))
        {
            cpp_pipeline::log_error(
                std::format("Error: Output path {} is not a directory",
                            output_folder_path.string()));
            return cpp_pipeline::ParseStatus::Error;
        }

        return cpp_pipeline::ParseStatus::Success;
    }
} // namespace

namespace cpp_pipeline
{
    ParseStatus parse_args(int argc, char *argv[], InferenceArgs &args)
    {
        std::optional<std::filesystem::path> model_folder_path;
        std::optional<std::filesystem::path> model_onnx_path;
        std::optional<std::filesystem::path> image_path;
        std::optional<std::filesystem::path> output_folder_path;

        // Discover cli args
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view arg = argv[i];

            if (arg == "-h" || arg == "--help")
            {
                print_help(argv);
                return ParseStatus::Help;
            }

            if (arg == "--model")
            {
                std::string_view value;
                if (!arg_is_given(i, argc, argv, arg, value))
                {
                    return ParseStatus::Error;
                }
                if (!arg_set_once(model_folder_path, arg, value))
                {
                    return ParseStatus::Error;
                }
            }
            else if (arg == "--image")
            {
                std::string_view value;
                if (!arg_is_given(i, argc, argv, arg, value))
                {
                    return ParseStatus::Error;
                }
                if (!arg_set_once(image_path, arg, value))
                {
                    return ParseStatus::Error;
                }
            }
            else if (arg == "--output")
            {
                std::string_view value;
                if (!arg_is_given(i, argc, argv, arg, value))
                {
                    return ParseStatus::Error;
                }
                if (!arg_set_once(output_folder_path, arg, value))
                {
                    return ParseStatus::Error;
                }
            }
            else
            {
                std::cerr << "Error: Unknown argument: " << arg << "\n";
                return ParseStatus::Error;
            }
        }

        // Check required args
        if (!model_folder_path)
        {
            std::cerr << "Error: --model is required.\n";
            return ParseStatus::Error;
        }
        if (!image_path)
        {
            std::cerr << "Error: --image is required.\n";
            return ParseStatus::Error;
        }

        // Validate args
        if (validate_model_folder_path(model_folder_path.value()) ==
            ParseStatus::Error)
        {
            return ParseStatus::Error;
        }
        model_onnx_path = model_folder_path.value() / "model.onnx";
        if (validate_model_onnx_path(model_onnx_path.value()) ==
            ParseStatus::Error)
        {
            return ParseStatus::Error;
        }
        if (validate_image_path(image_path.value()) == ParseStatus::Error)
        {
            return ParseStatus::Error;
        }
        if (!output_folder_path)
        {
            output_folder_path = kDefaultOutputFolder;
        }
        if (validate_output_folder_path(output_folder_path.value()) ==
            ParseStatus::Error)
        {
            return ParseStatus::Error;
        }

        args = InferenceArgs{model_folder_path.value(),
                             model_onnx_path.value(), image_path.value(),
                             output_folder_path.value()};
        return ParseStatus::Success;
    }
} // namespace cpp_pipeline
