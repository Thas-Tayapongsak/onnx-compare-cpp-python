#include "pipeline_core/pipeline.h"

#include "pipeline_core/stub.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    int elapsed_ms(Clock::time_point start)
    {
        const auto elapsed = Clock::now() - start;
        return static_cast<int>(std::llround(
            std::chrono::duration<double, std::milli>(elapsed).count()));
    }

    int median_ms(std::vector<int> samples)
    {
        std::sort(samples.begin(), samples.end());
        const std::size_t count = samples.size();
        if (count % 2U == 1U)
        {
            return samples[count / 2U];
        }
        return (samples[count / 2U - 1U] + samples[count / 2U]) / 2;
    }

    std::string utc_now()
    {
        const std::time_t time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm utc_time{};
        gmtime_s(&utc_time, &time);
        std::ostringstream text;
        text << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
        return text.str();
    }

    std::vector<std::vector<std::int64_t>>
    target_sizes_list(const pipeline_core::NDArray<std::int64_t> &target_sizes)
    {
        if (target_sizes.shape.size() != 2U)
        {
            throw std::runtime_error("target_sizes must have rank 2");
        }
        const std::size_t rows = target_sizes.shape[0];
        const std::size_t columns = target_sizes.shape[1];
        if (target_sizes.data.size() != rows * columns)
        {
            throw std::runtime_error(
                "target_sizes data does not match its shape");
        }

        std::vector<std::vector<std::int64_t>> values;
        for (std::size_t row = 0; row < rows; ++row)
        {
            std::vector<std::int64_t> value;
            value.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                value.push_back(target_sizes.data[row * columns + column]);
            }
            values.push_back(std::move(value));
        }
        return values;
    }

    std::string file_name(const std::filesystem::path &path)
    {
        return path.lexically_normal().filename().string();
    }
} // namespace

namespace pipeline_core
{
    RunState setup(const RunInput &run_input)
    {
        static_cast<void>(run_input);
        return RunState{};
    }

    InferencePayload preprocess(const InferencePayload &payload,
                                RunState &state)
    {
        static_cast<void>(state);
        return payload;
    }

    InstanceSegmentationInferenceResults
    inference(const InferencePayload &payload, RunState &state)
    {
        static_cast<void>(state);
        return run_stub(payload);
    }

    InstanceSegmentationInferenceResults
    postprocess(InstanceSegmentationInferenceResults results,
                const InferencePayload &payload)
    {
        static_cast<void>(payload);
        return results;
    }

    RunOutput run_pipeline(const RunInput &run_input)
    {
        std::vector<int> preprocess_samples;
        std::vector<int> inference_samples;
        std::vector<int> postprocess_samples;
        std::vector<int> total_samples;
        std::vector<InstanceSegmentationInferenceResults> results;
        std::optional<InferencePayload> first_payload;

        const std::string created_utc = utc_now();
        const Clock::time_point setup_start = Clock::now();
        RunState state = setup(run_input);
        const int setup_ms = elapsed_ms(setup_start);

        for (const InferencePayload &payload : run_input.inference_payloads)
        {
            const Clock::time_point total_start = Clock::now();

            Clock::time_point start = Clock::now();
            InferencePayload processed_payload = preprocess(payload, state);
            preprocess_samples.push_back(elapsed_ms(start));

            start = Clock::now();
            InstanceSegmentationInferenceResults raw_results =
                inference(processed_payload, state);
            inference_samples.push_back(elapsed_ms(start));

            start = Clock::now();
            InstanceSegmentationInferenceResults final_results =
                postprocess(std::move(raw_results), processed_payload);
            postprocess_samples.push_back(elapsed_ms(start));

            total_samples.push_back(elapsed_ms(total_start));
            results.push_back(std::move(final_results));

            if (!first_payload.has_value())
            {
                first_payload = std::move(processed_payload);
            }
        }

        if (!first_payload.has_value())
        {
            throw std::runtime_error("No inference payloads were run");
        }

        const RunTimings timings{
            setup_ms,
            median_ms(preprocess_samples),
            median_ms(inference_samples),
            median_ms(postprocess_samples),
            median_ms(total_samples),
        };
        MetaResults meta{
            "cpp",
            file_name(run_input.parsed_args.model_folder_path),
            file_name(run_input.parsed_args.image_path),
            first_payload->confidence_threshold,
            first_payload->nms_iou_threshold,
            target_sizes_list(first_payload->target_sizes),
            created_utc,
            timings.duration_ms,
            "stub",
        };
        return RunOutput{std::move(results), std::move(meta), timings};
    }
} // namespace pipeline_core
