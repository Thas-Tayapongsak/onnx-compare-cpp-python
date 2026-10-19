#include "pipeline_core/pipeline.h"

#include "pipeline_core/machine.h"
#include "pipeline_core/stub.h"
#include "run_support.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pipeline_core
{
    using detail::elapsed_ms;
    using detail::file_name;
    using detail::median_ms;
    using detail::target_sizes_list;
    using detail::utc_now;
    using Clock = detail::Clock;

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
        const MachineInfo machine = read_machine_info();
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
            .pipeline = "cpp",
            .model = file_name(run_input.parsed_args.model_folder_path),
            .input_image = file_name(run_input.parsed_args.image_path),
            .confidence_threshold = first_payload->confidence_threshold,
            .nms_iou_threshold = first_payload->nms_iou_threshold,
            .target_sizes = target_sizes_list(first_payload->target_sizes),
            .created_utc = created_utc,
            .setup_ms = timings.setup_ms,
            .preprocess_ms = timings.preprocess_ms,
            .inference_ms = timings.inference_ms,
            .postprocess_ms = timings.postprocess_ms,
            .duration_ms = timings.duration_ms,
            .execution_provider = "stub",
            .cpu = machine.cpu,
            .gpu = machine.gpu,
        };
        return RunOutput{std::move(results), std::move(meta), timings};
    }
} // namespace pipeline_core
