#include "pipeline_core/pipeline.h"

#include "config.h"
#include "pipeline_core/machine.h"
#include "pipeline_core/stub.h"
#include "run_support.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pipeline_core
{
    namespace
    {
        struct RunState
        {
            explicit RunState(const std::filesystem::path &model_path)
                : session(environment, model_path.c_str(), session_options)
            {
            }

            Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "cpp_pipeline"};
            Ort::SessionOptions session_options;
            Ort::Session session;
            double confidence_threshold{0.0};
            double nms_iou_threshold{1.0};
            NDArray<std::int64_t> target_sizes{{1, 2}, {0, 0}};
            detail::Standardization standardization;
        };

        std::unique_ptr<RunState> g_state;

        RunState &require_state()
        {
            if (!g_state)
            {
                throw std::runtime_error("Pipeline setup has not run");
            }
            return *g_state;
        }

        void setup(const RunInput &run_input)
        {
            const ParsedArgs &args = run_input.parsed_args;
            const double confidence_threshold =
                detail::read_confidence_threshold(args.model_folder_path);
            const detail::Standardization standardization =
                detail::read_standardization(args.model_folder_path);

            auto state = std::make_unique<RunState>(args.model_onnx_path);
            state->confidence_threshold = confidence_threshold;
            state->standardization = standardization;
            g_state = std::move(state);
        }

        InferencePayload preprocess(const InferencePayload &payload)
        {
            return payload;
        }

        InstanceSegmentationInferenceResults
        inference(const InferencePayload &payload)
        {
            return run_stub(payload);
        }

        InstanceSegmentationInferenceResults
        postprocess(InstanceSegmentationInferenceResults results)
        {
            return results;
        }
    } // namespace

    using detail::elapsed_ms;
    using detail::file_name;
    using detail::median_ms;
    using detail::target_sizes_list;
    using detail::utc_now;
    using Clock = detail::Clock;

    RunOutput run_pipeline(const RunInput &run_input, MemoryMonitor &memory)
    {
        if (run_input.inference_payloads.empty())
        {
            throw std::runtime_error("No inference payloads were run");
        }

        std::vector<int> preprocess_samples;
        std::vector<int> inference_samples;
        std::vector<int> postprocess_samples;
        std::vector<int> total_samples;
        std::vector<InstanceSegmentationInferenceResults> results;

        const std::string created_utc = utc_now();
        const MachineInfo machine = read_machine_info();

        memory.enter_stage("setup");
        const Clock::time_point setup_start = Clock::now();
        setup(run_input);
        RunState &state = require_state();
        const int setup_ms = elapsed_ms(setup_start);

        for (const InferencePayload &payload : run_input.inference_payloads)
        {
            memory.enter_stage("preprocess");
            const Clock::time_point total_start = Clock::now();

            Clock::time_point start = Clock::now();
            InferencePayload processed_payload = preprocess(payload);
            preprocess_samples.push_back(elapsed_ms(start));

            memory.enter_stage("inference");
            start = Clock::now();
            InstanceSegmentationInferenceResults raw_results =
                inference(processed_payload);
            inference_samples.push_back(elapsed_ms(start));

            memory.enter_stage("postprocess");
            start = Clock::now();
            InstanceSegmentationInferenceResults final_results =
                postprocess(std::move(raw_results));
            postprocess_samples.push_back(elapsed_ms(start));

            total_samples.push_back(elapsed_ms(total_start));
            results.push_back(std::move(final_results));
        }

        memory.enter_stage("output");

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
            .confidence_threshold = state.confidence_threshold,
            .nms_iou_threshold = state.nms_iou_threshold,
            .target_sizes = target_sizes_list(state.target_sizes),
            .created_utc = created_utc,
            .setup_ms = timings.setup_ms,
            .preprocess_ms = timings.preprocess_ms,
            .inference_ms = timings.inference_ms,
            .postprocess_ms = timings.postprocess_ms,
            .duration_ms = timings.duration_ms,
            .execution_provider = "stub",
            .cpu = machine.cpu,
            .gpu = machine.gpu,
            .setup_ram_gb = memory.stage_ram_gb("setup"),
            .preprocess_ram_gb = memory.stage_ram_gb("preprocess"),
            .inference_ram_gb = memory.stage_ram_gb("inference"),
            .postprocess_ram_gb = memory.stage_ram_gb("postprocess"),
            .ram_gb = memory.peak_ram_gb(),
            .setup_vram_gb = 0.0,
            .preprocess_vram_gb = 0.0,
            .inference_vram_gb = 0.0,
            .postprocess_vram_gb = 0.0,
            .vram_gb = 0.0,
        };
        return RunOutput{std::move(results), std::move(meta), timings};
    }
} // namespace pipeline_core
