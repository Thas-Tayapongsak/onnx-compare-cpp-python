#include "log.h"
#include "results.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

namespace
{
    nlohmann::ordered_json
    build_meta_json(const pipeline_core::MetaResults &meta)
    {
        nlohmann::ordered_json data;
        data["pipeline"] = meta.pipeline;
        data["model"] = meta.model;
        data["input_image"] = meta.input_image;
        data["confidence_threshold"] = meta.confidence_threshold;
        data["nms_iou_threshold"] = meta.nms_iou_threshold;
        data["target_sizes"] = meta.target_sizes;
        data["created_utc"] = meta.created_utc;
        data["setup_ms"] = meta.setup_ms;
        data["preprocess_ms"] = meta.preprocess_ms;
        data["inference_ms"] = meta.inference_ms;
        data["postprocess_ms"] = meta.postprocess_ms;
        data["duration_ms"] = meta.duration_ms;
        data["execution_provider"] = meta.execution_provider;
        data["cpu"] = meta.cpu;
        data["gpu"] = meta.gpu.has_value() ? nlohmann::ordered_json(*meta.gpu)
                                           : nlohmann::ordered_json(nullptr);
        data["setup_ram_gb"] = meta.setup_ram_gb;
        data["preprocess_ram_gb"] = meta.preprocess_ram_gb;
        data["inference_ram_gb"] = meta.inference_ram_gb;
        data["postprocess_ram_gb"] = meta.postprocess_ram_gb;
        data["ram_gb"] = meta.ram_gb;
        data["setup_vram_gb"] = meta.setup_vram_gb;
        data["preprocess_vram_gb"] = meta.preprocess_vram_gb;
        data["inference_vram_gb"] = meta.inference_vram_gb;
        data["postprocess_vram_gb"] = meta.postprocess_vram_gb;
        data["vram_gb"] = meta.vram_gb;
        return data;
    }
} // namespace

namespace cpp_pipeline
{
    void write_meta(const std::filesystem::path &output_dir,
                    const pipeline_core::MetaResults &meta)
    {
        const std::filesystem::path path = output_dir / "meta.json";
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not open {} for writing", path.string()));
        }

        file << build_meta_json(meta).dump();
        file.close();
        if (!file)
        {
            throw std::runtime_error(
                std::format("Could not write {}", path.string()));
        }
        log_info("`meta.json` written");
    }
} // namespace cpp_pipeline
