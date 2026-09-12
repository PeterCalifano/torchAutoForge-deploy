/**
 * @file testFilmCentroidingOnnx.cpp
 * @brief Image-plus-prior FiLM model contract checks on CPU and optional CUDA.
 */

#include <catch2/catch_test_macros.hpp>
#include <inference/model_facade.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <inference/task_adapters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;

    constexpr int64_t kImageHeight = 1536;
    constexpr int64_t kImageWidth = 2048;
    constexpr float kReferenceSizeMinM = 1.0e2F;
    constexpr float kReferenceSizeMaxM = 1.0e7F;
    constexpr float kReferenceSizeGamma = 0.85F;

    struct SSyntheticCentroidingSample
    {
        infer::SFloatTensor image{};
        infer::SFloatTensor prior_vector{};
    };

    [[nodiscard]] std::string GetOptionalFilmCentroidingOnnxPath()
    {
        if (const char* env_path = std::getenv("PTAFDEPLOY_FILM_CENTROIDING_ONNX"))
        {
            return env_path;
        }

        return {};
    }

    [[nodiscard]] float NormalizeReferenceSize(const float reference_size_m)
    {
        const float clipped_reference =
            std::clamp(reference_size_m, kReferenceSizeMinM, kReferenceSizeMaxM);
        const float log_min = std::log(kReferenceSizeMinM);
        const float log_max = std::log(kReferenceSizeMaxM);
        const float normalized = (std::log(clipped_reference) - log_min) / (log_max - log_min);
        return std::pow(std::clamp(normalized, 0.0F, 1.0F), kReferenceSizeGamma);
    }

    [[nodiscard]] SSyntheticCentroidingSample MakeLambertianCentroidingSample()
    {
        constexpr float centre_x_px = 1032.0F;
        constexpr float centre_y_px = 742.0F;
        constexpr float radius_px = 210.0F;
        constexpr float phase_angle_rad = 0.0F;
        constexpr float sun_direction_angle_rad = 0.0F;
        constexpr float reference_size_m = 1250.0F;

        std::vector<float> image_values(static_cast<size_t>(kImageHeight * kImageWidth), 0.0F);
        const float cos_phase = std::cos(phase_angle_rad);
        const float sin_phase = std::sin(phase_angle_rad);
        const float light_x = sin_phase * std::cos(sun_direction_angle_rad);
        const float light_y = sin_phase * std::sin(sun_direction_angle_rad);
        const float light_z = cos_phase;

        for (int64_t y = 0; y < kImageHeight; ++y)
        {
            for (int64_t x = 0; x < kImageWidth; ++x)
            {
                const float dx = (static_cast<float>(x) - centre_x_px) / radius_px;
                const float dy = (static_cast<float>(y) - centre_y_px) / radius_px;
                const float radial_sq = dx * dx + dy * dy;
                if (radial_sq >= 1.0F)
                {
                    continue;
                }

                const float normal_z = std::sqrt(1.0F - radial_sq);
                const float lambertian =
                    std::max(0.0F, dx * light_x + dy * light_y + normal_z * light_z);
                image_values[static_cast<size_t>(y * kImageWidth + x)] = lambertian;
            }
        }

        SSyntheticCentroidingSample sample;
        sample.image = infer::SFloatTensor{
            "image", {1, 1, kImageHeight, kImageWidth}, std::move(image_values)};
        sample.prior_vector = infer::SFloatTensor{
            "prior_vector",
            {1, 3},
            {phase_angle_rad, sun_direction_angle_rad, NormalizeReferenceSize(reference_size_m)}};
        return sample;
    }

    [[nodiscard]] fs::path WriteRuntimeManifest(const fs::path& onnx_path,
                                               const infer::EExecutionTarget execution_target)
    {
        const std::string target_name = infer::ToString(execution_target);
        const fs::path manifest_path =
            fs::temp_directory_path() /
            ("ptafdeploy_film_centroiding_" + target_name + "_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".ptafmodel");
        std::ofstream manifest;
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest.open(manifest_path);
        manifest << "schema_version = 1\n"
                 << "artifact_path = " << fs::absolute(onnx_path).string() << "\n"
                 << "role = centroiding\n"
                 << "preprocessing = external_lambertian_grayscale_nchw_with_film_prior\n"
                 << "postprocessing = normalized_centroid_xy_and_cob\n"
                 << "backend = onnxruntime\n"
                 << "artifact = onnx\n"
                 << "execution_target_priority = " << target_name << "\n"
                 << "allow_fallback = false\n"
                 << "device_id = 0\n"
                 << "intra_op_num_threads = 1\n"
                 << "inter_op_num_threads = 1\n";
        manifest.close();
        return manifest_path;
    }

    [[nodiscard]] const infer::SFloatTensor* FindTensorByName(
        const std::vector<infer::SFloatTensor>& tensors, const std::string& name)
    {
        const auto it =
            std::find_if(tensors.begin(), tensors.end(),
                         [&](const infer::SFloatTensor& tensor) { return tensor.name == name; });
        if (it == tensors.end())
        {
            throw std::runtime_error("Missing output tensor: " + name);
        }
        return &(*it);
    }

    void RunFilmCentroidingCase(const infer::EExecutionTarget execution_target,
                               const std::string& required_backend_fragment)
    {
        const fs::path onnx_path = GetOptionalFilmCentroidingOnnxPath();
        if (onnx_path.empty())
        {
            SKIP("Set PTAFDEPLOY_FILM_CENTROIDING_ONNX to run the FiLM centroiding ONNX contract "
                 "test.");
        }

        REQUIRE(fs::is_regular_file(onnx_path));
        const fs::path manifest_path = WriteRuntimeManifest(onnx_path, execution_target);
        struct Cleanup
        {
            fs::path path;
            ~Cleanup()
            {
                std::error_code ignored;
                fs::remove(path, ignored);
            }
        } cleanup{manifest_path};
        infer::CModelFacade model;
        model.LoadModelConfig(manifest_path.string());

        REQUIRE(model.GetRole() == "centroiding");
        REQUIRE(model.GetNumInputs() == 2);
        REQUIRE(model.GetNumOutputs() == 2);
        REQUIRE(model.GetInputInfo(0).name == "image");
        REQUIRE(model.GetInputInfo(0).shape ==
                std::vector<int64_t>{-1, 1, kImageHeight, kImageWidth});
        REQUIRE(model.GetInputInfo(1).name == "prior_vector");
        REQUIRE(model.GetInputInfo(1).shape == std::vector<int64_t>{-1, 3});
        REQUIRE(model.GetBackendDetail().find(required_backend_fragment) != std::string::npos);

        const SSyntheticCentroidingSample sample = MakeLambertianCentroidingSample();
        const std::vector<infer::SFloatTensor> outputs =
            model.InferFloatTensors({sample.image, sample.prior_vector});

        const infer::SFloatTensor* prediction = FindTensorByName(outputs, "prediction");
        const infer::SFloatTensor* centre_of_brightness =
            FindTensorByName(outputs, "centre_of_brightness");

        REQUIRE(prediction->shape == std::vector<int64_t>{1, 2});
        REQUIRE(centre_of_brightness->shape == std::vector<int64_t>{1, 2});
        REQUIRE(prediction->values.size() == 2);
        REQUIRE(centre_of_brightness->values.size() == 2);
        REQUIRE(std::isfinite(prediction->values[0]));
        REQUIRE(std::isfinite(prediction->values[1]));

        infer::SFeatureRowSchema feature_schema;
        feature_schema.x_index = 0U;
        feature_schema.y_index = 1U;
        const std::vector<infer::SFeature2D> cob_features =
            infer::DecodeFeatureRows(*centre_of_brightness, feature_schema);
        REQUIRE(cob_features.size() == 1U);

        // Synthetic illumination exercises the input contract, not learned accuracy.
        REQUIRE(std::isfinite(cob_features.front().position.x));
        REQUIRE(std::isfinite(cob_features.front().position.y));
    }
} // namespace

TEST_CASE("film_centroiding_onnx_runs_on_cpu_with_coherent_synthetic_inputs",
          "[inference][centroiding][film][ort][optional]")
{
    RunFilmCentroidingCase(infer::EExecutionTarget::cpu, "applied_ort_providers=cpu");
}

TEST_CASE("film_centroiding_onnx_runs_on_cuda_when_available",
          "[inference][centroiding][film][ort][cuda][optional]")
{
    if (!infer::onnxruntime::CInferenceManager_ORT::IsExecutionTargetAvailable(
            infer::EExecutionTarget::cuda))
        SKIP("CUDA execution provider is not available in this ONNX Runtime installation");
    RunFilmCentroidingCase(infer::EExecutionTarget::cuda, "applied_ort_providers=cuda");
}
