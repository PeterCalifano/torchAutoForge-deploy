/**
 * @file testReloadSafety.cpp
 * @brief Preserve public facade behavior across failed reloads and backend switches.
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <inference/inference_manager.h>
#include <inference/model_facade.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace infer = ptafdeploy::inference;
namespace fs = std::filesystem;

TEMPLATE_TEST_CASE("Public facades retain usable models after reload failure",
                   "[inference][reload]", infer::CInferenceManager, infer::CModelFacade)
{
    const auto onnx_path = fs::path(__FILE__).parent_path().parent_path() /
                           "matlab/testSamples/tracedSampleModel.onnx";
    const auto corrupt_path =
        fs::temp_directory_path() /
        ("ptaf-reload-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".onnx");
    struct SFileCleanup
    {
        fs::path path;
        ~SFileCleanup()
        {
            std::error_code error;
            fs::remove(path, error);
        }
    } cleanup{corrupt_path};
    {
        std::ofstream file;
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.open(corrupt_path);
        file << "invalid ONNX";
        file.close();
    }

    infer::SRuntimeConfig runtime;
    runtime.AddExecutionTarget(infer::EExecutionTarget::cpu);
    runtime.SetAllowFallback(false);
    const std::vector<float> input(11, 0.0F);
    TestType model;
    REQUIRE_THROWS(model.LoadModel(corrupt_path.string()));
    REQUIRE_THROWS(model.InferSingleFloatInput(input, {1, 11}));
    model.LoadModelWithRuntimeConfig(onnx_path.string(), runtime);
    const auto reference = model.InferSingleFloatInput(input, {1, 11});
    const auto backend_detail = model.GetBackendDetail();
    const auto input_info = model.GetInputInfo(0);

    // These failures occur after artifact dispatch, where the previous backend was lost.
    for (const auto& path : {corrupt_path.string(), corrupt_path.string() + ".missing.onnx",
                             corrupt_path.string() + ".missing.engine"})
    {
        REQUIRE_THROWS(model.LoadModel(path));
        REQUIRE(model.GetBackendDetail() == backend_detail);
        REQUIRE(model.GetInputInfo(0).name == input_info.name);
        REQUIRE(model.GetInputInfo(0).shape == input_info.shape);
        REQUIRE(model.InferSingleFloatInput(input, {1, 11}) == reference);
        if constexpr (std::is_same_v<TestType, infer::CModelFacade>)
        {
            const auto contract = model.GetContract();
            REQUIRE(contract.artifact_path == onnx_path.string());
            REQUIRE(contract.runtime.execution_target_priority ==
                    runtime.execution_target_priority);
            REQUIRE_FALSE(contract.runtime.allow_fallback);
        }
    }
    if constexpr (std::is_same_v<TestType, infer::CModelFacade>)
    {
        const auto original_contract = model.GetContract();
        auto replacement_runtime = runtime;
        replacement_runtime.SetDeviceId(7); // CPU execution still records the selected device.
        const auto invalid_role = static_cast<infer::EModelRole>(999);
        const infer::SModelRoleConfig invalid_config{invalid_role};

        // Cover each public role entry point, including valid artifacts with invalid contracts.
        REQUIRE_THROWS(model.LoadModelWithRole(onnx_path.string(), invalid_role));
        REQUIRE_THROWS(model.LoadModelWithRoleConfig(onnx_path.string(), invalid_config));
        REQUIRE_THROWS(model.LoadModelWithRoleAndRuntimeConfig(
            onnx_path.string(), invalid_role, replacement_runtime));
        REQUIRE_THROWS(model.LoadModelWithRoleConfigAndRuntimeConfig(
            onnx_path.string(), invalid_config, replacement_runtime));
        REQUIRE_THROWS(model.LoadModelConfig(corrupt_path.string()));
        REQUIRE_THROWS(model.LoadModelConfigWithRuntimeConfig(
            corrupt_path.string(), replacement_runtime));

        REQUIRE(model.GetBackendDetail() == backend_detail);
        REQUIRE(model.GetContract().role == original_contract.role);
        REQUIRE(model.GetContract().runtime.device_id == original_contract.runtime.device_id);
        REQUIRE(model.InferSingleFloatInput(input, {1, 11}) == reference);
    }
    REQUIRE_NOTHROW(model.LoadModelWithRuntimeConfig(onnx_path.string(), runtime));

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    const char* engine_path = std::getenv("PTAFDEPLOY_TENSORRT_TEST_ENGINE");
    if (engine_path != nullptr && *engine_path != '\0')
    {
        infer::SRuntimeConfig engine_runtime;
        engine_runtime.AddExecutionTarget(infer::EExecutionTarget::cuda);
        engine_runtime.SetAllowFallback(false);
        const char* device = std::getenv("PTAFDEPLOY_TENSORRT_TEST_DEVICE");
        engine_runtime.SetDeviceId(device ? std::stoi(device) : 0);
        model.LoadModelWithRuntimeConfig(engine_path, engine_runtime);
        const auto engine_reference = model.InferSingleFloatInput(input, {1, 11});
        const auto engine_detail = model.GetBackendDetail();
        REQUIRE_THROWS(model.LoadModel(corrupt_path.string()));
        REQUIRE(model.GetBackendDetail() == engine_detail);
        REQUIRE(model.InferSingleFloatInput(input, {1, 11}) == engine_reference);
        REQUIRE_NOTHROW(model.LoadModelWithRuntimeConfig(onnx_path.string(), runtime));
        REQUIRE(model.InferSingleFloatInput(input, {1, 11}) == reference);
    }
#endif
}
