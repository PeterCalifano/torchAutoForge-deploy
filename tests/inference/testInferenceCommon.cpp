/**
 * @file testInferenceCommon.cpp
 * @brief Target-owned tensor, artifact, and optional TensorRT behavior tests.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <inference/inference_common.h>
#include <inference/inference_manager.h>
#include <inference/tensorrt/tensorrt_inference_engine.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
#include <cuda_runtime_api.h>
namespace
{
    [[nodiscard]] int GetTensorRtTestDevice()
    {
        const char* device_env = std::getenv("PTAFDEPLOY_TENSORRT_TEST_DEVICE");
        return device_env == nullptr ? 0 : std::atoi(device_env);
    }
} // namespace
#endif

TEST_CASE("inference_common_tensor_helpers", "[inference][common]")
{
    using namespace ptafdeploy::inference;

    SECTION("element type sizes are mapped correctly")
    {
        REQUIRE(GetTensorElementTypeSize(ETensorElementType::boolean) == 1);
        REQUIRE(GetTensorElementTypeSize(ETensorElementType::float16) == 2);
        REQUIRE(GetTensorElementTypeSize(ETensorElementType::float32) == 4);
        REQUIRE(GetTensorElementTypeSize(ETensorElementType::float64) == 8);
    }

    SECTION("dynamic shapes are detected")
    {
        REQUIRE(HasDynamicShape({1, -1, 8}));
        REQUIRE_FALSE(HasDynamicShape({1, 2, 8}));
    }

    SECTION("owned tensor buffers allocate the expected number of bytes")
    {
        STensorDescriptor descriptor{
            "input", ETensorElementType::float32, {1, 11}, EMemoryLocation::host};
        STensorBuffer buffer = MakeOwnedTensorBuffer(descriptor);
        REQUIRE(buffer.bytes() == 44);
        REQUIRE(buffer.descriptor.name == "input");
    }

    SECTION("element and byte counts reject overflow")
    {
        REQUIRE_THROWS_AS(ComputeElementCount({std::numeric_limits<int64_t>::max(), 3}),
                          std::overflow_error);

        STensorDescriptor descriptor{"too_large",
                                     ETensorElementType::float64,
                                     {std::numeric_limits<int64_t>::max() / 4, 8},
                                     EMemoryLocation::host};
        REQUIRE_THROWS_AS(ComputeByteCount(descriptor), std::overflow_error);
    }

    SECTION("runtime target presets use enum-backed execution choices")
    {
        SRuntimeConfig runtime_config;
        runtime_config.UseCudaWithCpuFallback();
        REQUIRE(runtime_config.allow_fallback);
        REQUIRE(runtime_config.execution_target_priority ==
                std::vector<EExecutionTarget>{EExecutionTarget::cuda, EExecutionTarget::cpu});

        runtime_config.UseCpuOnly();
        REQUIRE_FALSE(runtime_config.allow_fallback);
        REQUIRE(runtime_config.execution_target_priority ==
                std::vector<EExecutionTarget>{EExecutionTarget::cpu});
    }

    SECTION("runtime config rejects impossible scalar values")
    {
        SRuntimeConfig runtime_config;
        REQUIRE_THROWS_WITH(runtime_config.SetDeviceId(-1),
                            Catch::Matchers::ContainsSubstring("device_id"));
        REQUIRE_THROWS_WITH(runtime_config.SetThreadCounts(-1, 1),
                            Catch::Matchers::ContainsSubstring("thread counts"));
        REQUIRE_THROWS_WITH(runtime_config.SetThreadCounts(1, -1),
                            Catch::Matchers::ContainsSubstring("thread counts"));
        REQUIRE_THROWS_WITH(runtime_config.SetTensorRtOptimizationProfileIndex(-1),
                            Catch::Matchers::ContainsSubstring("TensorRT optimization profile"));

        runtime_config.SetTensorRtOptimizationProfileIndex(3);
        REQUIRE(runtime_config.GetTensorRtOptimizationProfileIndex() == 3);
    }

    SECTION("common tensor validation supports positional and named inputs")
    {
        const std::vector<STensorDescriptor> expected_inputs{
            {"image", ETensorElementType::float32, {1, 3}, EMemoryLocation::host},
            {"state", ETensorElementType::float32, {1, 2}, EMemoryLocation::host}};
        const std::vector<float> image_values{1.0F, 2.0F, 3.0F};
        const std::vector<float> state_values{4.0F, 5.0F};

        const std::vector<STensorView> positional_inputs{
            {{"", ETensorElementType::float32, {1, 3}, EMemoryLocation::host},
             image_values.data(),
             image_values.size() * sizeof(float)},
            {{"", ETensorElementType::float32, {1, 2}, EMemoryLocation::host},
             state_values.data(),
             state_values.size() * sizeof(float)}};
        REQUIRE(OrderInputViews(expected_inputs, positional_inputs)[0] == &positional_inputs[0]);

        const std::vector<STensorView> named_inputs{
            {{"state", ETensorElementType::float32, {1, 2}, EMemoryLocation::host},
             state_values.data(),
             state_values.size() * sizeof(float)},
            {{"image", ETensorElementType::float32, {1, 3}, EMemoryLocation::host},
             image_values.data(),
             image_values.size() * sizeof(float)}};
        const std::vector<const STensorView*> ordered_named_inputs =
            OrderInputViews(expected_inputs, named_inputs);
        REQUIRE(ordered_named_inputs[0] == &named_inputs[1]);
        REQUIRE(ordered_named_inputs[1] == &named_inputs[0]);
        REQUIRE_NOTHROW(
            ValidateHostTensorView(expected_inputs[0], *ordered_named_inputs[0], "test"));
    }
}

TEST_CASE("inference_manager_detects_artifact_type", "[inference][facade]")
{
    using namespace ptafdeploy::inference;

    SECTION("Explicit backend and artifact selections must be compatible")
    {
        CInferenceManager manager;

        SInferenceOptions onnx_options;
        onnx_options.backend = EInferenceBackend::onnxruntime;
        onnx_options.artifact = EModelArtifact::tensorrt_engine;
        REQUIRE_THROWS_WITH(
            manager.LoadModel("fake.engine", onnx_options),
            Catch::Matchers::ContainsSubstring("incompatible backend and artifact"));

        SInferenceOptions tensorrt_options;
        tensorrt_options.backend = EInferenceBackend::tensorrt_engine;
        tensorrt_options.artifact = EModelArtifact::onnx;
        REQUIRE_THROWS_WITH(
            manager.LoadModel("fake.onnx", tensorrt_options),
            Catch::Matchers::ContainsSubstring("incompatible backend and artifact"));
    }

    SECTION("Explicit artifact selection must match the file extension")
    {
        CInferenceManager manager;
        SInferenceOptions options;
        options.artifact = EModelArtifact::onnx;

        REQUIRE_THROWS_WITH(
            manager.LoadModel("fake.engine", options),
            Catch::Matchers::ContainsSubstring("artifact selection does not match"));
    }

    SInferenceOptions options;
    options.backend = EInferenceBackend::tensorrt_engine;

    ptafdeploy::inference::CInferenceManager manager;
    REQUIRE_THROWS_WITH(manager.GetNumInputs(),
                        Catch::Matchers::ContainsSubstring("No inference backend has been loaded"));
    REQUIRE_THROWS_WITH(manager.Infer(std::vector<STensorView>{}),
                        Catch::Matchers::ContainsSubstring("No inference backend has been loaded"));
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    REQUIRE_THROWS_WITH(manager.LoadModel("fake.engine", options),
                        Catch::Matchers::ContainsSubstring("Could not open TensorRT engine file"));
#else
    REQUIRE_THROWS_WITH(manager.LoadModel("fake.engine", options),
                        Catch::Matchers::ContainsSubstring("TensorRT standalone backend"));
#endif
}

TEST_CASE("tensorrt_backend_reports_clear_load_error", "[inference][tensorrt]")
{
    ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine backend;
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    REQUIRE_THROWS_WITH(backend.LoadModel("placeholder.engine"),
                        Catch::Matchers::ContainsSubstring("Could not open TensorRT engine file"));
#else
    REQUIRE_THROWS_WITH(backend.LoadModel("placeholder.engine"),
                        Catch::Matchers::ContainsSubstring("TensorRT standalone backend"));
#endif
}

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
TEST_CASE("tensorrt_backend_runs_engine_when_fixture_is_provided", "[inference][tensorrt]")
{
    using namespace ptafdeploy::inference;

    const char* engine_path = std::getenv("PTAFDEPLOY_TENSORRT_TEST_ENGINE");
    if (engine_path == nullptr || std::string{engine_path}.empty())
    {
        SUCCEED(
            "PTAFDEPLOY_TENSORRT_TEST_ENGINE is not set; optional engine inference is not run.");
        return;
    }

    SRuntimeConfig runtime_config;
    runtime_config.SetBackend(EInferenceBackend::tensorrt_engine);
    runtime_config.SetArtifact(EModelArtifact::tensorrt_engine);
    runtime_config.SetDeviceId(GetTensorRtTestDevice());
    runtime_config.ClearExecutionTargetPriority();
    runtime_config.AddExecutionTarget(EExecutionTarget::tensorrt);

    CInferenceManager manager;
    manager.LoadModelWithRuntimeConfig(engine_path, runtime_config);
    REQUIRE(manager.GetNumInputs() == 1);
    REQUIRE(manager.GetNumOutputs() == 1);
    REQUIRE_THAT(manager.GetBackendDetail(),
                 Catch::Matchers::ContainsSubstring("backend=tensorrt_engine"));

    const std::vector<float> values(11, 0.0F);
    const std::vector<float> outputs = manager.InferSingleFloatInput(values, {1, 11});
    REQUIRE(outputs.size() == 2);
}
#endif

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
TEST_CASE("TensorRT target selection respects strict priority", "[inference][tensorrt]")
{
    using namespace ptafdeploy::inference;
    tensorrt::CInferenceManager_TensorRT_Engine backend;
    SInferenceOptions options;
    options.allow_fallback = false;
    options.execution_target_priority = {EExecutionTarget::cpu, EExecutionTarget::cuda};
    REQUIRE_THROWS_WITH(backend.LoadModel("absent.engine", options),
                        Catch::Matchers::ContainsSubstring("first target"));
    options.allow_fallback = true;
    options.execution_target_priority = {EExecutionTarget::cpu};
    REQUIRE_THROWS_WITH(backend.LoadModel("absent.engine", options),
                        Catch::Matchers::ContainsSubstring("CPU-only"));
}

TEST_CASE("TensorRT failed replacement preserves usable state", "[inference][tensorrt]")
{
    using namespace ptafdeploy::inference;
    const char* fixture = std::getenv("PTAFDEPLOY_TENSORRT_TEST_ENGINE");
    if (fixture == nullptr || std::string(fixture).empty())
        SKIP("Set PTAFDEPLOY_TENSORRT_TEST_ENGINE to a compatible engine");

    SInferenceOptions options;
    options.device_id = GetTensorRtTestDevice();
    options.allow_fallback = false;
    options.execution_target_priority = {EExecutionTarget::cuda};
    tensorrt::CInferenceManager_TensorRT_Engine backend;
    int caller_device = -1;
    REQUIRE(cudaGetDevice(&caller_device) == cudaSuccess);
    backend.LoadModel(fixture, options);
    int current_device = -1;
    REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
    REQUIRE(current_device == caller_device);
    const auto metadata = backend.GetModelMetadata();
    REQUIRE(metadata.inputs.size() == 1);
    auto descriptor = metadata.inputs.front();
    descriptor.shape = {1, 11}; // Same traced fixture contract as the existing engine smoke.
    std::vector<float> values(11, 0.0F);
    const std::vector<STensorView> inputs{
        {descriptor, values.data(), values.size() * sizeof(float)}};
    const auto reference = backend.Infer(inputs);

    const auto corrupt_path =
        std::filesystem::temp_directory_path() /
        ("ptaf-corrupt-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".engine");
    struct SFileCleanup
    {
        std::filesystem::path path;
        ~SFileCleanup()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{corrupt_path};
    {
        std::ofstream file;
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.open(corrupt_path);
        file << "invalid engine";
        file.close();
    }

    enum class EReloadFailure
    {
        missing_file,
        corrupt_engine,
        invalid_profile,
        rejected_targets
    };
    for (const auto failure : {EReloadFailure::missing_file, EReloadFailure::corrupt_engine,
                               EReloadFailure::invalid_profile, EReloadFailure::rejected_targets})
    {
        auto rejected_options = options;
        std::filesystem::path rejected_path = fixture;
        if (failure == EReloadFailure::missing_file)
            rejected_path = corrupt_path.string() + ".missing";
        if (failure == EReloadFailure::corrupt_engine)
            rejected_path = corrupt_path;
        if (failure == EReloadFailure::invalid_profile)
            rejected_options.tensorrt_optimization_profile_index = 999;
        if (failure == EReloadFailure::rejected_targets)
            rejected_options.execution_target_priority = {EExecutionTarget::cpu,
                                                          EExecutionTarget::cuda};
        REQUIRE_THROWS(backend.LoadModel(rejected_path, rejected_options));
        REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
        REQUIRE(current_device == caller_device);
        REQUIRE(backend.GetModelMetadata().backend_detail == metadata.backend_detail);
        REQUIRE(backend.GetModelMetadata().inputs.front().name == metadata.inputs.front().name);
        const auto result = backend.Infer(inputs);
        REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
        REQUIRE(current_device == caller_device);
        REQUIRE(result.size() == reference.size());
        for (size_t index = 0; index < result.size(); ++index)
        {
            REQUIRE(result[index].descriptor.shape == reference[index].descriptor.shape);
            REQUIRE(result[index].bytes() == reference[index].bytes());
            REQUIRE(std::memcmp(result[index].data(), reference[index].data(),
                                result[index].bytes()) == 0);
        }
    }

    options.allow_fallback = true;
    options.execution_target_priority = {EExecutionTarget::cpu, EExecutionTarget::cuda};
    REQUIRE_NOTHROW(backend.LoadModel(fixture, options));
    REQUIRE_THAT(backend.GetModelMetadata().backend_detail,
                 Catch::Matchers::ContainsSubstring("selected_target=cuda;skipped_target_count=1"));
    REQUIRE_NOTHROW(backend.Infer(inputs));
    options.allow_fallback = false;
    options.execution_target_priority = {EExecutionTarget::tensorrt};
    REQUIRE_NOTHROW(backend.LoadModel(fixture, options));
    REQUIRE_NOTHROW(backend.Infer(inputs));
    options.execution_target_priority.clear();
    REQUIRE_NOTHROW(backend.LoadModel(fixture, options));
    REQUIRE_NOTHROW(backend.Infer(inputs));
}
#endif
