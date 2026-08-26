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
#include <cstdlib>
#include <limits>

#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
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
