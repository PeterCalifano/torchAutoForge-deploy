/**
 * @file testModelFacade.cpp
 * @brief Target-owned model-role configuration and facade inference tests.
 */

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <inference/model_facade.h>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace fs = std::filesystem;
    using Catch::Matchers::ContainsSubstring;

    fs::path GetOrtFixturePath()
    {
        const fs::path test_source_path(__FILE__);
        return test_source_path.parent_path().parent_path() / "matlab" / "testSamples" /
               "tracedSampleModel.onnx";
    }

    fs::path GetModelConfigPath(const std::string& filename)
    {
        const fs::path test_source_path(__FILE__);
        return test_source_path.parent_path() / "model_configs" / filename;
    }

    void RequireReferenceOutput(const std::vector<float>& values)
    {
        REQUIRE(values.size() == 2);
        REQUIRE(values[0] == Catch::Approx(1.686690331F).margin(1.0e-5F));
        REQUIRE(values[1] == Catch::Approx(4.697796822F).margin(1.0e-5F));
    }

    bool ContainsRole(const std::vector<std::string>& roles, const std::string& role)
    {
        return std::find(roles.begin(), roles.end(), role) != roles.end();
    }
} // namespace

TEST_CASE("CModelFacade_exposes_role_contract", "[inference][model_facade]")
{
    const std::string model_path = GetOrtFixturePath().string();
    infer::CModelFacade model(model_path, infer::EModelRole::centroiding);

    const infer::SModelContract contract = model.GetContract();
    REQUIRE(contract.artifact_path == model_path);
    REQUIRE(contract.role == "centroiding");
    REQUIRE(contract.preprocessing == "caller_supplied_tensors");
    REQUIRE(contract.postprocessing == "raw_model_outputs");
    REQUIRE(contract.inputs.size() == 1);
    REQUIRE(contract.outputs.size() == 1);
    REQUIRE(contract.inputs.front().name == "input");

    REQUIRE(model.GetRole() == "centroiding");
    REQUIRE(model.GetNumInputs() == 1);
    REQUIRE(model.GetNumOutputs() == 1);
    REQUIRE(model.GetInputInfo(0).shape[1] == 11);
    REQUIRE(model.GetOutputInfo(0).shape[1] == 2);
}

TEST_CASE("CModelFacade_runs_float_inference_for_role_models", "[inference][model_facade]")
{
    infer::SModelRoleConfig config{infer::EModelRole::object_detection, "image_tensor_nchw",
                                   "raw_detection_tensor"};
    infer::CModelFacade model(GetOrtFixturePath().string(), config);

    REQUIRE(model.GetRole() == "object_detection");
    REQUIRE(model.GetPreprocessing() == "image_tensor_nchw");
    REQUIRE(model.GetPostprocessing() == "raw_detection_tensor");

    const std::vector<float> input_values{1.0F, 2.0F, 3.0F, 4.0F,  5.0F, 6.0F,
                                          7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
    RequireReferenceOutput(model.InferSingleFloatInput(input_values, {1, 11}));
}

TEST_CASE("CModelFacade_validates_roles_and_loaded_state", "[inference][model_facade]")
{
    SECTION("Recognized roles include executable and reserved contract names")
    {
        const std::vector<std::string> roles = infer::CModelFacade::GetRecognizedRoles();
        REQUIRE(ContainsRole(roles, "raw_tensor"));
        REQUIRE(ContainsRole(roles, "centroiding"));
        REQUIRE(ContainsRole(roles, "object_detection"));
        REQUIRE(ContainsRole(roles, "feature_matching"));
        REQUIRE(ContainsRole(roles, "tracking"));
        REQUIRE(ContainsRole(roles, "optical_flow"));
    }

    SECTION("Runtime config uses enums and rejects impossible device ids")
    {
        infer::SRuntimeConfig runtime_config;
        runtime_config.backend = infer::EInferenceBackend::onnxruntime;
        runtime_config.artifact = infer::EModelArtifact::onnx;
        runtime_config.execution_target_priority = {infer::EExecutionTarget::cpu};
        runtime_config.device_id = -1;

        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(model.LoadModelWithRoleAndRuntimeConfig(GetOrtFixturePath().string(),
                                                                    infer::EModelRole::centroiding,
                                                                    runtime_config),
                            ContainsSubstring("device_id"));
    }

    SECTION("Unloaded model reports clear error")
    {
        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(model.GetContract(), ContainsSubstring("No model has been loaded"));
        REQUIRE_THROWS_WITH(model.InferSingleFloatInput({}, {0}),
                            ContainsSubstring("No model has been loaded"));
    }
}

TEST_CASE("CModelFacade_loads_role_configs", "[inference][model_facade]")
{
    SECTION("Centroiding config loads model and runs fixture inference")
    {
        infer::CModelFacade model;
        model.LoadModelConfig(GetModelConfigPath("centroiding_fixture.ptafmodel").string());

        const infer::SModelContract contract = model.GetContract();
        REQUIRE(contract.role == "centroiding");
        REQUIRE(contract.preprocessing == "caller_supplied_tensors");
        REQUIRE(contract.postprocessing == "centroid_xy_float32");
        REQUIRE(contract.runtime.backend == infer::EInferenceBackend::onnxruntime);
        REQUIRE(contract.runtime.execution_target_priority ==
                std::vector<infer::EExecutionTarget>{infer::EExecutionTarget::cpu});
        REQUIRE(contract.backend_detail.find("applied_ort_providers=cpu") != std::string::npos);
        REQUIRE(contract.config_path.find("centroiding_fixture.ptafmodel") != std::string::npos);
        REQUIRE(contract.artifact_path.find("tracedSampleModel.onnx") != std::string::npos);

        const std::vector<float> input_values{1.0F, 2.0F, 3.0F, 4.0F,  5.0F, 6.0F,
                                              7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
        RequireReferenceOutput(model.InferSingleFloatInput(input_values, {1, 11}));
    }

    SECTION("Object-detection config dispatches through same facade without ORT specialization")
    {
        infer::CModelFacade model;
        model.LoadModelConfig(GetModelConfigPath("object_detection_fixture.ptafmodel").string());

        REQUIRE(model.GetRole() == "object_detection");
        REQUIRE(model.GetPreprocessing() == "caller_supplied_tensors");
        REQUIRE(model.GetPostprocessing() == "raw_detection_tensor");
        REQUIRE(model.GetNumInputs() == 1);
        REQUIRE(model.GetNumOutputs() == 1);
    }

    SECTION("Bad config keys are rejected")
    {
        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_unknown_key.ptafmodel").string()),
            ContainsSubstring("Unknown model config key"));
    }

    SECTION("Bad enum-like config values are rejected before model load")
    {
        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_execution_target.ptafmodel").string()),
            ContainsSubstring("Unsupported execution target"));
    }

    SECTION("Bad scalar runtime values are rejected before model load")
    {
        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_negative_device.ptafmodel").string()),
            ContainsSubstring("device_id"));
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_negative_threads.ptafmodel").string()),
            ContainsSubstring("thread counts"));
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(
                GetModelConfigPath("bad_negative_tensorrt_profile.ptafmodel").string()),
            ContainsSubstring("TensorRT optimization profile"));
        REQUIRE_THROWS_WITH(model.LoadModelConfig(
                                GetModelConfigPath("bad_negative_tensorrt_dla.ptafmodel").string()),
                            ContainsSubstring("Unknown model config key: tensorrt_dla_core"));
    }

    SECTION("Bad direct runtime thread values are rejected before backend load")
    {
        infer::SRuntimeConfig runtime_config;
        runtime_config.intra_op_num_threads = -1;

        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(
            model.LoadModelWithRuntimeConfig(GetOrtFixturePath().string(), runtime_config),
            ContainsSubstring("thread counts"));

        runtime_config.intra_op_num_threads = 1;
        runtime_config.tensorrt_optimization_profile_index = -1;
        REQUIRE_THROWS_WITH(
            model.LoadModelWithRuntimeConfig(GetOrtFixturePath().string(), runtime_config),
            ContainsSubstring("TensorRT optimization profile"));
    }

    SECTION("TensorRT profile selection remains validated value config even when another backend "
            "is used")
    {
        infer::CModelFacade model;
        model.LoadModelConfig(
            GetModelConfigPath("runtime_tensorrt_options_fixture.ptafmodel").string());

        const infer::SModelContract contract = model.GetContract();
        REQUIRE(contract.runtime.tensorrt_optimization_profile_index == 2);
    }

    SECTION("Manifest schema version is required and validated")
    {
        infer::CModelFacade model;
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_missing_schema.ptafmodel").string()),
            ContainsSubstring("schema_version"));
        REQUIRE_THROWS_WITH(
            model.LoadModelConfig(GetModelConfigPath("bad_schema_version.ptafmodel").string()),
            ContainsSubstring("Unsupported model config schema_version"));
    }
}
