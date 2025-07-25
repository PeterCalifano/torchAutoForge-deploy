#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>

using std::cout, std::endl;
#define MODEL_PATH "/home/peterc/devDir/ML-repos/torchAutoForge-deploy/tests/.test_samples/zealous-cow-471_epoch_2893_0.onnx" // DEVNOTE: relative paths won't work in the build folder

static const std::string model_path = MODEL_PATH;
TEST_CASE("CInferenceManager_ORT_startup_utils", "[inference]")
{

    SECTION("File not found or incorrect extension")
    {
        // Test with an invalid model path
        const std::string invalid_model_path = "invalid_model.onnx";
        REQUIRE_THROWS_AS(deploy_ort::CInferenceManager_ORT(invalid_model_path, false), std::invalid_argument);

        // Invalid extension
        const std::string invalid_extension_path = "CMakeLists.txt";
        REQUIRE_THROWS_AS(deploy_ort::CInferenceManager_ORT(invalid_extension_path, false), std::invalid_argument);
    }

    SECTION("Session setup and initialization")
    {
        // Test with a valid model path
        deploy_ort::CInferenceManager_ORT inference_manager(model_path, true, Ort::SessionOptions());
        REQUIRE_NOTHROW(inference_manager.initialize());
    }
};

TEST_CASE("CInferenceManager_ORT_infer", "[inference]")
{

    deploy_ort::CInferenceManager_ORT inference_manager(model_path, true, Ort::SessionOptions());
    SECTION("Session setup and model inference with dummy values")
    {
        std::runtime_error error("This test is not implemented yet.");
    }
}

/*
TEST_CASE_METHOD(fixtures::SObjectFixture, "test_template_method_fixture", "[test]")
{
    // Test print method
    SECTION("This_is_a_test_section")
    {
        cout << "This is a test section inside a fixture object, which contains the variable: " << fixtureVariable << "\n";
    }

    REQUIRE(1 == 1);
    REQUIRE(fixtureVariable == 0);
};
*/