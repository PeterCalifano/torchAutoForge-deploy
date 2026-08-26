/**
 * @file testCInferenceManager_ORT.cpp
 * @brief Target-owned ONNX Runtime metadata, inference, and facade tests.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstring>
#include <filesystem>
#include <inference/inference_manager.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    using Catch::Matchers::ContainsSubstring;

    fs::path GetOrtFixturePath()
    {
        const fs::path test_source_path(__FILE__);
        return test_source_path.parent_path().parent_path() / "matlab" / "testSamples" /
               "tracedSampleModel.onnx";
    }

    infer::STensorDescriptor MakeConcreteInputDescriptor(const infer::STensorDescriptor& descriptor)
    {
        infer::STensorDescriptor concrete_descriptor = descriptor;
        for (int64_t& dim : concrete_descriptor.shape)
        {
            if (dim < 0)
            {
                dim = 1;
            }
        }

        return concrete_descriptor;
    }

    infer::STensorBuffer MakeSequentialInputBuffer(const infer::STensorDescriptor& model_input,
                                                   const std::string& name = {})
    {
        infer::STensorDescriptor descriptor = MakeConcreteInputDescriptor(model_input);
        descriptor.name = name;

        infer::STensorBuffer input_buffer = infer::MakeOwnedTensorBuffer(descriptor);
        auto* input_values = reinterpret_cast<float*>(input_buffer.storage.data());
        for (size_t i = 0; i < infer::ComputeElementCount(input_buffer.descriptor.shape); ++i)
        {
            input_values[i] = static_cast<float>(i + 1);
        }

        return input_buffer;
    }

    std::vector<float> ReadFloatOutput(const infer::STensorBuffer& output_buffer)
    {
        REQUIRE(output_buffer.storage.size() % sizeof(float) == 0U);

        std::vector<float> values(output_buffer.storage.size() / sizeof(float));
        if (!values.empty())
        {
            std::memcpy(values.data(), output_buffer.storage.data(), output_buffer.storage.size());
        }
        return values;
    }

    void RequireReferenceOutput(const infer::STensorBuffer& output_buffer)
    {
        const std::vector<float> values = ReadFloatOutput(output_buffer);
        REQUIRE(values.size() == 2);
        REQUIRE(values[0] == Catch::Approx(1.686690331F).margin(1.0e-5F));
        REQUIRE(values[1] == Catch::Approx(4.697796822F).margin(1.0e-5F));
    }
} // namespace

TEST_CASE("CInferenceManager_ORT_loads_model_metadata", "[inference][ort]")
{
    const fs::path model_path = GetOrtFixturePath();
    SECTION("File not found or incorrect extension")
    {
        const std::string invalid_model_path = "invalid_model.onnx";
        ptafdeploy::inference::onnxruntime::CInferenceManager_ORT inference_manager;
        REQUIRE_THROWS_AS(inference_manager.LoadModel(invalid_model_path), std::invalid_argument);

        const std::string invalid_extension_path = "CMakeLists.txt";
        REQUIRE_THROWS_AS(inference_manager.LoadModel(invalid_extension_path),
                          std::invalid_argument);
    }

    SECTION("Session setup populates model metadata")
    {
        ptafdeploy::inference::onnxruntime::CInferenceManager_ORT inference_manager(model_path);
        const auto& metadata = inference_manager.GetModelMetadata();

        REQUIRE(metadata.backend == infer::EInferenceBackend::onnxruntime);
        REQUIRE(metadata.inputs.size() == 1);
        REQUIRE(metadata.outputs.size() == 1);
        REQUIRE(metadata.inputs.front().name == "input");
        REQUIRE(metadata.inputs.front().dtype == infer::ETensorElementType::float32);
        REQUIRE(metadata.inputs.front().shape.size() == 2);
        REQUIRE(metadata.inputs.front().shape[0] < 0);
        REQUIRE(metadata.inputs.front().shape[1] == 11);
        REQUIRE(metadata.outputs.front().dtype == infer::ETensorElementType::float32);
        REQUIRE(metadata.outputs.front().shape.size() == 2);
        REQUIRE(metadata.outputs.front().shape[0] < 0);
        REQUIRE(metadata.outputs.front().shape[1] == 2);
    }
}

TEST_CASE("CInferenceManager_ORT_runs_inference", "[inference][ort]")
{
    infer::onnxruntime::CInferenceManager_ORT inference_manager(GetOrtFixturePath());
    const auto& metadata = inference_manager.GetModelMetadata();

    infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(metadata.inputs.front());

    const std::vector<infer::STensorBuffer> outputs =
        inference_manager.Infer({input_buffer.AsView()});
    REQUIRE(outputs.size() == 1);
    REQUIRE(outputs.front().descriptor.shape == std::vector<int64_t>{1, 2});
    REQUIRE(outputs.front().descriptor.dtype == infer::ETensorElementType::float32);
    REQUIRE(outputs.front().bytes() == sizeof(float) * 2U);
    RequireReferenceOutput(outputs.front());
}

TEST_CASE("CInferenceManager_ORT_validates_input_contract", "[inference][ort]")
{
    infer::onnxruntime::CInferenceManager_ORT inference_manager(GetOrtFixturePath());
    const auto& metadata = inference_manager.GetModelMetadata();
    const infer::STensorDescriptor& model_input = metadata.inputs.front();

    SECTION("Named input mapping accepts model input name")
    {
        infer::STensorBuffer input_buffer =
            MakeSequentialInputBuffer(model_input, model_input.name);
        const std::vector<infer::STensorBuffer> outputs =
            inference_manager.Infer({input_buffer.AsView()});
        REQUIRE(outputs.size() == 1);
        RequireReferenceOutput(outputs.front());
    }

    SECTION("Mixed named and unnamed inputs are rejected before count mapping")
    {
        infer::STensorBuffer named_input = MakeSequentialInputBuffer(model_input, model_input.name);
        infer::STensorBuffer unnamed_input = MakeSequentialInputBuffer(model_input);
        REQUIRE_THROWS_WITH(inference_manager.Infer({named_input.AsView(), unnamed_input.AsView()}),
                            ContainsSubstring("Mixed named and unnamed"));
    }

    SECTION("Duplicate named inputs are rejected")
    {
        infer::STensorBuffer first_input = MakeSequentialInputBuffer(model_input, model_input.name);
        infer::STensorBuffer second_input =
            MakeSequentialInputBuffer(model_input, model_input.name);
        REQUIRE_THROWS_WITH(inference_manager.Infer({first_input.AsView(), second_input.AsView()}),
                            ContainsSubstring("Duplicate input tensor name"));
    }

    SECTION("Missing named input is rejected")
    {
        infer::STensorBuffer input_buffer =
            MakeSequentialInputBuffer(model_input, "not_the_model_input");
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_buffer.AsView()}),
                            ContainsSubstring("Missing input tensor: input"));
    }

    SECTION("Dtype mismatch is rejected")
    {
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        input_buffer.descriptor.dtype = infer::ETensorElementType::float64;
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_buffer.AsView()}),
                            ContainsSubstring("Input dtype mismatch"));
    }

    SECTION("Shape mismatch is rejected")
    {
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        input_buffer.descriptor.shape = {1, 10};
        input_buffer.storage.resize(infer::ComputeByteCount(input_buffer.descriptor));
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_buffer.AsView()}),
                            ContainsSubstring("Input shape mismatch"));
    }

    SECTION("Byte-count mismatch is rejected")
    {
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        infer::STensorView input_view = input_buffer.AsView();
        input_view.bytes -= sizeof(float);
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_view}),
                            ContainsSubstring("Input byte count mismatch"));
    }

    SECTION("Null data is rejected for non-empty tensors")
    {
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        infer::STensorView input_view = input_buffer.AsView();
        input_view.data = nullptr;
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_view}),
                            ContainsSubstring("data pointer is null"));
    }

    SECTION("Non-host memory is rejected")
    {
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        input_buffer.descriptor.location = infer::EMemoryLocation::device_cuda;
        REQUIRE_THROWS_WITH(inference_manager.Infer({input_buffer.AsView()}),
                            ContainsSubstring("Only host memory inputs"));
    }

    SECTION("Unloaded session is rejected")
    {
        infer::onnxruntime::CInferenceManager_ORT unloaded_manager;
        infer::STensorBuffer input_buffer = MakeSequentialInputBuffer(model_input);
        REQUIRE_THROWS_WITH(unloaded_manager.Infer({input_buffer.AsView()}),
                            ContainsSubstring("ONNX Runtime session has not been initialized"));
    }
}

TEST_CASE("CInferenceManager_facade_exposes_wrapper_safe_float_inference",
          "[inference][facade][ort]")
{
    infer::CInferenceManager inference_manager(GetOrtFixturePath().string());

    REQUIRE(inference_manager.GetNumInputs() == 1);
    REQUIRE(inference_manager.GetNumOutputs() == 1);

    const infer::STensorInfo input_info = inference_manager.GetInputInfo(0);
    REQUIRE(input_info.name == "input");
    REQUIRE(input_info.dtype == "float32");
    REQUIRE(input_info.shape.size() == 2);
    REQUIRE(input_info.shape[1] == 11);

    const std::vector<float> input_values{1.0F, 2.0F, 3.0F, 4.0F,  5.0F, 6.0F,
                                          7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
    const std::vector<float> output_values =
        inference_manager.InferSingleFloatInput(input_values, std::vector<int64_t>{1, 11});

    REQUIRE(output_values.size() == 2);
    REQUIRE(output_values[0] == Catch::Approx(1.686690331F).margin(1.0e-5F));
    REQUIRE(output_values[1] == Catch::Approx(4.697796822F).margin(1.0e-5F));
}
