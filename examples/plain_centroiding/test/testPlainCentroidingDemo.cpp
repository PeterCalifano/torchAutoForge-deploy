/**
 * @file testPlainCentroidingDemo.cpp
 * @brief Target-owned tests for the plain image-only centroiding integration.
 */

#include "plain_centroiding_support.h"

#include <inference/model_facade.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace demo = ptafdeploy::examples::plain_centroiding;
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;

    constexpr float kFixtureCenterX = 218.0F;
    constexpr float kFixtureCenterY = 91.0F;
    constexpr float kFixtureTolerancePixels = 5.0F;

    /** @brief Return the optional external plain ONNX path without taking ownership. */
    [[nodiscard]] fs::path GetOptionalModelPath()
    {
        const char* model_path = std::getenv("PTAFDEPLOY_PLAIN_CENTROIDING_ONNX");
        return model_path == nullptr ? fs::path{} : fs::path{model_path};
    }
} // namespace

TEST_CASE("plain_centroiding_prepares_grayscale_nchw_input",
          "[example][plain_centroiding]")
{
    cv::Mat image(2, 3, CV_8UC1);
    image.at<uint8_t>(0, 0) = 0U;
    image.at<uint8_t>(0, 1) = 64U;
    image.at<uint8_t>(0, 2) = 128U;
    image.at<uint8_t>(1, 0) = 192U;
    image.at<uint8_t>(1, 1) = 224U;
    image.at<uint8_t>(1, 2) = 255U;

    const infer::STensorInfo input_info{"image", "float32", {-1, 1, 2, 3}, "host"};
    const infer::SFloatTensor tensor = demo::PrepareImageTensor(image, input_info);

    REQUIRE(tensor.name == "image");
    REQUIRE(tensor.shape == std::vector<int64_t>{1, 1, 2, 3});
    REQUIRE(tensor.values.size() == 6U);
    REQUIRE(tensor.values[0] == Catch::Approx(0.0F));
    REQUIRE(tensor.values[1] == Catch::Approx(64.0F / 255.0F));
    REQUIRE(tensor.values[2] == Catch::Approx(128.0F / 255.0F));
    REQUIRE(tensor.values[3] == Catch::Approx(192.0F / 255.0F));
    REQUIRE(tensor.values[4] == Catch::Approx(224.0F / 255.0F));
    REQUIRE(tensor.values[5] == Catch::Approx(1.0F));
}

TEST_CASE("plain_centroiding_maps_normalized_output_to_image_coordinates",
          "[example][plain_centroiding]")
{
    const infer::SFloatTensor output{"prediction", {1, 2}, {0.25F, 0.75F}};
    const demo::SCentroidResult result = demo::DecodeCentroid(
        output,
        cv::Size{2048, 1536},
        cv::Size{320, 240});

    REQUIRE(result.normalized.x == Catch::Approx(0.25F));
    REQUIRE(result.normalized.y == Catch::Approx(0.75F));
    REQUIRE(result.model_input_pixels.x == Catch::Approx(512.0F));
    REQUIRE(result.model_input_pixels.y == Catch::Approx(1152.0F));
    REQUIRE(result.original_image_pixels.x == Catch::Approx(80.0F));
    REQUIRE(result.original_image_pixels.y == Catch::Approx(180.0F));
}

TEST_CASE("plain_centroiding_rejects_invalid_model_contracts",
          "[example][plain_centroiding]")
{
    const cv::Mat grayscale_image(2, 3, CV_8UC1, cv::Scalar{0});
    const cv::Mat color_image(2, 3, CV_8UC3, cv::Scalar{0, 0, 0});

    REQUIRE_THROWS_AS(
        demo::PrepareImageTensor(
            grayscale_image,
            infer::STensorInfo{"image", "float32", {-1, 3, 2, 3}, "host"}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::PrepareImageTensor(
            color_image,
            infer::STensorInfo{"image", "float32", {-1, 1, 2, 3}, "host"}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::DecodeCentroid(
            infer::SFloatTensor{"prediction", {1, 3}, {0.25F, 0.75F, 1.0F}},
            cv::Size{2048, 1536},
            cv::Size{320, 240}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::DecodeCentroid(
            infer::SFloatTensor{"prediction", {1, 2}, {1.01F, 0.75F}},
            cv::Size{2048, 1536},
            cv::Size{320, 240}),
        std::invalid_argument);
}

TEST_CASE("plain_centroiding_runs_external_onnx_on_bright_ellipse",
          "[example][plain_centroiding][ort][optional]")
{
    const fs::path model_path = GetOptionalModelPath();
    if (!fs::is_regular_file(model_path))
    {
        SKIP("Set PTAFDEPLOY_PLAIN_CENTROIDING_ONNX to the external plain ONNX model.");
    }

    infer::SRuntimeConfig runtime;
    runtime.UseCpuOnly();

    infer::CModelFacade model;
    model.LoadModelWithRoleAndRuntimeConfig(
        model_path.string(),
        infer::EModelRole::centroiding,
        runtime);

    REQUIRE(model.GetRole() == "centroiding");
    REQUIRE(model.GetNumInputs() == 1U);
    REQUIRE(model.GetNumOutputs() == 1U);
    REQUIRE(model.GetInputInfo(0).name == "image");
    REQUIRE(model.GetInputInfo(0).shape == std::vector<int64_t>{-1, 1, 1536, 2048});
    REQUIRE(model.GetOutputInfo(0).name == "prediction");
    REQUIRE(model.GetOutputInfo(0).shape == std::vector<int64_t>{-1, 2});

    const cv::Mat image = cv::imread(PTAFDEPLOY_PLAIN_CENTROIDING_FIXTURE,
                                     cv::IMREAD_GRAYSCALE);
    REQUIRE_FALSE(image.empty());

    const infer::SFloatTensor input = demo::PrepareImageTensor(image, model.GetInputInfo(0));
    const infer::SFloatTensor output = model.InferSingleFloatTensor(input);
    const demo::SCentroidResult result = demo::DecodeCentroid(
        output,
        cv::Size{static_cast<int>(input.shape[3]), static_cast<int>(input.shape[2])},
        image.size());

    const float error_x = result.original_image_pixels.x - kFixtureCenterX;
    const float error_y = result.original_image_pixels.y - kFixtureCenterY;
    const float error_pixels = std::hypot(error_x, error_y);

    REQUIRE(result.normalized.x >= 0.0F);
    REQUIRE(result.normalized.x <= 1.0F);
    REQUIRE(result.normalized.y >= 0.0F);
    REQUIRE(result.normalized.y <= 1.0F);
    REQUIRE(error_pixels < kFixtureTolerancePixels);
}
