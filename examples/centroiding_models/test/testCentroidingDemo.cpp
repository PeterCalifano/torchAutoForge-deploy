/**
 * @file testCentroidingDemo.cpp
 * @brief Target-owned tests for the image-only centroiding integration.
 */

#include "centroiding_io.h"
#include <fstream>
#include <rapidjson/document.h>

#include <inference/model_facade.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace demo = ptafdeploy::examples::centroiding_models;
    namespace fs = std::filesystem;
    namespace infer = ptafdeploy::inference;

    /** @brief Return the optional external plain ONNX path without taking ownership. */
    [[nodiscard]] fs::path GetOptionalModelPath()
    {
        const char* model_path = std::getenv("PTAFDEPLOY_PLAIN_CENTROIDING_ONNX");
        return model_path == nullptr ? fs::path{} : fs::path{model_path};
    }
} // namespace

TEST_CASE("centroiding_prepares_grayscale_nchw_input", "[example][centroiding]")
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

TEST_CASE("centroiding_maps_normalized_output_to_image_coordinates", "[example][centroiding]")
{
    const infer::SFloatTensor output{"prediction", {1, 2}, {0.25F, 0.75F}};
    const demo::SCentroidResult result =
        demo::DecodeCentroid(output, cv::Size{2048, 1536}, cv::Size{320, 240});

    REQUIRE(result.normalized.x == Catch::Approx(0.25F));
    REQUIRE(result.normalized.y == Catch::Approx(0.75F));
    REQUIRE(result.model_input_pixels.x == Catch::Approx(512.0F));
    REQUIRE(result.model_input_pixels.y == Catch::Approx(1152.0F));
    REQUIRE(result.original_image_pixels.x == Catch::Approx(80.0F));
    REQUIRE(result.original_image_pixels.y == Catch::Approx(180.0F));
}

TEST_CASE("centroiding_rejects_invalid_model_contracts", "[example][centroiding]")
{
    const cv::Mat grayscale_image(2, 3, CV_8UC1, cv::Scalar{0});
    const cv::Mat color_image(2, 3, CV_8UC3, cv::Scalar{0, 0, 0});

    REQUIRE_THROWS_AS(
        demo::PrepareImageTensor(grayscale_image,
                                 infer::STensorInfo{"image", "float32", {-1, 3, 2, 3}, "host"}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::PrepareImageTensor(color_image,
                                 infer::STensorInfo{"image", "float32", {-1, 1, 2, 3}, "host"}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::DecodeCentroid(infer::SFloatTensor{"prediction", {1, 3}, {0.25F, 0.75F, 1.0F}},
                             cv::Size{2048, 1536}, cv::Size{320, 240}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        demo::DecodeCentroid(infer::SFloatTensor{"prediction",
                                                 {1, 2},
                                                 {std::numeric_limits<float>::infinity(), 0.75F}},
                             cv::Size{2048, 1536}, cv::Size{320, 240}),
        std::invalid_argument);
}

TEST_CASE("centroiding_runs_external_onnx_on_bright_ellipse",
          "[example][centroiding][ort][optional]")
{
    const fs::path model_path = GetOptionalModelPath();
    if (!fs::is_regular_file(model_path))
    {
        SKIP("Set PTAFDEPLOY_PLAIN_CENTROIDING_ONNX to the external plain ONNX model.");
    }

    infer::SRuntimeConfig runtime;
    runtime.UseCpuOnly();

    infer::CModelFacade model;
    model.LoadModelWithRoleAndRuntimeConfig(model_path.string(), infer::EModelRole::centroiding,
                                            runtime);

    REQUIRE(model.GetRole() == "centroiding");
    REQUIRE(model.GetNumInputs() == 1U);
    REQUIRE(model.GetNumOutputs() == 1U);
    REQUIRE(model.GetInputInfo(0).name == "image");
    REQUIRE(model.GetInputInfo(0).shape == std::vector<int64_t>{-1, 1, 1536, 2048});
    REQUIRE(model.GetOutputInfo(0).name == "prediction");
    REQUIRE(model.GetOutputInfo(0).shape == std::vector<int64_t>{-1, 2});

    const cv::Mat image = cv::imread(PTAFDEPLOY_CENTROIDING_FIXTURE, cv::IMREAD_GRAYSCALE);
    REQUIRE_FALSE(image.empty());

    const infer::SFloatTensor input = demo::PrepareImageTensor(image, model.GetInputInfo(0));
    const infer::SFloatTensor output = model.InferSingleFloatTensor(input);
    const demo::SCentroidResult result = demo::DecodeCentroid(
        output, cv::Size{static_cast<int>(input.shape[3]), static_cast<int>(input.shape[2])},
        image.size());

    REQUIRE(std::isfinite(result.normalized.x));
    REQUIRE(std::isfinite(result.normalized.y));
    REQUIRE(result.original_image_pixels.x == Catch::Approx(result.normalized.x * image.cols));
    REQUIRE(result.original_image_pixels.y == Catch::Approx(result.normalized.y * image.rows));
}

TEST_CASE("centroiding_orders_natural_filenames_with_lexical_ties", "[example][centroiding]")
{
    std::vector<std::string> names{"frame10.png", "frame2.png", "frame02.png", "Frame2.png"};
    std::sort(names.begin(), names.end(), demo::NaturalLess);
    REQUIRE(names ==
            std::vector<std::string>{"Frame2.png", "frame02.png", "frame2.png", "frame10.png"});
    REQUIRE(demo::NaturalLess("f999999999999999999999", "f1000000000000000000000"));
}

TEST_CASE("centroiding_preserves_outside_predictions_in_json", "[example][centroiding]")
{
    const infer::SFloatTensor output{"prediction", {1, 2}, {1.25F, -0.5F}};
    const auto result = demo::DecodeCentroid(output, cv::Size{2048, 1536}, cv::Size{320, 240});
    REQUIRE_FALSE(demo::Inside(result, cv::Size{320, 240}));
    REQUIRE(result.original_image_pixels.x == 400.0);
    REQUIRE(result.original_image_pixels.y == -120.0);
    rapidjson::Document frame;
    frame.Parse(
        demo::FrameJson(0, "frame2.png", cv::Size{320, 240}, output, result, 12.0, "").c_str());
    REQUIRE_FALSE(frame.HasParseError());
    REQUIRE(frame["overlay"].IsNull());
    REQUIRE(frame["centroid"]["image_pixels"]["x"].GetDouble() == 400.0);
    REQUIRE_FALSE(frame["centroid"]["inside_image"].GetBool());
}

TEST_CASE("centroiding_selects_frames_and_preserves_incomplete_reports", "[example][centroiding]")
{
    const auto root = fs::temp_directory_path() /
                      ("ptaf-centroiding-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(root));
    struct Cleanup
    {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    } cleanup{root};
    REQUIRE_THROWS_AS(demo::SelectFrames(root), std::invalid_argument);
    for (const auto* name : {"frame10.png", "frame2.PNG", "frame02.png"})
        REQUIRE(cv::imwrite((root / name).string(), cv::Mat(32, 32, CV_8UC1, cv::Scalar(80))));
    fs::create_directory(root / "nested");
    REQUIRE(
        cv::imwrite((root / "nested/frame1.png").string(), cv::Mat(2, 2, CV_8UC1, cv::Scalar(0))));
    const auto frames = demo::SelectFrames(root);
    REQUIRE(frames.size() == 3);
    REQUIRE(frames[0].filename() == "frame02.png");
    REQUIRE(frames[1].filename() == "frame2.PNG");
    const auto destination = root / "results";
    demo::PrepareOutput(destination, root);
    demo::CReport report(destination, demo::MetadataJson(root, frames.size()));
    const infer::SFloatTensor output{"prediction", {1, 2}, {0.5F, 0.5F}};
    const auto result = demo::DecodeCentroid(output, cv::Size{32, 32}, cv::Size{32, 32});
    report.Append(
        demo::FrameJson(0, frames[0].filename(), cv::Size{32, 32}, output, result, 1.0, ""));
    {
        std::ofstream partial(destination / "frames.jsonl", std::ios::app);
        partial << "{uncommitted partial write";
    }
    report.Publish(false, R"({"stage":"decode","frame_index":1,"message":"unreadable"})");
    std::ifstream input(destination / "predictions.json");
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document document;
    document.Parse(text.c_str());
    REQUIRE_FALSE(document.HasParseError());
    REQUIRE(std::string(document["status"].GetString()) == "incomplete");
    REQUIRE(document["frames"].Size() == 1);
    REQUIRE_THROWS_AS(demo::PrepareOutput(destination, root), std::invalid_argument);
    REQUIRE_THROWS_AS(demo::PrepareOutput(root, root), std::invalid_argument);
    REQUIRE(demo::SelectFrames(root).size() == 3);
    demo::SaveOverlay(frames[0], destination / "overlay.png", result);
    const auto overlay = cv::imread((destination / "overlay.png").string());
    REQUIRE(overlay.size() == cv::Size(32, 32));
    REQUIRE(overlay.at<cv::Vec3b>(0, 0) == cv::Vec3b(80, 80, 80));
    REQUIRE(overlay.at<cv::Vec3b>(16, 16) == cv::Vec3b(255, 255, 255));
}

TEST_CASE("centroiding_keeps_boundary_and_large_finite_coordinates", "[example][centroiding]")
{
    const auto boundary =
        demo::DecodeCentroid(infer::SFloatTensor{"prediction", {1, 2}, {1.0F, 0.0F}},
                             cv::Size{2048, 1536}, cv::Size{320, 240});
    REQUIRE_FALSE(demo::Inside(boundary, cv::Size{320, 240}));
    REQUIRE(boundary.original_image_pixels.x == 320.0);
    const auto large = demo::DecodeCentroid(
        infer::SFloatTensor{"prediction", {1, 2}, {std::numeric_limits<float>::max(), 0.0F}},
        cv::Size{2048, 1536}, cv::Size{320, 240});
    REQUIRE(std::isfinite(large.model_input_pixels.x));
    REQUIRE(std::isfinite(large.original_image_pixels.x));
}
