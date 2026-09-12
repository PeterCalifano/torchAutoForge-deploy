/** @file test_inference_output.cpp
 * @brief Observable serialization and report failure behavior.
 */

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <limits>
#include <utils/inference_output/inference_output.h>
namespace inference_output = ptafdeploy::utils::inference_output;
namespace inference = ptafdeploy::inference;
namespace filesystem = std::filesystem;

TEST_CASE("JSON output validates storage and preserves numeric values", "[utils][output]")
{
    const inference::SFloatTensor tensor{"prediction", {1, 2}, {1.25F, -0.5F}};
    const auto json = inference_output::TensorValue(tensor);
    const auto& fields = std::get<inference_output::SJsonValue::Object>(json.value);
    REQUIRE(std::get<std::string>(fields.at("dtype").value) == "float32");
    const auto& values = std::get<inference_output::SJsonValue::Array>(fields.at("values").value);
    REQUIRE(std::get<double>(values.at(0).value) == 1.25);
    REQUIRE(std::get<double>(values.at(1).value) == -0.5);
    REQUIRE_THROWS_AS(inference_output::TensorValue(inference::SFloatTensor{"bad", {2}, {1.0F}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(inference_output::TensorValue(inference::SFloatTensor{
                          "bad", {1}, {std::numeric_limits<float>::infinity()}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(inference_output::Serialize(std::string(1, static_cast<char>(0xff))),
                      std::invalid_argument);
    inference::STensorDescriptor unsupported;
    unsupported.name = "half";
    unsupported.dtype = inference::ETensorElementType::float16;
    unsupported.shape = {1};
    const uint16_t half = 0;
    REQUIRE_THROWS_AS(
        inference_output::TensorValue(inference::STensorView{unsupported, &half, sizeof(half)}),
        std::invalid_argument);
    REQUIRE(inference_output::Serialize(uint64_t{18446744073709551615ULL}) ==
            "18446744073709551615");
    REQUIRE_THROWS_AS(inference_output::Serialize(std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(inference_output::FrameValue({0, "image.png", -1.0, {}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(inference_output::FrameValue({0, "image.png", 1.0, {{"index", 9}}}),
                      std::invalid_argument);
}

TEST_CASE("Incomplete reports omit uncommitted tails and retain custom fields", "[utils][output]")
{
    const auto test_directory =
        filesystem::temp_directory_path() /
        ("ptaf-output-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(filesystem::create_directory(test_directory));
    struct CTemporaryDirectoryCleanup
    {
        filesystem::path path;

        ~CTemporaryDirectoryCleanup()
        {
            std::error_code ignored;
            filesystem::remove_all(path, ignored);
        }
    } directory_cleanup{test_directory};

    inference_output::CReport report(test_directory, {1, {{"application", "generic"}}});
    report.Append({0,
                   "frame.png",
                   1.0,
                   {{"measurement", inference_output::SJsonValue::Object{{"valid", true}}}}});
    {
        std::ofstream tail(test_directory / "frames.jsonl", std::ios::app);
        tail << "INVALID_TAIL";
    }
    report.Publish(false, inference_output::SJsonValue::Object{{"message", "decode failed"}});
    std::ifstream stream(test_directory / "predictions.json");
    const std::string json{std::istreambuf_iterator<char>(stream), {}};
    REQUIRE(json.find("\"status\":\"incomplete\"") != std::string::npos);
    REQUIRE(json.find("\"valid\":true") != std::string::npos);
    REQUIRE(json.find("INVALID_TAIL") == std::string::npos);
    // A directory at the spool path deterministically makes append fail, even as root.
    filesystem::rename(test_directory / "frames.jsonl", test_directory / "saved.jsonl");
    REQUIRE(filesystem::create_directory(test_directory / "frames.jsonl"));
    REQUIRE_THROWS(report.Append({1, "next.png", 1.0, {}}));
    filesystem::remove(test_directory / "frames.jsonl");
    filesystem::rename(test_directory / "saved.jsonl", test_directory / "frames.jsonl");
    REQUIRE_THROWS_AS(report.Append({1, "next.png", 1.0, {}}), std::logic_error);
    REQUIRE_THROWS_AS(report.Publish(true), std::logic_error);
    REQUIRE_NOTHROW(
        report.Publish(false, inference_output::SJsonValue::Object{{"message", "append failed"}}));
    REQUIRE_THROWS_AS(inference_output::PrepareOutput(test_directory, test_directory / "input.png"),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(inference_output::CReport(test_directory, {}), std::invalid_argument);
}

TEST_CASE("Report reader preserves incomplete and singleton arrays and rejects malformed schemas",
          "[utils][output]")
{
    const auto test_directory =
        filesystem::temp_directory_path() /
        ("ptaf-reader-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(filesystem::create_directory(test_directory));
    struct CTemporaryDirectoryCleanup
    {
        filesystem::path path;

        ~CTemporaryDirectoryCleanup()
        {
            std::error_code ignored;
            filesystem::remove_all(path, ignored);
        }
    } directory_cleanup{test_directory};

    inference_output::CReport writer(test_directory, {1, {{"model", nullptr}}});
    const auto initial_report =
        inference_output::ReadInferenceReport(test_directory / "predictions.json");
    REQUIRE_FALSE(initial_report.complete);
    REQUIRE(initial_report.frames.empty());

    writer.Append({0,
                   "one.png",
                   2.0,
                   {{"array", inference_output::SJsonValue::Array{1}},
                    {"flag", true},
                    {"overlay", nullptr}}});
    writer.Publish(true);

    const auto completed_report =
        inference_output::ReadInferenceReport(test_directory / "predictions.json");
    REQUIRE(completed_report.complete);
    REQUIRE(completed_report.frames.size() == 1);
    REQUIRE(std::get<inference_output::SJsonValue::Array>(
                completed_report.frames[0].fields.at("array").value)
                .size() == 1);
    REQUIRE(std::get<bool>(completed_report.frames[0].fields.at("flag").value));

    // Malformed envelopes and duplicate JSON keys must fail before evaluation.
    const auto invalid_report_path = test_directory / "bad.json";
    for (const char* text : {"{\"schema_version\":2,\"status\":\"complete\",\"frames\":[]}",
                             "{\"schema_version\":1,\"status\":\"complete\",\"frames\":[{\"index\":"
                             "2,\"source\":\"x\",\"inference_ms\":1}]}",
                             "{\"key\":1,\"key\":2}", "{broken"})
    {
        {
            std::ofstream file(invalid_report_path);
            file << text;
        }
        REQUIRE_THROWS(inference_output::ReadInferenceReport(invalid_report_path));
    }
}

TEST_CASE("Report publication failure preserves the prior report and committed spool",
          "[utils][output]")
{
    const auto test_directory =
        filesystem::temp_directory_path() /
        ("ptaf-publish-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(filesystem::create_directory(test_directory));
    struct CTemporaryDirectoryCleanup
    {
        filesystem::path path;

        ~CTemporaryDirectoryCleanup()
        {
            std::error_code ignored;
            filesystem::remove_all(path, ignored);
        }
    } directory_cleanup{test_directory};

    inference_output::CReport writer(test_directory, {});

    writer.Append({0, "one.png", 1.0, {}});

    // Block publication without relying on platform-dependent permission failures.
    REQUIRE(filesystem::create_directory(test_directory / "predictions.json.tmp"));
    REQUIRE_THROWS(writer.Publish(true));
    REQUIRE_FALSE(
        inference_output::ReadInferenceReport(test_directory / "predictions.json").complete);
    REQUIRE(filesystem::exists(test_directory / "frames.jsonl"));

    // Removing the obstacle allows an incomplete report to retain the completed frame.
    filesystem::remove(test_directory / "predictions.json.tmp");
    writer.Publish(false, inference_output::SJsonValue::Object{{"stage", "output"}});
    REQUIRE(
        inference_output::ReadInferenceReport(test_directory / "predictions.json").frames.size() ==
        1);
}
