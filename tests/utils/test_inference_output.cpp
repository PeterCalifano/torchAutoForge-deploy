/** @file test_inference_output.cpp
 * @brief Observable serialization and report failure behavior.
 */
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <limits>
#include <utils/inference_output/inference_output.h>
namespace out = ptafdeploy::utils::inference_output;
namespace infer = ptafdeploy::inference;
namespace fs = std::filesystem;

TEST_CASE("JSON output validates storage and preserves numeric values", "[utils][output]")
{
    const infer::SFloatTensor tensor{"prediction", {1, 2}, {1.25F, -0.5F}};
    const auto json = out::TensorValue(tensor);
    const auto& fields = std::get<out::SJsonValue::Object>(json.value);
    REQUIRE(std::get<std::string>(fields.at("dtype").value) == "float32");
    const auto& values = std::get<out::SJsonValue::Array>(fields.at("values").value);
    REQUIRE(std::get<double>(values.at(0).value) == 1.25);
    REQUIRE(std::get<double>(values.at(1).value) == -0.5);
    REQUIRE_THROWS_AS(out::TensorValue(infer::SFloatTensor{"bad", {2}, {1.0F}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        out::TensorValue(infer::SFloatTensor{"bad", {1}, {std::numeric_limits<float>::infinity()}}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(out::Serialize(std::string(1, static_cast<char>(0xff))),
                      std::invalid_argument);
    infer::STensorDescriptor unsupported;
    unsupported.name = "half";
    unsupported.dtype = infer::ETensorElementType::float16;
    unsupported.shape = {1};
    const uint16_t half = 0;
    REQUIRE_THROWS_AS(out::TensorValue(infer::STensorView{unsupported, &half, sizeof(half)}),
                      std::invalid_argument);
    REQUIRE(out::Serialize(uint64_t{18446744073709551615ULL}) == "18446744073709551615");
    REQUIRE_THROWS_AS(out::Serialize(std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(out::FrameValue({0, "image.png", -1.0, {}}), std::invalid_argument);
    REQUIRE_THROWS_AS(out::FrameValue({0, "image.png", 1.0, {{"index", 9}}}),
                      std::invalid_argument);
}

TEST_CASE("Incomplete reports omit uncommitted tails and retain custom fields", "[utils][output]")
{
    const auto root = fs::temp_directory_path() /
                      ("ptaf-output-" +
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
    out::CReport report(root, {1, {{"application", "generic"}}});
    report.Append(
        {0, "frame.png", 1.0, {{"measurement", out::SJsonValue::Object{{"valid", true}}}}});
    {
        std::ofstream tail(root / "frames.jsonl", std::ios::app);
        tail << "INVALID_TAIL";
    }
    report.Publish(false, out::SJsonValue::Object{{"message", "decode failed"}});
    std::ifstream stream(root / "predictions.json");
    const std::string json{std::istreambuf_iterator<char>(stream), {}};
    REQUIRE(json.find("\"status\":\"incomplete\"") != std::string::npos);
    REQUIRE(json.find("\"valid\":true") != std::string::npos);
    REQUIRE(json.find("INVALID_TAIL") == std::string::npos);
    // A directory at the spool path deterministically makes append fail, even as root.
    fs::rename(root / "frames.jsonl", root / "saved.jsonl");
    REQUIRE(fs::create_directory(root / "frames.jsonl"));
    REQUIRE_THROWS(report.Append({1, "next.png", 1.0, {}}));
    fs::remove(root / "frames.jsonl");
    fs::rename(root / "saved.jsonl", root / "frames.jsonl");
    REQUIRE_THROWS_AS(report.Append({1, "next.png", 1.0, {}}), std::logic_error);
    REQUIRE_THROWS_AS(report.Publish(true), std::logic_error);
    REQUIRE_NOTHROW(report.Publish(false, out::SJsonValue::Object{{"message", "append failed"}}));
    REQUIRE_THROWS_AS(out::PrepareOutput(root, root / "input.png"), std::invalid_argument);
    REQUIRE_THROWS_AS(out::CReport(root, {}), std::invalid_argument);
}
