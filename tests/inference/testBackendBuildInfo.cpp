/**
 * @file testBackendBuildInfo.cpp
 * @brief Backend availability report checks for every backend configuration.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <inference/inference_build_info.h>
#include <inference/inference_config_parsing.h>
#include <inference/inference_manager.h>

#include <algorithm>
#include <string>

namespace infer = ptafdeploy::inference;

namespace
{
    [[nodiscard]] infer::SBackendBuildInfo Find(const infer::EInferenceBackend backend)
    {
        const auto report = infer::GetBackendBuildInfo();
        const auto found = std::find_if(report.begin(), report.end(), [backend](const auto& info)
                                        { return info.backend == backend; });
        REQUIRE(found != report.end());
        return *found;
    }
} // namespace

TEST_CASE("Backend build info reports both concrete backends", "[inference][build_info]")
{
    REQUIRE(infer::GetBackendBuildInfo().size() == 2U);
    REQUIRE_FALSE(infer::FormatBackendBuildInfo().empty());
}

TEST_CASE("Backend report order and identities are stable", "[inference][build_info]")
{
    const auto report = infer::GetBackendBuildInfo();
    REQUIRE(report.size() == 2U);

    REQUIRE(report[0].backend == infer::EInferenceBackend::onnxruntime);
    REQUIRE(report[0].name == "onnxruntime");

    REQUIRE(report[1].backend == infer::EInferenceBackend::tensorrt_engine);
    REQUIRE(report[1].name == "tensorrt_engine");
}

TEST_CASE("ONNX Runtime build info matches the configured option", "[inference][build_info]")
{
    const infer::SBackendBuildInfo info = Find(infer::EInferenceBackend::onnxruntime);
#if defined(PTAFDEPLOY_ENABLE_ORT)
    REQUIRE(info.compiled_in);
    REQUIRE_FALSE(info.version.empty());
    REQUIRE_FALSE(info.providers.empty());
    REQUIRE(info.note.empty());
    REQUIRE(infer::IsOrtExecutionTargetAvailable(infer::EExecutionTarget::cpu));
#else
    REQUIRE_FALSE(info.compiled_in);
    REQUIRE_FALSE(info.runtime_available);
    REQUIRE_FALSE(info.note.empty());
    REQUIRE_FALSE(infer::IsOrtExecutionTargetAvailable(infer::EExecutionTarget::cpu));

    infer::CInferenceManager manager;
    REQUIRE_THROWS_WITH(manager.LoadModel(std::string("missing.onnx")),
                        Catch::Matchers::ContainsSubstring(std::string(infer::kOrtNotBuiltHint)));
#endif
}

TEST_CASE("TensorRT build info matches the configured option", "[inference][build_info]")
{
    const infer::SBackendBuildInfo info = Find(infer::EInferenceBackend::tensorrt_engine);
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    REQUIRE(info.compiled_in);
    REQUIRE_FALSE(info.version.empty());
#else
    REQUIRE_FALSE(info.compiled_in);
    REQUIRE_FALSE(info.runtime_available);
    REQUIRE_FALSE(info.note.empty());

    infer::CInferenceManager manager;
    REQUIRE_THROWS_WITH(
        manager.LoadModel(std::string("missing.engine")),
        Catch::Matchers::ContainsSubstring(std::string(infer::kTensorRtNotBuiltHint)));
#endif
}

TEST_CASE("Artifact kind is detected from the file extension", "[inference][build_info]")
{
    REQUIRE(infer::DetectModelArtifact("model.onnx") == infer::EModelArtifact::onnx);
    REQUIRE(infer::DetectModelArtifact("model.engine") == infer::EModelArtifact::tensorrt_engine);
    REQUIRE(infer::DetectModelArtifact("model.plan") == infer::EModelArtifact::tensorrt_engine);
    REQUIRE_THROWS_WITH(infer::DetectModelArtifact("model.bin"),
                        Catch::Matchers::ContainsSubstring("Unsupported model artifact"));
}
