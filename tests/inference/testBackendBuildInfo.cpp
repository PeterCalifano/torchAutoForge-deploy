#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <inference/inference_build_info.h>
#include <inference/inference_manager.h>

#include <algorithm>
#include <string>

namespace infer = ptafdeploy::inference;

namespace
{
    [[nodiscard]] infer::SBackendBuildInfo Find(const std::string& name)
    {
        const auto report = infer::GetBackendBuildInfo();
        const auto found = std::find_if(report.begin(), report.end(),
                                        [&](const auto& info) { return info.name == name; });
        REQUIRE(found != report.end());
        return *found;
    }
} // namespace

TEST_CASE("Backend build info reports both concrete backends", "[inference][build_info]")
{
    REQUIRE(infer::GetBackendBuildInfo().size() == 2U);
    REQUIRE_FALSE(infer::FormatBackendBuildInfo().empty());
}

TEST_CASE("ONNX Runtime build info matches the configured option", "[inference][build_info]")
{
    const infer::SBackendBuildInfo info = Find("onnxruntime");
#if defined(PTAFDEPLOY_ENABLE_ORT)
    REQUIRE(info.compiled_in);
    REQUIRE_FALSE(info.version.empty());
    REQUIRE_FALSE(info.providers.empty());
    REQUIRE(info.note.empty());
#else
    REQUIRE_FALSE(info.compiled_in);
    REQUIRE_FALSE(info.runtime_available);
    REQUIRE_FALSE(info.note.empty());
    infer::CInferenceManager manager;
    REQUIRE_THROWS_WITH(manager.LoadModel(std::string("missing.onnx")),
                        Catch::Matchers::ContainsSubstring("not built"));
#endif
}

TEST_CASE("TensorRT build info matches the configured option", "[inference][build_info]")
{
    const infer::SBackendBuildInfo info = Find("tensorrt_engine");
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
    REQUIRE(info.compiled_in);
#else
    REQUIRE_FALSE(info.compiled_in);
    REQUIRE_FALSE(info.runtime_available);
    REQUIRE_FALSE(info.note.empty());
#endif
}
