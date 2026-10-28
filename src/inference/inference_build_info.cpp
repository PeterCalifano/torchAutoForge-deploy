/**
 * @file inference_build_info.cpp
 * @brief Backend availability report for the current build.
 */

#include "inference_build_info.h"

#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <inference/tensorrt/tensorrt_inference_engine.h>

#include <sstream>

namespace ptafdeploy::inference
{
    namespace
    {
        [[nodiscard]] SBackendBuildInfo MakeOrtInfo()
        {
            SBackendBuildInfo info;
            info.name = "onnxruntime";
#if defined(PTAFDEPLOY_ENABLE_ORT)
            info.compiled_in = true;
            info.version = onnxruntime::CInferenceManager_ORT::GetRuntimeVersion();
            info.providers = onnxruntime::CInferenceManager_ORT::GetAvailableProviders();
            info.runtime_available = !info.providers.empty();
#else
            info.note = "reconfigure with -Dautoforge_deploy_ENABLE_ORT=ON";
#endif
            return info;
        }

        [[nodiscard]] SBackendBuildInfo MakeTensorRtInfo()
        {
            SBackendBuildInfo info;
            info.name = "tensorrt_engine";
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
            info.compiled_in = true;
            info.version = tensorrt::CInferenceManager_TensorRT_Engine::GetRuntimeVersion();
            info.runtime_available = tensorrt::CInferenceManager_TensorRT_Engine::IsRuntimeAvailable();
#else
            info.note = "reconfigure with -Dautoforge_deploy_ENABLE_TENSORRT=ON";
#endif
            return info;
        }
    } // namespace

    std::vector<SBackendBuildInfo> GetBackendBuildInfo()
    {
        return {MakeOrtInfo(), MakeTensorRtInfo()};
    }

    std::string FormatBackendBuildInfo()
    {
        std::ostringstream report;
        for (const auto& info : GetBackendBuildInfo())
        {
            report << info.name << ": compiled=" << (info.compiled_in ? "yes" : "no");
            if (info.compiled_in)
            {
                report << " runtime_available=" << (info.runtime_available ? "yes" : "no")
                       << " version=" << info.version;
                if (!info.providers.empty())
                {
                    report << " providers=";
                    for (std::size_t i = 0; i < info.providers.size(); ++i)
                    {
                        report << (i == 0U ? "" : ",") << info.providers[i];
                    }
                }
            }
            else
            {
                report << " (" << info.note << ")";
            }
            report << '\n';
        }
        return report.str();
    }
} // namespace ptafdeploy::inference
