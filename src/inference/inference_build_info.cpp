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
            info.backend = EInferenceBackend::onnxruntime;
            info.name = ToString(info.backend);
#if defined(PTAFDEPLOY_ENABLE_ORT)
            info.compiled_in = true;
            info.version = onnxruntime::CInferenceManager_ORT::GetRuntimeVersion();
            info.providers = onnxruntime::CInferenceManager_ORT::GetAvailableProviders();
            info.runtime_available = !info.providers.empty();
#else
            info.note = kOrtNotBuiltHint;
#endif
            return info;
        }

        [[nodiscard]] SBackendBuildInfo MakeTensorRtInfo()
        {
            SBackendBuildInfo info;
            info.backend = EInferenceBackend::tensorrt_engine;
            info.name = ToString(info.backend);
#if defined(PTAFDEPLOY_ENABLE_TENSORRT)
            info.compiled_in = true;
            info.version = tensorrt::CInferenceManager_TensorRT_Engine::GetRuntimeVersion();
            info.runtime_available =
                tensorrt::CInferenceManager_TensorRT_Engine::IsRuntimeAvailable();
#else
            info.note = kTensorRtNotBuiltHint;
#endif
            return info;
        }
    } // namespace

    std::vector<SBackendBuildInfo> GetBackendBuildInfo()
    {
        return {MakeOrtInfo(), MakeTensorRtInfo()};
    }

    std::string FormatBackendBuildInfo(const std::vector<SBackendBuildInfo>& report)
    {
        std::ostringstream text;
        for (const auto& info : report)
        {
            text << info.name << ": compiled=" << (info.compiled_in ? "yes" : "no");
            if (info.compiled_in)
            {
                text << " runtime_available=" << (info.runtime_available ? "yes" : "no")
                     << " version=" << info.version;
                if (!info.providers.empty())
                {
                    text << " providers=";
                    for (std::size_t i = 0; i < info.providers.size(); ++i)
                    {
                        text << (i == 0U ? "" : ",") << info.providers[i];
                    }
                }
            }
            else
            {
                text << " (" << info.note << ")";
            }
            text << '\n';
        }

        return text.str();
    }

    std::string FormatBackendBuildInfo()
    {
        return FormatBackendBuildInfo(GetBackendBuildInfo());
    }

    bool IsOrtExecutionTargetAvailable(const EExecutionTarget target)
    {
        return onnxruntime::CInferenceManager_ORT::IsExecutionTargetAvailable(target);
    }
} // namespace ptafdeploy::inference
