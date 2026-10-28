/**
 * @file inference_build_info.h
 * @brief Wrapper-safe report of which inference backends a build provides.
 */

#pragma once

#include <string>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Availability of one inference backend in the current build.
     *
     * `compiled_in` reflects the build options; `runtime_available` additionally
     * requires the runtime environment (ORT library load, visible CUDA device).
     */
    struct SBackendBuildInfo
    {
        /** @brief Backend name as accepted by `ParseInferenceBackendName`. */
        std::string name{};
        /** @brief True when the backend was enabled at configure time. */
        bool compiled_in{false};
        /** @brief True when the backend is compiled in and usable on this host. */
        bool runtime_available{false};
        /** @brief Linked runtime version; empty when not compiled in. */
        std::string version{};
        /** @brief ORT execution providers; empty for other backends. */
        std::vector<std::string> providers{};
        /** @brief Reconfigure hint when not compiled in, otherwise empty. */
        std::string note{};
    };

    /**
     * @brief Report every known concrete backend (`onnxruntime`, `tensorrt_engine`).
     * @return One entry per backend, in a stable order.
     */
    [[nodiscard]] std::vector<SBackendBuildInfo> GetBackendBuildInfo();

    /**
     * @brief Render `GetBackendBuildInfo()` as human-readable text, one backend per line.
     */
    [[nodiscard]] std::string FormatBackendBuildInfo();
} // namespace ptafdeploy::inference
