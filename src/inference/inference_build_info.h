/**
 * @file inference_build_info.h
 * @brief Wrapper-safe report of which inference backends a build provides.
 */

#pragma once

#include <inference/inference_common.h>

#include <string>
#include <string_view>
#include <vector>

namespace ptafdeploy::inference
{
    /** @brief Action that makes the ONNX Runtime backend available in a build without it. */
    inline constexpr std::string_view kOrtNotBuiltHint =
        "reconfigure with -Dautoforge_deploy_ENABLE_ORT=ON";

    /** @brief Action that makes the TensorRT engine backend available in a build without it. */
    inline constexpr std::string_view kTensorRtNotBuiltHint =
        "reconfigure with -Dautoforge_deploy_ENABLE_TENSORRT=ON";

    /**
     * @brief Availability of one inference backend in the current build.
     *
     * `compiled_in` reflects the build options. `runtime_available` is backend
     * specific: ONNX Runtime reports true when it lists at least one execution
     * provider, which always holds once it is linked; TensorRT additionally
     * requires a visible CUDA device.
     */
    struct SBackendBuildInfo
    {
        /** @brief Concrete backend this entry describes. */
        EInferenceBackend backend{EInferenceBackend::auto_backend};
        /** @brief Backend name as accepted by `ParseInferenceBackendName`. */
        std::string name{};
        /** @brief True when the backend was enabled at configure time. */
        bool compiled_in{false};
        /** @brief True when compiled in and usable here; see the struct contract. */
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
     * @throws std::exception When the ONNX Runtime provider query fails.
     */
    [[nodiscard]] std::vector<SBackendBuildInfo> GetBackendBuildInfo();

    /**
     * @brief Render a backend report as human-readable text.
     * @param report Entries to render, in order.
     * @return Newline-terminated text with one line per entry.
     */
    [[nodiscard]] std::string FormatBackendBuildInfo(const std::vector<SBackendBuildInfo> &report);

    /**
     * @brief Render the report of the current build as human-readable text.
     * @return Newline-terminated text with one line per backend.
     * @throws std::exception When the ONNX Runtime provider query fails.
     */
    [[nodiscard]] std::string FormatBackendBuildInfo();

    /**
     * @brief Check whether ONNX Runtime exposes an execution target on this host.
     * @param target Backend-neutral execution target.
     * @return True when ORT is compiled in and an equivalent provider is available.
     * @throws std::exception When the ONNX Runtime provider query fails.
     */
    [[nodiscard]] bool IsOrtExecutionTargetAvailable(EExecutionTarget target);
} // namespace ptafdeploy::inference
