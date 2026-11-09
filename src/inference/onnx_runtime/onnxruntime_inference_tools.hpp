/**
 * @file onnxruntime_inference_tools.hpp
 * @brief ONNX Runtime backend for the generic inference contracts.
 */

#pragma once

#include <filesystem>
#include <inference/inference_common.h>

#include <memory>
#include <string>
#include <vector>

namespace ptafdeploy::inference::onnxruntime
{
    namespace fs = std::filesystem;
    struct SOrtState;

    /**
     * @brief ONNX Runtime backend implementation.
     *
     * Owns ORT environment/session state, extracts backend-neutral metadata,
     * validates host tensor views, and returns owned dense output buffers.
     * Provider selection is driven by `SInferenceOptions::execution_target_priority`.
     * When built with `autoforge_deploy_ENABLE_ORT=OFF` the class stays declared
     * but loading and inference throw an actionable "not built" error. ORT handles
     * stay behind an opaque state so the class layout and this installed header
     * do not depend on that option or on the ORT headers.
     */
    class CInferenceManager_ORT
    {
      public:
        /** @brief Construct an unloaded ONNX Runtime backend. */
        CInferenceManager_ORT();
        /**
         * @brief Construct the backend and load an ONNX model.
         * @param model_path Existing `.onnx` artifact.
         * @param options Threading, provider-priority, device, and fallback settings.
         * @throws std::exception When the file, options, provider selection, or
         *         ORT session creation is invalid.
         */
        explicit CInferenceManager_ORT(
            const fs::path& model_path,
            const ptafdeploy::inference::SInferenceOptions& options = {});
        /** @brief Release the ORT session and environment. */
        ~CInferenceManager_ORT();
        /** @brief Take over the source's loaded session, leaving the source unloaded. */
        CInferenceManager_ORT(CInferenceManager_ORT&& other) noexcept;
        /** @brief Replace this backend with the source's loaded session. */
        CInferenceManager_ORT& operator=(CInferenceManager_ORT&& other) noexcept;

        /**
         * @brief Load an ONNX model and extract its tensor metadata.
         * @param model_path Existing `.onnx` artifact.
         * @param options Threading, provider-priority, device, and fallback settings.
         * @throws std::exception When validation or ORT session creation fails.
         * @note Failure preserves the previous session, options, and metadata.
         */
        void LoadModel(const fs::path& model_path,
                       const ptafdeploy::inference::SInferenceOptions& options = {});

        /** @brief Return metadata extracted during the latest successful load. */
        [[nodiscard]] const ptafdeploy::inference::SModelMetadata&
        GetModelMetadata() const noexcept;

        /**
         * @brief Run inference with caller-owned host input memory.
         *
         * Input buffers are not copied before ORT execution. Output tensors are
         * copied into `STensorBuffer` because ORT owns returned tensor storage.
         * @param inputs Host tensor views, named or ordered to match the model.
         * @return Owned dense output tensors in model output order.
         * @throws std::exception When no session is loaded, inputs violate the
         *         model contract, or ORT execution fails.
         */
        [[nodiscard]] std::vector<ptafdeploy::inference::STensorBuffer>
        Infer(const std::vector<ptafdeploy::inference::STensorView>& inputs) const;

        /** @brief Return provider names available in the linked ORT build. */
        [[nodiscard]] static std::vector<std::string> GetAvailableProviders();

        /**
         * @brief Check whether the linked ORT build exposes a requested target.
         * @param target Backend-neutral execution target.
         * @return True when an equivalent ORT provider is available.
         */
        [[nodiscard]] static bool
        IsExecutionTargetAvailable(ptafdeploy::inference::EExecutionTarget target);

        /** @brief Return the linked ORT version string, or empty when ORT is not built. */
        [[nodiscard]] static std::string GetRuntimeVersion();

      protected:
        fs::path model_path_{};
        std::unique_ptr<SOrtState> state_{};
        ptafdeploy::inference::SInferenceOptions options_{};
        ptafdeploy::inference::SModelMetadata model_metadata_{};
    };
} // namespace ptafdeploy::inference::onnxruntime
