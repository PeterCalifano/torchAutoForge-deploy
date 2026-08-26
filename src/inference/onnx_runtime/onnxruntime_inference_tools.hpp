/**
 * @file onnxruntime_inference_tools.hpp
 * @brief ONNX Runtime backend for the generic inference contracts.
 */

#pragma once

#include <auxiliary/common_ops.h>
#include <inference/inference_common.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace ptafdeploy::inference::onnxruntime
{
    /**
     * @brief ONNX Runtime backend implementation.
     *
     * Owns ORT environment/session state, extracts backend-neutral metadata,
     * validates host tensor views, and returns owned dense output buffers.
     * Provider selection is driven by `SInferenceOptions::execution_target_priority`.
     */
    class CInferenceManager_ORT
    {
      public:
        /** @brief Construct an unloaded ONNX Runtime backend. */
        CInferenceManager_ORT() = default;
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

        /**
         * @brief Load an ONNX model and extract its tensor metadata.
         * @param model_path Existing `.onnx` artifact.
         * @param options Threading, provider-priority, device, and fallback settings.
         * @throws std::exception When validation or ORT session creation fails.
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

      protected:
        fs::path model_path_{};
        Ort::Env exec_env_{};
        Ort::SessionOptions session_options_{};
        std::unique_ptr<Ort::Session> session_ptr_{nullptr};
        Ort::AllocatorWithDefaultOptions allocator_{};
        ptafdeploy::inference::SInferenceOptions options_{};
        ptafdeploy::inference::SModelMetadata model_metadata_{};
    };
} // namespace ptafdeploy::inference::onnxruntime
