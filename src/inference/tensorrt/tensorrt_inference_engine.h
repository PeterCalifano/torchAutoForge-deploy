/**
 * @file tensorrt_inference_engine.h
 * @brief Optional serialized TensorRT engine runtime interface.
 */

#pragma once

#include <auxiliary/common_ops.h>
#include <inference/inference_common.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ptafdeploy::inference::tensorrt
{
    struct STensorRtState;

    /**
     * @brief Standalone TensorRT serialized-engine backend.
     *
     * Supports `.engine`/`.plan` artifacts when `ENABLE_TENSORRT=ON`. The
     * backend intentionally stays separate from ORT TensorRT Execution Provider
     * support; ORT provider selection remains inside the ORT backend.
     *
     * Runtime options cover CUDA device id and TensorRT optimization-profile
     * selection. DLA, dynamic-output allocators, and engine-build precision
     * policy are not part of the qualified contract. Current implementation
     * uses host-staged I/O with reusable CUDA buffers for concrete shapes.
     */
    class CInferenceManager_TensorRT_Engine
    {
      public:
        /** @brief Construct an unloaded TensorRT backend. */
        CInferenceManager_TensorRT_Engine();
        /**
         * @brief Construct the backend and load a serialized TensorRT engine.
         * @param model_path Existing serialized TensorRT engine path.
         * @param options Device and optimization-profile selections.
         * @throws std::exception When TensorRT support is unavailable or engine
         *         validation/deserialization fails.
         */
        explicit CInferenceManager_TensorRT_Engine(
            const fs::path& model_path,
            const ptafdeploy::inference::SInferenceOptions& options = {});
        /** @brief Release the TensorRT execution state and CUDA buffers. */
        ~CInferenceManager_TensorRT_Engine();

        /**
         * @brief Load a serialized `.engine` or `.plan` artifact.
         * @param model_path Existing serialized TensorRT engine path.
         * @param options Device and optimization-profile selections.
         * @throws std::exception When TensorRT is not built or loading fails.
         */
        void LoadModel(const fs::path& model_path,
                       const ptafdeploy::inference::SInferenceOptions& options = {});

        /** @brief Return metadata extracted during the latest successful load. */
        [[nodiscard]] const ptafdeploy::inference::SModelMetadata&
        GetModelMetadata() const noexcept;

        /**
         * @brief Run inference through the TensorRT execution context.
         * @param inputs Dense host tensor views matching the selected profile.
         * @return Host-owned dense output tensors.
         * @throws std::exception When TensorRT is unavailable, no engine is
         *         loaded, validation fails, or execution fails.
         */
        [[nodiscard]] std::vector<ptafdeploy::inference::STensorBuffer>
        Infer(const std::vector<ptafdeploy::inference::STensorView>& inputs) const;

      private:
        [[noreturn]] static void ThrowNotImplemented();

      private:
        fs::path model_path_{};
        ptafdeploy::inference::SInferenceOptions options_{};
        ptafdeploy::inference::SModelMetadata metadata_{};
        std::unique_ptr<STensorRtState> state_{};
        mutable std::mutex inference_mutex_{};
    };
} // namespace ptafdeploy::inference::tensorrt
