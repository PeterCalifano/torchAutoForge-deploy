/**
 * @file inference_manager.h
 * @brief Public backend-agnostic inference-manager facade.
 */

#pragma once

#include <auxiliary/common_ops.h>
#include <inference/inference_common.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <inference/tensorrt/tensorrt_inference_engine.h>

#include <string>
#include <variant>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Backend-dispatching inference manager.
     *
     * This class owns one concrete backend selected by artifact/backend config.
     * It exposes backend-neutral tensor descriptors/views and wrapper-safe
     * float32 helpers. Higher-level model semantics belong in `CModelFacade`.
     */
    class CInferenceManager
    {
      public:
        /** @brief Construct an unloaded manager. */
        CInferenceManager() = default;

        /**
         * @brief Construct a manager and load one model artifact.
         * @param model_path ONNX or serialized TensorRT artifact path.
         * @param options Backend selection and runtime options.
         * @throws std::exception When options, artifact selection, or backend
         *         loading fail. Backend exceptions propagate unchanged.
         */
        explicit CInferenceManager(const fs::path &model_path,
                                   const SInferenceOptions &options = {});

        /**
         * @brief Construct a manager and load one model with default options.
         * @param model_path ONNX or serialized TensorRT artifact path.
         * @throws std::exception When artifact selection or loading fails.
         */
        explicit CInferenceManager(const std::string &model_path);

        /**
         * @brief Select a backend from the artifact/options and load the model.
         * @param model_path ONNX or serialized TensorRT artifact path.
         * @param options Backend selection and runtime options.
         * @throws std::exception When validation or backend loading fails.
         * @note The selected backend replaces the previous backend before its
         *       load completes; transactional reload is a deferred enhancement.
         */
        void LoadModel(const fs::path &model_path,
                       const SInferenceOptions &options = {});

        /**
         * @brief Select a backend from the artifact and load with default options.
         * @param model_path ONNX or serialized TensorRT artifact path.
         * @throws std::exception When validation or backend loading fails.
         * @note The selected backend replaces the previous backend before its
         *       load completes; transactional reload is a deferred enhancement.
         */
        void LoadModel(const std::string &model_path);

        /**
         * @brief Load a model from wrapper-safe runtime configuration.
         * @param model_path Model artifact path.
         * @param runtime_config Enum-backed wrapper-safe runtime settings.
         * @throws std::exception When configuration validation or loading fails.
         */
        void LoadModelWithRuntimeConfig(const std::string &model_path,
                                        const SRuntimeConfig &runtime_config);

        /**
         * @brief Load a model with explicit ORT thread-count settings.
         * @param model_path Model artifact path.
         * @param intra_op_num_threads Intra-operation thread count; zero uses
         *        the backend default.
         * @param inter_op_num_threads Inter-operation thread count; zero uses
         *        the backend default.
         * @throws std::exception When counts are negative or loading fails.
         */
        void LoadModelWithThreadCounts(const std::string &model_path,
                                       int intra_op_num_threads,
                                       int inter_op_num_threads);

        /** @brief Return complete metadata for the loaded model.
         *  @return Backend-neutral model metadata owned by the manager.
         *  @throws std::runtime_error When no backend has been loaded. */
        [[nodiscard]] const SModelMetadata &GetModelMetadata() const;
        /** @brief Return the loaded backend/provider diagnostic string.
         *  @return Backend detail suitable for diagnostics, not capability proof.
         *  @throws std::runtime_error When no backend has been loaded. */
        [[nodiscard]] std::string GetBackendDetail() const;
        /** @brief Return the loaded model input count.
         *  @return Number of model inputs.
         *  @throws std::runtime_error When no backend has been loaded. */
        [[nodiscard]] size_t GetNumInputs() const;
        /** @brief Return the loaded model output count.
         *  @return Number of model outputs.
         *  @throws std::runtime_error When no backend has been loaded. */
        [[nodiscard]] size_t GetNumOutputs() const;
        /** @brief Return wrapper-safe metadata for one input.
         *  @param index Input index.
         *  @return Value-only input metadata.
         *  @throws std::out_of_range When `index` is invalid. */
        [[nodiscard]] STensorInfo GetInputInfo(size_t index) const;
        /** @brief Return wrapper-safe metadata for one output.
         *  @param index Output index.
         *  @return Value-only output metadata.
         *  @throws std::out_of_range When `index` is invalid. */
        [[nodiscard]] STensorInfo GetOutputInfo(size_t index) const;
        /** @brief Return wrapper-safe metadata for every input. @return Input metadata in model order. */
        [[nodiscard]] std::vector<STensorInfo> GetInputInfos() const;
        /** @brief Return wrapper-safe metadata for every output. @return Output metadata in model order. */
        [[nodiscard]] std::vector<STensorInfo> GetOutputInfos() const;

        /**
         * @brief Run dense tensor inference without copying host inputs.
         * @param inputs Host tensor views matching the loaded model contract.
         * @return Owned output buffers in model output order.
         * @throws std::exception When unloaded, validation fails, or execution
         *         fails. Backend exceptions propagate unchanged.
         */
        [[nodiscard]] std::vector<STensorBuffer> Infer(const std::vector<STensorView> &inputs) const;

        /**
         * @brief Run wrapper-friendly float32 inference.
         * @param inputs Owned float32 tensors matching the model contract.
         * @return Float32 outputs copied into wrapper-safe value types.
         * @throws std::runtime_error When an output is not float32.
         */
        [[nodiscard]] std::vector<SFloatTensor> InferFloatTensors(const std::vector<SFloatTensor> &inputs) const;
        /** @brief Run a one-input/one-output float32 model.
         *  @param input Sole wrapper-safe input tensor.
         *  @return Sole wrapper-safe output tensor.
         *  @throws std::runtime_error When the model does not produce one output. */
        [[nodiscard]] SFloatTensor InferSingleFloatTensor(const SFloatTensor &input) const;
        /** @brief Run a one-input/one-output float32 model from values and shape.
         *  @param values Flat float32 input values.
         *  @param shape Concrete input shape.
         *  @return Values from the sole output tensor. */
        [[nodiscard]] std::vector<float> InferSingleFloatInput(const std::vector<float> &values,
                                                               const std::vector<int64_t> &shape) const;

      private:
        using TBackendVariant = std::variant<std::monostate,
                                             ptafdeploy::inference::onnxruntime::CInferenceManager_ORT,
                                             ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine>;

        [[nodiscard]] static EModelArtifact DetectArtifactType(const fs::path &model_path);

      private:
      // TODO (PC) evaluate direct templating in place of std::variant usage.
        TBackendVariant backend_{};
    };
}
