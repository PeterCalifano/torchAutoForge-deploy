/**
 * @file model_facade.h
 * @brief Public role-level model configuration and inference facade.
 */

#pragma once

#include <inference/inference_common.h>
#include <inference/inference_manager.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Semantic model role settings above raw tensor inference.
     *
     * Pre/postprocessing names are declarative contract labels. Actual adapter
     * helpers live outside backend code so future roles can share them.
     */
    struct SModelRoleConfig
    {
        /** @brief Semantic role used by applications and integrations. */
        EModelRole role{EModelRole::raw_tensor};
        /** @brief Declarative preprocessing pipeline name. */
        std::string preprocessing{"caller_supplied_tensors"};
        /** @brief Declarative postprocessing pipeline name. */
        std::string postprocessing{"raw_model_outputs"};

        /** @brief Construct the raw-tensor role with default pipeline labels. */
        SModelRoleConfig() = default;
        /** @brief Construct default pipelines for a role. @param model_role Recognized semantic
         * role. */
        explicit SModelRoleConfig(EModelRole model_role);
        /**
         * @brief Construct an explicit semantic model-role contract.
         * @param model_role Recognized semantic role.
         * @param preprocessing_pipeline Declarative preprocessing name.
         * @param postprocessing_pipeline Declarative postprocessing name.
         */
        SModelRoleConfig(EModelRole model_role, std::string preprocessing_pipeline,
                         std::string postprocessing_pipeline);
    };

    /**
     * @brief Loaded model contract visible to wrappers and applications.
     *
     * Contains resolved artifact path, semantic role labels, runtime choices,
     * backend diagnostic detail, and wrapper-safe tensor metadata.
     */
    struct SModelContract
    {
        /** @brief Absolute manifest path, or empty when loaded directly. */
        std::string config_path{};
        /** @brief Resolved model artifact path. */
        std::string artifact_path{};
        /** @brief Stable semantic role name. */
        std::string role{"raw_tensor"};
        /** @brief Declarative preprocessing pipeline name. */
        std::string preprocessing{"caller_supplied_tensors"};
        /** @brief Declarative postprocessing pipeline name. */
        std::string postprocessing{"raw_model_outputs"};
        /** @brief Effective runtime configuration. */
        SRuntimeConfig runtime{};
        /** @brief Backend and provider diagnostic description. */
        std::string backend_detail{};
        /** @brief Wrapper-safe input metadata in model order. */
        std::vector<STensorInfo> inputs{};
        /** @brief Wrapper-safe output metadata in model order. */
        std::vector<STensorInfo> outputs{};
    };

    /**
     * @brief Public backend-agnostic model facade.
     *
     * Use this layer from MATLAB/Python and application code when model role
     * matters. It keeps backend handles private and exposes enum-validated
     * runtime configuration plus value-type tensor I/O.
     */
    class CModelFacade
    {
      public:
        /** @brief Construct an unloaded role-level facade. */
        CModelFacade() = default;
        /**
         * @brief Construct and load a raw-tensor model.
         * @param model_path Model artifact path.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        explicit CModelFacade(const std::string& model_path);
        /**
         * @brief Construct and load a model with one semantic role.
         * @param model_path Model artifact path.
         * @param role Recognized semantic role.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        CModelFacade(const std::string& model_path, EModelRole role);
        /**
         * @brief Construct and load a model with an explicit role contract.
         * @param model_path Model artifact path.
         * @param role_config Semantic pipeline labels and role.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        CModelFacade(const std::string& model_path, const SModelRoleConfig& role_config);

        /**
         * @brief Load a raw-tensor model.
         * @param model_path Model artifact path.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        void LoadModel(const std::string& model_path);
        /**
         * @brief Load a raw-tensor model with explicit runtime settings.
         * @param model_path Model artifact path.
         * @param runtime_config Complete runtime configuration.
         * @throws std::exception When configuration validation or backend loading fails.
         */
        void LoadModelWithRuntimeConfig(const std::string& model_path,
                                        const SRuntimeConfig& runtime_config);
        /**
         * @brief Load a model with one semantic role and default runtime settings.
         * @param model_path Model artifact path.
         * @param role Recognized semantic role.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        void LoadModelWithRole(const std::string& model_path, EModelRole role);
        /**
         * @brief Load a model with a semantic role and runtime settings.
         * @param model_path Model artifact path.
         * @param role Recognized semantic role.
         * @param runtime_config Complete runtime configuration.
         * @throws std::exception When configuration validation or backend loading fails.
         */
        void LoadModelWithRoleAndRuntimeConfig(const std::string& model_path, EModelRole role,
                                               const SRuntimeConfig& runtime_config);
        /**
         * @brief Load a model with explicit semantic pipeline labels.
         * @param model_path Model artifact path.
         * @param role_config Semantic pipeline labels and role.
         * @throws std::exception When artifact validation or backend loading fails.
         */
        void LoadModelWithRoleConfig(const std::string& model_path,
                                     const SModelRoleConfig& role_config);
        /**
         * @brief Load a model with complete semantic and runtime configuration.
         * @param model_path Model artifact path.
         * @param role_config Semantic pipeline labels and role.
         * @param runtime_config Complete runtime configuration.
         * @throws std::exception When configuration validation or backend
         *         loading fails.
         * @note Transactional reload is deferred; after a failed reload the
         *       previous contract must not be assumed usable.
         */
        void LoadModelWithRoleConfigAndRuntimeConfig(const std::string& model_path,
                                                     const SModelRoleConfig& role_config,
                                                     const SRuntimeConfig& runtime_config);
        /**
         * @brief Load a versioned `.ptafmodel` manifest.
         * @param config_path Manifest path; relative artifact paths resolve
         *        against its parent directory.
         * @throws std::exception When parsing, validation, or loading fails.
         */
        void LoadModelConfig(const std::string& config_path);
        /**
         * @brief Load a manifest while replacing all manifest runtime settings.
         * @param config_path Manifest path.
         * @param runtime_config Complete replacement runtime configuration.
         * @throws std::exception When parsing, validation, or loading fails.
         */
        void LoadModelConfigWithRuntimeConfig(const std::string& config_path,
                                              const SRuntimeConfig& runtime_config);

        /** @brief Return a copy of the loaded semantic/runtime contract.
         *  @throws std::runtime_error When no model contract is loaded. */
        [[nodiscard]] SModelContract GetContract() const;
        /** @brief Return the loaded model-role name. @return Stable role string. */
        [[nodiscard]] std::string GetRole() const;
        /** @brief Return the preprocessing label. @return Declarative pipeline name. */
        [[nodiscard]] std::string GetPreprocessing() const;
        /** @brief Return the postprocessing label. @return Declarative pipeline name. */
        [[nodiscard]] std::string GetPostprocessing() const;
        /** @brief Return backend/provider diagnostic details. @return Diagnostic string. */
        [[nodiscard]] std::string GetBackendDetail() const;
        /** @brief Return the loaded model input count. @return Number of inputs. */
        [[nodiscard]] size_t GetNumInputs() const;
        /** @brief Return the loaded model output count. @return Number of outputs. */
        [[nodiscard]] size_t GetNumOutputs() const;
        /** @brief Return input metadata. @param index Input index. @return Wrapper-safe metadata.
         */
        [[nodiscard]] STensorInfo GetInputInfo(size_t index) const;
        /** @brief Return output metadata. @param index Output index. @return Wrapper-safe metadata.
         */
        [[nodiscard]] STensorInfo GetOutputInfo(size_t index) const;

        /**
         * @brief Run named or ordered float32 host tensors through the model.
         * @param inputs Wrapper-safe float32 input tensors.
         * @return Wrapper-safe float32 output tensors.
         * @throws std::exception When unloaded, validation fails, or backend
         *         execution fails.
         */
        [[nodiscard]] std::vector<SFloatTensor>
        InferFloatTensors(const std::vector<SFloatTensor>& inputs) const;
        /**
         * @brief Run a one-input/one-output wrapper-safe float32 model.
         * @param input Sole model input.
         * @return Sole model output.
         */
        [[nodiscard]] SFloatTensor InferSingleFloatTensor(const SFloatTensor& input) const;
        /**
         * @brief Run a one-input/one-output model from flat values and shape.
         * @param values Flat float32 input values.
         * @param shape Concrete input shape.
         * @return Flat float32 output values.
         */
        [[nodiscard]] std::vector<float>
        InferSingleFloatInput(const std::vector<float>& values,
                              const std::vector<int64_t>& shape) const;

        /**
         * @brief Return every role name recognized by configuration parsing.
         *
         * Recognition does not imply that a preprocessing or postprocessing
         * adapter is executable for the role.
         * @return Stable role names accepted by configuration parsing.
         */
        [[nodiscard]] static std::vector<std::string> GetRecognizedRoles();

      private:
        CInferenceManager inference_manager_{};
        SModelContract contract_{};
    };
} // namespace ptafdeploy::inference
