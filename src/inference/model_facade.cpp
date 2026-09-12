/**
 * @file model_facade.cpp
 * @brief Model-role configuration, contract validation, and inference dispatch.
 */

#include "model_facade.h"

#include <inference/inference_config_parsing.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <utils/logging/CLogger.h>

namespace ptafdeploy::inference
{
    namespace
    {
        [[nodiscard]] ptafdeploy::logging::CLogger& GetLogger()
        {
            static ptafdeploy::logging::CLogger logger(
                "model_facade", ptafdeploy::logging::ELogLevel::Warning,
                ptafdeploy::logging::ELogColorMode::Disabled, std::clog, std::clog);
            static const bool environment_applied = logger.setLevelFromEnvironment();
            static_cast<void>(environment_applied);
            return logger;
        }

        [[nodiscard]] const std::vector<std::string>& RecognizedRoles()
        {
            static const std::vector<std::string> roles = []
            {
                constexpr std::array<EModelRole, 7> kSupportedModelRoles{
                    EModelRole::raw_tensor,
                    EModelRole::centroiding,
                    EModelRole::object_detection,
                    EModelRole::feature_matching,
                    EModelRole::tracking,
                    EModelRole::optical_flow,
                    EModelRole::custom};

                std::vector<std::string> values;
                values.reserve(kSupportedModelRoles.size());
                for (const EModelRole role : kSupportedModelRoles)
                {
                    values.push_back(ToString(role));
                }
                return values;
            }();
            return roles;
        }

        [[nodiscard]] std::string DefaultIfEmpty(const std::string& value,
                                                 const char* default_value)
        {
            return value.empty() ? std::string{default_value} : value;
        }

        [[nodiscard]] SModelRoleConfig NormalizeConfig(const SModelRoleConfig& role_config)
        {
            SModelRoleConfig normalized_config;
            normalized_config.role = role_config.role;
            normalized_config.preprocessing =
                DefaultIfEmpty(role_config.preprocessing, "caller_supplied_tensors");
            normalized_config.postprocessing =
                DefaultIfEmpty(role_config.postprocessing, "raw_model_outputs");
            return normalized_config;
        }

        [[nodiscard]] SModelContract MakeContract(const std::string& model_path,
                                                  const std::string& config_path,
                                                  const SModelRoleConfig& role_config,
                                                  const SRuntimeConfig& runtime_config,
                                                  const CInferenceManager& inference_manager)
        {
            SModelContract contract;
            contract.config_path = config_path;
            contract.artifact_path = model_path;
            contract.role = ToString(role_config.role);
            contract.preprocessing = role_config.preprocessing;
            contract.postprocessing = role_config.postprocessing;
            contract.runtime = runtime_config;
            contract.backend_detail = inference_manager.GetBackendDetail();
            contract.inputs = inference_manager.GetInputInfos();
            contract.outputs = inference_manager.GetOutputInfos();
            return contract;
        }

        [[nodiscard]] std::filesystem::path
        ResolveRelativePath(const std::filesystem::path& base_file, const std::string& path_value)
        {
            std::filesystem::path path{path_value};
            if (path.is_relative())
            {
                path = base_file.parent_path() / path;
            }

            return std::filesystem::weakly_canonical(path);
        }

        struct SParsedModelConfig
        {
            std::filesystem::path artifact_path{};
            SModelRoleConfig role_config{};
            SRuntimeConfig runtime_config{};
        };

        [[nodiscard]] SParsedModelConfig ReadModelConfigFile(const std::string& config_path)
        {
            const std::filesystem::path resolved_config_path =
                std::filesystem::absolute(config_path).lexically_normal();
            std::ifstream stream(resolved_config_path);
            if (!stream)
            {
                throw std::invalid_argument("Could not open model config file: " + config_path);
            }

            std::string artifact_path;
            SModelRoleConfig role_config;
            SRuntimeConfig runtime_config;
            bool has_schema_version = false;
            std::string line;
            size_t line_number = 0;
            while (std::getline(stream, line))
            {
                ++line_number;
                const size_t comment_pos = line.find('#');
                if (comment_pos != std::string::npos)
                {
                    line.erase(comment_pos);
                }

                line = TrimConfigToken(line);
                if (line.empty())
                {
                    continue;
                }

                const size_t separator_pos = line.find('=');
                if (separator_pos == std::string::npos)
                {
                    throw std::invalid_argument("Invalid model config line " +
                                                std::to_string(line_number) + " in " +
                                                resolved_config_path.string());
                }

                const std::string key = TrimConfigToken(line.substr(0, separator_pos));
                const std::string value = TrimConfigToken(line.substr(separator_pos + 1));
                if (key == "schema_version")
                {
                    has_schema_version = true;
                    if (ParseConfigInt(value, key) != 1)
                    {
                        throw std::invalid_argument("Unsupported model config schema_version: " +
                                                    value);
                    }
                }
                else if (key == "artifact_path")
                {
                    artifact_path = value;
                }
                else if (key == "role")
                {
                    role_config.role = ParseModelRoleName(value);
                }
                else if (key == "preprocessing")
                {
                    role_config.preprocessing = value;
                }
                else if (key == "postprocessing")
                {
                    role_config.postprocessing = value;
                }
                else if (key == "backend")
                {
                    runtime_config.backend = ParseInferenceBackendName(value);
                }
                else if (key == "artifact")
                {
                    runtime_config.artifact = ParseModelArtifactName(value);
                }
                else if (key == "allow_fallback")
                {
                    runtime_config.allow_fallback = ParseConfigBool(value, key);
                }
                else if (key == "execution_target_priority")
                {
                    runtime_config.execution_target_priority = ParseExecutionTargetPriority(value);
                }
                else if (key == "device_id")
                {
                    runtime_config.SetDeviceId(ParseConfigInt(value, key));
                }
                else if (key == "intra_op_num_threads")
                {
                    runtime_config.intra_op_num_threads = ParseConfigInt(value, key);
                }
                else if (key == "inter_op_num_threads")
                {
                    runtime_config.inter_op_num_threads = ParseConfigInt(value, key);
                }
                else if (key == "enable_profiling")
                {
                    runtime_config.enable_profiling = ParseConfigBool(value, key);
                }
                else if (key == "log_id")
                {
                    runtime_config.log_id = value;
                }
                else if (key == "tensorrt_optimization_profile_index")
                {
                    runtime_config.SetTensorRtOptimizationProfileIndex(ParseConfigInt(value, key));
                }
                else
                {
                    throw std::invalid_argument("Unknown model config key: " + key);
                }
            }

            if (!has_schema_version)
            {
                throw std::invalid_argument("Model config is missing required key: schema_version");
            }
            if (artifact_path.empty())
            {
                throw std::invalid_argument("Model config is missing required key: artifact_path");
            }

            return {ResolveRelativePath(resolved_config_path, artifact_path),
                    NormalizeConfig(role_config), runtime_config};
        }

        void EnsureModelLoaded(const SModelContract& contract)
        {
            if (contract.artifact_path.empty())
            {
                throw std::runtime_error("No model has been loaded.");
            }
        }
    } // namespace

    SModelRoleConfig::SModelRoleConfig(const EModelRole model_role) : role(model_role) {}

    SModelRoleConfig::SModelRoleConfig(const EModelRole model_role,
                                       std::string preprocessing_pipeline,
                                       std::string postprocessing_pipeline)
        : role(model_role), preprocessing(std::move(preprocessing_pipeline)),
          postprocessing(std::move(postprocessing_pipeline))
    {
    }

    CModelFacade::CModelFacade(const std::string& model_path)
    {
        LoadModel(model_path);
    }

    CModelFacade::CModelFacade(const std::string& model_path, const EModelRole role)
    {
        LoadModelWithRole(model_path, role);
    }

    CModelFacade::CModelFacade(const std::string& model_path, const SModelRoleConfig& role_config)
    {
        LoadModelWithRoleConfig(model_path, role_config);
    }

    void CModelFacade::LoadModel(const std::string& model_path)
    {
        LoadModelWithRoleConfig(model_path, SModelRoleConfig{});
    }

    void CModelFacade::LoadModelWithRuntimeConfig(const std::string& model_path,
                                                  const SRuntimeConfig& runtime_config)
    {
        LoadModelWithRoleConfigAndRuntimeConfig(model_path, SModelRoleConfig{}, runtime_config);
    }

    void CModelFacade::LoadModelWithRole(const std::string& model_path, const EModelRole role)
    {
        LoadModelWithRoleConfig(model_path, SModelRoleConfig{role});
    }

    void CModelFacade::LoadModelWithRoleAndRuntimeConfig(const std::string& model_path,
                                                         const EModelRole role,
                                                         const SRuntimeConfig& runtime_config)
    {
        LoadModelWithRoleConfigAndRuntimeConfig(model_path, SModelRoleConfig{role}, runtime_config);
    }

    void CModelFacade::LoadModelWithRoleConfig(const std::string& model_path,
                                               const SModelRoleConfig& role_config)
    {
        LoadModelWithRoleConfigAndRuntimeConfig(model_path, role_config, SRuntimeConfig{});
    }

    void CModelFacade::LoadModelWithRoleConfigAndRuntimeConfig(const std::string& model_path,
                                                               const SModelRoleConfig& role_config,
                                                               const SRuntimeConfig& runtime_config)
    {
        const SModelRoleConfig normalized_config = NormalizeConfig(role_config);
        inference_manager_.LoadModelWithRuntimeConfig(model_path, runtime_config);
        contract_ =
            MakeContract(model_path, {}, normalized_config, runtime_config, inference_manager_);
        GetLogger().info("Loaded role=", contract_.role, ", artifact=", contract_.artifact_path);
        GetLogger().debug("Preprocessing=", contract_.preprocessing,
                          ", postprocessing=", contract_.postprocessing);
    }

    void CModelFacade::LoadModelConfig(const std::string& config_path)
    {
        GetLogger().debug("Reading model configuration: ", config_path);
        const SParsedModelConfig parsed_config = ReadModelConfigFile(config_path);
        inference_manager_.LoadModelWithRuntimeConfig(parsed_config.artifact_path.string(),
                                                      parsed_config.runtime_config);
        contract_ = MakeContract(parsed_config.artifact_path.string(),
                                 std::filesystem::absolute(config_path).lexically_normal().string(),
                                 parsed_config.role_config, parsed_config.runtime_config,
                                 inference_manager_);
        GetLogger().info("Loaded config role=", contract_.role,
                         ", artifact=", contract_.artifact_path);
    }

    void CModelFacade::LoadModelConfigWithRuntimeConfig(const std::string& config_path,
                                                        const SRuntimeConfig& runtime_config)
    {
        GetLogger().debug("Reading model configuration with runtime override: ", config_path);
        const SParsedModelConfig parsed_config = ReadModelConfigFile(config_path);
        inference_manager_.LoadModelWithRuntimeConfig(parsed_config.artifact_path.string(),
                                                      runtime_config);
        contract_ = MakeContract(parsed_config.artifact_path.string(),
                                 std::filesystem::absolute(config_path).lexically_normal().string(),
                                 parsed_config.role_config, runtime_config, inference_manager_);
        GetLogger().info("Loaded config role=", contract_.role,
                         ", artifact=", contract_.artifact_path, " with runtime override");
    }

    SModelContract CModelFacade::GetContract() const
    {
        EnsureModelLoaded(contract_);
        return contract_;
    }

    std::string CModelFacade::GetRole() const
    {
        EnsureModelLoaded(contract_);
        return contract_.role;
    }

    std::string CModelFacade::GetPreprocessing() const
    {
        EnsureModelLoaded(contract_);
        return contract_.preprocessing;
    }

    std::string CModelFacade::GetPostprocessing() const
    {
        EnsureModelLoaded(contract_);
        return contract_.postprocessing;
    }

    std::string CModelFacade::GetBackendDetail() const
    {
        return inference_manager_.GetBackendDetail();
    }

    size_t CModelFacade::GetNumInputs() const
    {
        return inference_manager_.GetNumInputs();
    }

    size_t CModelFacade::GetNumOutputs() const
    {
        return inference_manager_.GetNumOutputs();
    }

    STensorInfo CModelFacade::GetInputInfo(const size_t index) const
    {
        return inference_manager_.GetInputInfo(index);
    }

    STensorInfo CModelFacade::GetOutputInfo(const size_t index) const
    {
        return inference_manager_.GetOutputInfo(index);
    }

    std::vector<SFloatTensor>
    CModelFacade::InferFloatTensors(const std::vector<SFloatTensor>& inputs) const
    {
        EnsureModelLoaded(contract_);
        GetLogger().trace("Running role=", contract_.role, " with ", inputs.size(),
                          " input tensor(s).");
        return inference_manager_.InferFloatTensors(inputs);
    }

    SFloatTensor CModelFacade::InferSingleFloatTensor(const SFloatTensor& input) const
    {
        EnsureModelLoaded(contract_);
        return inference_manager_.InferSingleFloatTensor(input);
    }

    std::vector<float> CModelFacade::InferSingleFloatInput(const std::vector<float>& values,
                                                           const std::vector<int64_t>& shape) const
    {
        EnsureModelLoaded(contract_);
        return inference_manager_.InferSingleFloatInput(values, shape);
    }

    std::vector<std::string> CModelFacade::GetRecognizedRoles()
    {
        return RecognizedRoles();
    }
} // namespace ptafdeploy::inference
