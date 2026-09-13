/**
 * @file model_facade.cpp
 * @brief Model-role configuration, contract validation, and inference dispatch.
 */

#include "model_facade.h"

#include <inference/inference_config_parsing.h>

#include <ptafmodel_schema.h>
#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string_view>
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

        [[nodiscard]] SParsedModelConfig ParseModelConfigJson(const std::string& json,
                                                              const std::string& config_path)
        {
            // A raw NUL can terminate the JSON reader before it examines trailing bytes.
            const auto nul_offset = json.find('\0');
            if (nul_offset != std::string::npos)
            {
                throw std::invalid_argument(config_path + ": raw NUL at byte " +
                                            std::to_string(nul_offset));
            }

            rapidjson::Document document;
            document.Parse<rapidjson::kParseValidateEncodingFlag>(json.data(), json.size());
            if (document.HasParseError())
            {
                throw std::invalid_argument(config_path + ": JSON error at byte " +
                                            std::to_string(document.GetErrorOffset()) + ": " +
                                            rapidjson::GetParseError_En(document.GetParseError()));
            }

            // Duplicate properties must not acquire parser-dependent last-value semantics.
            if (document.IsObject())
            {
                std::set<std::string_view> names;
                for (const auto& member : document.GetObject())
                {
                    const std::string_view name(member.name.GetString(),
                                                member.name.GetStringLength());
                    if (!names.insert(name).second)
                    {
                        throw std::invalid_argument(config_path + ": duplicate property /" +
                                                    std::string(name));
                    }
                    if (member.value.IsString() &&
                        std::string_view(member.value.GetString(), member.value.GetStringLength())
                                .find('\0') != std::string::npos)
                    {
                        throw std::invalid_argument(config_path + ": NUL in property /" +
                                                    std::string(name));
                    }
                }
            }

            // Compile once; each call owns its validator and document for concurrent use.
            static const rapidjson::SchemaDocument schema(
                []
                {
                    rapidjson::Document schema_document;
                    schema_document.Parse(kPtafModelSchema);
                    return schema_document;
                }());
            rapidjson::SchemaValidator validator(schema);
            if (!document.Accept(validator))
            {
                rapidjson::StringBuffer property;
                rapidjson::StringBuffer rule;
                validator.GetInvalidDocumentPointer().Stringify(property);
                validator.GetInvalidSchemaPointer().Stringify(rule);
                rapidjson::StringBuffer details;
                rapidjson::Writer<rapidjson::StringBuffer> error_writer(details);
                validator.GetError().Accept(error_writer);
                throw std::invalid_argument(config_path + ": invalid property " +
                                            property.GetString() + " (schema " + rule.GetString() +
                                            ", " + validator.GetInvalidSchemaKeyword() +
                                            "): " + details.GetString());
            }

            // Schema validation establishes the types and integer bounds used below.
            SModelRoleConfig role_config;
            SRuntimeConfig runtime;
            role_config.role = ParseModelRoleName(document["task"].GetString());
            if (document.HasMember("preprocessing"))
            {
                role_config.preprocessing = document["preprocessing"].GetString();
            }
            if (document.HasMember("postprocessing"))
            {
                role_config.postprocessing = document["postprocessing"].GetString();
            }
            if (document.HasMember("backend"))
            {
                runtime.backend = ParseInferenceBackendName(document["backend"].GetString());
            }
            if (document.HasMember("artifact"))
            {
                runtime.artifact = ParseModelArtifactName(document["artifact"].GetString());
            }
            if (document.HasMember("allow_fallback"))
            {
                runtime.allow_fallback = document["allow_fallback"].GetBool();
            }
            if (document.HasMember("enable_profiling"))
            {
                runtime.enable_profiling = document["enable_profiling"].GetBool();
            }
            if (document.HasMember("log_id"))
            {
                runtime.log_id = document["log_id"].GetString();
            }
            if (document.HasMember("device_id"))
            {
                runtime.SetDeviceId(document["device_id"].GetInt());
            }
            if (document.HasMember("intra_op_num_threads"))
            {
                runtime.intra_op_num_threads = document["intra_op_num_threads"].GetInt();
            }
            if (document.HasMember("inter_op_num_threads"))
            {
                runtime.inter_op_num_threads = document["inter_op_num_threads"].GetInt();
            }
            if (document.HasMember("tensorrt_optimization_profile_index"))
                runtime.SetTensorRtOptimizationProfileIndex(
                    document["tensorrt_optimization_profile_index"].GetInt());
            if (document.HasMember("execution_target_priority"))
            {
                for (const auto& target : document["execution_target_priority"].GetArray())
                    runtime.execution_target_priority.push_back(
                        ParseExecutionTargetName(target.GetString()));
            }

            const auto artifact_path = ResolveRelativePath(std::filesystem::absolute(config_path),
                                                           document["artifact_path"].GetString());
            const auto extension = artifact_path.extension().string();
            EModelArtifact detected_artifact = EModelArtifact::auto_artifact;
            if (extension == ".onnx")
            {
                detected_artifact = EModelArtifact::onnx;
            }
            else if (extension == ".engine" || extension == ".plan")
            {
                detected_artifact = EModelArtifact::tensorrt_engine;
            }

            if (detected_artifact != EModelArtifact::auto_artifact &&
                runtime.artifact != EModelArtifact::auto_artifact &&
                runtime.artifact != detected_artifact)
            {
                throw std::invalid_argument(config_path +
                                            ": /artifact does not match /artifact_path");
            }
            const EModelArtifact selected_artifact =
                detected_artifact == EModelArtifact::auto_artifact ? runtime.artifact
                                                                 : detected_artifact;
            if (selected_artifact != EModelArtifact::auto_artifact)
            {
                ValidateBackendArtifactCompatibility(runtime.backend, selected_artifact);
            }

            return {artifact_path, NormalizeConfig(role_config), runtime};
        }

        [[nodiscard]] SParsedModelConfig ReadModelConfigFile(const std::string& config_path)
        {
            std::ifstream stream(config_path);
            if (!stream)
                throw std::invalid_argument("Could not open model config file: " + config_path);
            const std::string json{std::istreambuf_iterator<char>(stream), {}};
            if (stream.bad())
                throw std::runtime_error("Could not read model config file: " + config_path);
            return ParseModelConfigJson(json, config_path);
        }

        void EnsureModelLoaded(const SModelContract& contract)
        {
            if (contract.artifact_path.empty())
            {
                throw std::runtime_error("No model has been loaded.");
            }
        }
    } // namespace

    SRuntimeConfig ReadPtafModelRuntimeConfig(const std::string& config_path)
    {
        return ReadModelConfigFile(config_path).runtime_config;
    }

    std::string GetPtafModelSchemaJson()
    {
        return kPtafModelSchema;
    }

    void ValidatePtafModelJson(const std::string& json, const std::string& config_path)
    {
        static_cast<void>(ParseModelConfigJson(json, config_path));
    }

    void ValidatePtafModelConfig(const std::string& config_path, const bool check_artifact)
    {
        const auto config = ReadModelConfigFile(config_path);
        if (check_artifact)
        {
            const auto extension = config.artifact_path.extension().string();
            if (extension != ".onnx" && extension != ".engine" && extension != ".plan")
                throw std::invalid_argument("Unsupported artifact extension: " + extension);
            if (!std::filesystem::is_regular_file(config.artifact_path))
                throw std::invalid_argument("Artifact is not a regular file: " +
                                            config.artifact_path.string());
        }
    }

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
        GetLogger().info("Loaded task=", contract_.role, ", artifact=", contract_.artifact_path);
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
        GetLogger().info("Loaded config task=", contract_.role,
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
        GetLogger().info("Loaded config task=", contract_.role,
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
        GetLogger().trace("Running task=", contract_.role, " with ", inputs.size(),
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
