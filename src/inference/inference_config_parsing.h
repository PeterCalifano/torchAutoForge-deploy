/**
 * @file inference_config_parsing.h
 * @brief Boundary parsers for enum-backed runtime and model configuration.
 */

#pragma once

#include <inference/inference_common.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptafdeploy::inference
{
    /**
     * @brief Trim ASCII whitespace around a manifest/CLI token.
     * @param value Source token.
     * @return Trimmed token, or an empty string for whitespace-only input.
     */
    [[nodiscard]] inline std::string TrimConfigToken(const std::string &value)
    {
        const auto begin = std::find_if_not(value.begin(),
                                            value.end(),
                                            [](const unsigned char c)
                                            {
                                                return std::isspace(c) != 0;
                                            });
        const auto end = std::find_if_not(value.rbegin(),
                                          value.rend(),
                                          [](const unsigned char c)
                                          {
                                              return std::isspace(c) != 0;
                                          }).base();
        if (begin >= end)
        {
            return {};
        }
        return std::string(begin, end);
    }

    /**
     * @brief Normalize a token for enum parsing.
     * @param token Source token.
     * @return Trimmed lowercase token with dashes replaced by underscores.
     */
    [[nodiscard]] inline std::string NormalizeConfigToken(std::string token)
    {
        token = TrimConfigToken(token);
        std::transform(token.begin(),
                       token.end(),
                       token.begin(),
                       [](const unsigned char c)
                       {
                           return static_cast<char>(std::tolower(c));
                       });
        std::replace(token.begin(), token.end(), '-', '_');
        return token;
    }

    /**
     * @brief Parse a manifest boolean value with strict error reporting.
     * @param value Boolean text.
     * @param key Config key included in error messages.
     * @return Parsed boolean.
     * @throws std::invalid_argument When the value is not a recognized boolean.
     */
    [[nodiscard]] inline bool ParseConfigBool(const std::string &value,
                                              const std::string &key)
    {
        const std::string normalized_value = NormalizeConfigToken(value);
        if (normalized_value == "true" || normalized_value == "on" ||
            normalized_value == "1" || normalized_value == "yes")
        {
            return true;
        }
        if (normalized_value == "false" || normalized_value == "off" ||
            normalized_value == "0" || normalized_value == "no")
        {
            return false;
        }

        throw std::invalid_argument("Invalid boolean value for model config key: " + key);
    }

    /**
     * @brief Parse an integer manifest value and reject trailing characters.
     * @param value Integer text.
     * @param key Config key included in error messages.
     * @return Parsed integer.
     * @throws std::invalid_argument When parsing fails or leaves characters.
     */
    [[nodiscard]] inline int ParseConfigInt(const std::string &value,
                                            const std::string &key)
    {
        try
        {
            size_t parsed_chars = 0;
            const std::string trimmed_value = TrimConfigToken(value);
            const int parsed_value = std::stoi(trimmed_value, &parsed_chars);
            if (parsed_chars != trimmed_value.size())
            {
                throw std::invalid_argument("trailing characters");
            }
            return parsed_value;
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument("Invalid integer value for model config key: " + key);
        }
    }

    /**
     * @brief Parse a backend enum from config text.
     * @param value Backend name or recognized alias.
     * @return Backend enum.
     * @throws std::invalid_argument When the name is unsupported.
     */
    [[nodiscard]] inline EInferenceBackend ParseInferenceBackendName(const std::string &value)
    {
        const std::string backend = NormalizeConfigToken(value);
        if (backend == "auto" || backend == "auto_backend")
        {
            return EInferenceBackend::auto_backend;
        }
        if (backend == "onnxruntime" || backend == "ort")
        {
            return EInferenceBackend::onnxruntime;
        }
        if (backend == "tensorrt" || backend == "tensorrt_engine" || backend == "engine")
        {
            return EInferenceBackend::tensorrt_engine;
        }
        throw std::invalid_argument("Unsupported runtime backend: " + value);
    }

    /**
     * @brief Parse an artifact enum from config text.
     * @param value Artifact name or recognized alias.
     * @return Artifact enum.
     * @throws std::invalid_argument When the name is unsupported.
     */
    [[nodiscard]] inline EModelArtifact ParseModelArtifactName(const std::string &value)
    {
        const std::string artifact = NormalizeConfigToken(value);
        if (artifact == "auto" || artifact == "auto_artifact")
        {
            return EModelArtifact::auto_artifact;
        }
        if (artifact == "onnx")
        {
            return EModelArtifact::onnx;
        }
        if (artifact == "tensorrt" || artifact == "tensorrt_engine" ||
            artifact == "engine" || artifact == "plan")
        {
            return EModelArtifact::tensorrt_engine;
        }
        throw std::invalid_argument("Unsupported model artifact type: " + value);
    }

    /**
     * @brief Parse an execution-target enum from config text.
     * @param value Target name or recognized alias.
     * @return Execution-target enum.
     * @throws std::invalid_argument When the name is unsupported.
     */
    [[nodiscard]] inline EExecutionTarget ParseExecutionTargetName(const std::string &value)
    {
        const std::string target = NormalizeConfigToken(value);
        if (target == "cpu")
        {
            return EExecutionTarget::cpu;
        }
        if (target == "cuda" || target == "gpu")
        {
            return EExecutionTarget::cuda;
        }
        if (target == "tensorrt")
        {
            return EExecutionTarget::tensorrt;
        }
        throw std::invalid_argument("Unsupported execution target: " + value);
    }

    /**
     * @brief Parse comma-separated execution-target priority.
     * @param value Comma-separated target names.
     * @return Priority-ordered target enums; empty tokens are ignored.
     * @throws std::invalid_argument When any non-empty target is unsupported.
     */
    [[nodiscard]] inline std::vector<EExecutionTarget> ParseExecutionTargetPriority(const std::string &value)
    {
        std::vector<EExecutionTarget> targets;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = TrimConfigToken(token);
            if (!token.empty())
            {
                targets.push_back(ParseExecutionTargetName(token));
            }
        }
        return targets;
    }

    /**
     * @brief Parse a recognized model role from config text.
     * @param value Model-role name.
     * @return Recognized role enum; recognition does not imply adapter support.
     * @throws std::invalid_argument When the role name is unsupported.
     */
    [[nodiscard]] inline EModelRole ParseModelRoleName(const std::string &value)
    {
        const std::string role = NormalizeConfigToken(value);
        if (role == "raw_tensor")
        {
            return EModelRole::raw_tensor;
        }
        if (role == "centroiding")
        {
            return EModelRole::centroiding;
        }
        if (role == "object_detection")
        {
            return EModelRole::object_detection;
        }
        if (role == "feature_matching")
        {
            return EModelRole::feature_matching;
        }
        if (role == "tracking")
        {
            return EModelRole::tracking;
        }
        if (role == "optical_flow")
        {
            return EModelRole::optical_flow;
        }
        if (role == "custom")
        {
            return EModelRole::custom;
        }
        throw std::invalid_argument("Unsupported model role: " + value);
    }
}
