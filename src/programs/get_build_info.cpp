/**
 * @file get_build_info.cpp
 * @brief Report which inference backends this build provides and validate requirements.
 */

#include <inference/inference_build_info.h>
#include <inference/inference_config_parsing.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <utils/logging/CLogger.h>

#include <tclap/CmdLine.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    struct SArguments
    {
        std::vector<infer::EExecutionTarget> required_targets{};
        std::vector<infer::EInferenceBackend> required_backends{};
    };

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("get_build_info", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    [[nodiscard]] std::vector<infer::EInferenceBackend> ParseBackendList(const std::string& text)
    {
        std::vector<infer::EInferenceBackend> backends;
        std::istringstream stream(text);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            if (token.empty())
            {
                continue;
            }
            const infer::EInferenceBackend backend = infer::ParseInferenceBackendName(token);
            if (backend == infer::EInferenceBackend::auto_backend)
            {
                throw std::invalid_argument(
                    "--require-backends expects concrete backends, not auto.");
            }
            backends.push_back(backend);
        }
        return backends;
    }

    [[nodiscard]] SArguments ParseArguments(const int argc, char** argv)
    {
        TCLAP::CmdLine command(
            "Report the inference backends compiled into this build, their runtime versions "
            "and ONNX Runtime execution providers. Optionally fail when required backends or "
            "execution targets are missing.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        TCLAP::ValueArg<std::string> required_backends(
            "", "require-backends",
            "Comma-separated backends that must be compiled in and runtime-available", false, "",
            "onnxruntime,tensorrt_engine", command);
        TCLAP::ValueArg<std::string> required_targets(
            "", "require-targets",
            "Comma-separated ONNX Runtime execution targets that must be available", false, "",
            "cpu,cuda,tensorrt", command);
        command.parse(argc, argv);

        SArguments arguments;
        if (required_backends.isSet())
        {
            arguments.required_backends = ParseBackendList(required_backends.getValue());
        }
        if (required_targets.isSet())
        {
            arguments.required_targets =
                infer::ParseExecutionTargetPriority(required_targets.getValue());
        }
        return arguments;
    }

    [[nodiscard]] int RequireBackends(const std::vector<infer::EInferenceBackend>& required)
    {
        const std::vector<infer::SBackendBuildInfo> report = infer::GetBackendBuildInfo();
        int status = 0;
        for (const infer::EInferenceBackend backend : required)
        {
            const std::string name = infer::ToString(backend);
            const auto found = std::find_if(report.begin(), report.end(),
                                            [&](const auto& info) { return info.name == name; });
            if (found == report.end() || !found->compiled_in || !found->runtime_available)
            {
                GetLogger().error("Required backend unavailable: ", name);
                status = 2;
            }
        }
        return status;
    }

    [[nodiscard]] int RequireTargets(const std::vector<infer::EExecutionTarget>& required_targets)
    {
        if (required_targets.empty())
        {
            return 0;
        }

        std::vector<infer::EExecutionTarget> missing_targets;
        for (const infer::EExecutionTarget target : required_targets)
        {
            if (!infer::onnxruntime::CInferenceManager_ORT::IsExecutionTargetAvailable(target))
            {
                missing_targets.push_back(target);
            }
        }

        if (!missing_targets.empty())
        {
            GetLogger().error("Missing ORT execution targets: ",
                              infer::JoinExecutionTargets(missing_targets));
            return 2;
        }

        std::cout << "required_targets_available=" << infer::JoinExecutionTargets(required_targets)
                  << "\n";
        return 0;
    }
} // namespace

/**
 * @brief Print the backend build report and optionally enforce requirements.
 * @return Zero when all required backends and execution targets are available.
 */
int main(int argc, char** argv)
{
    try
    {
        const SArguments arguments = ParseArguments(argc, argv);
        std::cout << infer::FormatBackendBuildInfo();
        const int backend_status = RequireBackends(arguments.required_backends);
        const int target_status = RequireTargets(arguments.required_targets);
        return backend_status != 0 ? backend_status : target_status;
    }
    catch (const TCLAP::ExitException& exit_request)
    {
        return exit_request.getExitStatus();
    }
    catch (const TCLAP::ArgException& error)
    {
        GetLogger().error("Invalid command line: ", error.error(), " for ", error.argId());
        return 1;
    }
    catch (const std::exception& error)
    {
        GetLogger().error(error.what());
        return 1;
    }
}
