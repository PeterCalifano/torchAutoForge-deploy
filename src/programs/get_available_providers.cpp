/**
 * @file get_available_providers.cpp
 * @brief Report and validate ONNX Runtime execution-provider availability.
 */

#include <inference/inference_config_parsing.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>
#include <utils/logging/CLogger.h>

#include <tclap/CmdLine.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("get_available_providers", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    void PrintProviders()
    {
        const std::vector<std::string> providers =
            infer::onnxruntime::CInferenceManager_ORT::GetAvailableProviders();
        for (const std::string& provider : providers)
        {
            std::cout << provider << "\n";
        }
    }

    [[nodiscard]] std::vector<infer::EExecutionTarget> ParseRequiredTargets(const int argc,
                                                                            char** argv)
    {
        TCLAP::CmdLine command(
            "Report ONNX Runtime execution providers and optionally require backend-neutral "
            "execution targets.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        TCLAP::ValueArg<std::string> required_targets(
            "", "require-targets", "Comma-separated execution targets that must be available",
            false, "", "cpu,cuda,tensorrt", command);
        command.parse(argc, argv);
        if (!required_targets.isSet())
        {
            return {};
        }

        return infer::ParseExecutionTargetPriority(required_targets.getValue());
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
 * @brief Print available ORT providers and optionally require target capabilities.
 * @return Zero when all requested execution targets are available.
 */
int main(int argc, char** argv)
{
    try
    {
        const std::vector<infer::EExecutionTarget> required_targets =
            ParseRequiredTargets(argc, argv);
        PrintProviders();
        return RequireTargets(required_targets);
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
