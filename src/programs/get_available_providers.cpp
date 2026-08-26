/**
 * @file get_available_providers.cpp
 * @brief Report and validate ONNX Runtime execution-provider availability.
 */

#include <inference/inference_config_parsing.h>
#include <inference/onnx_runtime/onnxruntime_inference_tools.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utils/logging/CLogger.h>
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

    void PrintUsage(const char* program_name)
    {
        std::cerr << "Usage: " << program_name << " [--require-targets cpu,cuda,tensorrt]\n";
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
        std::vector<infer::EExecutionTarget> required_targets;
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--require-targets")
            {
                if (i + 1 >= argc)
                {
                    throw std::invalid_argument("Missing value for --require-targets.");
                }
                required_targets = infer::ParseExecutionTargetPriority(argv[++i]);
            }
            else if (flag == "--help" || flag == "-h")
            {
                PrintUsage(argv[0]);
                std::exit(0);
            }
            else
            {
                throw std::invalid_argument("Unknown flag: " + flag);
            }
        }
        return required_targets;
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
    catch (const std::exception& error)
    {
        GetLogger().error(error.what());
        return 1;
    }
}
