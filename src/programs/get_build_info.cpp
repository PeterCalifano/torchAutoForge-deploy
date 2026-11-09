/**
 * @file get_build_info.cpp
 * @brief Report which inference backends this build provides and validate requirements.
 */

#include <inference/inference_build_info.h>
#include <inference/inference_config_parsing.h>
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

    /** @brief Validated requirement lists for one invocation. */
    struct SArguments
    {
        std::vector<infer::EExecutionTarget> required_targets{};
        std::vector<infer::EInferenceBackend> required_backends{};
    };

    /** @brief Return the process-wide logger configured from the environment once. */
    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("get_build_info", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /**
     * @brief Parse a comma-separated list of concrete backends.
     * @param text Backend names or recognized aliases.
     * @return Requested backends in command-line order.
     * @throws std::invalid_argument When a name is unknown or is `auto`.
     */
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

    /**
     * @brief Declare and collect one command line.
     * @param argc Number of command-line arguments.
     * @param argv Command-line argument values owned by the process.
     * @return Validated requirement lists independent of TCLAP types.
     * @throws TCLAP::ExitException When help or version output requests an early exit.
     * @throws TCLAP::ArgException If TCLAP rejects an option or value.
     * @throws std::invalid_argument If a backend or execution target is not recognized.
     */
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

    /**
     * @brief Find the report entry of one backend.
     * @param report Report of the current build.
     * @param backend Concrete backend to find.
     * @return Matching entry, or nullptr when the report has none.
     */
    [[nodiscard]] const infer::SBackendBuildInfo*
    FindBackend(const std::vector<infer::SBackendBuildInfo>& report,
                const infer::EInferenceBackend backend)
    {
        const auto found = std::find_if(report.begin(), report.end(), [backend](const auto& info)
                                        { return info.backend == backend; });
        return found == report.end() ? nullptr : &*found;
    }

    /**
     * @brief Require each listed backend to be compiled in and usable on this host.
     * @param report Report of the current build.
     * @param required Concrete backends that must be available.
     * @return Zero when all are available, otherwise two.
     */
    [[nodiscard]] int RequireBackends(const std::vector<infer::SBackendBuildInfo>& report,
                                      const std::vector<infer::EInferenceBackend>& required)
    {
        int status = 0;
        for (const infer::EInferenceBackend backend : required)
        {
            const infer::SBackendBuildInfo* info = FindBackend(report, backend);
            if (info != nullptr && info->compiled_in && info->runtime_available)
            {
                continue;
            }

            const std::string reason = (info != nullptr && !info->note.empty())
                                           ? info->note
                                           : "no usable runtime on this host";
            GetLogger().error("Required backend unavailable: ", infer::ToString(backend), " (",
                              reason, ")");
            status = 2;
        }

        return status;
    }

    /**
     * @brief Require each listed execution target to be exposed by ONNX Runtime.
     * @param report Report of the current build.
     * @param required_targets Execution targets that must be available.
     * @return Zero when ORT is built and all targets are available, otherwise two.
     */
    [[nodiscard]] int RequireTargets(const std::vector<infer::SBackendBuildInfo>& report,
                                     const std::vector<infer::EExecutionTarget>& required_targets)
    {
        if (required_targets.empty())
        {
            return 0;
        }

        // These targets are checked against ORT providers, so ORT must be built.
        const infer::SBackendBuildInfo* ort =
            FindBackend(report, infer::EInferenceBackend::onnxruntime);
        if (ort == nullptr || !ort->compiled_in)
        {
            GetLogger().error("--require-targets needs the ONNX Runtime backend, which is not "
                              "built: ",
                              infer::kOrtNotBuiltHint);
            return 2;
        }

        std::vector<infer::EExecutionTarget> missing_targets;
        for (const infer::EExecutionTarget target : required_targets)
        {
            if (!infer::IsOrtExecutionTargetAvailable(target))
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

        const std::vector<infer::SBackendBuildInfo> report = infer::GetBackendBuildInfo();
        std::cout << infer::FormatBackendBuildInfo(report);

        const int backend_status = RequireBackends(report, arguments.required_backends);
        const int target_status = RequireTargets(report, arguments.required_targets);
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
