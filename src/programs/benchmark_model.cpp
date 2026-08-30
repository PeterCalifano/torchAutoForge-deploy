/**
 * @file benchmark_model.cpp
 * @brief Generic model-facade inference benchmark for supported runtimes.
 *
 * The executable measures repeated `CModelFacade` inference with deterministic
 * zero-valued float inputs. It accepts model manifests or raw artifacts and
 * keeps backend selection behind the facade's runtime configuration.
 */

#include <inference/inference_config_parsing.h>
#include <inference/inference_tensor_parsing.h>
#include <inference/model_facade.h>
#include <utils/logging/CLogger.h>

#include <tclap/CmdLine.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    /** @brief Return the process-wide logger configured from the environment once. */
    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("benchmark_model", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    /** @brief Validated command-line state for one benchmark invocation. */
    struct SBenchmarkArgs
    {
        /** @brief Model manifest or raw supported artifact to load. */
        std::filesystem::path model_or_config_path{};

        /** @brief Backend-neutral runtime overrides. */
        infer::SRuntimeConfig runtime{};

        /** @brief Role assigned when loading a raw artifact. */
        infer::EModelRole role{infer::EModelRole::raw_tensor};

        /** @brief Optional concrete shape for the only model input. */
        std::vector<int64_t> single_input_shape_override{};

        /** @brief Untimed iterations used to stabilize runtime state. */
        int warmup_iterations{3};

        /** @brief Timed inference iterations used for the reported mean. */
        int iterations{20};

        /** @brief Whether explicit CLI values must replace manifest runtime policy. */
        bool runtime_overridden{false};
    };

    /**
     * @brief Declare, collect, and validate one benchmark command line.
     * @param argc Number of command-line arguments.
     * @param argv Command-line argument values owned by the process.
     * @return Validated benchmark state independent of TCLAP types.
     * @throws TCLAP::ExitException When help or version output requests an early exit.
     * @throws TCLAP::ArgException If TCLAP rejects an option or value.
     * @throws std::invalid_argument If iteration counts or semantic values are invalid.
     */
    [[nodiscard]] SBenchmarkArgs ParseArgs(const int argc, char** argv)
    {
        TCLAP::CmdLine command(
            "Benchmark supported model artifacts through CModelFacade with generated zero-valued "
            "float inputs.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        // TCLAP owns collection and help text only; semantic role, runtime, and
        // tensor-shape parsing remains in installed inference APIs.
        TCLAP::UnlabeledValueArg<std::string> model_path(
            "model",
            "Path to a model manifest or supported raw artifact",
            true,
            "",
            "model.ptafmodel|model.onnx|model.engine",
            command);
        TCLAP::ValueArg<int> iterations(
            "",
            "iterations",
            "Number of timed inference iterations",
            false,
            20,
            "positive integer",
            command);
        TCLAP::ValueArg<int> warmup(
            "",
            "warmup",
            "Number of untimed warmup iterations",
            false,
            3,
            "non-negative integer",
            command);
        TCLAP::ValueArg<std::string> targets(
            "",
            "targets",
            "Comma-separated execution-target priority",
            false,
            "",
            "cpu,cuda,tensorrt",
            command);
        TCLAP::ValueArg<int> device(
            "",
            "device",
            "Non-negative runtime device index",
            false,
            0,
            "non-negative integer",
            command);
        TCLAP::SwitchArg no_fallback(
            "",
            "no-fallback",
            "Reject runtime fallback to a lower-priority target",
            command,
            false);
        TCLAP::ValueArg<int> tensor_rt_profile(
            "",
            "trt-profile",
            "TensorRT optimization-profile index",
            false,
            0,
            "non-negative integer",
            command);
        TCLAP::ValueArg<std::string> backend(
            "",
            "backend",
            "Inference backend override",
            false,
            "",
            "auto|onnxruntime|tensorrt_engine",
            command);
        TCLAP::ValueArg<std::string> artifact(
            "",
            "artifact",
            "Model artifact override",
            false,
            "",
            "auto|onnx|tensorrt_engine",
            command);
        TCLAP::ValueArg<std::string> role(
            "",
            "role",
            "Model role for raw artifacts",
            false,
            "",
            "raw_tensor|centroiding|object_detection|custom",
            command);
        TCLAP::ValueArg<std::string> input_shape(
            "",
            "input-shape",
            "Concrete shape override for a single model input",
            false,
            "",
            "d0,d1,...",
            command);

        command.parse(argc, argv);

        SBenchmarkArgs args;
        args.model_or_config_path = model_path.getValue();
        args.iterations = iterations.getValue();
        args.warmup_iterations = warmup.getValue();

        if (args.iterations <= 0)
        {
            throw std::invalid_argument("--iterations must be positive.");
        }
        if (args.warmup_iterations < 0)
        {
            throw std::invalid_argument("--warmup must be non-negative.");
        }

        // Apply only explicit runtime overrides so manifest defaults remain authoritative.
        if (targets.isSet())
        {
            args.runtime.execution_target_priority =
                infer::ParseExecutionTargetPriority(targets.getValue());
        }
        if (device.isSet())
        {
            args.runtime.SetDeviceId(device.getValue());
        }
        if (tensor_rt_profile.isSet())
        {
            args.runtime.SetTensorRtOptimizationProfileIndex(tensor_rt_profile.getValue());
        }
        if (backend.isSet())
        {
            args.runtime.backend = infer::ParseInferenceBackendName(backend.getValue());
        }
        if (artifact.isSet())
        {
            args.runtime.artifact = infer::ParseModelArtifactName(artifact.getValue());
        }
        if (no_fallback.getValue())
        {
            args.runtime.SetAllowFallback(false);
        }
        args.runtime_overridden = targets.isSet() || device.isSet() || tensor_rt_profile.isSet() ||
                                  backend.isSet() || artifact.isSet() || no_fallback.getValue();

        if (role.isSet())
        {
            args.role = infer::ParseModelRoleName(role.getValue());
        }
        if (input_shape.isSet())
        {
            args.single_input_shape_override =
                infer::ParseTensorShape(input_shape.getValue(), "--input-shape");
        }

        return args;
    }

    /**
     * @brief Replace dynamic model dimensions with deterministic unit extents.
     * @param shape Model-declared dimensions using negative values for dynamic extents.
     * @return Concrete shape suitable for allocating a benchmark tensor.
     * @note Static dimensions are preserved exactly; each dynamic dimension becomes one.
     */
    [[nodiscard]] std::vector<int64_t> MakeConcreteShape(const std::vector<int64_t>& shape)
    {
        std::vector<int64_t> concrete_shape = shape;
        for (int64_t& dimension : concrete_shape)
        {
            if (dimension < 0)
            {
                dimension = 1;
            }
        }
        return concrete_shape;
    }

    /**
     * @brief Allocate deterministic zero-valued float inputs for a model contract.
     * @param contract Loaded role-level model contract in authoritative input order.
     * @param single_input_shape_override Optional concrete shape for a single-input model.
     * @return Owned float tensors in model input order.
     * @throws std::invalid_argument If an override is supplied for a multi-input model.
     * @throws std::runtime_error If an input dtype is not supported by this benchmark.
     * @throws std::overflow_error If a resolved tensor element count exceeds host capacity.
     */
    [[nodiscard]] std::vector<infer::SFloatTensor> MakeZeroInputs(
        const infer::SModelContract& contract,
        const std::vector<int64_t>& single_input_shape_override)
    {
        if (!single_input_shape_override.empty() && contract.inputs.size() != 1U)
        {
            throw std::invalid_argument(
                "--input-shape override is only supported for single-input models.");
        }

        std::vector<infer::SFloatTensor> inputs;
        inputs.reserve(contract.inputs.size());
        for (size_t input_index = 0U; input_index < contract.inputs.size(); ++input_index)
        {
            const infer::STensorInfo& input_info = contract.inputs[input_index];
            if (input_info.dtype != "float32")
            {
                throw std::runtime_error(
                    "benchmark_model currently supports float32 host inputs only.");
            }

            const bool use_model_shape = single_input_shape_override.empty();
            std::vector<int64_t> shape = use_model_shape
                                             ? MakeConcreteShape(input_info.shape)
                                             : single_input_shape_override;
            const size_t element_count = infer::ComputeElementCount(shape);
            inputs.emplace_back(input_info.name, std::move(shape),
                                std::vector<float>(element_count, 0.0F));
        }

        return inputs;
    }
} // namespace

/**
 * @brief Load a model-role configuration and report timed facade inference.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on successful benchmarking, nonzero on invalid input/runtime failure.
 */
int main(const int argc, char** argv)
{
    try
    {
        const SBenchmarkArgs args = ParseArgs(argc, argv);
        GetLogger().info("Loading benchmark artifact: ", args.model_or_config_path.string());

        infer::CModelFacade model;

        // Manifests retain their runtime policy unless at least one explicit
        // command-line override was supplied. Raw artifacts always require a role.
        if (args.model_or_config_path.extension() == ".ptafmodel")
        {
            if (args.runtime_overridden)
            {
                model.LoadModelConfigWithRuntimeConfig(args.model_or_config_path.string(),
                                                       args.runtime);
            }
            else
            {
                model.LoadModelConfig(args.model_or_config_path.string());
            }
        }
        else
        {
            model.LoadModelWithRoleAndRuntimeConfig(args.model_or_config_path.string(), args.role,
                                                    args.runtime);
        }

        const infer::SModelContract contract = model.GetContract();
        GetLogger().debug("Resolved role=", contract.role, ", inputs=", contract.inputs.size(),
                          ", outputs=", contract.outputs.size());
        const std::vector<infer::SFloatTensor> inputs =
            MakeZeroInputs(contract, args.single_input_shape_override);

        // Keep runtime initialization and cache effects outside the timed region.
        for (int i = 0; i < args.warmup_iterations; ++i)
        {
            static_cast<void>(model.InferFloatTensors(inputs));
        }

        const auto start_time = std::chrono::steady_clock::now();
        for (int i = 0; i < args.iterations; ++i)
        {
            static_cast<void>(model.InferFloatTensors(inputs));
        }
        const auto end_time = std::chrono::steady_clock::now();

        const auto elapsed_microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        const double avg_ms =
            static_cast<double>(elapsed_microseconds) /
            1000.0 /
            static_cast<double>(args.iterations);

        std::cout << "role=" << contract.role << '\n';
        std::cout << "backend=" << contract.backend_detail << '\n';
        std::cout << "inputs=" << contract.inputs.size() << '\n';
        std::cout << "outputs=" << contract.outputs.size() << '\n';
        std::cout << "warmup_iterations=" << args.warmup_iterations << '\n';
        std::cout << "iterations=" << args.iterations << '\n';
        std::cout << "avg_ms=" << avg_ms << '\n';
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

    return 0;
}
