/**
 * @file benchmark_model.cpp
 * @brief Generic model-facade inference benchmark for supported runtimes.
 */

#include <inference/inference_config_parsing.h>
#include <inference/model_facade.h>
#include <utils/logging/CLogger.h>

#include <tclap/CmdLine.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
    namespace infer = ptafdeploy::inference;
    namespace logging = ptafdeploy::logging;

    [[nodiscard]] logging::CLogger& GetLogger()
    {
        static logging::CLogger logger("benchmark_model", logging::ELogLevel::Info,
                                       logging::ELogColorMode::Disabled, std::clog, std::clog);
        static const bool environment_applied = logger.setLevelFromEnvironment();
        static_cast<void>(environment_applied);
        return logger;
    }

    struct SBenchmarkArgs
    {
        std::filesystem::path model_or_config_path{};
        infer::SRuntimeConfig runtime{};
        infer::EModelRole role{infer::EModelRole::raw_tensor};
        std::vector<int64_t> single_input_shape_override{};
        int warmup_iterations{3};
        int iterations{20};
        bool runtime_overridden{false};
    };

    [[nodiscard]] int ParseInt(const std::string& value, const std::string& flag_name)
    {
        try
        {
            size_t parsed_chars = 0;
            const int parsed_value = std::stoi(value, &parsed_chars);
            if (parsed_chars != value.size())
            {
                throw std::invalid_argument("trailing characters");
            }
            return parsed_value;
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("Invalid integer for " + flag_name + ": " + value);
        }
    }

    [[nodiscard]] std::vector<int64_t> ParseShape(const std::string& value)
    {
        std::vector<int64_t> shape;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            if (!token.empty())
            {
                shape.push_back(ParseInt(token, "--input-shape"));
            }
        }
        if (shape.empty())
        {
            throw std::invalid_argument("--input-shape must contain at least one dimension.");
        }
        return shape;
    }

    [[nodiscard]] SBenchmarkArgs ParseArgs(const int argc, char** argv)
    {
        TCLAP::CmdLine command(
            "Benchmark supported model artifacts through CModelFacade with generated zero-valued "
            "float inputs.",
            ' ', PTAFDEPLOY_CLI_VERSION);
        command.setExceptionHandling(false);

        TCLAP::UnlabeledValueArg<std::string> model_path(
            "model", "Path to a model manifest or supported raw artifact", true, "",
            "model.ptafmodel|model.onnx|model.engine", command);
        TCLAP::ValueArg<int> iterations(
            "", "iterations", "Number of timed inference iterations", false, 20,
            "positive integer", command);
        TCLAP::ValueArg<int> warmup(
            "", "warmup", "Number of untimed warmup iterations", false, 3,
            "non-negative integer", command);
        TCLAP::ValueArg<std::string> targets(
            "", "targets", "Comma-separated execution-target priority", false, "",
            "cpu,cuda,tensorrt", command);
        TCLAP::ValueArg<int> device(
            "", "device", "Non-negative runtime device index", false, 0,
            "non-negative integer", command);
        TCLAP::SwitchArg no_fallback(
            "", "no-fallback", "Reject runtime fallback to a lower-priority target", command,
            false);
        TCLAP::ValueArg<int> tensor_rt_profile(
            "", "trt-profile", "TensorRT optimization-profile index", false, 0,
            "non-negative integer", command);
        TCLAP::ValueArg<std::string> backend(
            "", "backend", "Inference backend override", false, "",
            "auto|onnxruntime|tensorrt_engine", command);
        TCLAP::ValueArg<std::string> artifact(
            "", "artifact", "Model artifact override", false, "",
            "auto|onnx|tensorrt_engine", command);
        TCLAP::ValueArg<std::string> role(
            "", "role", "Model role for raw artifacts", false, "",
            "raw_tensor|centroiding|object_detection|custom", command);
        TCLAP::ValueArg<std::string> input_shape(
            "", "input-shape", "Concrete shape override for a single model input", false, "",
            "d0,d1,...", command);

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
            args.single_input_shape_override = ParseShape(input_shape.getValue());
        }

        return args;
    }

    [[nodiscard]] std::vector<int64_t> MakeConcreteShape(const std::vector<int64_t>& shape)
    {
        std::vector<int64_t> concrete_shape = shape;
        for (int64_t& dim : concrete_shape)
        {
            if (dim < 0)
            {
                dim = 1;
            }
        }
        return concrete_shape;
    }

    [[nodiscard]] std::vector<infer::SFloatTensor>
    MakeZeroInputs(const infer::SModelContract& contract,
                   const std::vector<int64_t>& single_input_shape_override)
    {
        if (!single_input_shape_override.empty() && contract.inputs.size() != 1U)
        {
            throw std::invalid_argument(
                "--input-shape override is only supported for single-input models.");
        }

        std::vector<infer::SFloatTensor> inputs;
        inputs.reserve(contract.inputs.size());
        for (size_t i = 0; i < contract.inputs.size(); ++i)
        {
            const infer::STensorInfo& input_info = contract.inputs[i];
            if (input_info.dtype != "float32")
            {
                throw std::runtime_error(
                    "benchmark_model currently supports float32 host inputs only.");
            }

            std::vector<int64_t> shape = single_input_shape_override.empty()
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
 * @return Zero on successful benchmarking, nonzero on invalid input/runtime failure.
 */
int main(int argc, char** argv)
{
    try
    {
        const SBenchmarkArgs args = ParseArgs(argc, argv);
        GetLogger().info("Loading benchmark artifact: ", args.model_or_config_path.string());

        infer::CModelFacade model;
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

        for (int i = 0; i < args.warmup_iterations; ++i)
        {
            static_cast<void>(model.InferFloatTensors(inputs));
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < args.iterations; ++i)
        {
            static_cast<void>(model.InferFloatTensors(inputs));
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const double avg_ms =
            static_cast<double>(elapsed_us) / 1000.0 / static_cast<double>(args.iterations);

        std::cout << "role=" << contract.role << "\n";
        std::cout << "backend=" << contract.backend_detail << "\n";
        std::cout << "inputs=" << contract.inputs.size() << "\n";
        std::cout << "outputs=" << contract.outputs.size() << "\n";
        std::cout << "warmup_iterations=" << args.warmup_iterations << "\n";
        std::cout << "iterations=" << args.iterations << "\n";
        std::cout << "avg_ms=" << avg_ms << "\n";
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
    catch (const std::exception& e)
    {
        GetLogger().error(e.what());
        return 1;
    }

    return 0;
}
